"""LOSO evaluation.

Reports three operating points:
  - @0.5: naive threshold (still useful as a sanity baseline).
  - @fold-Youden: each fold uses the threshold chosen on its own inner val
    set during training. Stored in the fold's meta JSON. This is the honest
    real-time number — no peeking at the test set.
  - @post-processed: per-recording smoothing + hysteresis around the
    fold-Youden threshold. This is the number that approximates what the
    cueing belt would actually deliver.

Sample-level metrics (sensitivity, specificity, F1, MCC, PR-AUC, ROC-AUC)
report per-prediction performance. Event-level metrics — episode detection
rate, detection latency, and false alarms per hour — describe what the
patient actually experiences from the cueing belt: did it catch each
freezing episode and how often did it cry wolf during normal walking.
"""

import os
import sys
import json
import numpy as np
import torch

from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             average_precision_score, roc_auc_score)

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import (WINDOW_SIZES, PROCESSED_DATA_DIR, MODELS_DIR, BATCH_SIZE,
                    NUM_CHANNELS, KERNEL_SIZE, DROPOUT, NUM_INPUTS, NUM_CLASSES, SEED,
                    DROP_PATH, USE_SE, SMOOTH_WINDOW, HYSTERESIS_LOW, HYSTERESIS_HIGH,
                    SAMPLING_RATE, WINDOW_OVERLAP)
from data_pipeline.dataset import create_loso_dataloaders, get_all_subjects
from data_pipeline.dsp import RobustScaler
from models.tcn_model import HopeGaitTCN
from inference.postprocess import postprocess_predictions


def _collect_probs(model, loader, device):
    probs, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y_dense in loader:
            x = x.to(device)
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            # y_dense is (B, T) — last column is the causal target.
            targets.append(y_dense[:, -1].numpy() if y_dense.dim() == 2 else y_dense.numpy())
    if not probs:
        return np.array([]), np.array([])
    return np.concatenate(probs), np.concatenate(targets)


def _metrics_from_preds(targets, preds, threshold):
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = f1_score(targets, preds, zero_division=0)
    mcc_defined = (tp + fn) > 0 and (tn + fp) > 0 and (tp + fp) > 0 and (tn + fn) > 0
    mcc = matthews_corrcoef(targets, preds) if mcc_defined else 0.0
    return {'threshold': float(threshold),
            'sensitivity': float(sens), 'specificity': float(spec),
            'f1': float(f1), 'mcc': float(mcc),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}


def _metrics_at_threshold(targets, probs, thr):
    preds = (probs >= thr).astype(np.int64)
    return _metrics_from_preds(targets, preds, thr)


def _find_runs(arr, value):
    """Indices of contiguous runs of `value` in 1-D array. Returns list of
    (start, end) inclusive index pairs.
    """
    arr = np.asarray(arr).astype(np.int64)
    runs = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] == value:
            j = i
            while j < n and arr[j] == value:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def event_level_metrics(targets, preds, prediction_rate_hz=1.0):
    """Episode-level metrics for streaming FoG detection.

    Each contiguous run of 1s in `targets` is one FoG episode. An episode
    counts as detected if any positive prediction lands inside [start, end];
    detection latency is the index of the first such positive divided by
    `prediction_rate_hz`. A predicted run not overlapping any target episode
    is a false alarm, normalized to alarms per non-FoG hour.

    Returned dict keys: `n_episodes`, `episode_detection_rate`,
    `mean_detection_latency_s`, `median_detection_latency_s`, `false_alarms`,
    `false_alarms_per_hour`.
    """
    targets = np.asarray(targets).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    n = len(targets)
    if n == 0 or len(preds) != n:
        return {
            'n_episodes': 0,
            'episode_detection_rate': 0.0,
            'mean_detection_latency_s': None,
            'median_detection_latency_s': None,
            'false_alarms': 0,
            'false_alarms_per_hour': None,
        }

    target_episodes = _find_runs(targets, 1)
    pred_alarms = _find_runs(preds, 1)

    detected = 0
    latencies = []
    for start, end in target_episodes:
        # First positive prediction inside the episode counts as detection.
        # Any positive *before* the episode start is a false alarm, not a
        # detection — we only look in [start, end].
        slice_preds = preds[start:end + 1]
        hits = np.where(slice_preds == 1)[0]
        if hits.size > 0:
            detected += 1
            latencies.append(int(hits[0]))

    false_alarms = 0
    for ps, pe in pred_alarms:
        # Overlap = not (pe < target_start OR ps > target_end). Check all
        # target episodes; a pred run overlapping any target is not an alarm.
        overlapped = any(not (pe < ts or ps > te) for ts, te in target_episodes)
        if not overlapped:
            false_alarms += 1

    n_episodes = len(target_episodes)
    detection_rate = detected / n_episodes if n_episodes > 0 else 0.0
    mean_lat_s = float(np.mean(latencies)) / prediction_rate_hz if latencies else None
    median_lat_s = float(np.median(latencies)) / prediction_rate_hz if latencies else None

    non_fog_samples = int((targets == 0).sum())
    non_fog_hours = non_fog_samples / prediction_rate_hz / 3600.0
    fa_per_hour = false_alarms / non_fog_hours if non_fog_hours > 0 else None

    return {
        'n_episodes': int(n_episodes),
        'episode_detection_rate': float(detection_rate),
        'mean_detection_latency_s': mean_lat_s,
        'median_detection_latency_s': median_lat_s,
        'false_alarms': int(false_alarms),
        'false_alarms_per_hour': fa_per_hour,
    }


def _prediction_rate_hz(seq_length):
    """Predictions arrive at fs / step where step = window * (1 - overlap)."""
    step_samples = max(1, int(seq_length * (1.0 - WINDOW_OVERLAP)))
    return float(SAMPLING_RATE) / step_samples


def _load_fold_threshold(meta_path, fallback=0.5):
    if not os.path.exists(meta_path):
        return fallback
    with open(meta_path) as f:
        meta = json.load(f)
    return float(meta.get('val_threshold', fallback))


def evaluate_fold(test_subject, seq_length):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_dir = os.path.join(MODELS_DIR, f'win_{seq_length}')
    model_path = os.path.join(target_dir, f'hopegait_tcn_best_subj{test_subject}.pth')
    scaler_path = os.path.join(target_dir, f'scaler_subj{test_subject}.npz')
    meta_path = os.path.join(target_dir, f'fold_meta_subj{test_subject}.json')
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None

    data_dir = os.path.join(PROCESSED_DATA_DIR, f'win_{seq_length}')
    scaler = RobustScaler.load(scaler_path)
    _, _, test_loader, _, meta = create_loso_dataloaders(
        data_dir, test_subject=test_subject, batch_size=BATCH_SIZE,
        scaler=scaler, augment_train=False, seed=SEED)

    model = HopeGaitTCN(num_inputs=NUM_INPUTS, num_channels=NUM_CHANNELS,
                        kernel_size=KERNEL_SIZE, num_classes=NUM_CLASSES,
                        dropout=DROPOUT, drop_path=DROP_PATH, use_se=USE_SE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    probs, targets = _collect_probs(model, test_loader, device)
    fold_threshold = _load_fold_threshold(meta_path)
    return probs, targets, meta, fold_threshold


def evaluate_window(seq_length):
    data_dir = os.path.join(PROCESSED_DATA_DIR, f'win_{seq_length}')
    subjects = get_all_subjects(data_dir)
    if not subjects:
        return None

    per_subject = {}
    agg_targets, agg_probs = [], []
    pp_preds_all, pp_targets_all = [], []

    band = max(0.0, HYSTERESIS_HIGH - HYSTERESIS_LOW)
    pred_rate_hz = _prediction_rate_hz(seq_length)

    for subj in subjects:
        r = evaluate_fold(subj, seq_length)
        if r is None:
            continue
        probs, targets, _, fold_thr = r
        if len(targets) == 0:
            continue

        raw = _metrics_at_threshold(targets, probs, fold_thr)

        # Per-subject post-processing: smooth + hysteresis. The state machine
        # is run once per subject to avoid leaking state across recordings.
        pp_preds, _ = postprocess_predictions(
            probs, threshold=fold_thr, smooth_window=SMOOTH_WINDOW,
            hysteresis_band=band)
        pp = _metrics_from_preds(targets, pp_preds, fold_thr)

        # Event-level: each subject's predictions form a single stream so
        # episodes are detected per-subject (no cross-subject episode bleeds).
        events_pp = event_level_metrics(targets, pp_preds, prediction_rate_hz=pred_rate_hz)

        per_subject[subj] = {
            'at_fold_threshold': raw,
            'at_postprocessed': pp,
            'events_at_postprocessed': events_pp,
        }
        agg_targets.append(targets)
        agg_probs.append(probs)
        pp_preds_all.append(pp_preds)
        pp_targets_all.append(targets)

    if not per_subject:
        return None

    t = np.concatenate(agg_targets)
    p = np.concatenate(agg_probs)
    at_05 = _metrics_at_threshold(t, p, 0.5)
    pp_preds_cat = np.concatenate(pp_preds_all)
    pp_targets_cat = np.concatenate(pp_targets_all)
    at_pp = _metrics_from_preds(pp_targets_cat, pp_preds_cat, threshold=float('nan'))

    pr_auc = float(average_precision_score(t, p)) if len(np.unique(t)) > 1 else 0.0
    roc_auc = float(roc_auc_score(t, p)) if len(np.unique(t)) > 1 else 0.0

    # Aggregate event-level metrics across all subjects' post-processed streams.
    events_agg = event_level_metrics(pp_targets_cat, pp_preds_cat,
                                     prediction_rate_hz=pred_rate_hz)

    return {
        'window': seq_length,
        'n_subjects': len(per_subject),
        'prediction_rate_hz': pred_rate_hz,
        'at_0.5': at_05,
        'at_postprocessed': at_pp,
        'events_at_postprocessed': events_agg,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'smooth_window': SMOOTH_WINDOW,
        'hysteresis_band': band,
        'per_subject': per_subject,
    }


def _print_summary(r):
    w = r['window']
    print(f"\n=== Window {w} ({r['n_subjects']} subjects, "
          f"prediction_rate={r['prediction_rate_hz']:.2f} Hz) ===")
    a = r['at_0.5']
    print(f"  @0.5      sens={a['sensitivity']*100:5.1f}%  spec={a['specificity']*100:5.1f}%  "
          f"F1={a['f1']:.3f}  MCC={a['mcc']:+.3f}")
    b = r['at_postprocessed']
    print(f"  @post-pp  sens={b['sensitivity']*100:5.1f}%  spec={b['specificity']*100:5.1f}%  "
          f"F1={b['f1']:.3f}  MCC={b['mcc']:+.3f}  (smooth={r['smooth_window']}, band={r['hysteresis_band']:.2f})")
    print(f"  PR-AUC={r['pr_auc']:.3f}  ROC-AUC={r['roc_auc']:.3f}")
    e = r['events_at_postprocessed']
    mean_lat = f"{e['mean_detection_latency_s']:.2f} s" if e['mean_detection_latency_s'] is not None else "n/a"
    fa_hr = f"{e['false_alarms_per_hour']:.2f}" if e['false_alarms_per_hour'] is not None else "n/a"
    print(f"  events    n_episodes={e['n_episodes']}  detected={e['episode_detection_rate']*100:5.1f}%  "
          f"mean_latency={mean_lat}  false_alarms/h={fa_hr}")


def main():
    summary = {}
    for seq in WINDOW_SIZES:
        r = evaluate_window(seq)
        if r is None:
            continue
        summary[f'win_{seq}'] = r
        _print_summary(r)

    if not summary:
        print("No models evaluated. Did training finish?")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    out = os.path.join(MODELS_DIR, 'evaluation_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary -> {out}")

    best = max(summary.values(), key=lambda r: r['at_postprocessed']['mcc'])
    print(f"Best window by post-processed MCC: {best['window']} "
          f"(MCC={best['at_postprocessed']['mcc']:+.3f})")


if __name__ == "__main__":
    main()
