"""LOSO evaluation.

Reports three operating points:
  - @0.5: naive threshold (still useful as a sanity baseline).
  - @fold-Youden: each fold uses the threshold chosen on its own inner val
    set during training. Stored in the fold's meta JSON. This is the honest
    real-time number — no peeking at the test set.
  - @post-processed: per-recording smoothing + hysteresis around the
    fold-Youden threshold. This is the number that approximates what the
    cueing belt would actually deliver.
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
                    DROP_PATH, USE_SE, SMOOTH_WINDOW, HYSTERESIS_LOW, HYSTERESIS_HIGH)
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

        per_subject[subj] = {'at_fold_threshold': raw, 'at_postprocessed': pp}
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

    return {
        'window': seq_length,
        'n_subjects': len(per_subject),
        'at_0.5': at_05,
        'at_postprocessed': at_pp,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'smooth_window': SMOOTH_WINDOW,
        'hysteresis_band': band,
        'per_subject': per_subject,
    }


def _print_summary(r):
    w = r['window']
    print(f"\n=== Window {w} ({r['n_subjects']} subjects) ===")
    a = r['at_0.5']
    print(f"  @0.5      sens={a['sensitivity']*100:5.1f}%  spec={a['specificity']*100:5.1f}%  "
          f"F1={a['f1']:.3f}  MCC={a['mcc']:+.3f}")
    b = r['at_postprocessed']
    print(f"  @post-pp  sens={b['sensitivity']*100:5.1f}%  spec={b['specificity']*100:5.1f}%  "
          f"F1={b['f1']:.3f}  MCC={b['mcc']:+.3f}  (smooth={r['smooth_window']}, band={r['hysteresis_band']:.2f})")
    print(f"  PR-AUC={r['pr_auc']:.3f}  ROC-AUC={r['roc_auc']:.3f}")


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
