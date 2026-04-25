"""End-to-end pipeline smoke test on synthetic data.

Generates a tiny fake processed/win_<W>/ tree with 4 fake subjects, then runs
training for 2 epochs and evaluates. Used by CI to catch regressions in the
data shapes, model, training loop, and post-processing without needing the
real Stanford dataset.

Run:
    python scripts/smoke_train.py
"""

import os
import sys
import shutil
import tempfile
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)


def synth_fold(out_dir, subject, n_files=2, n_windows=24, win_size=64, seed=0):
    rng = np.random.default_rng(seed)
    for fid in range(n_files):
        # Walking-like: low-frequency sine on linear_acc.
        t = np.arange(win_size) / 100.0
        base = np.sin(2 * np.pi * 1.5 * t)[None, :, None]
        X = rng.standard_normal((n_windows, win_size, 9)).astype(np.float32) * 0.2
        X[:, :, 0:3] += base.astype(np.float32)
        # Half the windows are "freezing" (higher-frequency content).
        is_freeze = rng.integers(0, 2, size=n_windows).astype(np.int64)
        for i, f in enumerate(is_freeze):
            if f:
                X[i, :, 0:3] += np.sin(2 * np.pi * 5.0 * t)[:, None].astype(np.float32) * 0.5
        y = np.broadcast_to(is_freeze[:, None], (n_windows, win_size)).copy()
        np.save(os.path.join(out_dir, f'subj_{subject}_run{fid}_x.npy'), X)
        np.save(os.path.join(out_dir, f'subj_{subject}_run{fid}_y.npy'), y)


def main():
    work = tempfile.mkdtemp(prefix='hopegait_smoke_')
    processed = os.path.join(work, 'processed')
    models = os.path.join(work, 'models')
    win = 64
    win_dir = os.path.join(processed, f'win_{win}')
    os.makedirs(win_dir)
    for i, subj in enumerate(('A', 'B', 'C', 'D')):
        synth_fold(win_dir, subj, seed=i)

    os.environ['HOPEGAIT_PROCESSED_DATA_DIR'] = processed
    os.environ['HOPEGAIT_MODELS_DIR'] = models
    os.environ['HOPEGAIT_WINDOW_SIZES'] = '[64]'
    os.environ['HOPEGAIT_EPOCHS'] = '2'
    os.environ['HOPEGAIT_BATCH_SIZE'] = '8'
    os.environ['HOPEGAIT_NUM_CHANNELS'] = '[8, 16]'
    os.environ['HOPEGAIT_USE_AMP'] = 'false'
    os.environ['HOPEGAIT_DEVICE'] = 'cpu'

    # Importing after env vars are set so config.py picks them up.
    from training.train import main as train_main
    from training.evaluate import main as eval_main

    print(f"Smoke test workdir: {work}")
    # Reset argv so train.py's argparse doesn't pick up pytest/CI flags.
    sys.argv = [sys.argv[0]]
    train_main()
    eval_main()
    print("Smoke test OK.")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
