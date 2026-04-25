"""Brain-phase pipeline: preprocess -> FP32 train -> evaluate."""

import os
import sys
import glob
import argparse
import subprocess

from config import WINDOW_SIZES, PROCESSED_DATA_DIR, SRC_DIR


def is_preprocessed():
    if not os.path.exists(PROCESSED_DATA_DIR):
        return False
    for seq in WINDOW_SIZES:
        target = os.path.join(PROCESSED_DATA_DIR, f'win_{seq}')
        if not os.path.exists(target) or not glob.glob(os.path.join(target, '*.npy')):
            return False
    return True


def run_script(name, path):
    print(f"\n--- Running: {name} ---")
    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        sys.exit(f"Error: {name} failed. Halting pipeline.")


def main():
    parser = argparse.ArgumentParser(description="HopeGait Brain-phase pipeline.")
    parser.add_argument('--force-preprocess', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    args = parser.parse_args()

    preprocess_script = os.path.join(SRC_DIR, 'data_pipeline', 'preprocess.py')
    train_script = os.path.join(SRC_DIR, 'training', 'train.py')
    eval_script = os.path.join(SRC_DIR, 'training', 'evaluate.py')

    if args.force_preprocess or not is_preprocessed():
        run_script("Data Preprocessing", preprocess_script)
    else:
        print("Processed data detected. Skipping Preprocessing. (use --force-preprocess to redo)")

    if not args.skip_train:
        run_script("TCN Training (FP32)", train_script)
    if not args.skip_eval:
        run_script("Model Evaluation", eval_script)

    print("\nHopeGait Brain phase complete.")
    print("Next: run src/edge_conversion/ scripts to quantize + export to TFLite.")


if __name__ == "__main__":
    main()
