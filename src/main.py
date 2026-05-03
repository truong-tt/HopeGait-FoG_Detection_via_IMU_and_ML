"""Brain-phase pipeline: preprocess -> FP32 train -> evaluate -> (optional) quantize."""

import os
import sys
import glob
import argparse
import subprocess

from config import WINDOW_SIZES, PROCESSED_DATA_DIR, MODELS_DIR, SRC_DIR


def is_preprocessed():
    if not os.path.exists(PROCESSED_DATA_DIR):
        return False
    for seq in WINDOW_SIZES:
        target = os.path.join(PROCESSED_DATA_DIR, f'win_{seq}')
        if not os.path.exists(target) or not glob.glob(os.path.join(target, '*.npy')):
            return False
    return True


def run_script(name, path, extra_args=None):
    print(f"\n--- Running: {name} ---")
    cmd = [sys.executable, path]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: {name} failed. Halting pipeline.", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="HopeGait pipeline.")
    parser.add_argument('--force-preprocess', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument(
        '--quantize', action='store_true',
        help='After eval, run edge_conversion/quantize_model.py to produce '
             'int8 TFLite + C header. Requires `pip install -r requirements-edge.txt`.',
    )
    parser.add_argument(
        '--subject', default=None,
        help='Subject ID for the quantization step (required with --quantize).',
    )
    args = parser.parse_args()

    if args.quantize and not args.subject:
        parser.error("--quantize requires --subject <id>.")

    preprocess_script = os.path.join(SRC_DIR, 'data_pipeline', 'preprocess.py')
    train_script = os.path.join(SRC_DIR, 'training', 'train.py')
    eval_script = os.path.join(SRC_DIR, 'training', 'evaluate.py')
    quantize_script = os.path.join(SRC_DIR, 'edge_conversion', 'quantize_model.py')

    if args.force_preprocess or not is_preprocessed():
        run_script("Data Preprocessing", preprocess_script)
    else:
        print("Processed data detected. Skipping Preprocessing. (use --force-preprocess to redo)")

    if not args.skip_train:
        run_script("TCN Training (FP32)", train_script)
    if not args.skip_eval:
        run_script("Model Evaluation", eval_script)

    if args.quantize:
        run_script(
            "Edge Quantization (int8 TFLite + C header)",
            quantize_script,
            extra_args=['--subject', args.subject, '--window', str(WINDOW_SIZES[0])],
        )
        print(f"\nEdge artifacts written under {MODELS_DIR}/ "
              f"(hopegait.onnx, hopegait_int8.tflite, hopegait_model_data.h).")
    else:
        print("\nHopeGait Brain phase complete.")
        print("Next: re-run with `--quantize --subject <id>` for the Edge phase "
              "(needs requirements-edge.txt).")


if __name__ == "__main__":
    main()
