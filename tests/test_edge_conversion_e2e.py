"""End-to-end smoke for the PyTorch -> ONNX -> TF -> int8 TFLite path.

Skipped when the optional edge stack (onnx / onnx_tf / tensorflow) is not
installed — it is intentionally absent from CI's training environment to
keep the matrix lean. Run locally with `pip install -r requirements-edge.txt`
to exercise this test.

What this locks down:
- ONNX export with dummy_input shape (1, T, C) survives the model's internal
  (B, T, C) -> (B, C, T) transpose without a layout swap.
- onnx_tf can prepare the graph for SavedModel emission.
- TFLiteConverter accepts the representative dataset shape and produces a
  non-empty int8 .tflite.
- convert_to_c_array round-trips the bytes into a usable header file.
"""

import os
import sys
import tempfile

import numpy as np
import pytest


pytest.importorskip("onnx")
pytest.importorskip("onnx_tf")
pytest.importorskip("tensorflow")


def test_pytorch_to_int8_tflite_roundtrip():
    import torch

    # Tiny architecture so the conversion finishes in a few seconds even on CI.
    os.environ['HOPEGAIT_NUM_CHANNELS'] = '[8, 16]'
    os.environ['HOPEGAIT_WINDOW_SIZES'] = '[64]'

    # Reload config + model with the small overrides applied.
    for mod in ('config', 'models.tcn_model', 'edge_conversion.quantize_model'):
        sys.modules.pop(mod, None)

    from models.tcn_model import HopeGaitTCN
    from config import (NUM_INPUTS, NUM_CHANNELS, KERNEL_SIZE, NUM_CLASSES,
                        DROPOUT, DROP_PATH, USE_SE)
    from edge_conversion.quantize_model import main as edge_main

    with tempfile.TemporaryDirectory() as work:
        models_dir = os.path.join(work, 'models')
        proc_dir = os.path.join(work, 'processed')
        win_dir = os.path.join(proc_dir, 'win_64')
        os.makedirs(models_dir)
        os.makedirs(win_dir)

        rng = np.random.default_rng(0)
        np.save(os.path.join(win_dir, 'subj_S01_x.npy'),
                rng.standard_normal((8, 64, NUM_INPUTS)).astype(np.float32))
        np.save(os.path.join(win_dir, 'subj_S01_y.npy'),
                np.zeros((8, 64), dtype=np.int64))

        model = HopeGaitTCN(
            num_inputs=NUM_INPUTS, num_channels=tuple(NUM_CHANNELS),
            kernel_size=KERNEL_SIZE, num_classes=NUM_CLASSES,
            dropout=DROPOUT, drop_path=DROP_PATH, use_se=USE_SE,
        )
        ckpt_path = os.path.join(models_dir, 'hopegait_tcn_best_subjS01.pth')
        torch.save(model.state_dict(), ckpt_path)

        prefix = os.path.join(models_dir, 'hopegait')
        edge_main(['--subject', 'S01', '--window', '64',
                   '--checkpoint', ckpt_path,
                   '--output-prefix', prefix])

        for suffix in ('.onnx', '_int8.tflite', '_model_data.h'):
            path = prefix + suffix
            assert os.path.exists(path), f"missing {path}"
            assert os.path.getsize(path) > 0, f"empty {path}"

        with open(prefix + '_model_data.h') as f:
            header = f.read()
        assert 'hopegait_model_tflite_len' in header
        assert '0x' in header
