"""Lightweight tests for src/edge_conversion/quantize_model.py.

These run without TensorFlow / onnx / onnx_tf installed. They pin:
  - the module imports cleanly (heavy deps are guarded behind _load_edge_deps)
  - the CLI surface (flags exist, --help exits 0, missing flags error out)
  - the C-array encoder round-trips bytes correctly

Anything that needs the real edge stack lives in a separate, deps-gated path
and is not exercised here.
"""

import os
import re
import subprocess
import sys
import tempfile


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPT = os.path.join(ROOT, 'src', 'edge_conversion', 'quantize_model.py')


def test_module_imports_without_edge_stack():
    """The module must import even if onnx / onnx_tf / tensorflow are missing."""
    from edge_conversion import quantize_model  # noqa: F401

    assert hasattr(quantize_model, 'main')
    assert hasattr(quantize_model, 'parse_args')
    assert hasattr(quantize_model, 'convert_to_c_array')


def test_cli_help_lists_new_flags():
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(ROOT, 'src') + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(
        [sys.executable, SCRIPT, '--help'],
        capture_output=True, text=True, env=env, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for flag in ('--subject', '--window', '--checkpoint', '--output-prefix'):
        assert flag in out, f"--help missing {flag}:\n{out}"


def test_cli_missing_subject_errors():
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(ROOT, 'src') + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(
        [sys.executable, SCRIPT],
        capture_output=True, text=True, env=env, cwd=ROOT,
    )
    assert result.returncode != 0
    assert '--subject' in result.stderr


def test_parse_args_defaults():
    from edge_conversion.quantize_model import parse_args
    from config import WINDOW_SIZES

    args = parse_args(['--subject', '7'])
    assert args.subject == '7'
    assert args.window == WINDOW_SIZES[0]
    assert args.checkpoint is None
    assert args.output_prefix is None


def test_convert_to_c_array_round_trip():
    from edge_conversion.quantize_model import convert_to_c_array

    payload = bytes(range(40))  # 40 bytes -> 4 lines of 12 + 1 line of 4
    with tempfile.TemporaryDirectory() as d:
        tflite_path = os.path.join(d, 'tiny.tflite')
        header_path = os.path.join(d, 'model_data.h')
        with open(tflite_path, 'wb') as f:
            f.write(payload)
        convert_to_c_array(tflite_path, header_path)
        with open(header_path) as f:
            text = f.read()

    assert '#ifndef HOPEGAIT_MODEL_DATA_H' in text
    assert f'hopegait_model_tflite_len = {len(payload)};' in text
    hex_bytes = re.findall(r'0x[0-9a-f]{2}', text)
    parsed = [int(b, 16) for b in hex_bytes]
    assert parsed == list(payload)
