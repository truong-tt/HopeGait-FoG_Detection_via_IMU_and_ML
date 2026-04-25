"""PyTorch -> ONNX -> TF -> int8 TFLite -> C header. Edge-phase script."""

import os
import sys
import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import MODELS_DIR
from models.tcn_model import HopeGaitTCN

TARGET_SUBJECT = '3'
SEQ_LENGTH = 256
NUM_INPUTS = 11
PYTORCH_MODEL = os.path.join(MODELS_DIR, f'hopegait_tcn_best_subj{TARGET_SUBJECT}.pth')
ONNX_MODEL = os.path.join(MODELS_DIR, 'hopegait.onnx')
TF_MODEL_DIR = os.path.join(MODELS_DIR, 'tf_model')
TFLITE_MODEL = os.path.join(MODELS_DIR, 'hopegait_int8.tflite')
CPP_HEADER = os.path.join(MODELS_DIR, 'model_data.h')


def representative_data_gen():
    for _ in range(100):
        yield [np.random.randn(1, SEQ_LENGTH, NUM_INPUTS).astype(np.float32)]


def convert_to_c_array(tflite_path, header_path):
    with open(tflite_path, 'rb') as f:
        tflite_content = f.read()
    hex_lines = []
    for i in range(0, len(tflite_content), 12):
        chunk = tflite_content[i:i + 12]
        hex_lines.append('    ' + ', '.join(f'0x{b:02x}' for b in chunk))
    hex_array = ',\n'.join(hex_lines)

    c_code = f"""#ifndef HOPEGAIT_MODEL_DATA_H
#define HOPEGAIT_MODEL_DATA_H
extern const unsigned char hopegait_model_tflite[];
extern const unsigned int hopegait_model_tflite_len;
const unsigned char hopegait_model_tflite[] = {{
{hex_array}
}};
const unsigned int hopegait_model_tflite_len = {len(tflite_content)};
#endif
"""
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, 'w') as f:
        f.write(c_code)


def main():
    device = torch.device('cpu')
    model = HopeGaitTCN()
    model.load_state_dict(torch.load(PYTORCH_MODEL, map_location=device, weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, SEQ_LENGTH, NUM_INPUTS)
    torch.onnx.export(
        model, dummy_input, ONNX_MODEL,
        export_params=True, opset_version=13,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
    )

    onnx_model = onnx.load(ONNX_MODEL)
    prepare(onnx_model).export_graph(TF_MODEL_DIR)

    converter = tf.lite.TFLiteConverter.from_saved_model(TF_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open(TFLITE_MODEL, 'wb') as f:
        f.write(tflite_model)

    convert_to_c_array(TFLITE_MODEL, CPP_HEADER)


if __name__ == "__main__":
    main()