#!/usr/bin/env python3
"""Export and QDQ-quantize the CPU MobileI2V deployment graphs.

This script deliberately imports neither CUDA nor any GPU execution provider.  It
checks every graph twice: PyTorch vs FP32 ONNX and PyTorch vs INT8-QDQ ONNX.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader


class OneBatch(CalibrationDataReader):
    def __init__(self, batch): self.batch, self.done = batch, False
    def get_next(self):
        if self.done: return None
        self.done = True
        return self.batch


def compare(name, reference, candidate, atol, rtol):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    max_error = float(np.max(np.abs(reference - candidate)))
    if not np.allclose(reference, candidate, atol=atol, rtol=rtol):
        raise AssertionError(f"{name}: mismatch (max abs error {max_error:.6g})")
    return max_error


def export_one(name, module, args, input_names, output_name, output_dir):
    # CPU kernels require activations and convolution/linear weights to share
    # one dtype.  Release checkpoints may contain fp16 tensors, so normalize
    # floating modules to fp32 before generating the reference and ONNX graph.
    module = module.cpu().float().eval()
    with torch.inference_mode(): reference = module(*args).detach().float().numpy()
    onnx_path = output_dir / f"{name}.onnx"
    # Use the established TorchScript exporter: MobileI2V/Turbo-VAED contain
    # modules that cannot currently be captured by torch.export.  This also
    # avoids an undeclared onnxscript dependency.
    torch.onnx.export(module, args, onnx_path, input_names=input_names, output_names=[output_name], opset_version=18)
    feeds = {key: value.detach().cpu().numpy() for key, value in zip(input_names, args)}
    fp32 = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run([output_name], feeds)[0]
    fp32_error = compare(f"{name} FP32", reference, fp32, 1e-4, 1e-4)
    int8_path = output_dir / f"{name}.int8.qdq.onnx"
    quantize_static(str(onnx_path), str(int8_path), OneBatch(feeds), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"]).run([output_name], feeds)[0]
    int8_error = compare(f"{name} INT8-QDQ", reference, int8, 0.35, 0.15)
    return {"fp32_max_abs_error": fp32_error, "int8_qdq_max_abs_error": int8_error, "file": int8_path.name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True, help="module:function returning iterable (name, module, args, input_names, output_name)")
    parser.add_argument("--output-dir", default="models")
    ns = parser.parse_args(); module_name, function_name = ns.factory.split(":")
    factory = getattr(__import__(module_name, fromlist=[function_name]), function_name)
    output_dir = Path(ns.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    results = {name: export_one(name, model, inputs, names, output, output_dir) for name, model, inputs, names, output in factory()}
    (output_dir / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))

if __name__ == "__main__": main()
