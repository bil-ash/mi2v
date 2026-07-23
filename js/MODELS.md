# Quantized model package

The JS programs expect these four **INT8 QDQ** graphs in one directory:

* `qwen2_0.5b.int8.qdq.onnx`
* `denoiser.int8.qdq.onnx`
* `ltx_vae_encoder.int8.qdq.onnx`
* `turbo_vaed_decoder.int8.qdq.onnx`

The upstream checkpoint is available at <https://huggingface.co/hustvl/MobileI2V>. Keep generated ONNX artifacts out of regular Git; use Git LFS or release assets for distribution.

`validation.json`, emitted by the exporter, is the required proof that the CPU PyTorch output matched both the FP32 ONNX graph and the INT8-QDQ graph. The decoder entry must be Turbo-VAED's distilled decoder; do not export LTX's decoder.
