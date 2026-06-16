# Quantization

Quantization reduces the numerical precision of a model's weights (and sometimes
activations) to shrink memory use and speed up inference. Common targets are INT8,
FP8, and INT4, down from FP16 or BF16.

**Weight-only** methods such as GPTQ and AWQ quantize weights to four bits using a
small calibration dataset to minimize accuracy loss. They cut model memory by
roughly four times and help memory-bound decoding. **FP8** is supported natively on
newer GPUs (NVIDIA Hopper and Ada) and can quantize both weights and activations
with little quality loss.

The central trade-off is memory and latency versus accuracy. Aggressive low-bit
quantization can degrade quality on hard tasks, so the bit-width and method are
chosen per model and workload.
