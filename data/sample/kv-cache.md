# KV Cache

During autoregressive generation, a transformer attends over all previous tokens
at every step. The KV cache stores the key and value tensors for past tokens so
they are computed once and reused, turning each decoding step from quadratic into
linear work.

The catch is memory. KV cache size grows with batch size times sequence length
times the number of layers and attention heads, and at long context lengths it
becomes the dominant consumer of GPU memory, often larger than the model weights
for big batches.

Systems manage this pressure with PagedAttention (to avoid fragmentation), KV
cache quantization (storing keys and values in INT8 or FP8), and eviction or
offloading strategies for very long contexts.
