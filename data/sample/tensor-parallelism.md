# Tensor Parallelism

Tensor parallelism splits the individual weight matrices of a model across multiple
GPUs, so a model too large for one device can run and per-GPU memory and compute are
reduced. In the Megatron style, the rows or columns of each linear layer are
sharded, each GPU computes its partial result, and an all-reduce combines them.

Because an all-reduce happens inside every transformer block, tensor parallelism is
communication-intensive and is normally confined to GPUs connected by a fast
interconnect such as NVLink within a single node.

It is often combined with pipeline parallelism, which splits the model by layers
across nodes, and with data parallelism to scale training and inference across
large clusters.
