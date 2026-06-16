# Continuous Batching

Continuous batching (also called in-flight or iteration-level batching) is a
scheduling technique that maximizes GPU utilization when serving LLMs. Static
batching waits for every sequence in a batch to finish before starting the next
batch, which wastes cycles because generations have very different lengths.

With continuous batching, the scheduler operates at the granularity of a single
decoding step. As soon as one sequence emits its end-of-sequence token and exits,
a queued request takes its place in the batch without waiting for the rest.

This keeps the GPU busy and dramatically improves throughput for real workloads
with variable output lengths. It pairs naturally with PagedAttention, which makes
adding and removing sequences from the running batch inexpensive.
