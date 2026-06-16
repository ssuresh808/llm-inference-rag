# PagedAttention

PagedAttention is the attention algorithm introduced by vLLM to manage the
key-value (KV) cache efficiently during LLM inference. Instead of storing each
sequence's KV cache in one large contiguous buffer, PagedAttention splits it into
fixed-size **blocks** that can live anywhere in GPU memory, much like virtual
memory paging in an operating system.

This design nearly eliminates internal and external memory fragmentation, so the
server can fit many more concurrent sequences in the same amount of VRAM. Blocks
can also be **shared** across sequences with copy-on-write semantics, which makes
features like prefix caching and parallel sampling cheap.

The practical payoff is higher batch sizes and throughput: vLLM reports large
gains over naive contiguous KV cache management, with no change to model output.
