# Skill: Critical Analysis of LLM Inference-Optimization Techniques

Use this playbook when a question asks you to **evaluate, compare, or judge**
an inference-optimization technique (quantization, KV-cache management, paged
attention, continuous batching, speculative decoding, tensor/pipeline
parallelism, etc.) rather than merely define it.

## How to reason

1. **Name the objective.** Optimizations trade off along latency (TTFT vs.
   inter-token), throughput, memory footprint, and output quality. State which
   axis the technique targets before judging it.
2. **Surface the trade-off.** Every speedup costs something. Quantization →
   possible accuracy loss; speculative decoding → wasted compute on rejected
   drafts; aggressive batching → higher tail latency. Make the cost explicit.
3. **Check the regime.** A technique's value depends on the workload: batch
   size, sequence length, prefill- vs. decode-bound, model size, hardware
   (memory bandwidth vs. FLOPs bound). A win in one regime can be a loss in
   another.
4. **Demand evidence.** Prefer claims the retrieved passages actually support
   (reported speedups, conditions, baselines). Distinguish measured results
   from authors' framing. Note the baseline — "2x faster" is meaningless without
   one.
5. **State the limits.** Call out where the evidence is thin, the conditions
   under which the technique degrades, and what the source did *not* test.

## Output discipline

- Ground every quantitative claim in a retrieved passage; cite its source id.
- If the corpus does not support a comparison, say so plainly instead of
  filling the gap from memory.
- Prefer a short, structured verdict (objective → mechanism → trade-off →
  when it helps / when it hurts) over a flat summary.
