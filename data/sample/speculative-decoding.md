# Speculative Decoding

Speculative decoding lowers generation latency by letting a small, fast **draft**
model propose several future tokens that the large **target** model then verifies
in a single forward pass. Tokens the target model agrees with are accepted; the
first disagreement is corrected and the rest are discarded.

Because autoregressive decoding is memory-bandwidth bound, verifying several
proposed tokens at once costs little more than generating one, so accepted drafts
translate directly into speedups (often two times or more) with **identical**
output distribution to the target model.

Variants avoid a separate draft model: Medusa adds extra prediction heads, and
EAGLE predicts at the feature level. Self-speculative methods reuse a subset of the
target model's own layers as the drafter.
