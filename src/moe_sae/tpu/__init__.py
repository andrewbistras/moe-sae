"""JAX/TPU port of the SAE training core (spec §5–§12).

Division of responsibility (recorded decision): the torch implementation
remains the single source of truth for initialization, checkpoint
serialization, θ-EMA / dead-latent trackers, and the condition-C permutation —
`train_jax` wraps a torch `SAECondition` (CPU) and mirrors only the hot math
(encode / select / decode / losses / optimizer) in JAX. Parity is therefore by
construction everywhere except the jitted step, which tests/test_jax_parity.py
checks against the torch oracle component by component.
"""
