from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from moe_sae.contribution_transcoder_model import (
    Gemma4RoutedInputPrefixAdapter,
)
from tests.test_contribution_model import FakeModel


def test_source_only_capture_returns_exact_routed_norm_and_stops_before_experts():
    model = FakeModel()
    adapter = Gemma4RoutedInputPrefixAdapter(model)
    token_ids = jnp.asarray([[2, 3, 4]], dtype=jnp.int32)
    positions = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    segment_ids = jnp.ones_like(token_ids)
    sequence_ids = jnp.asarray([42], dtype=jnp.int32)

    captured = adapter(
        token_ids,
        positions,
        segment_ids,
        sequence_ids,
        valid_token_indices=jnp.asarray([0, 2], dtype=jnp.int32),
    )

    # The independent expression mirrors the fake layer's declared arithmetic:
    # embedding, fourteen prefix residual increments, target attention/norm,
    # attention residual, and finally only the routed pre-FFN norm.
    hidden = jnp.asarray(
        jnp.repeat(token_ids[..., None], 2_816, axis=-1),
        dtype=jnp.bfloat16,
    )
    for index in range(14):
        hidden = jnp.asarray(hidden + (index + 1) / 64.0, dtype=jnp.bfloat16)
    normalized = jnp.asarray(hidden * 0.875, dtype=jnp.bfloat16)
    attention = jnp.asarray(normalized + 0.25, dtype=jnp.bfloat16)
    attention = jnp.asarray(attention * 0.9375, dtype=jnp.bfloat16)
    residual = attention + hidden
    expected = jnp.asarray(residual * 0.75, dtype=jnp.bfloat16)
    expected = expected.reshape((-1, 2_816))[jnp.asarray([0, 2])]

    np.testing.assert_array_equal(np.asarray(captured.routed_input), expected)
    np.testing.assert_array_equal(np.asarray(captured.sequence_id), [42, 42])
    np.testing.assert_array_equal(np.asarray(captured.position), [0, 2])
    assert model.calls == list(range(14))

    target = model.decoder.layers[14]
    routed_moe = target.mlp.moe_block.routed_moe
    shared = target.mlp.moe_block.shared_experts
    assert routed_moe.calls == []
    assert routed_moe.gate.calls == 0
    assert shared.last_output is None


def test_source_only_capture_accepts_per_token_sequence_ids():
    model = FakeModel()
    adapter = Gemma4RoutedInputPrefixAdapter(model)
    token_ids = jnp.asarray([[2, 3]], dtype=jnp.int32)
    positions = jnp.asarray([[0, 1]], dtype=jnp.int32)
    segment_ids = jnp.ones_like(token_ids)
    sequence_ids = jnp.asarray([[10, 11]], dtype=jnp.int32)

    captured = adapter(token_ids, positions, segment_ids, sequence_ids)

    np.testing.assert_array_equal(np.asarray(captured.sequence_id), [10, 11])
    np.testing.assert_array_equal(np.asarray(captured.position), [0, 1])
    assert captured.routed_input.shape == (2, 2_816)
