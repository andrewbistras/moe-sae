from __future__ import annotations

import jax
import numpy as np
import pytest

from moe_sae.distributed import (
    DistributedBatch,
    DistributedConfig,
    create_mesh,
    init_distributed_state,
)
from moe_sae.gate1_distributed import (
    Gate1DistributedScorer,
    process_local_feature_state,
    process_local_token_scores,
)
from moe_sae.sae import SAEConfig


@pytest.mark.parametrize("condition", ["A", "B", "F"])
def test_gate1_token_scores_match_static_invariants(condition: str) -> None:
    devices = jax.devices()
    mesh = create_mesh(
        DistributedConfig(
            data_axis_size=1,
            feature_axis_size=1,
            microbatches_per_update=2,
        ),
        devices=devices[:1],
    )
    config = SAEConfig(
        condition=condition,
        hidden_dim=4,
        num_experts=4,
        top_routed=2,
        bank_width=4,
        k=2,
        parameter_dtype="float32",
        total_steps=1,
        warmup_steps=0,
    )
    distributed = DistributedConfig(
        data_axis_size=1,
        feature_axis_size=1,
        microbatches_per_update=2,
    )
    state = init_distributed_state(jax.random.PRNGKey(1), mesh, config, distributed)
    activation = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10
    ids = np.broadcast_to(
        np.array([[[0, 1]], [[2, 3]]], dtype=np.int32),
        (2, 3, 2),
    ).copy()
    valid = np.array([[True, True, True], [True, False, False]])
    batch = DistributedBatch(activation, ids, valid)
    scorer = Gate1DistributedScorer(
        mesh,
        config,
        distributed,
        a_capacity=4 if condition == "A" else None,
    )
    scores = scorer.score(
        state.params, batch, np.zeros(4, np.float32), process_local=False
    )
    sse, sst, l0, topk = process_local_token_scores(scores)
    assert sse.shape == (2, 3)
    assert sst.shape == (2, 3)
    assert np.isfinite(sse).all() and np.isfinite(sst).all()
    assert np.all(sse[~valid] == 0)
    assert np.all(sst[~valid] == 0)
    assert np.all(l0[valid] == 2)
    assert np.all(l0[~valid] == 0)
    assert np.all(topk)
    fired, overflow = process_local_feature_state(scores)
    assert fired.shape == (config.total_features,)
    assert fired.any()
    assert not overflow


def test_gate1_scorer_requires_a_capacity() -> None:
    mesh = create_mesh(
        DistributedConfig(data_axis_size=1, feature_axis_size=1),
        devices=jax.devices()[:1],
    )
    config = SAEConfig(
        condition="A",
        hidden_dim=4,
        num_experts=4,
        top_routed=2,
        bank_width=4,
        k=2,
        parameter_dtype="float32",
    )
    with pytest.raises(ValueError, match="fixed capacity"):
        Gate1DistributedScorer(mesh, config)
