from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from moe_sae.artifacts import (
    ArtifactDataset,
    ArtifactWriter,
    CaptureMetadata,
    float32_to_bf16_bits,
    publish_manifest,
)
from moe_sae.metrics import build_efficiency_report
from moe_sae.model import CaptureBatch
from moe_sae.pipeline import TrainingConfig, evaluate, train
from moe_sae.sae import SAEConfig, init_train_state


def _metadata(split: str) -> CaptureMetadata:
    return CaptureMetadata(
        checkpoint_sha256="a" * 64,
        maxtext_commit="48c7da3e28e8f57fe36ddf09d00eec53072a2ea0",
        dataset_fingerprint="2" * 64,
        split=split,
        config_sha256="b" * 64,
        environment_sha256="c" * 64,
        conversion_attestation_sha256="9" * 64,
    )


def _seal(root, split: str, activation: np.ndarray) -> ArtifactDataset:
    rows = activation.shape[0]
    route = np.tile(np.arange(8, dtype=np.uint8), (rows, 1))
    batch = CaptureBatch(
        activation=float32_to_bf16_bits(activation),
        expert_ids=route,
        router_weights=float32_to_bf16_bits(np.full((rows, 8), 1.0 / 8.0, np.float32)),
        sequence_id=np.arange(rows, dtype=np.int64),
        position=np.arange(rows, dtype=np.uint16),
    )
    writer = ArtifactWriter(root, "host-0", _metadata(split))
    writer.write_range(f"{split}-range", 0, rows, batch)
    writer.finalize_host()
    publish_manifest(
        root,
        _metadata(split),
        expected_hosts=1,
        expected_tokens=rows,
    )
    return ArtifactDataset(root)


def test_synthetic_capture_train_eval_report_end_to_end(tmp_path) -> None:
    rng = np.random.default_rng(9)
    basis = rng.normal(size=(2, 2_816)).astype(np.float32)
    train_activation = rng.normal(size=(8, 2)).astype(np.float32) @ basis
    eval_activation = rng.normal(size=(4, 2)).astype(np.float32) @ basis
    training_capture = _seal(tmp_path / "train", "train", train_activation)
    heldout_capture = _seal(tmp_path / "eval", "eval", eval_activation)

    evaluations = []
    for condition, chip_seconds in (("A", 2.0), ("B", 1.0), ("F", 4.0)):
        sae = SAEConfig(
            condition,
            hidden_dim=2_816,
            num_experts=128,
            top_routed=8,
            bank_width=16,
            k=64,
            parameter_dtype="float32",
            warmup_steps=1,
            total_steps=2,
            peak_lr=1e-3,
            min_lr=1e-4,
        )
        config = TrainingConfig(
            sae,
            seed=1,
            total_tokens=8,
            effective_batch_tokens=4,
            microbatch_tokens=2,
            checkpoint_every_tokens=4,
            health_gate_tokens=4,
            live_feature_fraction=0.0,
            loss_ratio_threshold=2.0,
        )
        state = init_train_state(jax.random.PRNGKey(1), sae)
        state = state._replace(
            params=state.params._replace(bias=jnp.full_like(state.params.bias, 1_000.0))
        )
        trained = train(
            config,
            training_capture.iter_batches(),
            training_capture.mean,
            state=state,
        )
        result = evaluate(
            trained.state.params,
            config,
            heldout_capture.iter_batches(),
            training_capture.mean,
            max_tokens=heldout_capture.total_tokens,
            expected_sst=heldout_capture.sst_about(training_capture.mean),
        )
        evaluations.append(
            {
                "condition": condition,
                "seed": 1,
                "ev": result.explained_variance,
                "nmse": result.nmse,
                "l0": result.l0,
                "dead_rate": result.dead_rate,
                "chip_seconds": chip_seconds,
                "wall_seconds": chip_seconds,
                "tokens_per_second": 8.0 / chip_seconds,
            }
        )

    report = build_efficiency_report(
        evaluations,
        harvester={"chip_seconds": 10.0},
    )

    assert report["schema_version"] == "gemma4-router-efficiency-report-v1"
    assert set(report["conditions"]) == {"A", "B", "F"}
    assert report["primary"]["cost"]["cost_ratio_percent"] == 50.0
    assert "secondary_end_to_end_cost" in report
