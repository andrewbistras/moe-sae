from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from moe_sae import contribution_evaluation_worker as worker
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.contribution_evaluation_worker import (
    Experiment2EvaluationError,
    _ContentStore,
    _load_run,
    _validate_training_result,
)
from moe_sae.checkpointing import (
    checkpoint_payload_uri,
    make_pointer,
    publish_milestone_pointer,
)
from moe_sae.contribution_score_artifacts import ScoreArtifactRef
from moe_sae.contribution_training_worker import (
    TRAINING_RESULT_SCHEMA_VERSION,
    TrainingWorkerPlan,
)


def _plan() -> TrainingWorkerPlan:
    return TrainingWorkerPlan(
        phase_name="phase2_mechanism_pilot",
        condition="A",
        seed=0,
        total_tokens=32,
        planned_updates=2,
        warmup_updates=1,
        global_microbatch_tokens=8,
        microbatches_per_update=2,
        checkpoint_tokens=(32,),
        randomization_key=0,
    )


def _result(plan: TrainingWorkerPlan) -> dict[str, object]:
    identity_body = {
        "schema_version": "gemma4-contribution-sae-run-identity-v2",
        "plan": {
            "phase_name": plan.phase_name,
            "condition": plan.condition,
            "seed": plan.seed,
            "total_tokens": plan.total_tokens,
            "planned_updates": plan.planned_updates,
            "warmup_updates": plan.warmup_updates,
            "global_microbatch_tokens": plan.global_microbatch_tokens,
            "microbatches_per_update": plan.microbatches_per_update,
            "checkpoint_tokens": list(plan.checkpoint_tokens),
            "randomization_key": plan.randomization_key,
        },
        "input_mode": "resident",
        "performance_identity": {
            "graph_sha256": "1" * 64,
            "hardware_sha256": "2" * 64,
            "config_sha256": "3" * 64,
        },
        "performance_identity_sha256": "4" * 64,
        "admission": {
            "basis": "bounded_ungated_resident",
            "evidence_sha256": None,
        },
    }
    identity = {
        **identity_body,
        "run_identity_sha256": canonical_sha256(identity_body),
    }
    body: dict[str, object] = {
        "schema_version": TRAINING_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "run_identity": identity,
        "steps": plan.planned_updates,
        "tokens": plan.total_tokens,
        "chip_seconds": 12.5,
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def test_training_result_accepts_opaque_v2_run_identity() -> None:
    plan = _plan()
    assert _validate_training_result(_result(plan), plan)["tokens"] == 32


def test_training_result_rejects_nonfinal_or_tampered_evidence() -> None:
    plan = _plan()
    value = _result(plan)
    value["tokens"] = 31
    with pytest.raises(Experiment2EvaluationError, match="self-hash"):
        _validate_training_result(value, plan)


def test_load_run_selects_named_nominal_curve_checkpoint(
    tmp_path: Path,
) -> None:
    plan = TrainingWorkerPlan(
        phase_name="learning_horizon_10m",
        condition="A",
        seed=0,
        total_tokens=48,
        planned_updates=3,
        warmup_updates=1,
        global_microbatch_tokens=8,
        microbatches_per_update=2,
        checkpoint_tokens=(17, 48),
        randomization_key=0,
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    result = _result(plan)
    (run_root / "training-result.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )
    payload_root = tmp_path / "payloads"
    pointer = make_pointer(
        kind="latest",
        payload_uri=checkpoint_payload_uri(
            payload_root,
            result["run_identity"],
            step=2,
            tokens_seen=32,
            attempt_id="a" * 32,
        ),
        step=2,
        tokens_seen=32,
        attempt_id="a" * 32,
        run_identity=result["run_identity"],
    )
    publish_milestone_pointer(
        run_root / "checkpoints",
        pointer,
        checkpoint_target_tokens=17,
    )

    loaded_result, loaded_pointer = _load_run(
        run_root,
        plan,
        str(payload_root),
        checkpoint_target_tokens=17,
    )
    assert loaded_result == result
    assert loaded_pointer["tokens_seen"] == 32


def test_content_store_is_create_only_and_hash_verified(tmp_path: Path) -> None:
    blob = b"paired evidence"

    reference = ScoreArtifactRef(
        sha256=hashlib.sha256(blob).hexdigest(),
        size_bytes=len(blob),
        media_type="application/x.moe-sae-score-manifest+json",
    )
    store = _ContentStore(tmp_path)
    store.retain(reference, blob)
    store.retain(reference, blob)
    assert store(reference) == blob
    (tmp_path / reference.sha256).write_bytes(b"corrupt")
    with pytest.raises(Experiment2EvaluationError, match="size changed"):
        store(reference)


def test_component_and_routed_endpoints_receive_their_distinct_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component_mean = object()
    routed_mean = object()
    evidence = type("Evidence", (), {"scores": object()})()
    routed_scores = object()
    observed: dict[str, object] = {}

    def recompose(scores, _sidecar, mean):
        observed["routed_scores"] = scores
        observed["routed_mean"] = mean
        return routed_scores

    def evaluate(
        params,
        source,
        mean,
        mesh,
        **kwargs,
    ):
        observed.update(
            params=params,
            source=source,
            component_mean=mean,
            mesh=mesh,
        )
        kwargs["row_sink"](evidence)
        return "evaluation"

    paired: list[tuple[object, object]] = []
    monkeypatch.setattr(worker, "routed_recomposition_token_scores", recompose)
    monkeypatch.setattr(worker, "evaluate_contribution_input", evaluate)
    result = worker._evaluate_condition_endpoints(
        "params",
        "source",
        component_mean,
        routed_mean,
        "mesh",
        sidecar={},
        checkpoint_sha256="1" * 64,
        capacity_receipts=(),
        distributed="distributed",
        paired_row_sink=lambda left, right: paired.append((left, right)),
    )
    assert result == "evaluation"
    assert observed["component_mean"] is component_mean
    assert observed["routed_mean"] is routed_mean
    assert paired == [(evidence, routed_scores)]


def test_checkpoint_capacity_is_read_from_immutable_orbax_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Checkpointer:
        def metadata(self, uri):
            assert uri == "gs://bucket/final"
            return types.SimpleNamespace(
                custom_metadata={"training_config": {"a_capacity": 73}}
            )

        def close(self):
            return None

    checkpoint = types.ModuleType("orbax.checkpoint")
    checkpoint.StandardCheckpointer = Checkpointer
    orbax = types.ModuleType("orbax")
    orbax.checkpoint = checkpoint
    monkeypatch.setitem(sys.modules, "orbax", orbax)
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", checkpoint)
    assert (
        worker._checkpoint_a_capacity({"payload_uri": "gs://bucket/final"})
        == 73
    )
