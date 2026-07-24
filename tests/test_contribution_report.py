from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math

import numpy as np
import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.contribution_input import ContributionInputReceipt
from moe_sae.contribution_report import (
    BOOTSTRAP_REPLICATES,
    ContributionReportError,
    ContributionRunEvaluation,
    CostLedger,
    LearningCurvePoint,
    PairedSequenceScores,
    RunHealthEvidence,
    SharedQEvaluation,
    aggregate_paired_token_scores,
    build_contribution_report,
    build_phase1_receipt,
    paired_sequence_bootstrap,
    validate_contribution_report,
    validate_phase1_receipt,
    write_contribution_report,
)
from moe_sae.contribution_sae import (
    pilot_contribution_config,
    production_contribution_config,
)
from moe_sae.contribution_score_artifacts import (
    ScoreArtifactRef,
    seal_aggregated_score_row_shard,
    seal_score_manifest,
)

PROTOCOL = "1" * 64
SIDECAR = "2" * 64
PHASE1_TRAIN_CAPTURE = hashlib.sha256(b"phase1-train-capture").hexdigest()
PHASE1_EVAL_CAPTURE = hashlib.sha256(b"phase1-eval-capture").hexdigest()
PHASE2_TRAIN_CAPTURE = hashlib.sha256(b"phase2-train-capture").hexdigest()
PHASE2_EVAL_CAPTURE = hashlib.sha256(b"phase2-eval-capture").hexdigest()
PHASE3_TRAIN_CAPTURE = hashlib.sha256(b"phase3-train-capture").hexdigest()
PHASE3_EVAL_CAPTURE = hashlib.sha256(b"phase3-eval-capture").hexdigest()
PHASE4_TRAIN_CAPTURE = hashlib.sha256(b"phase4-train-capture").hexdigest()
PHASE4_EVAL_CAPTURE = hashlib.sha256(b"phase4-eval-capture").hexdigest()

PLAN = {
    "phase2": {
        "scale": "pilot",
        "seeds": (0,),
        "train": 2_000_000,
        "eval": 250_000,
        "updates": 123,
        "warmup": 12,
        "tail": 15_232,
        "checkpoints": (2_000_000,),
        "train_capture": PHASE2_TRAIN_CAPTURE,
        "eval_capture": PHASE2_EVAL_CAPTURE,
        "capture_tokens": 2_250_000,
    },
    "phase3": {
        "scale": "full",
        "seeds": (0,),
        "train": 10_000_000,
        "eval": 1_000_000,
        "updates": 611,
        "warmup": 61,
        "tail": 10_624,
        "checkpoints": (2_000_000, 5_000_000, 10_000_000),
        "train_capture": PHASE3_TRAIN_CAPTURE,
        "eval_capture": PHASE3_EVAL_CAPTURE,
        "capture_tokens": 11_000_000,
    },
    "phase4": {
        "scale": "full",
        "seeds": (1, 2, 3),
        "train": 50_000_000,
        "eval": 2_000_000,
        "updates": 3_052,
        "warmup": 250,
        "tail": 3_968,
        "checkpoints": (
            2_000_000,
            5_000_000,
            10_000_000,
            20_000_000,
            30_000_000,
            40_000_000,
            50_000_000,
        ),
        "train_capture": PHASE4_TRAIN_CAPTURE,
        "eval_capture": PHASE4_EVAL_CAPTURE,
        "capture_tokens": 52_000_000,
    },
}

DEFAULT_COMPONENT = {"A": 0.70, "R": 0.60, "B": 0.50, "F": 0.80}
DEFAULT_ROUTED = {"A": 0.68, "R": 0.58, "B": 0.50, "F": 0.78}


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def rehash(value: dict) -> dict:
    unsigned = {key: item for key, item in value.items() if key != "report_sha256"}
    value["report_sha256"] = canonical_sha256(unsigned)
    return value


def cost_ledger(label: str, chip_seconds: float, chips: int = 16) -> CostLedger:
    return CostLedger(
        timing_receipt_sha256=digest(f"timing:{label}"),
        chip_count=chips,
        timed_wall_seconds=chip_seconds / chips,
        marginal_chip_seconds=chip_seconds,
        compilation_chip_seconds=2.0,
        checkpoint_serialization_chip_seconds=3.0,
        evaluation_chip_seconds=4.0,
    )


def health(label: str, *, passed: bool = True) -> RunHealthEvidence:
    return RunHealthEvidence(
        health_receipt_sha256=digest(f"health:{label}"),
        finite_state=passed,
        exact_l0=True,
        no_capacity_overflow=True,
        no_token_drops=True,
        falling_loss=True,
        coverage_passed=True,
    )


def curve(
    phase: str,
    label: str,
    checkpoints: tuple[int, ...],
    final_ev: float,
    *,
    final_checkpoint_sha256: str,
) -> tuple[LearningCurvePoint, ...]:
    return tuple(
        LearningCurvePoint(
            training_tokens=token,
            sse=100.0 * (1.0 - final_ev + 0.02 * (len(checkpoints) - index - 1)),
            sst=100.0,
            checkpoint_sha256=(
                final_checkpoint_sha256
                if index == len(checkpoints) - 1
                else digest(f"{label}:checkpoint:{token}")
            ),
            evaluation_capture_manifest_sha256=PLAN[phase]["eval_capture"],
        )
        for index, token in enumerate(checkpoints)
    )


def input_receipts(
    phase: str,
    condition: str,
    *,
    seed: int,
    process_count: int = 4,
) -> tuple[ContributionInputReceipt, ...]:
    plan = PLAN[phase]
    randomization_key = seed + 17 if condition == "R" else 0
    factory = (
        pilot_contribution_config
        if plan["scale"] == "pilot"
        else production_contribution_config
    )
    layout = factory(condition, randomization_key=randomization_key)
    layout_sha256 = canonical_sha256(dataclasses.asdict(layout))
    result = []
    for process_index in range(process_count):
        unsigned = {
            "schema_version": "gemma4-contribution-training-input-v1",
            "capture_manifest_sha256": plan["train_capture"],
            "condition": condition,
            "randomization_key": randomization_key,
            "layout_sha256": layout_sha256,
            "total_tokens": plan["train"],
            "global_microbatch_tokens": 4_096,
            "microbatches_per_update": 4,
            "planned_updates": plan["updates"],
            "process_count": process_count,
            "process_index": process_index,
            "local_rows_per_microbatch": 4_096 // process_count,
        }
        result.append(
            ContributionInputReceipt(
                **unsigned,
                receipt_sha256=canonical_sha256(unsigned),
            )
        )
    return tuple(result)


def run(
    phase: str,
    condition: str,
    *,
    seed: int,
    component_ev: float,
    routed_ev: float,
    cost: float,
    q_checkpoint_sha256: str,
    healthy: bool = True,
) -> ContributionRunEvaluation:
    plan = PLAN[phase]
    evaluation_tokens = plan["eval"]
    l0 = 2 if plan["scale"] == "pilot" else 8
    checkpoint_sha256 = digest(f"{phase}:{seed}:{condition}:checkpoint")
    return ContributionRunEvaluation(
        condition=condition,
        seed=seed,
        scale=plan["scale"],
        evaluation_tokens=evaluation_tokens,
        training_tokens=plan["train"],
        optimizer_updates=plan["updates"],
        warmup_updates=plan["warmup"],
        masked_tail_tokens=plan["tail"],
        checkpoint_tokens=plan["checkpoints"],
        run_manifest_sha256=digest(f"{phase}:{seed}:{condition}:run"),
        config_sha256=digest(f"{phase}:{seed}:{condition}:config"),
        checkpoint_sha256=checkpoint_sha256,
        shared_q_checkpoint_sha256=q_checkpoint_sha256,
        input_receipts=input_receipts(
            phase,
            condition,
            seed=seed,
        ),
        component_sse=evaluation_tokens * (1.0 - component_ev),
        component_sst=float(evaluation_tokens),
        routed_sse=evaluation_tokens * (1.0 - routed_ev),
        routed_sst=float(evaluation_tokens),
        full_oracle_shared_sse=2.0 * evaluation_tokens * (1.0 - routed_ev),
        full_oracle_shared_sst=2.0 * evaluation_tokens,
        full_learned_shared_sse=2.0 * evaluation_tokens * (1.0 - routed_ev),
        full_learned_shared_sst=2.0 * evaluation_tokens,
        l0_per_contribution=l0,
        total_l0=8 * l0,
        dead_rate=0.01,
        fired_feature_fraction=0.99,
        macro_expert_ev=component_ev - 0.01,
        learning_curve=curve(
            phase,
            f"{phase}:{seed}:{condition}",
            plan["checkpoints"],
            component_ev,
            final_checkpoint_sha256=checkpoint_sha256,
        ),
        timing=cost_ledger(f"{phase}:{seed}:{condition}", cost),
        health=health(f"{phase}:{seed}:{condition}", passed=healthy),
    )


def q(
    phase: str, seed: int, *, checkpoint_sha256: str | None = None
) -> SharedQEvaluation:
    plan = PLAN[phase]
    selected_checkpoint = checkpoint_sha256 or digest(f"{phase}:{seed}:q")
    return SharedQEvaluation(
        seed=seed,
        evaluation_tokens=plan["eval"],
        training_tokens=plan["train"],
        checkpoint_tokens=plan["checkpoints"],
        checkpoint_sha256=selected_checkpoint,
        config_sha256=digest(f"{phase}:{seed}:q:config"),
        training_manifest_sha256=plan["train_capture"],
        evaluation_capture_manifest_sha256=plan["eval_capture"],
        width=8_192,
        l0=64,
        sse=0.2 * plan["eval"],
        sst=float(plan["eval"]),
        dead_rate=0.01,
        firing_entropy=4.0,
        firing_gini=0.2,
        decoder_coherence=0.1,
        effective_rank=100.0,
        learning_curve=curve(
            phase,
            f"{phase}:{seed}:q",
            plan["checkpoints"],
            0.8,
            final_checkpoint_sha256=selected_checkpoint,
        ),
        timing=cost_ledger(f"{phase}:{seed}:q", 5.0),
    )


def rows_for(
    phase: str,
    *,
    component: dict[int, dict[str, float]] | None = None,
    routed: dict[int, dict[str, float]] | None = None,
    a_cost: dict[int, float] | None = None,
    unhealthy: set[tuple[int, str]] | None = None,
) -> tuple[list[ContributionRunEvaluation], list[SharedQEvaluation]]:
    rows: list[ContributionRunEvaluation] = []
    shared: list[SharedQEvaluation] = []
    for seed in PLAN[phase]["seeds"]:
        q_hash = digest(f"{phase}:{seed}:q")
        selected_component = DEFAULT_COMPONENT if component is None else component[seed]
        selected_routed = DEFAULT_ROUTED if routed is None else routed[seed]
        costs = {
            "A": 30.0 if a_cost is None else a_cost[seed],
            "R": 29.0,
            "B": 20.0,
            "F": 100.0,
        }
        for condition in ("A", "R", "B", "F"):
            rows.append(
                run(
                    phase,
                    condition,
                    seed=seed,
                    component_ev=selected_component[condition],
                    routed_ev=selected_routed[condition],
                    cost=costs[condition],
                    q_checkpoint_sha256=q_hash,
                    healthy=(seed, condition) not in (unhealthy or set()),
                )
            )
        shared.append(q(phase, seed, checkpoint_sha256=q_hash))
    return rows, shared


def sequence_scores(
    phase: str,
    rows: list[ContributionRunEvaluation],
) -> list[PairedSequenceScores]:
    tokens = PLAN[phase]["eval"]
    clusters = math.ceil(tokens / 1_024)
    counts = np.full(clusters, 1_024, dtype=np.uint16)
    counts[-1] = tokens - 1_024 * (clusters - 1)
    ids = np.arange(clusters, dtype=np.int64)
    token_identity = digest(f"{phase}:heldout-token-identity")
    by_key = {(row.condition, row.seed): row for row in rows}
    result = []
    for seed in PLAN[phase]["seeds"]:
        for endpoint in ("component", "routed"):
            sst = counts.astype(np.float64)
            condition_sse = {
                condition: sst
                * (
                    getattr(by_key[(condition, seed)], f"{endpoint}_sse")
                    / getattr(by_key[(condition, seed)], f"{endpoint}_sst")
                )
                for condition in ("A", "R", "B", "F")
            }
            result.append(
                PairedSequenceScores(
                    seed=seed,
                    endpoint=endpoint,
                    capture_manifest_sha256=PLAN[phase]["eval_capture"],
                    token_identity_sha256=token_identity,
                    sequence_id=ids,
                    token_count=counts,
                    common_sst=sst,
                    condition_sse=condition_sse,
                )
            )
    return result


class ScoreResolver:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def retain(self, reference: ScoreArtifactRef, blob: bytes) -> None:
        self.objects[reference.sha256] = blob

    def __call__(self, reference: ScoreArtifactRef) -> bytes:
        return self.objects[reference.sha256]


def prefix_position_bitmaps(counts: np.ndarray) -> np.ndarray:
    result = np.zeros((counts.size, 16), dtype=np.uint64)
    for word in range(16):
        remaining = np.clip(counts.astype(np.int64) - 64 * word, 0, 64)
        full = remaining == 64
        partial = (remaining > 0) & ~full
        result[full, word] = np.iinfo(np.uint64).max
        if np.any(partial):
            result[partial, word] = np.left_shift(
                np.uint64(1),
                remaining[partial].astype(np.uint64),
            ) - np.uint64(1)
    return result


def retained_score_artifacts(
    phase: str,
    rows: list[ContributionRunEvaluation],
    scores: list[PairedSequenceScores],
) -> tuple[list[ScoreArtifactRef], ScoreResolver]:
    if any(
        item.capture_manifest_sha256 != PLAN[phase]["eval_capture"] for item in scores
    ):
        raise ContributionReportError(
            "test score evidence is not bound to the evaluation capture"
        )
    if len({item.token_identity_sha256 for item in scores}) != 1:
        raise ContributionReportError(
            "test scores do not use one scientific token inventory"
        )
    resolver = ScoreResolver()
    by_score = {(item.seed, item.endpoint): item for item in scores}
    by_run = {(item.seed, item.condition): item for item in rows}
    references = []
    for seed in PLAN[phase]["seeds"]:
        component = by_score[(seed, "component")]
        routed = by_score[(seed, "routed")]
        assert np.array_equal(component.sequence_id, routed.sequence_id)
        assert np.array_equal(component.token_count, routed.token_count)
        bitmap = prefix_position_bitmaps(component.token_count)
        for condition in ("A", "R", "B", "F"):
            row = by_run[(seed, condition)]
            randomization_key = row.input_receipts[0].randomization_key
            layout_sha256 = row.input_receipts[0].layout_sha256
            unsigned_receipt = {
                "schema_version": "gemma4-contribution-training-input-v1",
                "capture_manifest_sha256": PLAN[phase]["eval_capture"],
                "condition": condition,
                "randomization_key": randomization_key,
                "layout_sha256": layout_sha256,
                "total_tokens": PLAN[phase]["eval"],
                "global_microbatch_tokens": 4_096,
                "microbatches_per_update": 4,
                "planned_updates": 1,
                "process_count": 1,
                "process_index": 0,
                "local_rows_per_microbatch": 4_096,
            }
            evaluation_receipt = ContributionInputReceipt(
                **unsigned_receipt,
                receipt_sha256=canonical_sha256(unsigned_receipt),
            )
            row_reference, row_blob = seal_aggregated_score_row_shard(
                condition=condition,
                seed=seed,
                process_index=0,
                update_index=0,
                evaluation_capture_manifest_sha256=PLAN[phase]["eval_capture"],
                checkpoint_sha256=row.checkpoint_sha256,
                recomposition_sidecar_sha256=SIDECAR,
                centering_mean_sha256=digest(f"{phase}:common-centering-mean"),
                evaluation_input_receipt_sha256=(evaluation_receipt.receipt_sha256),
                sequence_id=component.sequence_id,
                position_bitmap=bitmap,
                token_count=component.token_count,
                component_sse=component.condition_sse[condition],
                component_sst=component.common_sst,
                routed_sse=routed.condition_sse[condition],
                routed_sst=routed.common_sst,
            )
            resolver.retain(row_reference, row_blob)
            manifest_reference, manifest_blob = seal_score_manifest(
                (row_reference,),
                resolver,
                phase=phase,
                protocol_sha256=PROTOCOL,
                condition=condition,
                seed=seed,
                randomization_key=randomization_key,
                training_capture_manifest_sha256=PLAN[phase]["train_capture"],
                evaluation_capture_manifest_sha256=PLAN[phase]["eval_capture"],
                recomposition_sidecar_sha256=SIDECAR,
                run_manifest_sha256=row.run_manifest_sha256,
                config_sha256=row.config_sha256,
                checkpoint_sha256=row.checkpoint_sha256,
                layout_sha256=layout_sha256,
                centering_mean_sha256=digest(f"{phase}:common-centering-mean"),
                training_input_receipt_sha256s=tuple(
                    item.receipt_sha256 for item in row.input_receipts
                ),
                evaluation_input_receipts=(evaluation_receipt,),
                expected_tokens=PLAN[phase]["eval"],
            )
            resolver.retain(manifest_reference, manifest_blob)
            references.append(manifest_reference)
    return references, resolver


def phase1_receipt(*, passing: bool = True):
    checks = {
        "all_capture_exactness_checks_pass": passing,
        "bf16_roundtrip_bit_exact": True,
        "interrupted_resume_shard_identity": True,
        "multihost_ranges_no_gaps_or_duplicates": True,
        "no_routed_token_drops": True,
        "writer_queue_bounded": True,
        "persisted_goodput_ge_0.85_device_output": True,
        "no_recompilation_after_warmup": True,
        "auxiliary_tokens_per_second_ge_0.90_writer_disabled": True,
    }
    return build_phase1_receipt(
        protocol_sha256=PROTOCOL,
        training_capture_manifest_sha256=PHASE1_TRAIN_CAPTURE,
        evaluation_capture_manifest_sha256=PHASE1_EVAL_CAPTURE,
        recomposition_sidecar_sha256=SIDECAR,
        checks=checks,
        useful_prefix_mfu=0.51,
        device_output_tokens_per_second=1_100.0,
        persisted_tokens_per_second=1_000.0,
        writer_disabled_tokens_per_second=1_100.0,
        auxiliary_tokens_per_second=1_000.0,
        max_writer_backlog_batches=1,
        created_at="2026-07-24T00:00:00+00:00",
    )


def harvester(phase: str):
    wall_seconds = PLAN[phase]["capture_tokens"] / 1_000.0
    return {
        "training_capture_manifest_sha256": PLAN[phase]["train_capture"],
        "evaluation_capture_manifest_sha256": PLAN[phase]["eval_capture"],
        "timing_receipt_sha256": digest(f"{phase}:harvester:timing"),
        "training_tokens": PLAN[phase]["train"],
        "evaluation_tokens": PLAN[phase]["eval"],
        "capture_tokens": PLAN[phase]["capture_tokens"],
        "chip_count": 16,
        "useful_prefix_mfu": 0.51,
        "device_output_tokens_per_second": 1_100.0,
        "persisted_tokens_per_second": 1_000.0,
        "input_wait_fraction": 0.01,
        "max_writer_backlog_batches": 1,
        "wall_seconds": wall_seconds,
        "chip_seconds": 16.0 * wall_seconds,
        "compilation_chip_seconds": 16.0,
        "serialization_chip_seconds": 8.0,
    }


def build(
    phase: str,
    *,
    rows: list[ContributionRunEvaluation] | None = None,
    scores: list[PairedSequenceScores] | None = None,
    shared: list[SharedQEvaluation] | None = None,
    parent=None,
    phase1=None,
    authoritative: bool = True,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    capture_parents: tuple[str, str] | None = None,
    score_artifact_bundle: tuple[
        list[ScoreArtifactRef],
        ScoreResolver,
    ]
    | None = None,
):
    if rows is None or shared is None:
        rows, shared = rows_for(phase)
    if scores is None:
        scores = sequence_scores(phase, rows)
    if phase == "phase2":
        selected_parents = capture_parents or (
            PHASE1_TRAIN_CAPTURE,
            PHASE1_EVAL_CAPTURE,
        )
        selected_phase1 = phase1 if phase1 is not None else phase1_receipt()
    elif phase == "phase3":
        selected_parents = capture_parents or (
            PHASE2_TRAIN_CAPTURE,
            PHASE2_EVAL_CAPTURE,
        )
        selected_phase1 = None
        if parent is None:
            parent = build("phase2")
    else:
        selected_parents = capture_parents or (
            PHASE3_TRAIN_CAPTURE,
            PHASE3_EVAL_CAPTURE,
        )
        selected_phase1 = None
        if parent is None:
            parent = build("phase3")
    score_references, score_resolver = (
        retained_score_artifacts(
            phase,
            rows,
            scores,
        )
        if score_artifact_bundle is None
        else score_artifact_bundle
    )
    return build_contribution_report(
        rows,
        score_references,
        shared,
        score_artifact_resolver=score_resolver,
        phase=phase,
        protocol_sha256=PROTOCOL,
        training_capture_manifest_sha256=PLAN[phase]["train_capture"],
        evaluation_capture_manifest_sha256=PLAN[phase]["eval_capture"],
        parent_training_capture_manifest_sha256=selected_parents[0],
        parent_evaluation_capture_manifest_sha256=selected_parents[1],
        training_capture_lineage_receipt_sha256=digest(
            f"{phase}:train-capture-lineage"
        ),
        evaluation_capture_lineage_receipt_sha256=digest(
            f"{phase}:eval-capture-lineage"
        ),
        recomposition_sidecar_sha256=SIDECAR,
        harvester=harvester(phase),
        phase1_receipt=selected_phase1,
        parent_report=parent,
        authoritative=authoritative,
        bootstrap_replicates=bootstrap_replicates,
        created_at="2026-07-24T00:00:00+00:00",
    )


def test_token_scores_are_clustered_by_sequence_and_deterministic():
    sequence_ids = np.asarray([10, 10, 11, 11, 11, 12], dtype=np.int64)
    positions = np.asarray([0, 1, 0, 1, 2, 0], dtype=np.uint16)
    sst = np.ones(6, dtype=np.float64)
    errors = {
        "A": np.asarray([0.1, 0.1, 0.2, 0.2, 0.2, 0.3]),
        "R": np.asarray([0.2, 0.2, 0.3, 0.3, 0.3, 0.4]),
        "B": np.asarray([0.4, 0.4, 0.5, 0.5, 0.5, 0.6]),
        "F": np.asarray([0.05, 0.05, 0.1, 0.1, 0.1, 0.2]),
    }
    scores = aggregate_paired_token_scores(
        sequence_ids,
        positions,
        sst,
        errors,
        seed=0,
        endpoint="component",
        capture_manifest_sha256=PHASE2_EVAL_CAPTURE,
    )
    first = paired_sequence_bootstrap(
        scores,
        left="A",
        right="B",
        replicates=2_000,
        bootstrap_seed=17,
        chunk_size=31,
    )
    second = paired_sequence_bootstrap(
        scores,
        left="A",
        right="B",
        replicates=2_000,
        bootstrap_seed=17,
        chunk_size=31,
    )
    assert first == second
    assert first.clusters == 3
    assert first.tokens == 6
    assert first.effect == pytest.approx(0.3)
    assert first.ci95_low > 0


def test_token_score_aggregation_rejects_duplicate_token_identity():
    ids = np.asarray([10, 10], dtype=np.int64)
    positions = np.asarray([0, 0], dtype=np.uint16)
    errors = {condition: np.ones(2) for condition in ("A", "R", "B", "F")}
    with pytest.raises(ContributionReportError, match="duplicate"):
        aggregate_paired_token_scores(
            ids,
            positions,
            np.ones(2),
            errors,
            seed=0,
            endpoint="component",
            capture_manifest_sha256=PHASE2_EVAL_CAPTURE,
        )


def test_phase1_receipt_derives_performance_checks_and_is_self_hashed():
    receipt = phase1_receipt()
    assert validate_phase1_receipt(receipt) == receipt
    assert receipt["status"] == "passed"

    checks = dict(receipt["checks"])
    checks["persisted_goodput_ge_0.85_device_output"] = False
    with pytest.raises(ContributionReportError, match="disagrees"):
        build_phase1_receipt(
            protocol_sha256=PROTOCOL,
            training_capture_manifest_sha256=PHASE1_TRAIN_CAPTURE,
            evaluation_capture_manifest_sha256=PHASE1_EVAL_CAPTURE,
            recomposition_sidecar_sha256=SIDECAR,
            checks=checks,
            useful_prefix_mfu=0.51,
            device_output_tokens_per_second=1_100.0,
            persisted_tokens_per_second=1_000.0,
            writer_disabled_tokens_per_second=1_100.0,
            auxiliary_tokens_per_second=1_000.0,
            max_writer_backlog_batches=1,
        )


def test_phase1_validation_rejects_rehashed_nonpositive_throughput():
    receipt = phase1_receipt()
    tampered = copy.deepcopy(receipt)
    tampered["performance"].update(
        {
            "device_output_tokens_per_second": -1.0,
            "persisted_tokens_per_second": 0.0,
            "writer_disabled_tokens_per_second": -1.0,
            "auxiliary_tokens_per_second": 0.0,
        }
    )
    unsigned = {key: item for key, item in tampered.items() if key != "receipt_sha256"}
    tampered["receipt_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ContributionReportError, match="strictly positive"):
        validate_phase1_receipt(tampered)


def test_phase2_pass_is_explicitly_exploratory_and_self_hashed():
    report = build("phase2")
    assert report["status"] == "passed"
    assert report["gate"]["passed"] is True
    assert report["primary"]["gain_recovery_percent"] == pytest.approx(2 / 3 * 100)
    assert report["primary"]["cost_ratio_percent"] == 30.0
    assert report["primary"]["delta_route"] == pytest.approx(0.1)
    assert report["conditions"]["A"]["layout"]["candidates_per_token"] == 2_048
    assert "Seed-0 phase2" in report["headline"]
    assert "permanently excluded" in report["headline"]
    assert not report["headline"].startswith("Expert-aligned A recovers")
    assert validate_contribution_report(report) == report


def test_f_is_explicitly_route_rank_conditioned_and_geometry_matches():
    report = build("phase2")
    layouts = {name: row["layout"] for name, row in report["conditions"].items()}
    assert layouts["A"]["candidates_per_token"] == layouts["B"]["candidates_per_token"]
    assert layouts["A"]["parameter_count"] == layouts["F"]["parameter_count"]
    assert layouts["A"]["candidate_exposure_percent_of_f"] == 6.25
    assert layouts["R"]["bank_identity"] == "per_sequence_randomized_expert_id"
    assert layouts["F"]["reference_description"] == (
        "full-width route-rank-conditioned component dictionary"
    )


def test_gain_is_unclipped_negative_and_undefined_by_policy():
    over_component = {0: {**DEFAULT_COMPONENT, "A": 0.90, "F": 0.60}}
    rows, shared = rows_for("phase2", component=over_component)
    over = build(
        "phase2",
        rows=rows,
        shared=shared,
        scores=sequence_scores("phase2", rows),
    )
    assert over["primary"]["gain_recovery_percent"] == pytest.approx(400.0)

    negative_component = {0: {**DEFAULT_COMPONENT, "A": 0.40}}
    rows, shared = rows_for("phase2", component=negative_component)
    negative = build(
        "phase2",
        rows=rows,
        shared=shared,
        scores=sequence_scores("phase2", rows),
    )
    assert negative["primary"]["gain_recovery_percent"] == pytest.approx(-100.0 / 3.0)

    undefined_component = {0: {**DEFAULT_COMPONENT, "F": 0.40}}
    rows, shared = rows_for("phase2", component=undefined_component)
    undefined = build(
        "phase2",
        rows=rows,
        shared=shared,
        scores=sequence_scores("phase2", rows),
    )
    assert undefined["primary"]["gain_recovery_percent"] is None
    assert undefined["primary"]["gain_status"] == "undefined"
    assert undefined["status"] == "failed"


def test_positive_but_unstable_denominator_is_exploratory():
    rows, shared = rows_for("phase2")
    scores = sequence_scores("phase2", rows)
    component = next(item for item in scores if item.endpoint == "component")
    counts = component.token_count.astype(np.float64)
    signs = np.where(np.arange(component.clusters) % 2 == 0, 1.0, -1.0)
    weighted_mean = float(np.sum(signs * counts) / np.sum(counts))
    rate_difference = 0.40 * (signs - weighted_mean) + 0.002
    condition_sse = dict(component.condition_sse)
    condition_sse["F"] = counts * (0.5 - rate_difference)
    replacement = PairedSequenceScores(
        seed=component.seed,
        endpoint="component",
        capture_manifest_sha256=component.capture_manifest_sha256,
        token_identity_sha256=component.token_identity_sha256,
        sequence_id=component.sequence_id,
        token_count=component.token_count,
        common_sst=component.common_sst,
        condition_sse=condition_sse,
    )
    scores[scores.index(component)] = replacement
    f_index = next(index for index, item in enumerate(rows) if item.condition == "F")
    f_sse = float(condition_sse["F"].sum())
    rows[f_index] = dataclasses.replace(
        rows[f_index],
        component_sse=f_sse,
    )
    report = build("phase2", rows=rows, scores=scores, shared=shared)
    assert report["primary"]["mean_f_minus_b"] > 0
    assert report["primary"]["gain_status"] == "unstable"
    assert report["primary"]["paired_f_minus_b_ci95"][0] < 0
    assert report["status"] == "failed"


def test_exact_phase_scale_and_stage_lineage_are_required():
    rows, shared = rows_for("phase2")
    rows[0] = dataclasses.replace(rows[0], evaluation_tokens=249_999)
    with pytest.raises(ContributionReportError, match="evaluation_tokens"):
        build(
            "phase2",
            rows=rows,
            shared=shared,
            scores=sequence_scores("phase2", rows_for("phase2")[0]),
        )

    rows3, shared3 = rows_for("phase3")
    score_refs3, score_resolver3 = retained_score_artifacts(
        "phase3",
        rows3,
        sequence_scores("phase3", rows3),
    )
    with pytest.raises(ContributionReportError, match="requires a passed phase2"):
        build_contribution_report(
            rows3,
            score_refs3,
            shared3,
            score_artifact_resolver=score_resolver3,
            phase="phase3",
            protocol_sha256=PROTOCOL,
            training_capture_manifest_sha256=PHASE3_TRAIN_CAPTURE,
            evaluation_capture_manifest_sha256=PHASE3_EVAL_CAPTURE,
            parent_training_capture_manifest_sha256=PHASE2_TRAIN_CAPTURE,
            parent_evaluation_capture_manifest_sha256=PHASE2_EVAL_CAPTURE,
            training_capture_lineage_receipt_sha256=digest("train lineage"),
            evaluation_capture_lineage_receipt_sha256=digest("eval lineage"),
            recomposition_sidecar_sha256=SIDECAR,
            harvester=harvester("phase3"),
        )

    failed_component = {0: {**DEFAULT_COMPONENT, "A": 0.40}}
    failed_rows, failed_q = rows_for("phase2", component=failed_component)
    failed_parent = build(
        "phase2",
        rows=failed_rows,
        shared=failed_q,
        scores=sequence_scores("phase2", failed_rows),
    )
    assert failed_parent["status"] == "failed"
    with pytest.raises(ContributionReportError, match="not an authoritative passed"):
        build(
            "phase3",
            rows=rows3,
            shared=shared3,
            scores=sequence_scores("phase3", rows3),
            parent=failed_parent,
        )


def test_training_input_receipts_bind_geometry_condition_and_train_capture():
    rows, shared = rows_for("phase2")
    r_index = next(index for index, row in enumerate(rows) if row.condition == "R")
    bad_receipts = []
    for receipt in rows[r_index].input_receipts:
        unsigned = {
            **dataclasses.asdict(receipt),
            "layout_sha256": digest("fixed-global-wrong-layout"),
        }
        unsigned.pop("receipt_sha256")
        bad_receipts.append(
            ContributionInputReceipt(
                **unsigned,
                receipt_sha256=canonical_sha256(unsigned),
            )
        )
    rows[r_index] = dataclasses.replace(
        rows[r_index],
        input_receipts=tuple(bad_receipts),
    )
    with pytest.raises(ContributionReportError, match="layout is not canonical"):
        build(
            "phase2",
            rows=rows,
            scores=sequence_scores("phase2", rows),
            shared=shared,
        )

    rows, shared = rows_for("phase2")
    a_index = next(index for index, row in enumerate(rows) if row.condition == "A")
    bad_receipts = []
    for receipt in rows[a_index].input_receipts:
        unsigned = {
            **dataclasses.asdict(receipt),
            "capture_manifest_sha256": digest("different-training-capture"),
        }
        unsigned.pop("receipt_sha256")
        bad_receipts.append(
            ContributionInputReceipt(
                **unsigned,
                receipt_sha256=canonical_sha256(unsigned),
            )
        )
    rows[a_index] = dataclasses.replace(
        rows[a_index],
        input_receipts=tuple(bad_receipts),
    )
    with pytest.raises(ContributionReportError, match="common training capture"):
        build(
            "phase2",
            rows=rows,
            scores=sequence_scores("phase2", rows),
            shared=shared,
        )


def test_retained_scores_bind_checkpoint_run_config_and_training_inputs():
    rows, shared = rows_for("phase2")
    scores = sequence_scores("phase2", rows)
    bundle = retained_score_artifacts("phase2", rows, scores)

    changed = list(rows)
    a_index = next(index for index, row in enumerate(changed) if row.condition == "A")
    changed[a_index] = dataclasses.replace(
        changed[a_index],
        checkpoint_sha256=digest("different checkpoint"),
        learning_curve=tuple(
            dataclasses.replace(
                point,
                checkpoint_sha256=(
                    digest("different checkpoint")
                    if index == len(changed[a_index].learning_curve) - 1
                    else point.checkpoint_sha256
                ),
            )
            for index, point in enumerate(changed[a_index].learning_curve)
        ),
    )
    with pytest.raises(ContributionReportError, match="retained score manifest"):
        build(
            "phase2",
            rows=changed,
            scores=scores,
            shared=shared,
            score_artifact_bundle=bundle,
        )

    changed = list(rows)
    r_index = next(index for index, row in enumerate(changed) if row.condition == "R")
    randomization_key = 999
    layout_sha256 = canonical_sha256(
        dataclasses.asdict(
            pilot_contribution_config(
                "R",
                randomization_key=randomization_key,
            )
        )
    )
    receipts = []
    for receipt in changed[r_index].input_receipts:
        unsigned = {
            **dataclasses.asdict(receipt),
            "randomization_key": randomization_key,
            "layout_sha256": layout_sha256,
        }
        unsigned.pop("receipt_sha256")
        receipts.append(
            ContributionInputReceipt(
                **unsigned,
                receipt_sha256=canonical_sha256(unsigned),
            )
        )
    changed[r_index] = dataclasses.replace(
        changed[r_index],
        input_receipts=tuple(receipts),
    )
    with pytest.raises(
        ContributionReportError,
        match="retained score manifest",
    ):
        build(
            "phase2",
            rows=changed,
            scores=scores,
            shared=shared,
            score_artifact_bundle=bundle,
        )


def test_full_ffn_endpoints_require_one_exact_target_sst():
    rows, shared = rows_for("phase2")
    rows = [
        dataclasses.replace(
            row,
            full_learned_shared_sst=row.full_learned_shared_sst + 1.0,
        )
        for row in rows
    ]
    with pytest.raises(ContributionReportError, match="same exact full-FFN"):
        build(
            "phase2",
            rows=rows,
            scores=sequence_scores("phase2", rows),
            shared=shared,
        )


def test_phase3_failed_gate_never_emits_confirmatory_headline():
    parent = build("phase2")
    rows, shared = rows_for("phase3", a_cost={0: 60.0})
    a_index = next(index for index, row in enumerate(rows) if row.condition == "A")
    collapsed = tuple(
        dataclasses.replace(
            point,
            sse=100.0 * (1.0 - ev),
            sst=100.0,
        )
        for point, ev in zip(
            rows[a_index].learning_curve,
            (0.80, 0.78, 0.70),
            strict=True,
        )
    )
    rows[a_index] = dataclasses.replace(rows[a_index], learning_curve=collapsed)
    report = build(
        "phase3",
        rows=rows,
        scores=sequence_scores("phase3", rows),
        shared=shared,
        parent=parent,
    )
    assert report["status"] == "failed"
    assert report["gate"]["checks"]["cost_ratio_below_50_percent"] is False
    assert report["gate"]["checks"]["learning_curve_no_material_late_collapse"] is False
    assert not report["headline"].startswith("Expert-aligned A recovers")
    assert "Seed-0 phase3 gate failed" in report["headline"]


def test_learning_curve_points_are_checkpoint_and_capture_bound():
    rows, shared = rows_for("phase3")
    a_index = next(index for index, row in enumerate(rows) if row.condition == "A")
    curve_points = list(rows[a_index].learning_curve)
    curve_points[-1] = dataclasses.replace(
        curve_points[-1],
        checkpoint_sha256=digest("wrong checkpoint"),
    )
    rows[a_index] = dataclasses.replace(
        rows[a_index],
        learning_curve=tuple(curve_points),
    )
    with pytest.raises(ContributionReportError, match="final curve point"):
        build(
            "phase3",
            rows=rows,
            shared=shared,
            scores=sequence_scores("phase3", rows),
            parent=build("phase2"),
        )


def test_phase4_requires_paired_confirmatory_evidence_and_is_hierarchical():
    phase3 = build("phase3")
    report = build("phase4", parent=phase3)
    assert report["status"] == "headline-ready"
    assert report["gate"]["passed"] is True
    assert report["primary"]["paired_f_minus_b_interval_basis"].startswith(
        "hierarchical"
    )
    assert len(report["hierarchical_seed_sequence_effects"]) == 4
    assert report["headline"].startswith("Expert-aligned A recovers")
    assert "route-rank-conditioned F" in report["headline"]


def test_phase4_negative_seed_health_or_excess_cost_suppresses_headline():
    phase3 = build("phase3")
    component = {
        1: {**DEFAULT_COMPONENT, "A": 0.55},
        2: dict(DEFAULT_COMPONENT),
        3: dict(DEFAULT_COMPONENT),
    }
    rows, shared = rows_for(
        "phase4",
        component=component,
        a_cost={1: 120.0, 2: 120.0, 3: 120.0},
        unhealthy={(2, "A")},
    )
    report = build(
        "phase4",
        rows=rows,
        scores=sequence_scores("phase4", rows),
        shared=shared,
        parent=phase3,
    )
    assert report["status"] == "confirmatory-failed"
    assert report["gate"]["checks"]["a_minus_r_positive_with_ci"] is False
    assert report["gate"]["checks"]["all_runs_healthy"] is False
    assert report["gate"]["checks"]["cost_ratio_below_50_percent"] is False
    assert not report["headline"].startswith("Expert-aligned A recovers")


def test_non_authoritative_report_can_never_pass_or_headline():
    report = build(
        "phase2",
        authoritative=False,
        bootstrap_replicates=200,
    )
    assert report["status"] == "non-authoritative"
    assert report["gate"]["passed"] is False
    assert report["gate"]["checks"]["authoritative_evidence"] is False
    assert report["headline"].startswith("Non-authoritative")


def test_score_capture_inventory_and_pooled_sse_are_bound():
    rows, shared = rows_for("phase2")
    scores = sequence_scores("phase2", rows)
    first = scores[0]
    scores[0] = dataclasses.replace(
        first,
        capture_manifest_sha256=digest("wrong capture"),
    )
    with pytest.raises(ContributionReportError, match="not bound"):
        build("phase2", rows=rows, scores=scores, shared=shared)

    scores = sequence_scores("phase2", rows)
    first = scores[0]
    scores[0] = dataclasses.replace(
        first,
        token_identity_sha256=digest("different token inventory"),
    )
    with pytest.raises(ContributionReportError, match="one scientific token"):
        build("phase2", rows=rows, scores=scores, shared=shared)

    scores = sequence_scores("phase2", rows)
    first = scores[0]
    changed = dict(first.condition_sse)
    changed["A"] = changed["A"] + 1.0
    scores[0] = dataclasses.replace(first, condition_sse=changed)
    with pytest.raises(ContributionReportError, match="does not match pooled SSE"):
        build("phase2", rows=rows, scores=scores, shared=shared)


def test_all_confirmatory_seeds_must_share_sst_and_token_inventory():
    rows, shared = rows_for("phase4")
    seed2_indices = [index for index, row in enumerate(rows) if row.seed == 2]
    for index in seed2_indices:
        rows[index] = dataclasses.replace(
            rows[index],
            component_sst=rows[index].component_sst + 1.0,
        )
    with pytest.raises(ContributionReportError, match="all conditions and seeds"):
        build(
            "phase4",
            rows=rows,
            shared=shared,
            scores=sequence_scores("phase4", rows_for("phase4")[0]),
        )


def test_shared_q_checkpoint_and_capture_are_common_and_exact():
    rows, shared = rows_for("phase2")
    rows[0] = dataclasses.replace(
        rows[0],
        shared_q_checkpoint_sha256=digest("condition-specific-q"),
    )
    with pytest.raises(ContributionReportError, match="one common Q"):
        build(
            "phase2",
            rows=rows,
            shared=shared,
            scores=sequence_scores("phase2", rows),
        )

    rows, shared = rows_for("phase2")
    shared[0] = dataclasses.replace(
        shared[0],
        evaluation_capture_manifest_sha256=digest("wrong-eval-capture"),
    )
    with pytest.raises(ContributionReportError, match="report capture"):
        build(
            "phase2",
            rows=rows,
            shared=shared,
            scores=sequence_scores("phase2", rows),
        )

    with pytest.raises(ContributionReportError, match="8,192"):
        dataclasses.replace(q("phase2", 0), width=4_096)


def test_c2_is_median_of_paired_seed_ratios_and_cost_totals_are_absolute():
    phase3 = build("phase3")
    rows, shared = rows_for(
        "phase4",
        a_cost={1: 10.0, 2: 30.0, 3: 80.0},
    )
    report = build(
        "phase4",
        rows=rows,
        scores=sequence_scores("phase4", rows),
        shared=shared,
        parent=phase3,
    )
    assert report["primary"]["per_seed_cost_ratio_percent"] == {
        "1": 10.0,
        "2": 30.0,
        "3": 80.0,
    }
    assert report["primary"]["cost_ratio_percent"] == 30.0
    system = report["secondary_end_to_end_cost"]
    assert system["one_seed"]["1"]["a_total_chip_seconds"] == 832_015.0
    assert system["one_seed"]["1"]["f_total_chip_seconds"] == 832_105.0
    assert system["all_available_seeds"]["a_total_chip_seconds"] == 832_135.0
    assert system["all_available_seeds"]["f_total_chip_seconds"] == 832_315.0
    assert system["excluded_costs"]["condition_compilation_chip_seconds"] == 24.0


def test_report_validation_rejects_semantic_tampering_even_when_rehashed():
    report = build("phase2")
    tampered = copy.deepcopy(report)
    tampered["status"] = "headline-ready"
    rehash(tampered)
    with pytest.raises(ContributionReportError, match="status is inconsistent"):
        validate_contribution_report(tampered)

    tampered = copy.deepcopy(report)
    tampered["headline"] = "Expert-aligned A recovers a spectacular result."
    rehash(tampered)
    with pytest.raises(ContributionReportError, match="headline is inconsistent"):
        validate_contribution_report(tampered)


def test_report_resolver_recomputes_intervals_from_retained_score_rows():
    rows, shared = rows_for("phase2")
    scores = sequence_scores("phase2", rows)
    references, resolver = retained_score_artifacts("phase2", rows, scores)
    report = build(
        "phase2",
        rows=rows,
        scores=scores,
        shared=shared,
    )
    assert report["score_manifest_references"] == [
        item.to_dict() for item in references
    ]
    assert (
        validate_contribution_report(
            report,
            score_artifact_resolver=resolver,
        )
        == report
    )

    tampered = copy.deepcopy(report)
    tampered["paired_sequence_effects"][0]["ci95_low"] += 0.001
    tampered["paired_sequence_effects"][0]["ci95_high"] += 0.001
    rehash(tampered)
    with pytest.raises(ContributionReportError, match="paired intervals differ"):
        validate_contribution_report(
            tampered,
            score_artifact_resolver=resolver,
        )

    missing = copy.deepcopy(report)
    missing["score_manifest_references"][0]["sha256"] = digest(
        "unretained score manifest"
    )
    rehash(missing)
    with pytest.raises(
        ContributionReportError,
        match="retained sequence-score evidence is invalid",
    ):
        validate_contribution_report(
            missing,
            score_artifact_resolver=resolver,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["primary"].update(
                {
                    "gain_recovery_percent": 9_999.0,
                    "cost_ratio_percent": 0.001,
                }
            ),
            "primary G2/C2",
        ),
        (
            lambda report: report["gate"].update(
                {"checks": {"made_up": True}, "passed": True}
            ),
            "gate checks",
        ),
        (
            lambda report: report["paired_sequence_effects"][0].update(
                {"score_artifact_sha256": digest("forged score")}
            ),
            "differs from its score artifact",
        ),
        (
            lambda report: report["secondary_end_to_end_cost"][
                "all_available_seeds"
            ].update({"a_total_chip_seconds": 1.0}),
            "secondary cost totals",
        ),
    ],
)
def test_deep_validator_rejects_rehashed_derived_evidence(mutate, message):
    report = build("phase2")
    mutate(report)
    if "gain_recovery_percent" in report["primary"]:
        report["headline"] = (
            "Seed-0 phase2 gate passed: exploratory G2=9999.0% and "
            "C2=0.0%. Seed 0 is permanently excluded from the confirmatory "
            "headline."
            if report["primary"]["gain_recovery_percent"] == 9_999.0
            else report["headline"]
        )
    rehash(report)
    with pytest.raises(ContributionReportError, match=message):
        validate_contribution_report(report)


def test_harvester_goodput_must_match_tokens_and_wall_time():
    rows, shared = rows_for("phase2")
    bad_harvester = harvester("phase2")
    bad_harvester["wall_seconds"] = 25.0
    bad_harvester["chip_seconds"] = 400.0
    score_refs, score_resolver = retained_score_artifacts(
        "phase2",
        rows,
        sequence_scores("phase2", rows),
    )
    with pytest.raises(ContributionReportError, match="capture_tokens / wall_seconds"):
        build_contribution_report(
            rows,
            score_refs,
            shared,
            score_artifact_resolver=score_resolver,
            phase="phase2",
            protocol_sha256=PROTOCOL,
            training_capture_manifest_sha256=PHASE2_TRAIN_CAPTURE,
            evaluation_capture_manifest_sha256=PHASE2_EVAL_CAPTURE,
            parent_training_capture_manifest_sha256=PHASE1_TRAIN_CAPTURE,
            parent_evaluation_capture_manifest_sha256=PHASE1_EVAL_CAPTURE,
            training_capture_lineage_receipt_sha256=digest("train lineage"),
            evaluation_capture_lineage_receipt_sha256=digest("eval lineage"),
            recomposition_sidecar_sha256=SIDECAR,
            harvester=bad_harvester,
            phase1_receipt=phase1_receipt(),
        )


def test_atomic_report_round_trip_and_hash_tamper_rejection(tmp_path):
    report = build("phase2")
    path = write_contribution_report(tmp_path / "report.json", report)
    assert json.loads(path.read_text()) == report
    assert "NaN" not in path.read_text()

    tampered = copy.deepcopy(report)
    tampered["primary"]["gain_recovery_percent"] = -1.0
    with pytest.raises(ContributionReportError, match="self-hash"):
        validate_contribution_report(tampered)
