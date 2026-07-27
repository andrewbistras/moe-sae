"""Scientific scoring/report tests for Experiment-3 transcoders."""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    TOP_K_EXPERTS,
    build_recomposition_sidecar,
    float32_to_bf16_bits,
)
from moe_sae.contribution_recomposition import SUPPORTED_ACCUMULATION_ORDER
from moe_sae.contribution_transcoder_evaluation import (
    CONDITIONS,
    ENDPOINTS,
    ContributionTranscoderEvaluationError,
    ContributionTranscoderDistributedEvaluator,
    ContributionTranscoderTokenScores,
    TranscoderPairedSequenceScores,
    aggregate_paired_sequence_scores,
    expert_rank_diagnostics,
    score_transcoder_predictions,
)
from moe_sae.contribution_sae import ContributionSAEConfig
from moe_sae.contribution_training import ContributionOptimizerConfig
from moe_sae.contribution_transcoder import forward_contribution_transcoder
from moe_sae.contribution_transcoder_distributed import (
    init_contribution_transcoder_distributed_state,
)
from moe_sae.distributed import DistributedConfig, create_mesh
from moe_sae.contribution_transcoder_report import (
    ContributionTranscoderReportError,
    RESULT_SCHEMA_VERSION,
    TranscoderTrainingLedger,
    build_condition_summary,
    build_transcoder_scientific_result,
    render_transcoder_report,
    validate_transcoder_scientific_result,
)
from moe_sae.contribution_transcoder_config import PROTOCOL_ID


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _sidecar() -> dict:
    ones = float32_to_bf16_bits(np.ones((HIDDEN_SIZE,), np.float32))
    return build_recomposition_sidecar(
        checkpoint_sha256="1" * 64,
        maxtext_fork_commit="2" * 40,
        source_fingerprints={
            "checkpoint": "1" * 64,
            "fork_tree": "3" * 64,
        },
        routed_rmsnorm_scale=ones,
        shared_rmsnorm_scale=ones,
        outer_ffn_rmsnorm_scale=ones,
        layer_scalar=float32_to_bf16_bits(np.ones((1,), np.float32)),
        accumulation_order=SUPPORTED_ACCUMULATION_ORDER,
    )


def test_host_scoring_uses_exact_common_shared_branch_and_padding() -> None:
    rng = np.random.default_rng(19)
    target = rng.normal(
        0.0,
        0.25,
        size=(3, TOP_K_EXPERTS, HIDDEN_SIZE),
    ).astype(np.float32)
    weights = np.broadcast_to(
        np.asarray(
            [[0.25, 0.1875, 0.15625, 0.125, 0.09375, 0.078125, 0.0625, 0.046875]],
            np.float32,
        ),
        (3, TOP_K_EXPERTS),
    ).copy()
    shared = float32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(3, HIDDEN_SIZE)).astype(np.float32)
    )
    valid = np.asarray([True, True, False])
    exact = score_transcoder_predictions(
        expert_output=target,
        reconstruction=target.copy(),
        router_weights=weights,
        shared_output_bf16_bits=shared,
        target_mean=np.zeros(HIDDEN_SIZE, np.float32),
        routed_mean=np.zeros(HIDDEN_SIZE, np.float32),
        oracle_shared_mean=np.zeros(HIDDEN_SIZE, np.float32),
        valid_mask=valid,
        recomposition_sidecar=_sidecar(),
    )

    assert exact.tokens == 2
    for endpoint in ENDPOINTS:
        sse, sst = exact.endpoint(endpoint)
        np.testing.assert_array_equal(sse, 0.0)
        assert float(sst[valid].sum()) > 0
        assert sst[-1] == 0
    np.testing.assert_array_equal(exact.route_component_sse, 0.0)

    changed_target = target.copy()
    changed_target[:2, :, 0] += np.float32(0.5)
    changed = score_transcoder_predictions(
        expert_output=target,
        reconstruction=changed_target,
        router_weights=weights,
        shared_output_bf16_bits=shared,
        target_mean=np.zeros(HIDDEN_SIZE, np.float32),
        routed_mean=np.zeros(HIDDEN_SIZE, np.float32),
        oracle_shared_mean=np.zeros(HIDDEN_SIZE, np.float32),
        valid_mask=valid,
        recomposition_sidecar=_sidecar(),
    )
    assert float(changed.component_sse.sum()) > 0
    assert float(changed.routed_sse.sum()) > 0
    assert float(changed.oracle_shared_sse.sum()) > 0
    assert changed.component_sse[-1] == 0
    assert changed.routed_sse[-1] == 0
    assert changed.oracle_shared_sse[-1] == 0


@pytest.mark.parametrize("condition", ["A", "R", "B", "F"])
@pytest.mark.parametrize("encoder_bias", [10.0, -1.0e6])
def test_distributed_transcoder_evaluator_uses_source_only_operand(
    condition: str,
    encoder_bias: float,
) -> None:
    layout = ContributionSAEConfig(
        condition=condition,  # type: ignore[arg-type]
        hidden_dim=3,
        num_experts=4,
        top_routed=2,
        bank_width=8 if condition == "F" else 4,
        k_per_contribution=2,
        randomization_key=17,
    )
    distributed = DistributedConfig(microbatches_per_update=2)
    mesh = create_mesh(distributed)
    optimizer = ContributionOptimizerConfig(
        parameter_dtype="float32",
        warmup_steps=0,
        total_steps=2,
    )
    state = init_contribution_transcoder_distributed_state(
        jax.random.key(3),
        mesh,
        layout,
        optimizer,
        distributed,
    )
    state = state._replace(
        params=state.params._replace(
            encoder=(
                jnp.zeros_like(state.params.encoder)
                if encoder_bias < 0
                else state.params.encoder
            ),
            encoder_bias=encoder_bias * jnp.ones_like(state.params.encoder_bias),
        )
    )
    source = np.arange(18, dtype=np.float32).reshape(6, 3) / 20.0
    target = np.arange(36, dtype=np.float32).reshape(6, 2, 3) / 30.0
    expert_banks = np.asarray(
        [[0, 1], [2, 3], [0, 2], [1, 3], [3, 2], [0, 1]],
        np.int32,
    )
    rank_banks = np.broadcast_to(np.arange(2), (6, 2)).astype(np.int32)
    banks = expert_banks if condition in {"A", "R"} else rank_banks
    weights = np.broadcast_to(
        np.asarray([[0.625, 0.375]], np.float32),
        (6, 2),
    ).copy()
    valid = np.asarray([True, True, True, True, False, False])
    microbatches = [
        {
            "routed_input": source[:3],
            "expert_output": target[:3],
            "bank_ids": banks[:3],
            "router_weights": weights[:3],
            "valid_mask": valid[:3],
        },
        {
            "routed_input": source[3:],
            "expert_output": target[3:],
            "bank_ids": banks[3:],
            "router_weights": weights[3:],
            "valid_mask": valid[3:],
        },
    ]
    source_mean = np.asarray([0.1, -0.2, 0.05], np.float32)
    target_mean = np.asarray([-0.1, 0.15, 0.02], np.float32)
    evaluator = ContributionTranscoderDistributedEvaluator(
        mesh,
        layout,
        distributed,
        bank_capacity=3 if condition in {"A", "R"} else None,
    )
    aux, chunk = evaluator.step(
        state.params,
        microbatches,
        source_mean,
        target_mean,
    )
    expected_reconstruction = np.asarray(
        forward_contribution_transcoder(
            state.params,
            jnp.asarray(source),
            jnp.asarray(banks),
            jnp.asarray(source_mean),
            jnp.asarray(target_mean),
            layout,
        ).reconstruction
    )
    expected_raw = np.einsum(
        "nkh,nk->nh",
        expected_reconstruction,
        weights,
        dtype=np.float32,
    ).reshape(2, 3, 3)
    expected_raw[1, 1:] = 0

    assert int(aux.valid_source_tokens) == 4
    assert int(aux.valid_components) == 8
    assert int(aux.l0_sum) == 4 * layout.total_l0
    assert bool(aux.topk_valid)
    assert not bool(aux.capacity_overflow)
    np.testing.assert_allclose(
        chunk.raw_routed_reconstruction,
        expected_raw,
        rtol=3e-5,
        atol=3e-5,
    )
    assert evaluator.complete_feature_mask(aux).shape == (
        layout.total_features,
    )


def _token_scores(
    component_sse: np.ndarray,
    *,
    common_sst: np.ndarray | None = None,
) -> ContributionTranscoderTokenScores:
    selected_sse = np.asarray(component_sse, dtype=np.float64)
    selected_sst = (
        np.asarray(common_sst, dtype=np.float64)
        if common_sst is not None
        else np.full(selected_sse.shape, 100.0, np.float64)
    )
    route_sse = np.broadcast_to(
        selected_sse[:, None] / TOP_K_EXPERTS,
        (selected_sse.size, TOP_K_EXPERTS),
    ).copy()
    route_sst = np.broadcast_to(
        selected_sst[:, None] / TOP_K_EXPERTS,
        (selected_sse.size, TOP_K_EXPERTS),
    ).copy()
    return ContributionTranscoderTokenScores(
        component_sse=selected_sse,
        component_sst=selected_sst,
        routed_sse=selected_sse * 1.1,
        routed_sst=selected_sst * 1.2,
        oracle_shared_sse=selected_sse * 0.9,
        oracle_shared_sst=selected_sst * 0.8,
        route_component_sse=route_sse,
        route_component_sst=route_sst,
        valid_mask=np.ones(selected_sse.shape, np.bool_),
        recomposition_sidecar_sha256=_digest("sidecar"),
    )


def _paired_scores(
    *,
    f_sse: np.ndarray | None = None,
) -> TranscoderPairedSequenceScores:
    condition_sse = {
        "A": np.asarray([30.0, 32.0, 28.0, 30.0]),
        "R": np.asarray([70.0, 68.0, 72.0, 70.0]),
        "B": np.asarray([50.0, 52.0, 48.0, 50.0]),
        "F": (
            np.asarray(f_sse, dtype=np.float64)
            if f_sse is not None
            else np.asarray([40.0, 42.0, 38.0, 40.0])
        ),
    }
    sequence = np.asarray([10, 10, 20, 20], np.int64)
    position = np.asarray([0, 1, 0, 1], np.uint16)
    return aggregate_paired_sequence_scores(
        scores={
            condition: _token_scores(values)
            for condition, values in condition_sse.items()
        },
        sequence_id=sequence,
        position=position,
        source_manifest_sha256=_digest("eval-source"),
        target_manifest_sha256=_digest("eval-target"),
    )


def test_paired_sequence_artifact_round_trip_and_manifest_binding() -> None:
    scores = _paired_scores()
    payload = scores.to_dict()
    restored = TranscoderPairedSequenceScores.from_dict(payload)

    assert restored.score_artifact_sha256 == scores.score_artifact_sha256
    assert restored.tokens == 4
    assert restored.clusters == 2
    assert restored.endpoint_ev("A", "component") == pytest.approx(0.7)
    assert restored.endpoint_ev("F", "component") == pytest.approx(0.6)
    assert restored.source_manifest_sha256 == _digest("eval-source")
    assert restored.target_manifest_sha256 == _digest("eval-target")

    tampered = copy.deepcopy(payload)
    tampered["condition_sse"]["A"]["component"][0] += 1.0
    with pytest.raises(
        ContributionTranscoderEvaluationError,
        match="self-hash",
    ):
        TranscoderPairedSequenceScores.from_dict(tampered)


def test_paired_aggregation_rejects_condition_specific_sst() -> None:
    scores = {condition: _token_scores(np.ones(2)) for condition in CONDITIONS}
    scores["R"] = _token_scores(
        np.ones(2),
        common_sst=np.asarray([100.0, 101.0]),
    )
    with pytest.raises(
        ContributionTranscoderEvaluationError,
        match="SST is not common",
    ):
        aggregate_paired_sequence_scores(
            scores=scores,
            sequence_id=np.asarray([1, 1], np.int64),
            position=np.asarray([0, 1], np.uint16),
            source_manifest_sha256=_digest("source"),
            target_manifest_sha256=_digest("target"),
        )


def test_expert_and_rank_diagnostics_are_target_id_based() -> None:
    scores = _token_scores(np.asarray([10.0, 20.0]))
    ids = np.asarray(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 8, 9, 10, 11, 12, 13],
        ],
        np.uint8,
    )
    result = expert_rank_diagnostics(scores, ids)

    assert result["supported_experts"] == 14
    assert result["expert_route_count"][0] == 2
    assert len(result["frequency_quartiles"]) == 4
    assert len(result["router_rank"]) == TOP_K_EXPERTS
    assert result["router_rank"][0]["route_count"] == 2


def _ledger(condition: str, chip_seconds: float) -> TranscoderTrainingLedger:
    chips = 16
    marginal_wall = chip_seconds / chips
    return TranscoderTrainingLedger(
        training_result_sha256=_digest(f"training:{condition}"),
        timing_receipt_sha256=_digest(f"timing:{condition}"),
        chip_count=chips,
        preprocessing_wall_seconds=1.0,
        optimizer_wall_seconds=marginal_wall - 1.0,
        marginal_wall_seconds=marginal_wall,
        marginal_chip_seconds=chip_seconds,
        compilation_wall_seconds=2.0,
        checkpoint_serialization_wall_seconds=3.0,
        evaluation_wall_seconds=4.0,
        total_wall_seconds=marginal_wall + 9.0,
        tokens_per_second=2_000_000 / marginal_wall,
        useful_mfu=0.15,
    )


def _result(
    scores: TranscoderPairedSequenceScores,
    *,
    replicates: int = 200,
) -> dict:
    summaries = {
        condition: build_condition_summary(
            scores,
            condition=condition,
            checkpoint_sha256=_digest(f"checkpoint:{condition}"),
            evaluation_receipt_sha256=_digest(f"evaluation:{condition}"),
            fired_feature_fraction=0.98,
            macro_expert_ev=0.25,
            supported_experts=128,
            diagnostics_sha256=_digest(f"diagnostics:{condition}"),
        )
        for condition in CONDITIONS
    }
    ledgers = {
        "A": _ledger("A", 1_600.0),
        "R": _ledger("R", 1_620.0),
        "B": _ledger("B", 800.0),
        "F": _ledger("F", 3_200.0),
    }
    return build_transcoder_scientific_result(
        scores=scores,
        summaries=summaries,
        ledgers=ledgers,
        source_revision="a" * 40,
        config_sha256=_digest("config"),
        train_source_manifest_sha256=_digest("train-source"),
        train_target_manifest_sha256=_digest("train-target"),
        common_sst_receipt_sha256=_digest("sst"),
        shared_harvest_receipt_sha256=_digest("harvest"),
        shared_harvest_chip_seconds=400.0,
        bootstrap_replicates=replicates,
    )


def test_report_recomputes_direct_a_f_gain_cost_and_effect_retention() -> None:
    scores = _paired_scores()
    result = _result(scores)

    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert result["protocol_id"] == PROTOCOL_ID
    assert result["status"] == "transfer_passed"
    assert result["primary"]["g3"]["status"] == "stable"
    assert result["primary"]["g3"]["gain_recovery_percent"] == pytest.approx(
        200.0
    )
    assert result["primary"]["c3_percent"] == pytest.approx(50.0)
    assert result["primary"]["effect_retention"]["alignment_ratio"] > 0
    direct = [
        row
        for row in result["paired_effects"]
        if row["endpoint"] == "component"
        and row["left"] == "A"
        and row["right"] == "F"
    ]
    assert len(direct) == 1
    assert direct[0]["effect"] == pytest.approx(0.1)
    assert result["parents"]["eval_source_manifest_sha256"] == _digest(
        "eval-source"
    )
    assert result["parents"]["eval_target_manifest_sha256"] == _digest(
        "eval-target"
    )
    assert (
        validate_transcoder_scientific_result(result, scores=scores)
        == result
    )
    markdown = render_transcoder_report(result)
    assert "200.00% (stable)" in markdown
    assert result["result_sha256"] in markdown


def test_result_validation_rejects_old_or_mislabeled_protocol() -> None:
    scores = _paired_scores()
    result = _result(scores)
    changed = copy.deepcopy(result)
    changed["protocol_id"] = "gemma4-contribution-transcoder-v1-20260727"
    unsigned = {
        name: value
        for name, value in changed.items()
        if name != "result_sha256"
    }
    from moe_sae.contribution_artifacts import canonical_sha256

    changed["result_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(
        ContributionTranscoderReportError,
        match="protocol",
    ):
        validate_transcoder_scientific_result(changed)


def test_g3_is_undefined_when_f_minus_b_is_nonpositive() -> None:
    scores = _paired_scores(
        f_sse=np.asarray([55.0, 57.0, 53.0, 55.0])
    )
    result = _result(scores)

    assert result["primary"]["g3"]["gain_recovery_percent"] is None
    assert result["primary"]["g3"]["status"] == "undefined"
    assert not result["verdict"]["passed"]


def test_result_replay_rejects_self_consistent_but_false_effect() -> None:
    scores = _paired_scores()
    result = _result(scores)
    changed = copy.deepcopy(result)
    changed["paired_effects"][0]["effect"] += 0.01
    unsigned = {
        name: value for name, value in changed.items() if name != "result_sha256"
    }
    from moe_sae.contribution_artifacts import canonical_sha256

    changed["result_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(
        ContributionTranscoderReportError,
        match="replayed",
    ):
        validate_transcoder_scientific_result(changed, scores=scores)
