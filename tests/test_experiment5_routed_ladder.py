from __future__ import annotations

import copy

import numpy as np
import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_routed_ladder import (
    BOOTSTRAP_REPLICATES,
    CAVEATS,
    CHECKPOINT_TOKENS,
    CONDITION_GEOMETRIES,
    ENDPOINTS,
    ExistingHReferences,
    Experiment5RoutedLadderError,
    ImmutableArtifactReference,
    PANELS,
    PROTOCOL_ID,
    RoutedLadderEvaluationBundleAccounting,
    RoutedLadderTrainingCost,
    RoutedLadderTokenScores,
    RoutedLadderTrajectoryAccounting,
    RoutedLadderWallCashAccounting,
    SEEDS,
    build_experiment5_routed_ladder_report,
    hierarchical_a512_relu_minus_a256_relu_bootstrap,
    hierarchical_a512_softplus_minus_a512_relu_bootstrap,
    paired_a512_relu_minus_a256_relu_bootstrap,
    paired_a512_softplus_minus_a512_relu_bootstrap,
    validate_experiment5_routed_ladder_report,
)


def _token_scores(
    condition: str,
    *,
    panel: str = "transcoder",
    seed: int = 1,
    checkpoint_tokens: int = 10_000_000,
    positions: np.ndarray | None = None,
    sst_scale: float = 1.0,
    a512_relu_gain: float = 0.05,
    a512_softplus_gain: float = 0.1,
) -> RoutedLadderTokenScores:
    sequence_id = np.asarray([10, 10, 11, 11, 12, 12], dtype=np.int64)
    selected_position = (
        np.asarray([0, 1, 0, 1, 0, 1], dtype=np.uint16)
        if positions is None
        else positions
    )
    endpoint_sst = {
        "component": sst_scale
        * np.asarray([1.0, 2.0, 1.5, 1.5, 2.0, 2.0], dtype=np.float64),
        "routed": sst_scale
        * np.asarray([2.0, 2.0, 3.0, 1.0, 2.5, 1.5], dtype=np.float64),
        "oracle_shared": sst_scale
        * np.asarray([3.0, 1.0, 2.0, 2.0, 1.0, 3.0], dtype=np.float64),
    }
    baseline_nmse = {
        "component": 0.50,
        "routed": 0.40,
        "oracle_shared": 0.30,
    }
    if condition == "A256ReLU":
        nmse = {
            endpoint: value + a512_relu_gain
            for endpoint, value in baseline_nmse.items()
        }
    elif condition == "A512ReLU":
        nmse = baseline_nmse
    else:
        nmse = {
            endpoint: value - a512_softplus_gain
            for endpoint, value in baseline_nmse.items()
        }
    endpoint_sse = {
        endpoint: values * nmse[endpoint]
        for endpoint, values in endpoint_sst.items()
    }
    return RoutedLadderTokenScores(
        condition=condition,  # type: ignore[arg-type]
        panel=panel,  # type: ignore[arg-type]
        seed=seed,
        checkpoint_tokens=checkpoint_tokens,
        source_revision="a" * 40,
        capture_manifest_sha256="b" * 64,
        recomposition_sidecar_sha256="c" * 64,
        sequence_id=sequence_id,
        position=selected_position,
        endpoint_sse=endpoint_sse,
        endpoint_sst=endpoint_sst,
    )


def _score(ev: float, sst: float) -> dict[str, float]:
    nmse = 1.0 - ev
    return {"sse": nmse * sst, "sst": sst, "nmse": nmse, "ev": ev}


def _artifact(kind: str, character: str) -> ImmutableArtifactReference:
    return ImmutableArtifactReference(
        kind=kind,
        schema_version=f"fixture-{kind}-v1",
        uri=f"gs://fixture/{kind}/{character * 64}.json",
        sha256=character * 64,
    )


def _h_reference(
    source: RoutedLadderTokenScores,
    *,
    token_identity_sha256: str | None = None,
    learned: bool = True,
) -> ExistingHReferences:
    aggregates = {endpoint: source.aggregate(endpoint) for endpoint in ENDPOINTS}
    endpoint_scores = {
        "component": _score(0.65, aggregates["component"]["sst"]),
        "routed": _score(0.75, aggregates["routed"]["sst"]),
        "oracle_shared": _score(0.85, aggregates["oracle_shared"]["sst"]),
        "learned_full_ffn": (
            _score(0.82, aggregates["oracle_shared"]["sst"])
            if learned
            else None
        ),
    }
    sst_proofs = {
        endpoint: source.endpoint_sst_sha256(endpoint) for endpoint in ENDPOINTS
    }
    sst_proofs["learned_full_ffn"] = (
        sst_proofs["oracle_shared"] if learned else None
    )
    return ExistingHReferences(
        panel=source.panel,
        seed=source.seed,
        checkpoint_tokens=source.checkpoint_tokens,
        source_revision="d" * 40,
        training_source_revision="4" * 40,
        h_config_sha256="5" * 64,
        training_result_sha256="6" * 64,
        training_result_file_sha256="7" * 64,
        checkpoint_pointer_sha256="8" * 64,
        checkpoint_pointer_file_sha256="9" * 64,
        score_artifact_sha256="0" * 64,
        token_identity_sha256=(
            source.token_identity_sha256
            if token_identity_sha256 is None
            else token_identity_sha256
        ),
        evaluation_tokens=source.tokens,
        packed_sequences=source.packed_sequences,
        endpoint_scores=endpoint_scores,
        endpoint_sst_sha256=sst_proofs,
        scientific_result=_artifact("scientific_result", "e"),
        diagnostics=(
            _artifact("paired_scores", "f"),
            _artifact("training_diagnostics", "1"),
        ),
    )


def _report_inputs_reference() -> ImmutableArtifactReference:
    return ImmutableArtifactReference(
        kind="report_inputs",
        schema_version="fixture-report-inputs-v1",
        uri="gs://fixture/report-inputs.json",
        sha256="a" * 64,
    )


_A256_MANIFEST_SHA = "2" * 64
_H_MANIFEST_SHA = "3" * 64


def _training_cost(
    *,
    resource_id: str,
    wall_seconds: float = 10.0,
) -> RoutedLadderTrainingCost:
    return RoutedLadderTrainingCost(
        resource_id=resource_id,
        accelerator_type="v6e-16",
        chip_count=16,
        marginal_wall_seconds=wall_seconds,
        chip_seconds=16 * wall_seconds,
        chip_seconds_scope="training",
        chip_seconds_excludes=("compile", "checkpoint"),
    )


def _trajectory_accounting(
    *,
    include_h_cost: bool = False,
) -> list[RoutedLadderTrajectoryAccounting]:
    rows: list[RoutedLadderTrajectoryAccounting] = []
    for panel in PANELS:
        for condition in ("A256ReLU", "A512ReLU", "A512Softplus", "H"):
            for seed in SEEDS:
                historical_a256 = (
                    panel == "sae"
                    and condition == "A256ReLU"
                    and seed == 0
                )
                if historical_a256:
                    origin = "historical_experiment4_seed0_reference"
                    manifest_kind = "a256_reference_manifest"
                    manifest_sha = _A256_MANIFEST_SHA
                    training_revision = "4" * 40
                    result_sha = "8" * 64
                    result_file_sha = "9" * 64
                    cost_status = "measured_historical_training_result"
                    cost = _training_cost(
                        resource_id="historical-a256",
                    )
                elif condition == "H":
                    origin = "historical_experiment5_reference"
                    manifest_kind = "h_reference_manifest"
                    manifest_sha = _H_MANIFEST_SHA
                    training_revision = "4" * 40
                    result_sha = "6" * 64
                    result_file_sha = "7" * 64
                    cost_status = (
                        "measured_historical_training_result"
                        if include_h_cost
                        else "not_supplied_by_h_reference_manifest"
                    )
                    cost = (
                        _training_cost(
                            resource_id=f"historical-h-{panel}-seed{seed}",
                        )
                        if include_h_cost
                        else None
                    )
                else:
                    origin = "protocol_v2_new_training"
                    manifest_kind = None
                    manifest_sha = None
                    training_revision = "a" * 40
                    digest = (
                        100
                        + 20 * PANELS.index(panel)
                        + 3
                        * ("A256ReLU", "A512ReLU", "A512Softplus").index(
                            condition
                        )
                        + seed
                    )
                    result_sha = f"{digest:064x}"
                    result_file_sha = f"{digest + 40:064x}"
                    cost_status = "measured_v2_training_result"
                    cost = _training_cost(
                        resource_id=f"v2-{panel}-{condition}-seed{seed}",
                    )
                rows.append(
                    RoutedLadderTrajectoryAccounting(
                        panel=panel,  # type: ignore[arg-type]
                        condition=condition,  # type: ignore[arg-type]
                        seed=seed,
                        origin=origin,
                        training_source_revision=training_revision,
                        reference_manifest_kind=manifest_kind,
                        reference_manifest_sha256=manifest_sha,
                        parent_training_result_sha256=result_sha,
                        parent_training_result_file_sha256=result_file_sha,
                        cost_status=cost_status,
                        training_cost=cost,
                    )
                )
    return rows


def _bundle_accounting() -> list[RoutedLadderEvaluationBundleAccounting]:
    rows: list[RoutedLadderEvaluationBundleAccounting] = []
    for panel in PANELS:
        for seed in SEEDS:
            index = 200 + 10 * PANELS.index(panel) + seed
            rows.append(
                RoutedLadderEvaluationBundleAccounting(
                    panel=panel,  # type: ignore[arg-type]
                    seed=seed,
                    source_revision="a" * 40,
                    operator_manifest_sha256=f"{index:064x}",
                    bundle_spec_sha256=f"{index + 20:064x}",
                    bundle_result_sha256=f"{index + 40:064x}",
                    bundle_result_file_sha256=f"{index + 60:064x}",
                    accelerator_type="v6e-16",
                    chip_count=16,
                    successful_attempt_wall_seconds=5.0,
                    successful_attempt_chip_seconds=80.0,
                )
            )
    return rows


def _wall_cash_accounting() -> RoutedLadderWallCashAccounting:
    return RoutedLadderWallCashAccounting(
        campaign_wall_seconds=90.0,
        wall_scope=(
            "durable_campaign_clock_through_evaluation_completion"
        ),
        cash_usd=1.25,
        cash_currency="USD",
        cash_basis="controller_cost_projection",
        accounting_source=ImmutableArtifactReference(
            kind="cost_projection",
            schema_version="fixture-cost-projection-v1",
            uri="gs://fixture/cost-projection.json",
            sha256="4" * 64,
        ),
    )


def _report_accounting_inputs(
    *,
    include_h_cost: bool = False,
) -> dict:
    return {
        "trajectory_accounting": _trajectory_accounting(
            include_h_cost=include_h_cost
        ),
        "evaluation_bundle_accounting": _bundle_accounting(),
        "a256_reference_manifest_sha256": _A256_MANIFEST_SHA,
        "h_reference_manifest_sha256": _H_MANIFEST_SHA,
        "wall_cash_accounting": _wall_cash_accounting(),
    }


def _full_report_inputs(
    *,
    gains: dict[int, float] | None = None,
    learned: bool = True,
) -> tuple[
    list[RoutedLadderTokenScores],
    list[ExistingHReferences],
    dict[tuple[str, str, int, int], RoutedLadderTokenScores],
]:
    selected_gains = gains or {seed: 0.1 for seed in SEEDS}
    scores: list[RoutedLadderTokenScores] = []
    h_references: list[ExistingHReferences] = []
    by_role: dict[
        tuple[str, str, int, int],
        RoutedLadderTokenScores,
    ] = {}
    for panel in PANELS:
        for seed in SEEDS:
            for checkpoint_tokens in CHECKPOINT_TOKENS:
                a512_relu = _token_scores(
                    "A512ReLU",
                    panel=panel,
                    seed=seed,
                    checkpoint_tokens=checkpoint_tokens,
                )
                a256_relu = _token_scores(
                    "A256ReLU",
                    panel=panel,
                    seed=seed,
                    checkpoint_tokens=checkpoint_tokens,
                )
                a512_softplus = _token_scores(
                    "A512Softplus",
                    panel=panel,
                    seed=seed,
                    checkpoint_tokens=checkpoint_tokens,
                    a512_softplus_gain=selected_gains[seed],
                )
                scores.extend((a256_relu, a512_relu, a512_softplus))
                h_references.append(_h_reference(a512_softplus, learned=learned))
                by_role[("A256ReLU", panel, seed, checkpoint_tokens)] = a256_relu
                by_role[("A512ReLU", panel, seed, checkpoint_tokens)] = a512_relu
                by_role[
                    ("A512Softplus", panel, seed, checkpoint_tokens)
                ] = a512_softplus
    return scores, h_references, by_role


def _replace_score(
    scores: list[RoutedLadderTokenScores],
    replacement: RoutedLadderTokenScores,
) -> list[RoutedLadderTokenScores]:
    return [
        replacement
        if (
            row.condition,
            row.panel,
            row.seed,
            row.checkpoint_tokens,
        )
        == (
            replacement.condition,
            replacement.panel,
            replacement.seed,
            replacement.checkpoint_tokens,
        )
        else row
        for row in scores
    ]


def test_condition_geometries_freeze_scale_then_activation_ladder() -> None:
    a256_relu = CONDITION_GEOMETRIES["A256ReLU"]
    a512_relu = CONDITION_GEOMETRIES["A512ReLU"]
    a512_softplus = CONDITION_GEOMETRIES["A512Softplus"]
    hybrid = CONDITION_GEOMETRIES["H"]

    assert (
        a256_relu.total_features,
        a256_relu.candidates_per_token,
        a256_relu.exact_l0,
        a256_relu.trainable_parameters,
    ) == (32_768, 2_048, 16, 184_582_144)
    assert (
        a512_relu.total_features,
        a512_relu.candidates_per_token,
        a512_relu.exact_l0,
        a512_relu.trainable_parameters,
    ) == (65_536, 4_096, 32, 369_164_288)
    assert (
        a512_softplus.total_features,
        a512_softplus.candidates_per_token,
        a512_softplus.exact_l0,
        a512_softplus.trainable_parameters,
    ) == (65_536, 4_096, 32, 369_164_288)
    assert (
        hybrid.total_features,
        hybrid.candidates_per_token,
        hybrid.exact_l0,
        hybrid.trainable_parameters,
    ) == (69_632, 8_192, 64, 392_237_056)
    assert a512_relu.report_row() | {
        "condition": "A512Softplus",
        "encoder_activation": "softplus",
    } == a512_softplus.report_row()
    assert a512_relu.encoder_activation == "relu"
    assert a256_relu.encoder_activation == "relu"
    assert a512_softplus.encoder_activation == hybrid.encoder_activation == (
        "softplus"
    )
    assert (
        a256_relu.shared_treatment
        == a512_relu.shared_treatment
        == a512_softplus.shared_treatment
        == "oracle_exact"
    )
    assert hybrid.shared_treatment == "learned_sparse_dictionary"
    assert PROTOCOL_ID == (
        "gemma4-experiment5-routed-activation-ladder-v2-20260729"
    )


def test_scores_aggregate_and_pair_on_packed_sequences_deterministically() -> None:
    a256_relu = _token_scores("A256ReLU")
    a512_relu = _token_scores("A512ReLU")
    a512_softplus = _token_scores("A512Softplus")
    scale = paired_a512_relu_minus_a256_relu_bootstrap(
        a256_relu,
        a512_relu,
        endpoint="routed",
    )

    assert a512_relu.aggregate("routed")["nmse"] == pytest.approx(0.4)
    assert a512_softplus.aggregate("routed")["ev"] == pytest.approx(0.7)
    first = paired_a512_softplus_minus_a512_relu_bootstrap(
        a512_relu,
        a512_softplus,
        endpoint="routed",
    )
    second = paired_a512_softplus_minus_a512_relu_bootstrap(
        a512_relu,
        a512_softplus,
        endpoint="routed",
    )

    assert first == second
    assert first["estimate_ev_difference"] == pytest.approx(0.1)
    assert first["ci95_lower"] == pytest.approx(0.1)
    assert first["ci95_upper"] == pytest.approx(0.1)
    assert first["bootstrap_replicates"] == BOOTSTRAP_REPLICATES
    assert first["bootstrap_unit"] == "packed_sequence"
    assert isinstance(first["bootstrap_seed"], int)
    assert scale["estimate_ev_difference"] == pytest.approx(0.05)
    assert scale["contrast"] == "A512ReLU-A256ReLU"


def test_hierarchical_bootstrap_reports_three_seed_mean_and_positive_count() -> None:
    pairs = [
        (
            _token_scores(
                "A512ReLU",
                panel="sae",
                seed=seed,
                checkpoint_tokens=5_000_000,
            ),
            _token_scores(
                "A512Softplus",
                panel="sae",
                seed=seed,
                checkpoint_tokens=5_000_000,
                a512_softplus_gain=gain,
            ),
        )
        for seed, gain in zip(SEEDS, (-0.02, 0.1, 0.2), strict=True)
    ]
    first = hierarchical_a512_softplus_minus_a512_relu_bootstrap(
        pairs,
        endpoint="component",
    )
    second = hierarchical_a512_softplus_minus_a512_relu_bootstrap(
        pairs,
        endpoint="component",
    )

    assert first == second
    assert first["estimate_mean_ev_difference"] == pytest.approx(
        (-0.02 + 0.1 + 0.2) / 3
    )
    assert first["positive_seed_count"] == 2
    assert first["seeds"] == [0, 1, 2]
    assert first["bootstrap_unit"] == "seed_then_packed_sequence"
    assert first["ci95_lower"] <= first["estimate_mean_ev_difference"]
    assert first["ci95_upper"] >= first["estimate_mean_ev_difference"]

    scale_pairs = [
        (
            _token_scores(
                "A256ReLU",
                panel="sae",
                seed=seed,
                checkpoint_tokens=5_000_000,
                a512_relu_gain=gain,
            ),
            _token_scores(
                "A512ReLU",
                panel="sae",
                seed=seed,
                checkpoint_tokens=5_000_000,
            ),
        )
        for seed, gain in zip(SEEDS, (0.01, 0.05, 0.09), strict=True)
    ]
    scale = hierarchical_a512_relu_minus_a256_relu_bootstrap(
        scale_pairs,
        endpoint="component",
    )
    assert scale["estimate_mean_ev_difference"] == pytest.approx(0.05)
    assert scale["positive_seed_count"] == 3


def test_report_binds_h_points_but_keeps_h_gaps_descriptive() -> None:
    scores, h_references, _by_role = _full_report_inputs()
    report = build_experiment5_routed_ladder_report(
        score_sets=list(reversed(scores)),
        h_references=list(reversed(h_references)),
        report_inputs=_report_inputs_reference(),
        **_report_accounting_inputs(),
    )

    assert validate_experiment5_routed_ladder_report(report) == report
    assert len(report["cohorts"]) == 18
    assert sum(len(row["conditions"]) for row in report["cohorts"]) == 72
    assert report["report_inputs"] == _report_inputs_reference().report_row()
    assert report["interpretation"]["headline_authorized"] is False
    assert report["interpretation"]["caveats"] == list(CAVEATS)
    cohort = next(
        row
        for row in report["cohorts"]
        if row["identity"]
        == {
            "panel": "transcoder",
            "seed": 1,
            "checkpoint_tokens": 10_000_000,
        }
    )
    assert cohort["conditions"]["A256ReLU"]["endpoints"]["component"]["ev"] == (
        pytest.approx(0.45)
    )
    assert cohort["conditions"]["A256ReLU"]["trajectory_origin"] == (
        "protocol_v2_new_training"
    )
    assert cohort["conditions"]["A512ReLU"]["endpoints"]["component"]["ev"] == (
        pytest.approx(0.5)
    )
    assert cohort["conditions"]["A512Softplus"]["endpoints"]["oracle_shared"]["ev"] == (
        pytest.approx(0.8)
    )
    assert cohort["conditions"]["A512ReLU"]["encoder_activation"] == "relu"
    assert (
        cohort["conditions"]["A512Softplus"]["encoder_activation"]
        == cohort["conditions"]["H"]["encoder_activation"]
        == "softplus"
    )
    assert cohort["conditions"]["H"]["endpoints"]["learned_full_ffn"]["ev"] == (
        pytest.approx(0.82)
    )
    gaps = cohort["descriptive_noncausal_h_gaps"]
    assert gaps["h_oracle_minus_a512_softplus_oracle"] == {
        "status": "descriptive_noncausal",
        "endpoint": "oracle_shared",
        "estimate_ev_difference": pytest.approx(0.05),
        "paired_interval_available": False,
    }
    assert gaps["h_oracle_minus_h_learned"] == {
        "status": "descriptive_noncausal",
        "endpoint": "full_ffn_common_sst",
        "estimate_ev_difference": pytest.approx(0.03),
        "paired_interval_available": False,
    }
    hierarchical = next(
        row
        for row in report["across_seed_a512_softplus_minus_a512_relu"]
        if row["identity"]
        == {
            "panel": "transcoder",
            "checkpoint_tokens": 10_000_000,
        }
    )["paired_mean_a512_softplus_minus_a512_relu"]["oracle_shared"]
    assert hierarchical["estimate_mean_ev_difference"] == pytest.approx(0.1)
    assert hierarchical["positive_seed_count"] == 3
    scale_hierarchical = next(
        row
        for row in report["across_seed_a512_relu_minus_a256_relu"]
        if row["identity"]
        == {
            "panel": "transcoder",
            "checkpoint_tokens": 10_000_000,
        }
    )["paired_mean_a512_relu_minus_a256_relu"]["oracle_shared"]
    assert scale_hierarchical["estimate_mean_ev_difference"] == pytest.approx(0.05)
    assert scale_hierarchical["positive_seed_count"] == 3

    historical = next(
        row
        for row in report["cohorts"]
        if row["identity"]
        == {
            "panel": "sae",
            "seed": 0,
            "checkpoint_tokens": 2_000_000,
        }
    )
    assert historical["conditions"]["A256ReLU"]["trajectory_origin"] == (
        "historical_experiment4_seed0_reference"
    )
    fresh_transcoder_seed0 = next(
        row
        for row in report["cohorts"]
        if row["identity"]
        == {
            "panel": "transcoder",
            "seed": 0,
            "checkpoint_tokens": 2_000_000,
        }
    )
    assert (
        fresh_transcoder_seed0["conditions"]["A256ReLU"]["trajectory_origin"]
        == "protocol_v2_new_training"
    )
    assert report["interpretation"]["a256_seed_provenance"] == {
        "sae_seed_0": "historical Experiment-4 immutable reference",
        "sae_seeds_1_2": "new protocol-v2 training",
        "transcoder_seeds_0_1_2": "new protocol-v2 training",
        "excluded_transcoder_reference": (
            "Experiment-4 transcoder A256 seed 0 used Softplus"
        ),
        "three_seed_summary_policy": (
            "unavailable until all three exact cells in a panel validate"
        ),
    }

    tampered = copy.deepcopy(report)
    tampered["cohorts"][0]["conditions"]["A512Softplus"]["endpoints"]["routed"][
        "ev"
    ] = 0.99
    with pytest.raises(Experiment5RoutedLadderError, match="self-hash"):
        validate_experiment5_routed_ladder_report(tampered)


def test_complete_report_rejects_missing_h_learned_point() -> None:
    scores, h_references, _by_role = _full_report_inputs(learned=False)

    with pytest.raises(
        Experiment5RoutedLadderError,
        match="requires every H learned-full",
    ):
        build_experiment5_routed_ladder_report(
            score_sets=scores,
            h_references=h_references,
            report_inputs=_report_inputs_reference(),
            **_report_accounting_inputs(),
        )


def test_report_rejects_relabeling_fresh_transcoder_a256_as_historical() -> None:
    scores, h_references, _by_role = _full_report_inputs()
    report = build_experiment5_routed_ladder_report(
        score_sets=scores,
        h_references=h_references,
        report_inputs=_report_inputs_reference(),
        **_report_accounting_inputs(),
    )
    tampered = copy.deepcopy(report)
    selected = next(
        row
        for row in tampered["cohorts"]
        if row["identity"]
        == {
            "panel": "transcoder",
            "seed": 0,
            "checkpoint_tokens": 2_000_000,
        }
    )
    selected["conditions"]["A256ReLU"]["trajectory_origin"] = (
        "historical_experiment4_seed0_reference"
    )
    body = {
        key: value for key, value in tampered.items() if key != "report_sha256"
    }
    tampered["report_sha256"] = canonical_sha256(body)

    with pytest.raises(Experiment5RoutedLadderError, match="identity changed"):
        validate_experiment5_routed_ladder_report(tampered)


def test_report_rejects_nonidentical_a_token_rows() -> None:
    scores, h_references, _by_role = _full_report_inputs()
    changed_positions = np.asarray([0, 1, 0, 1, 0, 2], dtype=np.uint16)
    a512_softplus = _token_scores(
        "A512Softplus",
        positions=changed_positions,
    )

    with pytest.raises(Experiment5RoutedLadderError, match="token identities"):
        build_experiment5_routed_ladder_report(
            score_sets=_replace_score(scores, a512_softplus),
            h_references=h_references,
            report_inputs=_report_inputs_reference(),
            **_report_accounting_inputs(),
        )


def test_report_rejects_different_per_token_common_sst() -> None:
    scores, h_references, _by_role = _full_report_inputs()
    a512_softplus = _token_scores("A512Softplus", sst_scale=2.0)

    with pytest.raises(Experiment5RoutedLadderError, match="target SST rows"):
        build_experiment5_routed_ladder_report(
            score_sets=_replace_score(scores, a512_softplus),
            h_references=h_references,
            report_inputs=_report_inputs_reference(),
            **_report_accounting_inputs(),
        )


def test_report_rejects_h_without_the_same_token_and_sst_proofs() -> None:
    scores, h_references, by_role = _full_report_inputs()
    a512_softplus = by_role[("A512Softplus", "transcoder", 1, 10_000_000)]
    bad_h = _h_reference(a512_softplus, token_identity_sha256="9" * 64)
    selected_h = [
        bad_h if row.identity == bad_h.identity else row
        for row in h_references
    ]

    with pytest.raises(Experiment5RoutedLadderError, match="token identity"):
        build_experiment5_routed_ladder_report(
            score_sets=scores,
            h_references=selected_h,
            report_inputs=_report_inputs_reference(),
            **_report_accounting_inputs(),
        )


def test_report_rejects_an_incomplete_curve_grid() -> None:
    scores, h_references, _by_role = _full_report_inputs()

    with pytest.raises(Experiment5RoutedLadderError, match="exact 18-cohort"):
        build_experiment5_routed_ladder_report(
            score_sets=scores[:-3],
            h_references=h_references[:-1],
            report_inputs=_report_inputs_reference(),
            **_report_accounting_inputs(),
        )


def test_token_scores_reject_negative_scientific_scores() -> None:
    valid = _token_scores("A512ReLU")
    bad_sse = dict(valid.endpoint_sse)
    bad_sse["component"] = np.asarray(
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    with pytest.raises(
        Experiment5RoutedLadderError,
        match="finite nonnegative",
    ):
        RoutedLadderTokenScores(
            condition="A512ReLU",
            panel=valid.panel,
            seed=valid.seed,
            checkpoint_tokens=valid.checkpoint_tokens,
            source_revision=valid.source_revision,
            capture_manifest_sha256=valid.capture_manifest_sha256,
            recomposition_sidecar_sha256=valid.recomposition_sidecar_sha256,
            sequence_id=valid.sequence_id,
            position=valid.position,
            endpoint_sse=bad_sse,
            endpoint_sst=valid.endpoint_sst,
        )
