from __future__ import annotations

import copy

import numpy as np
import pytest

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
    RoutedLadderTokenScores,
    SEEDS,
    build_experiment5_routed_ladder_report,
    hierarchical_a512_softplus_minus_a512_relu_bootstrap,
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
    nmse = (
        baseline_nmse
        if condition == "A512ReLU"
        else {
            endpoint: value - a512_softplus_gain
            for endpoint, value in baseline_nmse.items()
        }
    )
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
                a512_softplus = _token_scores(
                    "A512Softplus",
                    panel=panel,
                    seed=seed,
                    checkpoint_tokens=checkpoint_tokens,
                    a512_softplus_gain=selected_gains[seed],
                )
                scores.extend((a512_relu, a512_softplus))
                h_references.append(_h_reference(a512_softplus, learned=learned))
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


def test_condition_geometries_isolate_activation_between_full_sized_a_arms() -> None:
    a512_relu = CONDITION_GEOMETRIES["A512ReLU"]
    a512_softplus = CONDITION_GEOMETRIES["A512Softplus"]
    hybrid = CONDITION_GEOMETRIES["H"]

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
    assert a512_softplus.encoder_activation == hybrid.encoder_activation == (
        "softplus"
    )
    assert (
        a512_relu.shared_treatment
        == a512_softplus.shared_treatment
        == "oracle_exact"
    )
    assert hybrid.shared_treatment == "learned_sparse_dictionary"
    assert PROTOCOL_ID == (
        "gemma4-experiment5-routed-activation-ladder-v1-20260729"
    )


def test_scores_aggregate_and_pair_on_packed_sequences_deterministically() -> None:
    a512_relu = _token_scores("A512ReLU")
    a512_softplus = _token_scores("A512Softplus")

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
    assert first["bootstrap_seed"] == 11_962_227_367_045_583_585


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


def test_report_binds_h_points_but_keeps_h_gaps_descriptive() -> None:
    scores, h_references, _by_role = _full_report_inputs()
    report = build_experiment5_routed_ladder_report(
        score_sets=list(reversed(scores)),
        h_references=list(reversed(h_references)),
        report_inputs=_report_inputs_reference(),
    )

    assert validate_experiment5_routed_ladder_report(report) == report
    assert len(report["cohorts"]) == 18
    assert sum(len(row["conditions"]) for row in report["cohorts"]) == 54
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
        )


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
        )


def test_report_rejects_different_per_token_common_sst() -> None:
    scores, h_references, _by_role = _full_report_inputs()
    a512_softplus = _token_scores("A512Softplus", sst_scale=2.0)

    with pytest.raises(Experiment5RoutedLadderError, match="target SST rows"):
        build_experiment5_routed_ladder_report(
            score_sets=_replace_score(scores, a512_softplus),
            h_references=h_references,
            report_inputs=_report_inputs_reference(),
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
        )


def test_report_rejects_an_incomplete_curve_grid() -> None:
    scores, h_references, _by_role = _full_report_inputs()

    with pytest.raises(Experiment5RoutedLadderError, match="exact 18-cohort"):
        build_experiment5_routed_ladder_report(
            score_sets=scores[:-2],
            h_references=h_references[:-1],
            report_inputs=_report_inputs_reference(),
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
