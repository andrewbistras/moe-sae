from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from moe_sae.experiment5_routed_ladder import (
    ImmutableArtifactReference,
    RoutedLadderHTokenScores,
    h_token_scores_to_existing_reference,
)
from moe_sae.experiment5_routed_ladder_evaluation_worker import (
    BUNDLE_CONDITIONS,
    BUNDLE_SPEC_SCHEMA_VERSION,
    CHECKPOINT_TOKENS,
    ENDPOINT_CENTERING_ALGORITHM_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    RoutedLadderEvaluationError,
    _cached_endpoint_training_means,
    _endpoint_centering_cache_identity,
    load_merged_scores,
    load_process_evidence,
    parse_resident_bundle_spec,
    main,
    merge_process_evidence,
    seal_resident_bundle_result,
    seal_merged_result,
    seal_process_evidence,
    run_resident_bundle,
)
from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.experiment5_routed_ladder import PROTOCOL_ID


SOURCE_REVISION = "a" * 40
TRAINING_REVISION = "b" * 40
TARGET_MANIFEST = "c" * 64
RECOMPOSITION_SIDECAR = "d" * 64
DIGESTS = {
    name: f"{index:064x}"
    for index, name in enumerate(
        (
            "config",
            "config_file",
            "training_result",
            "training_result_file",
            "training_contract",
            "training_contract_file",
            "checkpoint",
            "checkpoint_file",
            "target",
            "sidecar",
            "centering",
            "input0",
            "input1",
        ),
        start=1,
    )
}


def _endpoint_values(
    *,
    tokens: int = 10,
) -> tuple[np.ndarray, str, np.ndarray, str, dict]:
    routed = np.linspace(-1.0, 1.0, 2_816, dtype=np.float32)
    oracle = np.linspace(2.0, 4.0, 2_816, dtype=np.float32)

    def digest(domain: bytes, value: np.ndarray) -> str:
        selected = hashlib.sha256()
        selected.update(domain + b"\0")
        selected.update(
            value.astype("<f4", copy=False).tobytes(order="C")
        )
        return selected.hexdigest()

    routed_sha = digest(
        b"gemma4-transcoder-routed-endpoint-mean-v1",
        routed,
    )
    oracle_sha = digest(
        b"gemma4-transcoder-oracle-shared-endpoint-mean-v1",
        oracle,
    )
    body = {
        "schema_version": ENDPOINT_CENTERING_ALGORITHM_SCHEMA_VERSION,
        "training_target_manifest_sha256": TARGET_MANIFEST,
        "recomposition_sidecar_sha256": RECOMPOSITION_SIDECAR,
        "tokens": tokens,
        "routed_mean_sha256": routed_sha,
        "oracle_shared_mean_sha256": oracle_sha,
        "reduction_dtype": "float64_high_residual_low_float32_collective",
    }
    from moe_sae.contribution_artifacts import canonical_sha256

    return (
        routed,
        routed_sha,
        oracle,
        oracle_sha,
        {**body, "receipt_sha256": canonical_sha256(body)},
    )


def _scores(
    scale: float = 1.0,
    *,
    include_learned: bool = False,
) -> SimpleNamespace:
    valid = np.ones((2,), dtype=np.bool_)
    return SimpleNamespace(
        valid_mask=valid,
        component_sse=np.asarray([2.0, 3.0]) * scale,
        component_sst=np.asarray([10.0, 12.0]),
        routed_sse=np.asarray([1.5, 2.5]) * scale,
        routed_sst=np.asarray([9.0, 11.0]),
        oracle_shared_sse=np.asarray([1.0, 2.0]) * scale,
        oracle_shared_sst=np.asarray([8.0, 10.0]),
        learned_full_ffn_sse=(
            np.asarray([1.2, 2.2]) * scale if include_learned else None
        ),
        learned_full_ffn_sst=(
            np.asarray([8.0, 10.0]) if include_learned else None
        ),
    )


def _seal_lane(
    root: Path,
    *,
    process_index: int,
    condition: str = "A512ReLU",
    checkpoint_tokens: int = 2_000_000,
) -> Path:
    sequence = np.asarray(
        [10 + process_index, 10 + process_index],
        dtype=np.int64,
    )
    position = np.asarray([0, 1], dtype=np.uint16)
    seal_process_evidence(
        root,
        panel="sae",
        condition=condition,
        seed=0,
        checkpoint_tokens=checkpoint_tokens,
        checkpoint_actual_tokens=2_015_232,
        process_index=process_index,
        process_count=2,
        source_revision=SOURCE_REVISION,
        training_source_revision=TRAINING_REVISION,
        config_sha256=DIGESTS["config"],
        config_file_sha256=DIGESTS["config_file"],
        h_config_sha256=DIGESTS["config"] if condition == "H" else None,
        h_config_file_sha256=(
            DIGESTS["config_file"] if condition == "H" else None
        ),
        training_result_sha256=DIGESTS["training_result"],
        training_contract_sha256=DIGESTS["training_contract"],
        checkpoint_sha256=DIGESTS["checkpoint"],
        parent_training_result_sha256=DIGESTS["training_result"],
        parent_training_result_file_sha256=DIGESTS[
            "training_result_file"
        ],
        parent_run_contract_sha256=(
            None if condition == "H" else DIGESTS["training_contract"]
        ),
        parent_run_contract_file_sha256=(
            None if condition == "H" else DIGESTS["training_contract_file"]
        ),
        parent_checkpoint_pointer_sha256=DIGESTS["checkpoint"],
        parent_checkpoint_pointer_file_sha256=DIGESTS[
            "checkpoint_file"
        ],
        target_manifest_sha256=DIGESTS["target"],
        source_manifest_sha256=None,
        recomposition_sidecar_sha256=DIGESTS["sidecar"],
        centering_receipt_sha256=DIGESTS["centering"],
        input_receipt_sha256=DIGESTS[f"input{process_index}"],
        learned_input_receipt_sha256=(
            DIGESTS[f"input{process_index}"] if condition == "H" else None
        ),
        evaluation_wall_seconds=1.0 + process_index,
        sequence_id=sequence,
        position=position,
        scores=_scores(
            1.0 + 0.1 * process_index,
            include_learned=condition == "H",
        ),
    )
    return root / "process-evidence.json"


def test_endpoint_centering_cache_miss_then_validated_hit(
    tmp_path: Path,
) -> None:
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return _endpoint_values()

    first, first_status = _cached_endpoint_training_means(
        tmp_path,
        target_manifest_sha256=TARGET_MANIFEST,
        recomposition_sidecar_sha256=RECOMPOSITION_SIDECAR,
        expected_tokens=10,
        compute=compute,
    )
    second, second_status = _cached_endpoint_training_means(
        tmp_path,
        target_manifest_sha256=TARGET_MANIFEST,
        recomposition_sidecar_sha256=RECOMPOSITION_SIDECAR,
        expected_tokens=10,
        compute=compute,
    )

    assert (first_status, second_status, calls) == ("miss", "hit", 1)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[2], second[2])
    assert first[1] == second[1]
    assert first[3:] == second[3:]


def test_endpoint_centering_cache_force_recompute_is_create_only_replay(
    tmp_path: Path,
) -> None:
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return _endpoint_values()

    _cached_endpoint_training_means(
        tmp_path,
        target_manifest_sha256=TARGET_MANIFEST,
        recomposition_sidecar_sha256=RECOMPOSITION_SIDECAR,
        expected_tokens=10,
        compute=compute,
    )
    replay, status = _cached_endpoint_training_means(
        tmp_path,
        target_manifest_sha256=TARGET_MANIFEST,
        recomposition_sidecar_sha256=RECOMPOSITION_SIDECAR,
        expected_tokens=10,
        compute=compute,
        force_recompute=True,
    )

    assert status == "miss"
    assert calls == 2
    assert replay[4]["tokens"] == 10


def test_endpoint_centering_cache_rejects_tampered_data_object(
    tmp_path: Path,
) -> None:
    _cached_endpoint_training_means(
        tmp_path,
        target_manifest_sha256=TARGET_MANIFEST,
        recomposition_sidecar_sha256=RECOMPOSITION_SIDECAR,
        expected_tokens=10,
        compute=_endpoint_values,
    )
    identity = _endpoint_centering_cache_identity(
        target_manifest_sha256=TARGET_MANIFEST,
        recomposition_sidecar_sha256=RECOMPOSITION_SIDECAR,
    )
    root = (
        tmp_path
        / "endpoint-centering"
        / identity["cache_key_sha256"]
    )
    metadata = json.loads(
        (root / "centering-cache.json").read_text(encoding="ascii")
    )
    data_path = root / metadata["data_object"]
    payload = bytearray(data_path.read_bytes())
    payload[-1] ^= 1
    data_path.write_bytes(payload)

    with pytest.raises(
        RoutedLadderEvaluationError,
        match="content identity",
    ):
        _cached_endpoint_training_means(
            tmp_path,
            target_manifest_sha256=TARGET_MANIFEST,
            recomposition_sidecar_sha256=RECOMPOSITION_SIDECAR,
            expected_tokens=10,
            compute=lambda: pytest.fail(
                "tampered cache must fail rather than recompute"
            ),
        )


def test_process_merge_and_final_score_round_trip(tmp_path: Path) -> None:
    rows = [
        load_process_evidence(
            _seal_lane(tmp_path / f"process-p{index:05d}", process_index=index)
        )
        for index in range(2)
    ]
    summary, tensors = merge_process_evidence(rows)

    assert summary["tokens"] == 4
    assert summary["packed_sequences"] == 2
    assert summary["process_evidence_sha256s"] == [
        rows[0][0]["evidence_sha256"],
        rows[1][0]["evidence_sha256"],
    ]
    result = seal_merged_result(
        tmp_path / "final",
        summary=summary,
        tensors=tensors,
        expected_tokens=4,
        expected_token_identity_sha256=summary["token_identity_sha256"],
    )

    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert result["condition"] == "A512ReLU"
    assert result["endpoints"]["component"]["sse"] == pytest.approx(10.5)
    scores = load_merged_scores(
        tmp_path / "final" / "evaluation-result.json"
    )
    assert scores.condition == "A512ReLU"
    assert scores.tokens == 4
    assert scores.packed_sequences == 2
    assert scores.token_identity_sha256 == summary["token_identity_sha256"]
    assert scores.score_artifact_sha256 == result["score_artifact_sha256"]
    assert {
        name: scores.endpoint_sst_sha256(name)
        for name in ("component", "routed", "oracle_shared")
    } == result["endpoint_sst_sha256"]


def test_h_read_only_scores_round_trip_all_four_endpoints(
    tmp_path: Path,
) -> None:
    rows = [
        load_process_evidence(
            _seal_lane(
                tmp_path / f"h-process-p{index:05d}",
                process_index=index,
                condition="H",
            )
        )
        for index in range(2)
    ]
    summary, tensors = merge_process_evidence(rows)
    result = seal_merged_result(
        tmp_path / "h-final",
        summary=summary,
        tensors=tensors,
        expected_tokens=4,
    )
    scores = load_merged_scores(
        tmp_path / "h-final" / "evaluation-result.json"
    )

    assert isinstance(scores, RoutedLadderHTokenScores)
    assert set(result["endpoints"]) == {
        "component",
        "routed",
        "oracle_shared",
        "learned_full_ffn",
    }
    assert np.array_equal(
        scores.endpoint_sst["learned_full_ffn"],
        scores.endpoint_sst["oracle_shared"],
    )
    assert (
        scores.endpoint_sst_sha256("learned_full_ffn")
        == scores.endpoint_sst_sha256("oracle_shared")
    )
    reference = h_token_scores_to_existing_reference(
        scores,
        training_source_revision="b" * 40,
        h_config_sha256="c" * 64,
        training_result_sha256="d" * 64,
        training_result_file_sha256="e" * 64,
        checkpoint_pointer_sha256="f" * 64,
        checkpoint_pointer_file_sha256="0" * 64,
        scientific_result=ImmutableArtifactReference(
            kind="scientific_result",
            schema_version=RESULT_SCHEMA_VERSION,
            uri="/h-final/evaluation-result.json",
            sha256=result["result_sha256"],
        ),
        diagnostics=(
            ImmutableArtifactReference(
                kind="token_scores",
                schema_version=result["score_schema_version"],
                uri=result["score_data_object"],
                sha256=result["score_data_sha256"],
            ),
        ),
    )
    assert reference.panel == "sae"
    assert reference.endpoint_scores["learned_full_ffn"] == (
        scores.aggregate("learned_full_ffn")
    )
    assert reference.endpoint_sst_sha256["learned_full_ffn"] == (
        reference.endpoint_sst_sha256["oracle_shared"]
    )
    assert reference.score_artifact_sha256 == scores.score_artifact_sha256
    assert reference.training_source_revision == "b" * 40


def test_merge_rejects_checkpoint_substitution(tmp_path: Path) -> None:
    left = load_process_evidence(
        _seal_lane(tmp_path / "left", process_index=0)
    )
    right = load_process_evidence(
        _seal_lane(
            tmp_path / "right",
            process_index=1,
            checkpoint_tokens=5_000_000,
        )
    )

    with pytest.raises(
        RoutedLadderEvaluationError,
        match="checkpoint_tokens differs",
    ):
        merge_process_evidence([left, right])


def test_final_seal_requires_common_held_out_identity(tmp_path: Path) -> None:
    rows = [
        load_process_evidence(
            _seal_lane(tmp_path / f"lane-{index}", process_index=index)
        )
        for index in range(2)
    ]
    summary, tensors = merge_process_evidence(rows)

    with pytest.raises(
        RoutedLadderEvaluationError,
        match="common cohort proof",
    ):
        seal_merged_result(
            tmp_path / "final",
            summary=summary,
            tensors=tensors,
            expected_tokens=4,
            expected_token_identity_sha256="f" * 64,
        )


def test_process_evidence_detects_tensor_tampering(tmp_path: Path) -> None:
    path = _seal_lane(tmp_path / "lane", process_index=0)
    metadata, _ = load_process_evidence(path)
    data_path = path.parent / metadata["data_object"]
    blob = bytearray(data_path.read_bytes())
    blob[-1] ^= 1
    data_path.write_bytes(blob)

    with pytest.raises(
        RoutedLadderEvaluationError,
        match="content identity",
    ):
        load_process_evidence(path)


def test_cli_entry_passes_one_exact_cell_to_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = None

    def fake_run(args):
        nonlocal observed
        observed = args
        return 7

    monkeypatch.setattr(
        "moe_sae.experiment5_routed_ladder_evaluation_worker.run",
        fake_run,
    )
    code = main(
        [
            "--config",
            "/config.yaml",
            "--panel",
            "sae",
            "--condition",
            "A512Softplus",
            "--seed",
            "2",
            "--checkpoint-tokens",
            "10000000",
            "--source-revision",
            SOURCE_REVISION,
            "--training-source-revision",
            TRAINING_REVISION,
            "--train-target-root",
            "/train",
            "--eval-target-root",
            "/eval",
            "--train-target-object-prefix",
            "gs://bucket/train",
            "--eval-target-object-prefix",
            "gs://bucket/eval",
            "--cache-root",
            "/cache",
            "--cache-max-bytes",
            "1024",
            "--run-root",
            "/run",
            "--checkpoint-payload-root",
            "gs://bucket/checkpoints",
            "--recomposition-sidecar",
            "/sidecar.json",
            "--result-root",
            "/result",
            "--process-index",
            "3",
        ]
    )

    assert code == 7
    assert observed.panel == "sae"
    assert observed.condition == "A512Softplus"
    assert observed.seed == 2
    assert observed.checkpoint_tokens == 10_000_000
    assert observed.process_index == 3


def _bundle_spec(
    tmp_path: Path,
    *,
    panel: str = "sae",
    seed: int = 0,
) -> dict:
    cells = []
    for condition in BUNDLE_CONDITIONS:
        for checkpoint_tokens in CHECKPOINT_TOKENS:
            historical_a256 = (
                panel == "sae"
                and seed == 0
                and condition == "A256ReLU"
            )
            reference_kind = (
                "a256"
                if historical_a256
                else "h"
                if condition == "H"
                else "none"
            )
            cells.append(
                {
                    "condition": condition,
                    "checkpoint_tokens": checkpoint_tokens,
                    "training_source_revision": (
                        TRAINING_REVISION
                        if reference_kind != "none"
                        else SOURCE_REVISION
                    ),
                    "run_root": str(
                        tmp_path
                        / "runs"
                        / panel
                        / condition
                        / f"seed{seed}"
                    ),
                    "checkpoint_payload_root": (
                        f"gs://bucket/checkpoints/{panel}/{condition}/seed{seed}"
                    ),
                    "result_root": str(
                        tmp_path
                        / "results"
                        / panel
                        / condition
                        / f"seed{seed}"
                        / str(checkpoint_tokens)
                    ),
                    "reference_kind": reference_kind,
                    "reference_config": (
                        f"/remote/configs/{reference_kind}-{panel}.yaml"
                        if reference_kind != "none"
                        else None
                    ),
                    "reference_run_contract_sha256": (
                        "7" * 64 if reference_kind == "a256" else None
                    ),
                    "reference_run_contract_file_sha256": (
                        "8" * 64 if reference_kind == "a256" else None
                    ),
                    "parent_training_result_sha256": "1" * 64,
                    "parent_training_result_file_sha256": "2" * 64,
                    "parent_run_contract_sha256": (
                        None
                        if reference_kind == "h"
                        else "7" * 64
                        if reference_kind == "a256"
                        else "9" * 64
                    ),
                    "parent_run_contract_file_sha256": (
                        None
                        if reference_kind == "h"
                        else "8" * 64
                        if reference_kind == "a256"
                        else "a" * 64
                    ),
                    "parent_checkpoint_pointer_sha256": "b" * 64,
                    "parent_checkpoint_pointer_file_sha256": "c" * 64,
                }
            )
    unsigned = {
        "schema_version": BUNDLE_SPEC_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "panel": panel,
        "seed": seed,
        "source_revision": SOURCE_REVISION,
        "config": f"/remote/configs/ladder-{panel}-v2.yaml",
        "bundle_result_root": str(
            tmp_path / "bundle-results" / panel / f"seed{seed}"
        ),
        "operator_manifest_sha256": "e" * 64,
        "cells": cells,
    }
    return {
        **unsigned,
        "bundle_spec_sha256": canonical_sha256(unsigned),
    }


def test_resident_bundle_spec_is_exact_and_panel_specific(
    tmp_path: Path,
) -> None:
    sae = parse_resident_bundle_spec(_bundle_spec(tmp_path))
    transcoder = parse_resident_bundle_spec(
        _bundle_spec(tmp_path, panel="transcoder")
    )

    assert len(sae.cells) == 12
    assert [cell.identity for cell in sae.cells] == [
        (condition, checkpoint_tokens)
        for condition in BUNDLE_CONDITIONS
        for checkpoint_tokens in CHECKPOINT_TOKENS
    ]
    assert {
        cell.reference_kind
        for cell in sae.cells
        if cell.condition == "A256ReLU"
    } == {"a256"}
    assert {
        cell.reference_kind
        for cell in transcoder.cells
        if cell.condition == "A256ReLU"
    } == {"none"}


def test_resident_bundle_rejects_reordering_and_transcoder_a256_parent(
    tmp_path: Path,
) -> None:
    reordered = _bundle_spec(tmp_path)
    reordered["cells"][0], reordered["cells"][1] = (
        reordered["cells"][1],
        reordered["cells"][0],
    )
    reordered["bundle_spec_sha256"] = canonical_sha256(
        {
            name: value
            for name, value in reordered.items()
            if name != "bundle_spec_sha256"
        }
    )
    with pytest.raises(
        RoutedLadderEvaluationError,
        match="deterministic 12-cell order",
    ):
        parse_resident_bundle_spec(reordered)

    transcoder = _bundle_spec(tmp_path, panel="transcoder")
    transcoder["cells"][0]["reference_kind"] = "a256"
    transcoder["cells"][0]["reference_config"] = "/remote/configs/a256.yaml"
    transcoder["cells"][0]["training_source_revision"] = TRAINING_REVISION
    transcoder["bundle_spec_sha256"] = canonical_sha256(
        {
            name: value
            for name, value in transcoder.items()
            if name != "bundle_spec_sha256"
        }
    )
    with pytest.raises(
        RoutedLadderEvaluationError,
        match="reference role changed",
    ):
        parse_resident_bundle_spec(transcoder)

    missing_pin = _bundle_spec(tmp_path)
    missing_pin["cells"][0]["reference_run_contract_sha256"] = None
    missing_pin["bundle_spec_sha256"] = canonical_sha256(
        {
            name: value
            for name, value in missing_pin.items()
            if name != "bundle_spec_sha256"
        }
    )
    with pytest.raises(
        RoutedLadderEvaluationError,
        match="reference_run_contract_sha256",
    ):
        parse_resident_bundle_spec(missing_pin)


def test_resident_bundle_completion_binds_all_twelve_cells(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = parse_resident_bundle_spec(_bundle_spec(tmp_path))

    def completion(_bundle, cell):
        return {
            "condition": cell.condition,
            "checkpoint_tokens": cell.checkpoint_tokens,
            "training_source_revision": cell.training_source_revision,
            "parent_training_result_sha256": (
                cell.parent_training_result_sha256
            ),
            "parent_training_result_file_sha256": (
                cell.parent_training_result_file_sha256
            ),
            "parent_run_contract_sha256": (
                cell.parent_run_contract_sha256
            ),
            "parent_run_contract_file_sha256": (
                cell.parent_run_contract_file_sha256
            ),
            "parent_checkpoint_pointer_sha256": (
                cell.parent_checkpoint_pointer_sha256
            ),
            "parent_checkpoint_pointer_file_sha256": (
                cell.parent_checkpoint_pointer_file_sha256
            ),
            "result_root": cell.result_root,
            "result_sha256": f"{cell.checkpoint_tokens // 1_000_000:064x}",
            "result_file_sha256": "1" * 64,
            "score_artifact_sha256": "2" * 64,
            "score_data_sha256": "3" * 64,
            "score_data_bytes": 123,
        }

    monkeypatch.setattr(
        "moe_sae.experiment5_routed_ladder_evaluation_worker."
        "_resident_cell_completion",
        completion,
    )
    rows = [completion(bundle, cell) for cell in bundle.cells]
    timing = {
        "accelerator_type": "v6e-16",
        "chip_count": 16,
        "successful_attempt_wall_seconds": 2.0,
        "successful_attempt_chip_seconds": 32.0,
        "timing_scope": (
            "successful_resident_worker_after_cli_parse_through_cell_completion"
        ),
    }
    result = seal_resident_bundle_result(bundle, rows, timing=timing)

    assert result["status"] == "bundle_evaluated"
    assert len(result["cell_results"]) == 12
    with pytest.raises(
        RoutedLadderEvaluationError,
        match="incomplete or reordered",
    ):
        seal_resident_bundle_result(
            bundle,
            list(reversed(rows)),
            timing=timing,
        )


def test_cli_dispatches_resident_bundle_without_single_cell_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed = None

    def fake_run(args):
        nonlocal observed
        observed = args
        return 11

    monkeypatch.setattr(
        "moe_sae.experiment5_routed_ladder_evaluation_worker."
        "run_resident_bundle",
        fake_run,
    )
    spec = _bundle_spec(tmp_path)
    code = main(
        [
            "--config",
            spec["config"],
            "--panel",
            "sae",
            "--seed",
            "0",
            "--source-revision",
            SOURCE_REVISION,
            "--train-target-root",
            "/train",
            "--eval-target-root",
            "/eval",
            "--train-target-object-prefix",
            "gs://bucket/train",
            "--eval-target-object-prefix",
            "gs://bucket/eval",
            "--cache-root",
            "/cache",
            "--cache-max-bytes",
            "1024",
            "--recomposition-sidecar",
            "/sidecar.json",
            "--process-index",
            "0",
            "--bundle-spec-json",
            json.dumps(spec),
            "--attempt-id",
            "attempt-1",
        ]
    )

    assert code == 11
    assert observed.condition is None
    assert observed.checkpoint_tokens is None
    assert observed.attempt_id == "attempt-1"


def test_resident_bundle_skips_completed_cells_and_clears_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import jax

    spec = _bundle_spec(tmp_path)
    bundle = parse_resident_bundle_spec(spec)
    completed = {cell.identity for cell in bundle.cells[:2]}
    executed = []
    observed_parent_pins = {}
    clear_calls = 0

    def completion(_bundle, cell):
        if cell.identity not in completed:
            return None
        return {
            "condition": cell.condition,
            "checkpoint_tokens": cell.checkpoint_tokens,
            "training_source_revision": cell.training_source_revision,
            "parent_training_result_sha256": (
                cell.parent_training_result_sha256
            ),
            "parent_training_result_file_sha256": (
                cell.parent_training_result_file_sha256
            ),
            "parent_run_contract_sha256": (
                cell.parent_run_contract_sha256
            ),
            "parent_run_contract_file_sha256": (
                cell.parent_run_contract_file_sha256
            ),
            "parent_checkpoint_pointer_sha256": (
                cell.parent_checkpoint_pointer_sha256
            ),
            "parent_checkpoint_pointer_file_sha256": (
                cell.parent_checkpoint_pointer_file_sha256
            ),
            "result_root": cell.result_root,
            "result_sha256": "1" * 64,
            "result_file_sha256": "2" * 64,
            "score_artifact_sha256": "3" * 64,
            "score_data_sha256": "4" * 64,
            "score_data_bytes": 123,
        }

    def fake_run(args, **_kwargs):
        identity = (args.condition, args.checkpoint_tokens)
        executed.append(identity)
        observed_parent_pins[identity] = (
            args.parent_training_result_sha256,
            args.parent_training_result_file_sha256,
            args.parent_run_contract_sha256,
            args.parent_run_contract_file_sha256,
            args.parent_checkpoint_pointer_sha256,
            args.parent_checkpoint_pointer_file_sha256,
        )
        Path(args.result_root).mkdir(parents=True)
        completed.add(identity)
        return 0

    def clear_caches():
        nonlocal clear_calls
        clear_calls += 1

    monkeypatch.setattr(
        "moe_sae.experiment5_routed_ladder_evaluation_worker."
        "_resident_cell_completion",
        completion,
    )
    monkeypatch.setattr(
        "moe_sae.experiment5_routed_ladder_evaluation_worker.run",
        fake_run,
    )
    monkeypatch.setattr(
        "moe_sae.contribution_training_worker._collective_records",
        lambda row: [row],
    )
    monkeypatch.setattr(
        "moe_sae.contribution_training_worker._collective_barrier",
        lambda _name: None,
    )
    monkeypatch.setattr(
        "moe_sae.runtime.initialize_jax_distributed",
        lambda: None,
    )
    monkeypatch.setattr(
        "moe_sae.runtime.require_accelerator_runtime",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setattr(jax, "clear_caches", clear_caches)

    code = run_resident_bundle(
        SimpleNamespace(
            attempt_id="attempt-1",
            bundle_spec_json=json.dumps(spec),
            panel=bundle.panel,
            seed=bundle.seed,
            source_revision=bundle.source_revision,
            config=bundle.config,
            accelerator_type="v6e-16",
            peak_flops_per_second=1,
            process_index=0,
        )
    )

    assert code == 0
    assert executed == [cell.identity for cell in bundle.cells[2:]]
    for cell in bundle.cells[2:]:
        assert observed_parent_pins[cell.identity] == (
            cell.parent_training_result_sha256,
            cell.parent_training_result_file_sha256,
            cell.parent_run_contract_sha256,
            cell.parent_run_contract_file_sha256,
            cell.parent_checkpoint_pointer_sha256,
            cell.parent_checkpoint_pointer_file_sha256,
        )
    assert clear_calls == 1
    assert (
        Path(bundle.bundle_result_root) / "bundle-result.json"
    ).is_file()
