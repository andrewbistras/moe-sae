from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from moe_sae.contribution_artifacts import canonical_sha256
from moe_sae.result_registry import (
    ResultRegistryError,
    build_registry,
    load_registry,
    make_cell,
    make_contrast,
    make_cost_space,
    make_fragment,
    make_score_space,
    make_source,
    validate_registry,
    write_registry,
)


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


def _source() -> dict:
    return make_source(
        kind="synthetic_closure_report",
        schema_version="synthetic-closure-v1",
        scientific_sha256=_digest("report"),
    )


def _score_space(*, endpoint: str = "component", target: str = "target") -> dict:
    return make_score_space(
        model_id="google/gemma-4-26B-A4B",
        model_checkpoint_sha256=_digest("model"),
        layer_one_indexed=15,
        target_id=target,
        endpoint=endpoint,
        evaluation_manifest_sha256=_digest("evaluation"),
        token_identity_sha256=_digest("tokens"),
        centering_id="train-mean-v1",
        sst_id=f"{target}-sst-v1",
    )


def _cost_space(*, accelerator: str = "v4-32") -> dict:
    generation = accelerator.split("-", maxsplit=1)[0]
    return make_cost_space(
        measurement_id="marginal-sae-training-v1",
        accelerator_type=accelerator,
        accelerator_generation=generation,
        chip_count=4,
        included=["optimizer-steps", "input-stalls"],
        excluded=["compilation", "checkpoint-write", "evaluation"],
        effective_batch_tokens=100,
        microbatch_tokens=25,
    )


def _cell(
    *,
    source: dict,
    score_space: dict,
    cost_space: dict,
    condition_id: str,
    condition_label: str,
    ev: float,
) -> dict:
    sst = 100.0
    nmse = 1.0 - ev
    return make_cell(
        experiment_id="experiment5-a512-closure",
        protocol_id="a512-closure-v1",
        panel="sae",
        condition_id=condition_id,
        condition_label=condition_label,
        seed=1,
        checkpoint_tokens=100,
        actual_training_tokens=100,
        encoder_source_id="selected-expert-output",
        activation_policy_id="relu-inclusive-zero-topk",
        geometry={
            "bank_identity": "true-expert-id",
            "bank_count": 128,
            "bank_width": 512,
            "total_features": 65_536,
            "candidates_per_token": 4_096,
            "selected_slots": 64,
            "positive_l0": 40.0,
            "trainable_parameters": 369_164_288,
        },
        endpoints={
            "component": {
                "score_space_id": score_space["score_space_id"],
                "sse": nmse * sst,
                "sst": sst,
                "nmse": nmse,
                "ev": ev,
                "sst_sha256": _digest("component-sst"),
            }
        },
        cost={
            "cost_space_id": cost_space["cost_space_id"],
            "chip_seconds": 40.0,
            "wall_seconds": 10.0,
            "throughput_tokens": 100,
            "tokens_per_second": 10.0,
            "useful_mfu": 0.2,
        },
        provenance={
            "registry_source_id": source["source_id"],
            "result_sha256": _digest(f"result-{condition_id}"),
            "checkpoint_sha256": _digest(f"checkpoint-{condition_id}"),
            "source_revision": "a" * 40,
        },
    )


def _fragment(
    *,
    accelerator: str = "v4-32",
    second_score_space: bool = False,
    interpreted_cost: bool = True,
) -> dict:
    source = _source()
    left_score = _score_space()
    right_score = (
        _score_space(target="other-target") if second_score_space else left_score
    )
    left_cost = _cost_space()
    right_cost = _cost_space(accelerator=accelerator)
    left = _cell(
        source=source,
        score_space=left_score,
        cost_space=left_cost,
        condition_id="experiment5-closure.sae.a512-relu",
        condition_label="A512-ReLU",
        ev=0.7,
    )
    right = _cell(
        source=source,
        score_space=right_score,
        cost_space=right_cost,
        condition_id="experiment5-closure.sae.f65536-relu",
        condition_label="F65536-ReLU",
        ev=0.6,
    )
    quality = make_contrast(
        experiment_id="experiment5-a512-closure",
        kind="quality_difference",
        label="A512-ReLU minus F65536-ReLU",
        left_cell_ids=[left["cell_id"]],
        right_cell_ids=[right["cell_id"]],
        endpoint="component",
        aggregation="point_ev_difference",
        estimate=0.1,
        interval=None,
        interpreted=True,
        source_artifact_sha256=source["scientific_sha256"],
    )
    comparable = left_cost["cost_space_id"] == right_cost["cost_space_id"]
    cost = make_contrast(
        experiment_id="experiment5-a512-closure",
        kind="cost_ratio",
        label="A512-ReLU/F65536-ReLU measured cost",
        left_cell_ids=[left["cell_id"]],
        right_cell_ids=[right["cell_id"]],
        endpoint=None,
        aggregation="paired_median_percent_ratio",
        estimate=100.0 if comparable and interpreted_cost else None,
        interval=None,
        interpreted=comparable and interpreted_cost,
        source_artifact_sha256=source["scientific_sha256"],
    )
    return make_fragment(
        sources=[source],
        score_spaces=list(
            {row["score_space_id"]: row for row in (left_score, right_score)}.values()
        ),
        cost_spaces=list(
            {row["cost_space_id"]: row for row in (left_cost, right_cost)}.values()
        ),
        cells=[left, right],
        contrasts=[quality, cost],
    )


def test_registry_round_trip_and_create_only_output(tmp_path: Path) -> None:
    registry = build_registry([_fragment()])
    output = tmp_path / "registry.json"

    assert write_registry(output, registry) == output.resolve()
    assert write_registry(output, registry) == output.resolve()
    assert load_registry(output) == registry

    other = build_registry([_fragment(accelerator="v5e-16", interpreted_cost=False)])
    with pytest.raises(ResultRegistryError, match="refusing to replace"):
        write_registry(output, other)
    assert load_registry(output) == registry

    changed = copy.deepcopy(registry)
    changed["registry_sha256"] = "0" * 64
    output.write_text(json.dumps(changed), encoding="ascii")
    with pytest.raises(ResultRegistryError, match="self-hash"):
        load_registry(output)


@pytest.mark.parametrize("condition_id", ["F", "A", "unqualified"])
def test_bare_or_unqualified_condition_id_is_rejected(condition_id: str) -> None:
    source = _source()
    score = _score_space()
    cost = _cost_space()
    with pytest.raises(ResultRegistryError, match="must be qualified"):
        _cell(
            source=source,
            score_space=score,
            cost_space=cost,
            condition_id=condition_id,
            condition_label=condition_id,
            ev=0.5,
        )


def test_quality_contrast_cannot_cross_score_spaces() -> None:
    with pytest.raises(ResultRegistryError, match="crosses score spaces"):
        _fragment(second_score_space=True)


def test_interpreted_cost_contrast_cannot_cross_hardware() -> None:
    source = _source()
    score = _score_space()
    v4 = _cost_space(accelerator="v4-32")
    v5 = _cost_space(accelerator="v5e-16")
    left = _cell(
        source=source,
        score_space=score,
        cost_space=v4,
        condition_id="experiment5-closure.sae.a512-relu",
        condition_label="A512-ReLU",
        ev=0.7,
    )
    right = _cell(
        source=source,
        score_space=score,
        cost_space=v5,
        condition_id="experiment5-closure.sae.f65536-relu",
        condition_label="F65536-ReLU",
        ev=0.6,
    )
    contrast = make_contrast(
        experiment_id="experiment5-a512-closure",
        kind="cost_ratio",
        label="invalid cross-hardware ratio",
        left_cell_ids=[left["cell_id"]],
        right_cell_ids=[right["cell_id"]],
        endpoint=None,
        aggregation="paired_median_percent_ratio",
        estimate=100.0,
        interval=None,
        interpreted=True,
        source_artifact_sha256=source["scientific_sha256"],
    )
    with pytest.raises(ResultRegistryError, match="cross-hardware/cost-space"):
        make_fragment(
            sources=[source],
            score_spaces=[score],
            cost_spaces=[v4, v5],
            cells=[left, right],
            contrasts=[contrast],
        )


def test_cross_hardware_raw_costs_are_allowed_without_an_interpreted_ratio() -> None:
    fragment = _fragment(
        accelerator="v5e-16",
        interpreted_cost=False,
    )
    ratio = next(
        row for row in fragment["contrasts"].values() if row["kind"] == "cost_ratio"
    )
    assert ratio["estimate"] is None
    assert ratio["interpreted"] is False


def test_score_and_cost_arithmetic_fail_closed() -> None:
    fragment = _fragment()
    registry = build_registry([fragment])
    cell_id, cell = next(iter(registry["cells"].items()))

    bad_score = copy.deepcopy(cell)
    bad_score["endpoints"]["component"]["ev"] = 0.123
    bad_score_body = {
        key: value for key, value in bad_score.items() if key != "cell_id"
    }
    bad_score["cell_id"] = f"cell:{canonical_sha256(bad_score_body)}"
    bad_fragment = copy.deepcopy(fragment)
    del bad_fragment["cells"][cell_id]
    bad_fragment["cells"][bad_score["cell_id"]] = bad_score
    with pytest.raises(ResultRegistryError, match="1-NMSE"):
        build_registry([bad_fragment])

    bad_cost = copy.deepcopy(cell)
    bad_cost["cost"]["chip_seconds"] = 41.0
    bad_cost_body = {key: value for key, value in bad_cost.items() if key != "cell_id"}
    bad_cost["cell_id"] = f"cell:{canonical_sha256(bad_cost_body)}"
    bad_fragment = copy.deepcopy(fragment)
    del bad_fragment["cells"][cell_id]
    bad_fragment["cells"][bad_cost["cell_id"]] = bad_cost
    with pytest.raises(ResultRegistryError, match="wall × chips"):
        build_registry([bad_fragment])


def test_fragments_merge_for_future_closure_artifacts() -> None:
    first = _fragment()
    second = copy.deepcopy(first)
    second["contrasts"] = {}
    registry = build_registry([first, second])

    assert len(registry["sources"]) == 1
    assert len(registry["score_spaces"]) == 1
    assert len(registry["cost_spaces"]) == 1
    assert len(registry["cells"]) == 2


def test_registry_self_hash_covers_all_records() -> None:
    registry = build_registry([_fragment()])
    changed = copy.deepcopy(registry)
    changed["cells"].pop(next(iter(changed["cells"])))
    with pytest.raises(ResultRegistryError, match="self-hash"):
        validate_registry(changed)
