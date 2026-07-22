from __future__ import annotations

from pathlib import Path

import pytest

from moe_sae.sae_paths import SAEOperationPaths, SAEPathError


def digest(character: str = "a") -> str:
    return character * 64


def test_operation_paths_are_content_addressed_and_separate_smoke(
    tmp_path: Path,
) -> None:
    paths = SAEOperationPaths(tmp_path / "artifacts", digest())
    operation = (
        (tmp_path / "artifacts").resolve()
        / "operator"
        / "sae"
        / "operations"
        / digest()
    )
    numerical = operation / "artifacts"

    assert paths.operation_root == operation
    assert paths.numerical_root == numerical
    assert paths.run_root("A", 2) == numerical / "training" / "A" / "seed-2"
    assert paths.evaluation_path("F", 3) == (
        numerical / "evaluations" / "F-seed-3.json"
    )
    assert paths.metrics_path("B", 1) == (
        numerical / "training" / "B" / "seed-1" / "metrics.jsonl"
    )
    assert paths.smoke_root("A") == numerical / "smoke" / "A"
    assert paths.smoke_root("A") != paths.run_root("A", 1)
    assert paths.report_path == numerical / "report.json"

    # The policy is side-effect free; phase code owns publication.
    assert not operation.exists()


def test_operation_digest_selects_a_disjoint_namespace(tmp_path: Path) -> None:
    first = SAEOperationPaths(tmp_path, digest("a"))
    second = SAEOperationPaths(tmp_path, digest("b"))

    assert first.operation_root != second.operation_root
    assert first.run_root("A", 1) != second.run_root("A", 1)
    assert first.report_path != second.report_path


@pytest.mark.parametrize(
    "value",
    (
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        "../" + "a" * 64,
        "a" * 32 + "/" + "b" * 31,
    ),
)
def test_invalid_operation_digest_is_rejected(tmp_path: Path, value: str) -> None:
    with pytest.raises(SAEPathError, match="operation_sha256"):
        SAEOperationPaths(tmp_path, value)


@pytest.mark.parametrize(
    "condition",
    ("", "a", "C", "../A", "A/../../outside", "/A", 1, None),
)
def test_invalid_or_traversing_condition_is_rejected(
    tmp_path: Path, condition: object
) -> None:
    paths = SAEOperationPaths(tmp_path, digest())
    with pytest.raises(SAEPathError, match="condition"):
        paths.run_root(condition, 1)  # type: ignore[arg-type]
    with pytest.raises(SAEPathError, match="condition"):
        paths.evaluation_path(condition, 1)  # type: ignore[arg-type]
    with pytest.raises(SAEPathError, match="condition"):
        paths.smoke_root(condition)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", (-1, 0, 4, True, False, 1.0, "1", None))
def test_invalid_seed_is_rejected(tmp_path: Path, seed: object) -> None:
    paths = SAEOperationPaths(tmp_path, digest())
    with pytest.raises(SAEPathError, match="seed"):
        paths.run_root("A", seed)  # type: ignore[arg-type]
    with pytest.raises(SAEPathError, match="seed"):
        paths.evaluation_path("A", seed)  # type: ignore[arg-type]
    with pytest.raises(SAEPathError, match="seed"):
        paths.metrics_path("A", seed)  # type: ignore[arg-type]


def test_existing_symlink_cannot_escape_operation_root(tmp_path: Path) -> None:
    paths = SAEOperationPaths(tmp_path / "artifacts", digest())
    outside = tmp_path / "outside"
    outside.mkdir()
    paths.operation_root.mkdir(parents=True)
    (paths.operation_root / "artifacts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SAEPathError, match="escapes"):
        _ = paths.numerical_root


def test_artifact_root_is_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = SAEOperationPaths(Path("nested") / ".." / "artifacts", digest())
    assert paths.artifact_root == (tmp_path / "artifacts").resolve()
