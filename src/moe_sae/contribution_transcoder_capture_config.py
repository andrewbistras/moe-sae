"""Strict configuration boundary for routed-input overlay capture.

The completed Experiment-3 v2 protocol is already fully bound and remains
valid for its original 600-second recovery path.  The 10M learning-horizon
protocol has one unavoidable dependency cycle:

* its executable training config must pin the sealed routed-input overlay;
* the overlay must be captured against the newly sealed contribution target.

The capture contract therefore permits exactly one unresolved *external*
identity: the 10M train overlay which this operation is about to create.  Its
target manifest must already be pinned.  The template also retains a
self-receipt placeholder so its non-self-referential canonical projection can
be named later.  After capture, the ordinary learning-horizon finalizer inserts
the sealed overlay SHA and that projection hash to produce the executable
training config.  No accelerator-facing training loader accepts this
intermediate contract.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

import yaml

from moe_sae.contribution_transcoder_config import (
    SCHEMA_VERSION as TRANSCODER_V2_SCHEMA_VERSION,
    load_contribution_transcoder_config,
)
from moe_sae.learning_horizon_config import (
    PENDING_SOURCE_CAPTURE_CONTRACT,
    PENDING_TRAIN_SOURCE,
    PENDING_TRAIN_TARGET,
    SCHEMA_VERSION as HORIZON_SCHEMA_VERSION,
    LearningHorizonConfigError,
    learning_horizon_config_from_mapping,
    load_learning_horizon_config,
)


TRANSCODER_V2_MAXIMUM_RUNTIME_SECONDS = 600
HORIZON_MAXIMUM_RUNTIME_SECONDS = 3_600
HORIZON_TRAIN_TOKENS = 10_000_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TranscoderOverlayCaptureConfigError(ValueError):
    """The selected config is not a safe routed-input capture contract."""


def _read_yaml(path: str | Path) -> tuple[Path, Any]:
    selected = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise TranscoderOverlayCaptureConfigError(
            f"cannot load overlay capture config {selected}"
        ) from error
    return selected, value


def load_transcoder_overlay_capture_config(path: str | Path) -> Any:
    """Load either frozen v2 or the target-bound 10M capture contract.

    Returning the existing worker-compatible config objects keeps this
    boundary deliberately small.  The schema version in ``config.raw``
    distinguishes the two supported protocols.
    """

    selected, raw = _read_yaml(path)
    if not isinstance(raw, dict):
        raise TranscoderOverlayCaptureConfigError(
            f"overlay capture config {selected} must be a mapping"
        )
    schema = raw.get("schema_version")
    if schema == TRANSCODER_V2_SCHEMA_VERSION:
        return load_contribution_transcoder_config(selected)
    if schema != HORIZON_SCHEMA_VERSION:
        raise TranscoderOverlayCaptureConfigError(
            f"unsupported overlay capture config schema {schema!r}"
        )
    try:
        config = learning_horizon_config_from_mapping(
            raw,
            expected_scientific_model="transcoder",
            require_finalized=False,
        )
    except LearningHorizonConfigError as error:
        raise TranscoderOverlayCaptureConfigError(
            "invalid 10M overlay capture contract"
        ) from error

    train_target = config.parents["train_target_manifest_sha256"]
    train_source = config.parents["train_source_overlay_manifest_sha256"]
    capture_receipt = config.parents[
        "source_overlay_capture_contract_sha256"
    ]
    if train_target == PENDING_TRAIN_TARGET:
        raise TranscoderOverlayCaptureConfigError(
            "10M overlay capture requires its sealed target manifest SHA-256"
        )
    if train_source != PENDING_TRAIN_SOURCE:
        raise TranscoderOverlayCaptureConfigError(
            "10M overlay capture contract must have exactly the train source "
            "overlay SHA-256 pending"
        )
    if capture_receipt != PENDING_SOURCE_CAPTURE_CONTRACT:
        raise TranscoderOverlayCaptureConfigError(
            "10M overlay capture contract must leave its self-identity receipt "
            "pending until the executable config is finalized"
        )
    if config.parents["train_tokens"] != HORIZON_TRAIN_TOKENS:
        raise TranscoderOverlayCaptureConfigError(
            "10M overlay capture contract has an unexpected token count"
        )
    return config


def finalize_transcoder_overlay_capture_config(
    template_path: str | Path,
    destination_path: str | Path,
    *,
    train_target_manifest_sha256: str,
) -> Any:
    """Publish the non-self-referential, target-bound capture contract.

    This is intentionally distinct from finalizing an executable horizon
    training config.  It replaces only the already-sealed target identity and
    preserves the overlay and capture-receipt placeholders.
    """

    template = load_learning_horizon_config(
        template_path,
        expected_scientific_model="transcoder",
        require_finalized=False,
    )
    if (
        template.parents["train_target_manifest_sha256"]
        != PENDING_TRAIN_TARGET
        or template.parents["train_source_overlay_manifest_sha256"]
        != PENDING_TRAIN_SOURCE
        or template.parents["source_overlay_capture_contract_sha256"]
        != PENDING_SOURCE_CAPTURE_CONTRACT
    ):
        raise TranscoderOverlayCaptureConfigError(
            "overlay capture finalization requires the untouched horizon template"
        )
    if (
        not isinstance(train_target_manifest_sha256, str)
        or _SHA256_RE.fullmatch(train_target_manifest_sha256) is None
    ):
        raise TranscoderOverlayCaptureConfigError(
            "train target manifest SHA-256 must be lowercase hexadecimal"
        )

    raw = template.as_dict()
    raw["artifacts"][
        "train_target_manifest_sha256"
    ] = train_target_manifest_sha256
    projected = learning_horizon_config_from_mapping(
        raw,
        expected_scientific_model="transcoder",
        require_finalized=False,
    )

    destination = Path(destination_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = load_transcoder_overlay_capture_config(destination)
        if existing.config_hash != projected.config_hash:
            raise TranscoderOverlayCaptureConfigError(
                "capture-contract destination already has different content"
            )
        return existing
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    payload = yaml.safe_dump(projected.as_dict(), sort_keys=False)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = load_transcoder_overlay_capture_config(destination)
            if existing.config_hash != projected.config_hash:
                raise TranscoderOverlayCaptureConfigError(
                    "concurrent capture-contract publication differs"
                ) from None
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return load_transcoder_overlay_capture_config(destination)


def overlay_capture_runtime_cap_seconds(config: Any, *, split: str) -> int:
    """Return the protocol-scoped hard runtime cap.

    Only a new 10M *train* overlay receives the longer bound.  The frozen v2
    path, including any recovery capture, remains capped at ten minutes.
    Evaluation for the horizon study reuses its already-sealed 250k overlay
    and is intentionally not recaptured.
    """

    schema = config.raw.get("schema_version")
    if schema == TRANSCODER_V2_SCHEMA_VERSION:
        return TRANSCODER_V2_MAXIMUM_RUNTIME_SECONDS
    if schema == HORIZON_SCHEMA_VERSION:
        if split != "train":
            raise TranscoderOverlayCaptureConfigError(
                "the 10M horizon reuses the sealed 250k eval overlay; only its "
                "train overlay may be captured"
            )
        return HORIZON_MAXIMUM_RUNTIME_SECONDS
    raise TranscoderOverlayCaptureConfigError(
        f"unsupported overlay capture config schema {schema!r}"
    )


def overlay_capture_attention(config: Any) -> str:
    """Return the unchanged Gemma prefix attention selection."""

    model = config.raw.get("model")
    if isinstance(model, dict):
        attention = model.get("attention", "autoselected")
        if isinstance(attention, str) and attention:
            return attention
    raise TranscoderOverlayCaptureConfigError(
        "overlay capture config has no valid attention selection"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--template", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--train-target-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "finalize":  # pragma: no cover - argparse is exhaustive.
        raise AssertionError("unknown command")
    config = finalize_transcoder_overlay_capture_config(
        args.template,
        args.output,
        train_target_manifest_sha256=args.train_target_manifest_sha256,
    )
    print(
        json.dumps(
            {
                "capture_config": str(
                    Path(args.output).expanduser().resolve()
                ),
                "capture_contract_sha256": config.config_hash,
                "train_target_manifest_sha256": config.parents[
                    "train_target_manifest_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "HORIZON_MAXIMUM_RUNTIME_SECONDS",
    "HORIZON_TRAIN_TOKENS",
    "TRANSCODER_V2_MAXIMUM_RUNTIME_SECONDS",
    "TranscoderOverlayCaptureConfigError",
    "finalize_transcoder_overlay_capture_config",
    "load_transcoder_overlay_capture_config",
    "overlay_capture_attention",
    "overlay_capture_runtime_cap_seconds",
]


if __name__ == "__main__":
    raise SystemExit(main())
