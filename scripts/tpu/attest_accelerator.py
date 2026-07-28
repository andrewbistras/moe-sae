#!/usr/bin/env python3
"""Freeze and attest one already-created TPU slice incarnation."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from moe_sae.contribution_artifacts import canonical_json_bytes
from moe_sae.regional_execution import (
    RegionalExecutionError,
    assert_spot_tpu_description,
    atomic_write_evidence,
    build_accelerator_contract,
)


class AcceleratorAttestationError(RuntimeError):
    """The live TPU cannot be converted into the requested exact contract."""


def _describe_command(args: argparse.Namespace) -> list[str]:
    return [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "describe",
        args.tpu_name,
        f"--project={args.project}",
        f"--zone={args.zone}",
        "--format=json",
    ]


def attest(args: argparse.Namespace) -> dict[str, object]:
    completed = subprocess.run(
        _describe_command(args),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise AcceleratorAttestationError(f"cannot describe TPU{suffix}")
    try:
        description = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise AcceleratorAttestationError(
            "gcloud TPU description is not JSON"
        ) from error
    if not isinstance(description, dict):
        raise AcceleratorAttestationError("TPU description must be an object")
    create_time = description.get("createTime")
    contract = build_accelerator_contract(
        run_id=args.run_id,
        scientific_model=args.scientific_model,
        condition=args.condition,
        project=args.project,
        tpu_name=args.tpu_name,
        resource_create_time=str(create_time),
        region=args.zone.rsplit("-", 1)[0],
        zone=args.zone,
        accelerator_type=args.accelerator_type,
        provisioning_model=(
            "ON_DEMAND" if args.accelerator_type == "v4-32" else "SPOT"
        ),
    )
    attestation = assert_spot_tpu_description(description, contract)
    output = Path(args.output_root).expanduser().resolve()
    atomic_write_evidence(output / "tpu-description.json", description)
    atomic_write_evidence(output / "accelerator-contract.json", contract)
    atomic_write_evidence(output / "spot-attestation.json", attestation)
    return {
        "description": description,
        "contract": contract,
        "attestation": attestation,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only exact TPU incarnation attestation"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scientific-model", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--tpu-name", required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument(
        "--accelerator-type",
        choices=("v4-32", "v6e-16", "v5litepod-16"),
        default="v6e-16",
    )
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = attest(args)
        print(canonical_json_bytes(result).decode("ascii"), flush=True)
        return 0
    except (
        AcceleratorAttestationError,
        RegionalExecutionError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"[accelerator-attestation] ERROR: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
