"""Narrow compatibility proof between capture producers and SAE consumers.

Operational fixes are allowed to change the repository and execution
environment after an immutable capture has been sealed.  They must not silently
change the model adapter, data interpretation, artifact schema, SAE objective,
or report mathematics.  This module compares those components directly at the
producer and consumer Git revisions instead of treating the whole environment
fingerprint as a scientific identity.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


CAPTURE_COMPATIBILITY_SCHEMA = "gemma4-capture-consumer-compatibility-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")

# These files are the scientific/data/schema implementation.  CLI,
# orchestration, storage, and checkpoint transport code are deliberately not
# in this list: changing one of those is the reason this compatibility boundary
# exists.  Changes to any file below require a new capture or an explicitly
# revised scientific contract.
# Only components which produced or define interpretation of the immutable
# activation values must match the producer.  SAE optimization/evaluation code
# was not involved in capture; its full current environment is instead bound
# into run/checkpoint/report identities.
SCIENTIFIC_COMPONENT_FILES = (
    "configs/gemma4_26b_a4b.yaml",
    "src/moe_sae/config.py",
    "src/moe_sae/data.py",
    "src/moe_sae/model.py",
)

# harvest.py and runtime.py also contain operational machinery.  Hash only the
# nodes which define scientific token selection and capture values.
SCIENTIFIC_COMPONENT_SYMBOLS: Mapping[str, tuple[str, ...]] = {
    # ArtifactDataset and its cache/prefetch path are consumer operations and
    # may evolve after capture.  Bind the wire schema, writer, statistics, and
    # sealing/validation code which actually produced and defines the captured
    # values, not the entire mixed producer/consumer module.
    "src/moe_sae/artifacts.py": (
        "SCHEMA_VERSION",
        "MAX_SHARD_TOKENS",
        "HIDDEN_SIZE",
        "NUM_EXPERTS",
        "TOP_K_EXPERTS",
        "TARGET_LAYER_ONE_INDEXED",
        "DEFAULT_CAPTURE_SITE",
        "_STATS_COUNT",
        "_STATS_SUM",
        "_STATS_SQUARED_NORM",
        "_DATA_KEYS",
        "_FILE_KEYS",
        "float32_to_bf16_bits",
        "bf16_bits_to_float32",
        "_as_bf16_bits",
        "_integer_wire_array",
        "_normalize_batch",
        "_token_count",
        "_batch_tensors",
        "CaptureMetadata",
        "_metadata_dict",
        "_batch_stats",
        "_validate_upload_receipt",
        "ArtifactWriter",
        "_validate_entry_shape",
        "_load_shard",
        "publish_manifest",
        "load_manifest",
    ),
    "src/moe_sae/runtime.py": (
        "TARGET_SEQUENCE_LENGTH",
        "TARGET_LAYER_ONE_INDEXED",
        "TARGET_TRAIN_TOKENS",
        "TARGET_EVAL_TOKENS",
        "token_quota",
        "scientific_token_mask",
        "compact_capture",
    ),
    "src/moe_sae/harvest.py": (
        "EXCLUDED_TOKEN_IDS",
        "DEFAULT_SEQUENCE_LENGTH",
        "host_token_offset",
        "capture_metadata",
        "_capture_range_id",
        "_compact_adapter_output",
    ),
}


class CompatibilityError(RuntimeError):
    """Capture producer and SAE consumer cannot be proven compatible."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _git(
    repo_root: Path, arguments: Sequence[str], *, text: bool = False
) -> bytes | str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=text,
            timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise CompatibilityError(
            "cannot read the repository history needed for capture compatibility"
        ) from error
    return completed.stdout


def _source_bytes(repo_root: Path, path: str, revision: str | None) -> bytes:
    if revision is None:
        try:
            return (repo_root / path).read_bytes()
        except OSError as error:
            raise CompatibilityError(f"missing compatibility component: {path}") from error
    if not _COMMIT_RE.fullmatch(revision):
        raise CompatibilityError("producer revision is not a full Git commit")
    value = _git(repo_root, ["show", f"{revision}:{path}"])
    assert isinstance(value, bytes)
    return value


def _symbol_nodes(source: bytes, path: str) -> dict[str, ast.AST]:
    try:
        tree = ast.parse(source.decode("utf-8"), filename=path)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise CompatibilityError(f"cannot parse compatibility component: {path}") from error
    result: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node
    return result


def scientific_code_fingerprint(
    repo_root: str | Path, *, revision: str | None = None
) -> dict[str, Any]:
    """Hash the exact scientific/data/schema components at one revision."""

    root = Path(repo_root).resolve()
    components: dict[str, str] = {}
    for path in SCIENTIFIC_COMPONENT_FILES:
        components[f"file:{path}"] = hashlib.sha256(
            _source_bytes(root, path, revision)
        ).hexdigest()
    for path, names in SCIENTIFIC_COMPONENT_SYMBOLS.items():
        source = _source_bytes(root, path, revision)
        nodes = _symbol_nodes(source, path)
        for name in names:
            node = nodes.get(name)
            if node is None:
                raise CompatibilityError(
                    f"compatibility symbol {name!r} is missing from {path}"
                )
            normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
            components[f"symbol:{path}:{name}"] = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()
    return {
        "policy": "exact-selected-components-v1",
        "components": components,
        "sha256": _sha256_json(components),
    }


def repository_content_sha256_at_revision(
    repo_root: str | Path, revision: str
) -> str:
    """Reproduce runtime.repository_content_sha256 for a clean Git tree."""

    if not _COMMIT_RE.fullmatch(revision):
        raise CompatibilityError("repository revision is not a full Git commit")
    root = Path(repo_root).resolve()
    listing = _git(root, ["ls-tree", "-r", "--name-only", "-z", revision])
    assert isinstance(listing, bytes)
    paths = sorted(item for item in listing.split(b"\0") if item)
    digest = hashlib.sha256()
    for encoded in paths:
        path = encoded.decode("utf-8", errors="surrogateescape")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        blob = _git(root, ["show", f"{revision}:{path}"])
        assert isinstance(blob, bytes)
        digest.update(blob)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    result = str(value)
    if not _SHA256_RE.fullmatch(result):
        raise CompatibilityError(f"{label} is not a SHA-256")
    return result


def _capture_record(
    manifest: Mapping[str, Any], *, split: str, expected_tokens: int
) -> dict[str, Any]:
    metadata = manifest.get("metadata")
    if (
        manifest.get("schema_version") != "gemma4-router-aggregate-v1"
        or manifest.get("sealed") is not True
        or int(manifest.get("total_tokens", -1)) != expected_tokens
        or not isinstance(metadata, Mapping)
        or metadata.get("split") != split
    ):
        raise CompatibilityError(f"{split} capture is not the expected sealed schema")
    return {
        "manifest_sha256": _digest(
            manifest.get("manifest_sha256"), f"{split} manifest"
        ),
        "metadata_sha256": _sha256_json(dict(metadata)),
        "dataset_fingerprint": _digest(
            metadata.get("dataset_fingerprint"), f"{split} dataset"
        ),
        "tokens": expected_tokens,
    }


def build_capture_compatibility(
    *,
    repo_root: str | Path,
    config_sha256: str,
    model_id: str,
    maxtext_commit: str,
    checkpoint_sha256: str,
    conversion_attestation_sha256: str,
    activation_storage_attestation_sha256: str,
    train_manifest: Mapping[str, Any],
    eval_manifest: Mapping[str, Any],
    train_tokens: int,
    eval_tokens: int,
    benchmark_record: Mapping[str, Any],
    consumer_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an exact producer-to-consumer compatibility proof."""

    train = _capture_record(train_manifest, split="train", expected_tokens=train_tokens)
    heldout = _capture_record(eval_manifest, split="eval", expected_tokens=eval_tokens)
    train_metadata = train_manifest["metadata"]
    eval_metadata = eval_manifest["metadata"]
    fixed = {
        "config_sha256": _digest(config_sha256, "config"),
        "model_id": model_id,
        "maxtext_commit": maxtext_commit,
        "checkpoint_sha256": _digest(checkpoint_sha256, "checkpoint"),
        "conversion_attestation_sha256": _digest(
            conversion_attestation_sha256, "conversion attestation"
        ),
        "activation_storage_attestation_sha256": _digest(
            activation_storage_attestation_sha256, "activation storage attestation"
        ),
        "target_layer_one_indexed": 15,
        "capture_site": "routed_moe.post_feedforward_layernorm_2",
        "hidden_size": 2816,
        "num_experts": 128,
        "top_k_experts": 8,
    }
    metadata_expected = {
        "config_sha256": fixed["config_sha256"],
        "model_id": model_id,
        "maxtext_commit": maxtext_commit,
        "checkpoint_sha256": fixed["checkpoint_sha256"],
        "conversion_attestation_sha256": fixed[
            "conversion_attestation_sha256"
        ],
        "storage_attestation_sha256": fixed[
            "activation_storage_attestation_sha256"
        ],
        "target_layer_one_indexed": 15,
        "capture_site": fixed["capture_site"],
        "hidden_size": 2816,
        "num_experts": 128,
        "top_k_experts": 8,
    }
    for split, metadata in (("train", train_metadata), ("eval", eval_metadata)):
        changed = {
            key: (metadata.get(key), expected)
            for key, expected in metadata_expected.items()
            if metadata.get(key) != expected
        }
        if changed:
            raise CompatibilityError(
                f"{split} capture model/config/schema identity changed: {changed}"
            )

    producer_environment_sha256 = _digest(
        train_metadata.get("environment_sha256"), "capture producer environment"
    )
    if eval_metadata.get("environment_sha256") != producer_environment_sha256:
        raise CompatibilityError("train/eval captures have different producers")
    benchmark_environment = benchmark_record.get("environment")
    if not isinstance(benchmark_environment, Mapping):
        raise CompatibilityError("benchmark lacks its producer environment")
    producer_revision = str(benchmark_environment.get("git_head", ""))
    producer_repository = _digest(
        benchmark_environment.get("repository_content_sha256"),
        "producer repository content",
    )
    if (
        benchmark_environment.get("sha256") != producer_environment_sha256
        or benchmark_environment.get("repository_dirty") is not False
        or not _COMMIT_RE.fullmatch(producer_revision)
        or repository_content_sha256_at_revision(repo_root, producer_revision)
        != producer_repository
    ):
        raise CompatibilityError(
            "benchmark cannot prove the clean capture-producer Git tree"
        )

    consumer_environment_sha256 = _digest(
        consumer_environment.get("sha256"), "consumer environment"
    )
    consumer_revision = str(consumer_environment.get("git_head", ""))
    consumer_repository = _digest(
        consumer_environment.get("repository_content_sha256"),
        "consumer repository content",
    )
    if (
        consumer_environment.get("repository_dirty") is not False
        or not _COMMIT_RE.fullmatch(consumer_revision)
        or repository_content_sha256_at_revision(repo_root, consumer_revision)
        != consumer_repository
    ):
        raise CompatibilityError("SAE consumer is not a clean, identifiable Git tree")

    producer_code = scientific_code_fingerprint(
        repo_root, revision=producer_revision
    )
    consumer_code = scientific_code_fingerprint(repo_root)
    if producer_code["sha256"] != consumer_code["sha256"]:
        changed = sorted(
            name
            for name, digest in producer_code["components"].items()
            if consumer_code["components"].get(name) != digest
        )
        raise CompatibilityError(
            "scientific/data/schema code changed since capture: " + ", ".join(changed)
        )

    payload = {
        "schema_version": CAPTURE_COMPATIBILITY_SCHEMA,
        "contract": fixed,
        "captures": {"train": train, "eval": heldout},
        "producer": {
            "environment_sha256": producer_environment_sha256,
            "git_head": producer_revision,
            "repository_content_sha256": producer_repository,
            "scientific_code_sha256": producer_code["sha256"],
        },
        "consumer": {
            "environment_sha256": consumer_environment_sha256,
            "git_head": consumer_revision,
            "repository_content_sha256": consumer_repository,
            "scientific_code_sha256": consumer_code["sha256"],
        },
        "scientific_components": consumer_code,
        "compatible": True,
    }
    payload["compatibility_sha256"] = _sha256_json(payload)
    return payload


def validate_capture_compatibility(
    value: Mapping[str, Any], **expected: Any
) -> str:
    """Recompute the proof and require byte-for-byte semantic equality."""

    payload = dict(value)
    claimed = payload.pop("compatibility_sha256", None)
    if (
        value.get("schema_version") != CAPTURE_COMPATIBILITY_SCHEMA
        or value.get("compatible") is not True
        or not isinstance(claimed, str)
        or claimed != _sha256_json(payload)
    ):
        raise CompatibilityError("capture compatibility self-hash/schema is invalid")
    rebuilt = build_capture_compatibility(**expected)
    if dict(value) != rebuilt:
        raise CompatibilityError("capture compatibility no longer matches its inputs")
    return claimed


__all__ = [
    "CAPTURE_COMPATIBILITY_SCHEMA",
    "CompatibilityError",
    "SCIENTIFIC_COMPONENT_FILES",
    "SCIENTIFIC_COMPONENT_SYMBOLS",
    "build_capture_compatibility",
    "repository_content_sha256_at_revision",
    "scientific_code_fingerprint",
    "validate_capture_compatibility",
]
