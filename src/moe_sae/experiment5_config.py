"""Strict executable configuration boundary for Gemma Experiment 5.

The checked-in YAML files are the scientific contract.  This module exposes a
small compatibility view used by the existing resumable TPU operator while
rejecting any document that changes the frozen H/R/B/F layouts, token budget,
optimizer, parent artifacts, or full-FFN objective.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from moe_sae.contribution_artifacts import canonical_sha256

SCHEMA_VERSION = "gemma4-routed-shared-hybrid-run-config-v1"
PROTOCOL_ID = "gemma4-routed-shared-hybrid-v1-20260728"
PHASE_NAME = "routed_shared_hybrid_10m"
PANELS = ("sae", "transcoder")
CONDITIONS = ("H", "R", "B", "F")

TRAIN_TARGET_SHA256 = (
    "f29aaf45d9ff93be08cc3d175cdcea9fa4aecc37fc0e256d09720e88e1ae838c"
)
EVAL_TARGET_SHA256 = (
    "31454e5ce8693521c41a2a95f2ef88d40c5d26c726a0efac5c26d882ed341f5e"
)
TRAIN_SOURCE_SHA256 = (
    "8f27dcf9c1dc0c7f3ffa2e2941daec13d8a72c7204077f09420e15aacb02ffe0"
)
EVAL_SOURCE_SHA256 = (
    "f68e2fe6e4cf568e694925697dd5d65fe10160a52d357c641892de85ee5c3da7"
)
SOURCE_CAPTURE_CONTRACT_SHA256 = (
    "fe0b004ca6e48c9eb8163237a5ea20f45b33acdbd9a3a267a9fb4e670ed2c954"
)
RECOMPOSITION_SIDECAR_SHA256 = (
    "37c7b8c012f8fc0c1f63210986c0b3071a261261c88ee4bc1f502220ce3b8526"
)
RECOMPOSITION_SIDECAR_FILE_SHA256 = (
    "6486288677e81665b21815879729c424c6acb8c7d0e45be264b961a9d652fe3d"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_TRAIN_TOKENS = 10_000_000
_EVAL_TOKENS = 250_000
_EFFECTIVE_BATCH = 16_384
_MICROBATCH = 4_096
_UPDATES = 611
_WARMUP = 61
_TAIL_MASK = 10_624
_RESTART_TOKENS = tuple(range(1_000_000, 10_000_001, 1_000_000))
_EVAL_CHECKPOINTS = (2_000_000, 5_000_000, 10_000_000)


class Experiment5ConfigError(ValueError):
    """An Experiment-5 config is incomplete or scientifically incompatible."""


def experiment5_randomization_key(seed: int) -> int:
    """Return the protocol- and seed-separated uint64 key for condition R.

    The existing bijection primitive mixes this key with each packed sequence
    ID, completing the frozen ``protocol_id | seed | sequence_id`` key.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in (0, 1, 2):
        raise Experiment5ConfigError("randomization seed must be one of 0/1/2")
    digest = hashlib.sha256(
        f"{PROTOCOL_ID}\0seed={seed}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Experiment5ConfigError(f"{name} must be a mapping")
    return value


def _exact(value: Any, expected: Any, *, name: str) -> None:
    if value != expected:
        raise Experiment5ConfigError(
            f"{name} is frozen to {expected!r}, got {value!r}"
        )


def _sha(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Experiment5ConfigError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Experiment5ConfigError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Experiment5ConfigError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class Experiment5Provenance:
    maxtext_fork_commit: str


@dataclass(frozen=True, slots=True)
class Experiment5Model:
    hidden_size: int
    num_routed_experts: int
    experts_per_token: int


@dataclass(frozen=True, slots=True)
class Experiment5Runtime:
    effective_batch_tokens: int
    microbatch_tokens: int
    capacity_margin_fraction: float
    accelerator_min_useful_mfu: float
    accelerator_qualification_max_seconds: int
    direct_runtime_max_seconds: int


@dataclass(frozen=True, slots=True)
class Experiment5Optimizer:
    parameter_dtype: str
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    global_gradient_clip: float
    peak_learning_rate: float
    minimum_learning_rate: float


@dataclass(frozen=True, slots=True)
class Experiment5Layout:
    condition: str
    architecture: str
    total_features: int
    candidates_per_token: int
    total_l0: int
    trainable_parameters: int
    routed_bank_width: int | None
    routed_topk_per_component: int | None
    shared_bank_width: int | None
    shared_topk_per_token: int | None
    global_topk: int | None

    @property
    def bank_width(self) -> int:
        """Compatibility width for generic checkpoint/worker metadata."""

        return self.total_features

    @property
    def topk_per_component(self) -> int:
        return self.total_l0


@dataclass(frozen=True, slots=True)
class Experiment5Phase:
    name: str
    layout_set: str
    seeds: tuple[int, ...]
    train_tokens: int
    eval_tokens: int
    planned_updates: int
    warmup_updates: int
    tail_mask_tokens: int
    restart_checkpoint_tokens: tuple[int, ...]
    evaluation_checkpoint_tokens: tuple[int, ...]
    initialization: str

    @property
    def checkpoint_tokens(self) -> tuple[int, ...]:
        return self.restart_checkpoint_tokens


@dataclass(frozen=True, slots=True)
class Experiment5Config:
    raw: Mapping[str, Any]
    config_hash: str
    protocol_id: str
    panel: Literal["sae", "transcoder"]
    scientific_model: Literal["sae", "transcoder"]
    encoder_activation: str
    provenance: Experiment5Provenance
    model: Experiment5Model
    runtime: Experiment5Runtime
    optimizer: Experiment5Optimizer
    layouts: tuple[Experiment5Layout, ...]
    parents: Mapping[str, Any]
    experiment_phase: Experiment5Phase
    recomposition_sidecar_sha256: str

    def validate(self) -> None:
        if canonical_sha256(self.raw) != self.config_hash:
            raise Experiment5ConfigError("Experiment-5 config identity changed")
        if self.protocol_id != PROTOCOL_ID or self.panel not in PANELS:
            raise Experiment5ConfigError("Experiment-5 protocol identity drifted")
        if tuple(layout.condition for layout in self.layouts) != CONDITIONS:
            raise Experiment5ConfigError("layouts must be exactly H/R/B/F")

    def phase(self, name: str) -> Experiment5Phase:
        if name != PHASE_NAME:
            raise Experiment5ConfigError(
                f"Experiment 5 has one phase named {PHASE_NAME!r}"
            )
        return self.experiment_phase

    def layouts_for_phase(self, name: str) -> tuple[Experiment5Layout, ...]:
        self.phase(name)
        return self.layouts

    def as_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(self.raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )


def _layout(value: Any, *, index: int) -> Experiment5Layout:
    raw = _mapping(value, name=f"layouts[{index}]")
    condition = raw.get("condition")
    expected = CONDITIONS[index]
    _exact(condition, expected, name=f"layouts[{index}].condition")
    architecture = raw.get("architecture")
    if condition in {"H", "R"}:
        _exact(architecture, "routed_shared_hybrid", name=f"{condition}.architecture")
        _exact(raw.get("routed_bank_count"), 128, name=f"{condition}.routed_bank_count")
        _exact(raw.get("routed_bank_width"), 512, name=f"{condition}.routed_bank_width")
        _exact(
            raw.get("routed_visible_banks_per_token"),
            8,
            name=f"{condition}.routed_visible_banks_per_token",
        )
        _exact(
            raw.get("routed_topk_per_component"),
            4,
            name=f"{condition}.routed_topk_per_component",
        )
        _exact(raw.get("shared_bank_count"), 1, name=f"{condition}.shared_bank_count")
        _exact(raw.get("shared_bank_width"), 4096, name=f"{condition}.shared_bank_width")
        _exact(raw.get("shared_topk_per_token"), 32, name=f"{condition}.shared_topk")
        expected_identity = (
            "true_expert_id"
            if condition == "H"
            else "per_sequence_randomized_expert_id"
        )
        _exact(
            raw.get("routed_bank_identity"),
            expected_identity,
            name=f"{condition}.routed_bank_identity",
        )
        routed_width = 512
        routed_k = 4
        shared_width = 4096
        shared_k = 32
        global_k = None
        expected_candidates = 8192
    else:
        _exact(architecture, "global", name=f"{condition}.architecture")
        expected_width = 8192 if condition == "B" else 69632
        _exact(raw.get("bank_count"), 1, name=f"{condition}.bank_count")
        _exact(raw.get("bank_width"), expected_width, name=f"{condition}.bank_width")
        _exact(raw.get("topk_per_token"), 64, name=f"{condition}.topk_per_token")
        routed_width = None
        routed_k = None
        shared_width = None
        shared_k = None
        global_k = 64
        expected_candidates = expected_width
    expected_features = 69632 if condition in {"H", "R", "F"} else 8192
    expected_parameters = 392_237_056 if expected_features == 69632 else 46_145_536
    _exact(raw.get("total_features"), expected_features, name=f"{condition}.features")
    _exact(
        raw.get("candidates_per_token"),
        expected_candidates,
        name=f"{condition}.candidates",
    )
    _exact(raw.get("exact_total_l0"), 64, name=f"{condition}.L0")
    _exact(
        raw.get("trainable_parameters"),
        expected_parameters,
        name=f"{condition}.parameters",
    )
    return Experiment5Layout(
        condition=str(condition),
        architecture=str(architecture),
        total_features=expected_features,
        candidates_per_token=expected_candidates,
        total_l0=64,
        trainable_parameters=expected_parameters,
        routed_bank_width=routed_width,
        routed_topk_per_component=routed_k,
        shared_bank_width=shared_width,
        shared_topk_per_token=shared_k,
        global_topk=global_k,
    )


def _validate_document(value: Any, *, expected_panel: str | None) -> Experiment5Config:
    top = _mapping(value, name="config")
    required_top = {
        "schema_version",
        "protocol_id",
        "panel",
        "parent_protocol_id",
        "provenance",
        "model",
        "artifacts",
        "training",
        "centering",
        "optimizer",
        "layouts",
        "randomized_control",
        "execution",
        "budget",
        "evaluation",
    }
    if set(top) != required_top:
        raise Experiment5ConfigError(
            f"config fields differ: expected {sorted(required_top)}, got {sorted(top)}"
        )
    _exact(top["schema_version"], SCHEMA_VERSION, name="schema_version")
    _exact(top["protocol_id"], PROTOCOL_ID, name="protocol_id")
    panel = top["panel"]
    if panel not in PANELS or (expected_panel is not None and panel != expected_panel):
        raise Experiment5ConfigError("panel differs from the requested executable")

    provenance = _mapping(top["provenance"], name="provenance")
    fork = provenance.get("maxtext_fork_commit")
    if not isinstance(fork, str) or _GIT_RE.fullmatch(fork) is None:
        raise Experiment5ConfigError("maxtext_fork_commit must be a Git SHA")
    _exact(
        provenance.get("model_id"),
        "google/gemma-4-26B-A4B",
        name="provenance.model_id",
    )

    model = _mapping(top["model"], name="model")
    for key, expected in (
        ("target_layer_one_indexed", 15),
        ("hidden_size", 2816),
        ("num_routed_experts", 128),
        ("experts_per_token", 8),
        ("shared_experts", 1),
    ):
        _exact(model.get(key), expected, name=f"model.{key}")

    artifacts = _mapping(top["artifacts"], name="artifacts")
    _exact(
        _sha(artifacts.get("train_target_manifest_sha256"), name="train target"),
        TRAIN_TARGET_SHA256,
        name="artifacts.train_target_manifest_sha256",
    )
    _exact(
        _sha(artifacts.get("eval_target_manifest_sha256"), name="eval target"),
        EVAL_TARGET_SHA256,
        name="artifacts.eval_target_manifest_sha256",
    )
    sidecar_sha = _sha(
        artifacts.get("recomposition_sidecar_content_sha256"),
        name="artifacts.recomposition_sidecar_content_sha256",
    )
    _exact(
        sidecar_sha,
        RECOMPOSITION_SIDECAR_SHA256,
        name="artifacts.recomposition_sidecar_content_sha256",
    )
    _exact(
        _sha(
            artifacts.get("recomposition_sidecar_file_sha256"),
            name="artifacts.recomposition_sidecar_file_sha256",
        ),
        RECOMPOSITION_SIDECAR_FILE_SHA256,
        name="artifacts.recomposition_sidecar_file_sha256",
    )
    if panel == "transcoder":
        for key, expected in (
            ("train_source_overlay_manifest_sha256", TRAIN_SOURCE_SHA256),
            ("eval_source_overlay_manifest_sha256", EVAL_SOURCE_SHA256),
            (
                "source_overlay_capture_contract_sha256",
                SOURCE_CAPTURE_CONTRACT_SHA256,
            ),
        ):
            _exact(
                _sha(artifacts.get(key), name=f"artifacts.{key}"),
                expected,
                name=f"artifacts.{key}",
            )
    else:
        for key in (
            "train_source_overlay_manifest_sha256",
            "eval_source_overlay_manifest_sha256",
        ):
            _exact(artifacts.get(key), None, name=f"artifacts.{key}")

    training = _mapping(top["training"], name="training")
    for key, expected in (
        ("phase_name", PHASE_NAME),
        ("seeds", [0, 1, 2]),
        ("initialization", "from_scratch"),
        ("encoder_activation", "relu" if panel == "sae" else "softplus"),
        ("train_tokens", _TRAIN_TOKENS),
        ("eval_tokens", _EVAL_TOKENS),
        ("effective_batch_tokens", _EFFECTIVE_BATCH),
        ("microbatch_tokens", _MICROBATCH),
        ("planned_updates", _UPDATES),
        ("warmup_updates", _WARMUP),
        ("tail_mask_tokens", _TAIL_MASK),
        ("objective_target", "exact_full_ffn_output_before_residual"),
        ("objective", "normalized_full_ffn_sse"),
        ("topk_requires_strictly_positive", True),
        ("capacity_overflow_policy", "hard_error"),
        ("token_drop_policy", "hard_error"),
    ):
        _exact(training.get(key), expected, name=f"training.{key}")

    optimizer = _mapping(top["optimizer"], name="optimizer")
    frozen_optimizer = {
        "parameter_dtype": "bfloat16",
        "matmul_dtype": "bfloat16",
        "moment_dtype": "float32",
        "reduction_dtype": "float32",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "global_gradient_clip": 1.0,
        "peak_learning_rate": 3.0e-4,
        "minimum_learning_rate": 3.0e-5,
        "schedule": "linear_warmup_cosine_decay",
        "decoder_init": "unit_norm_columns",
        "encoder_init": "decoder_transpose",
        "encoder_bias_init": "zeros",
        "tangent_decoder_gradient_projection": True,
        "decoder_column_renormalization": True,
    }
    for key, expected in frozen_optimizer.items():
        _exact(optimizer.get(key), expected, name=f"optimizer.{key}")

    raw_layouts = top["layouts"]
    if not isinstance(raw_layouts, list) or len(raw_layouts) != len(CONDITIONS):
        raise Experiment5ConfigError("layouts must be a four-item list")
    layouts = tuple(_layout(item, index=index) for index, item in enumerate(raw_layouts))

    execution = _mapping(top["execution"], name="execution")
    _exact(
        execution.get("accelerator_mfu_gate_enforced"),
        False,
        name="execution.accelerator_mfu_gate_enforced",
    )
    _exact(
        tuple(execution.get("restart_checkpoint_tokens", ())),
        _RESTART_TOKENS,
        name="execution.restart_checkpoint_tokens",
    )
    _exact(execution.get("new_bulk_capture_allowed"), False, name="new harvest")
    _exact(execution.get("cross_region_bulk_copy_allowed"), False, name="bulk copy")
    runtime = Experiment5Runtime(
        effective_batch_tokens=_EFFECTIVE_BATCH,
        microbatch_tokens=_MICROBATCH,
        capacity_margin_fraction=_finite(
            training.get("capacity_margin_fraction"),
            name="training.capacity_margin_fraction",
        ),
        accelerator_min_useful_mfu=_finite(
            execution.get("accelerator_min_useful_mfu"),
            name="execution.accelerator_min_useful_mfu",
        ),
        accelerator_qualification_max_seconds=int(
            execution.get("accelerator_qualification_max_seconds")
        ),
        direct_runtime_max_seconds=int(execution.get("direct_runtime_max_seconds")),
    )
    _exact(runtime.capacity_margin_fraction, 0.10, name="capacity margin")
    _exact(runtime.accelerator_min_useful_mfu, 0.10, name="minimum MFU")
    _exact(runtime.accelerator_qualification_max_seconds, 600, name="qualification")
    _exact(runtime.direct_runtime_max_seconds, 43_200, name="direct runtime")

    evaluation = _mapping(top["evaluation"], name="evaluation")
    _exact(
        tuple(evaluation.get("checkpoint_tokens", ())),
        _EVAL_CHECKPOINTS,
        name="evaluation.checkpoint_tokens",
    )
    _exact(evaluation.get("primary_endpoint"), "full_ffn", name="primary endpoint")
    _exact(evaluation.get("common_target_sst"), True, name="common target SST")

    budget = _mapping(top["budget"], name="budget")
    _exact(_finite(budget.get("incremental_cash_hard_limit"), name="cost cap"), 50.0, name="cost cap")
    _exact(budget.get("stop_before_projected_exceedance"), True, name="budget stop")

    parents = {
        "train_target_manifest_sha256": TRAIN_TARGET_SHA256,
        "eval_target_manifest_sha256": EVAL_TARGET_SHA256,
        "train_source_overlay_manifest_sha256": (
            TRAIN_SOURCE_SHA256 if panel == "transcoder" else None
        ),
        "eval_source_overlay_manifest_sha256": (
            EVAL_SOURCE_SHA256 if panel == "transcoder" else None
        ),
        "source_overlay_capture_contract_sha256": (
            SOURCE_CAPTURE_CONTRACT_SHA256 if panel == "transcoder" else None
        ),
    }
    raw_copy = json.loads(
        json.dumps(top, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    result = Experiment5Config(
        raw=raw_copy,
        config_hash=canonical_sha256(raw_copy),
        protocol_id=PROTOCOL_ID,
        panel=panel,
        scientific_model=panel,
        encoder_activation=str(training["encoder_activation"]),
        provenance=Experiment5Provenance(maxtext_fork_commit=fork),
        model=Experiment5Model(2816, 128, 8),
        runtime=runtime,
        optimizer=Experiment5Optimizer(
            parameter_dtype="bfloat16",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1.0e-8,
            global_gradient_clip=1.0,
            peak_learning_rate=3.0e-4,
            minimum_learning_rate=3.0e-5,
        ),
        layouts=layouts,
        parents=parents,
        experiment_phase=Experiment5Phase(
            name=PHASE_NAME,
            layout_set="hybrid",
            seeds=(0, 1, 2),
            train_tokens=_TRAIN_TOKENS,
            eval_tokens=_EVAL_TOKENS,
            planned_updates=_UPDATES,
            warmup_updates=_WARMUP,
            tail_mask_tokens=_TAIL_MASK,
            restart_checkpoint_tokens=_RESTART_TOKENS,
            evaluation_checkpoint_tokens=_EVAL_CHECKPOINTS,
            initialization="from_scratch",
        ),
        recomposition_sidecar_sha256=sidecar_sha,
    )
    result.validate()
    return result


def load_experiment5_config(
    path: str | Path,
    *,
    expected_panel: Literal["sae", "transcoder"] | None = None,
) -> Experiment5Config:
    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise Experiment5ConfigError(f"cannot read Experiment-5 config {source}") from error
    return _validate_document(raw, expected_panel=expected_panel)


__all__ = [
    "CONDITIONS",
    "PHASE_NAME",
    "PROTOCOL_ID",
    "RECOMPOSITION_SIDECAR_SHA256",
    "SCHEMA_VERSION",
    "Experiment5Config",
    "Experiment5ConfigError",
    "Experiment5Layout",
    "experiment5_randomization_key",
    "load_experiment5_config",
]
