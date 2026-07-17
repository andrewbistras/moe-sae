"""Single-source configuration for the Gemma 4 reconstruction experiment.

The previous project exposed many partially-overlapping configuration objects.
This module deliberately has one immutable configuration.  Model architecture
facts are pins, while scale and filesystem fields can be changed for smoke tests.
Unknown configuration keys are rejected instead of being silently ignored.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping


MODEL_ID = "google/gemma-4-26B-A4B"
MODEL_REVISION = "24548b62aa021d562695c04aaf7758a1ea47990b"
MAXTEXT_COMMIT = "48c7da3e28e8f57fe36ddf09d00eec53072a2ea0"
C4_REPOSITORY = "allenai/c4"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
CONDITIONS = ("A", "B", "F")


@dataclass(frozen=True, slots=True)
class Gemma4Config:
    """Validated configuration for capture, SAE training, and evaluation.

    The flat shape is intentional: it makes a run's complete contract easy to
    hash, diff, serialize, and pass through a CLI without precedence rules.
    """

    schema_version: str = "gemma4_v1"

    # Model and capture site pins.
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    tokenizer_path: str = MODEL_ID
    maxtext_commit: str = MAXTEXT_COMMIT
    maxtext_model_name: str = "gemma4-26b"
    num_decoder_layers: int = 30
    hidden_size: int = 2816
    num_query_heads: int = 16
    num_kv_heads: int = 8
    global_num_kv_heads: int = 2
    head_dim: int = 256
    global_head_dim: int = 512
    mlp_dim: int = 2112
    moe_mlp_dim: int = 704
    num_routed_experts: int = 128
    experts_per_token: int = 8
    num_shared_experts: int = 1
    target_layer_one_indexed: int = 15
    sequence_length: int = 1024
    dtype: str = "bfloat16"
    weight_dtype: str = "bfloat16"
    use_multimodal: bool = False
    scan_layers: bool = False
    pure_nnx_decoder: bool = True
    dropout_rate: float = 0.0
    load_balance_loss_weight: float = 0.0
    sliding_window_size: int = 1024
    share_kv_projections: bool = True
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True
    float32_gate_logits: bool = False

    # Deterministic text corpus and artifact contract.
    dataset_name: str = C4_REPOSITORY
    dataset_config: str = "en"
    dataset_revision: str = C4_REVISION
    train_split: str = "train"
    eval_split: str = "validation"
    train_tokens: int = 50_000_000
    eval_tokens: int = 2_000_000
    shard_tokens: int = 131_072
    artifacts_root: str = "artifacts/gemma4_v1"

    # A/B/F dictionary contract.
    conditions: tuple[str, ...] = CONDITIONS
    bank_width: int = 1024
    topk_features: int = 64
    seeds: tuple[int, ...] = (1, 2, 3)

    # Optimizer and schedule.
    effective_batch_tokens: int = 16_384
    microbatch_tokens: int = 4_096
    warmup_updates: int = 250
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    grad_clip: float = 1.0
    checkpoint_every_tokens: int = 10_000_000

    # Performance gates.
    min_harvester_mfu: float = 0.50
    benchmark_warmup_steps: int = 5
    benchmark_steps: int = 200
    benchmark_worker_timeout_seconds: int = 3_600
    canary_tokens: int = 2_000_000
    canary_goodput_fraction: float = 0.90
    max_input_wait_fraction: float = 0.05
    max_writer_backlog_batches: int = 2
    health_gate_tokens: int = 5_000_000
    min_live_feature_fraction: float = 0.95
    min_control_free_bytes: int = 10_000_000_000
    min_local_buffer_free_bytes: int = 128 * 1024**3
    gcs_probe_bytes_per_process: int = 8 * 1024**3
    gcs_probe_block_bytes: int = 8 * 1024**2
    min_gcs_upload_bytes_per_second: float = 4_000_000_000.0
    min_gcs_download_bytes_per_second: float = 4_000_000_000.0
    capture_cache_max_bytes: int = 100 * 1024**3

    def __post_init__(self) -> None:
        # YAML represents tuples as lists.  Normalize at the boundary while
        # retaining an immutable public object.
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        self.validate()

    @property
    def target_layer_index(self) -> int:
        """Zero-indexed MaxText layer index (block 15 -> ``layers[14]``)."""

        return self.target_layer_one_indexed - 1

    @property
    def total_features(self) -> int:
        return self.num_routed_experts * self.bank_width

    @property
    def visible_features(self) -> int:
        return self.experts_per_token * self.bank_width

    @property
    def candidate_exposure(self) -> float:
        return self.experts_per_token / self.num_routed_experts

    @property
    def planned_updates(self) -> int:
        return math.ceil(self.train_tokens / self.effective_batch_tokens)

    @property
    def final_padding_tokens(self) -> int:
        return self.planned_updates * self.effective_batch_tokens - self.train_tokens

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def width_for_condition(self, condition: str) -> int:
        """Number of dictionary features stored by a condition."""

        self._validate_condition(condition)
        return self.visible_features if condition == "B" else self.total_features

    def candidates_per_token(self, condition: str) -> int:
        """Number of encoder scores evaluated for one token."""

        self._validate_condition(condition)
        return self.total_features if condition == "F" else self.visible_features

    def parameter_count(self, condition: str) -> int:
        """Encoder + decoder + encoder-bias parameter count."""

        width = self.width_for_condition(condition)
        return width * (2 * self.hidden_size + 1)

    def _validate_condition(self, condition: str) -> None:
        if condition not in self.conditions:
            raise ValueError(
                f"condition must be one of {self.conditions}, got {condition!r}"
            )

    def validate(self) -> None:
        """Fail closed on architecture drift or internally inconsistent runs."""

        pinned: dict[str, Any] = {
            "schema_version": "gemma4_v1",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_path": MODEL_ID,
            "maxtext_commit": MAXTEXT_COMMIT,
            "maxtext_model_name": "gemma4-26b",
            "num_decoder_layers": 30,
            "hidden_size": 2816,
            "num_query_heads": 16,
            "num_kv_heads": 8,
            "global_num_kv_heads": 2,
            "head_dim": 256,
            "global_head_dim": 512,
            "mlp_dim": 2112,
            "moe_mlp_dim": 704,
            "num_routed_experts": 128,
            "experts_per_token": 8,
            "num_shared_experts": 1,
            "target_layer_one_indexed": 15,
            "sequence_length": 1024,
            "dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "use_multimodal": False,
            "scan_layers": False,
            "pure_nnx_decoder": True,
            "dropout_rate": 0.0,
            "load_balance_loss_weight": 0.0,
            "sliding_window_size": 1024,
            "share_kv_projections": True,
            "use_post_attn_norm": True,
            "use_post_ffw_norm": True,
            "float32_gate_logits": False,
            "dataset_name": C4_REPOSITORY,
            "dataset_config": "en",
            "dataset_revision": C4_REVISION,
            "train_split": "train",
            "eval_split": "validation",
        }
        for name, expected in pinned.items():
            actual = getattr(self, name)
            if actual != expected:
                raise ValueError(f"{name} is pinned to {expected!r}, got {actual!r}")

        if self.conditions != CONDITIONS:
            raise ValueError(
                f"conditions are exactly {CONDITIONS}; got {self.conditions}"
            )
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty tuple of unique integers")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in self.seeds
        ):
            raise ValueError("every seed must be a non-negative integer")

        positive_ints = (
            "train_tokens",
            "eval_tokens",
            "shard_tokens",
            "bank_width",
            "topk_features",
            "effective_batch_tokens",
            "microbatch_tokens",
            "warmup_updates",
            "checkpoint_every_tokens",
            "benchmark_warmup_steps",
            "benchmark_steps",
            "benchmark_worker_timeout_seconds",
            "canary_tokens",
            "max_writer_backlog_batches",
            "health_gate_tokens",
            "min_control_free_bytes",
            "min_local_buffer_free_bytes",
            "gcs_probe_bytes_per_process",
            "gcs_probe_block_bytes",
            "capture_cache_max_bytes",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if self.topk_features > self.visible_features:
            raise ValueError(
                "topk_features cannot exceed A/B's visible candidate count"
            )
        if self.effective_batch_tokens % self.microbatch_tokens:
            raise ValueError(
                "effective_batch_tokens must be divisible by microbatch_tokens"
            )
        if self.shard_tokens > self.train_tokens:
            raise ValueError("shard_tokens cannot exceed train_tokens")
        if self.shard_tokens > 131_072:
            raise ValueError(
                "shard_tokens cannot exceed the 131,072-row artifact limit"
            )
        if self.canary_tokens > self.train_tokens:
            raise ValueError("canary_tokens cannot exceed train_tokens")
        if self.health_gate_tokens > self.train_tokens:
            raise ValueError("health_gate_tokens cannot exceed train_tokens")
        if self.warmup_updates >= self.planned_updates:
            raise ValueError(
                "warmup_updates must be smaller than the number of training updates"
            )

        if not (0.0 < self.min_learning_rate <= self.learning_rate):
            raise ValueError(
                "learning rates must satisfy 0 < min_learning_rate <= learning_rate"
            )
        if not (0.0 < self.adam_beta1 < 1.0 and 0.0 < self.adam_beta2 < 1.0):
            raise ValueError("Adam betas must lie strictly between zero and one")
        if self.adam_epsilon <= 0.0 or self.grad_clip <= 0.0:
            raise ValueError("adam_epsilon and grad_clip must be positive")
        for name in (
            "min_harvester_mfu",
            "canary_goodput_fraction",
            "max_input_wait_fraction",
            "min_live_feature_fraction",
        ):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if self.min_harvester_mfu < 0.50:
            raise ValueError("min_harvester_mfu cannot weaken the 50% hard gate")
        if self.benchmark_warmup_steps != 5 or self.benchmark_steps < 200:
            raise ValueError(
                "harvester benchmarks require five warmups and at least 200 measured steps"
            )
        if self.canary_goodput_fraction < 0.90:
            raise ValueError("canary_goodput_fraction cannot weaken the 90% gate")
        if self.max_input_wait_fraction > 0.05:
            raise ValueError("max_input_wait_fraction cannot weaken the 5% gate")
        if self.max_writer_backlog_batches != 2:
            raise ValueError("max_writer_backlog_batches is fixed at two")
        if self.min_live_feature_fraction < 0.95:
            raise ValueError(
                "min_live_feature_fraction cannot weaken the 95% health gate"
            )
        if self.min_control_free_bytes < 10_000_000_000:
            raise ValueError("min_control_free_bytes cannot weaken the 10 GB gate")
        if self.min_local_buffer_free_bytes < 128 * 1024**3:
            raise ValueError(
                "min_local_buffer_free_bytes cannot weaken the 128 GiB gate"
            )
        if self.gcs_probe_bytes_per_process < 8 * 1024**3:
            raise ValueError("GCS probe must transfer at least 8 GiB per controller")
        if self.gcs_probe_block_bytes < 1024**2:
            raise ValueError("GCS probe block size must be at least 1 MiB")
        for name in (
            "min_gcs_upload_bytes_per_second",
            "min_gcs_download_bytes_per_second",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.min_gcs_upload_bytes_per_second < 4_000_000_000.0:
            raise ValueError("GCS upload gate cannot be below 4 GB/s")
        if self.min_gcs_download_bytes_per_second < 4_000_000_000.0:
            raise ValueError("GCS download gate cannot be below 4 GB/s")
        if self.capture_cache_max_bytes < 100 * 1024**3:
            raise ValueError("capture cache must hold at least 100 GiB per controller")
        required_local = (
            self.capture_cache_max_bytes + 2 * self.gcs_probe_bytes_per_process
        )
        if self.min_local_buffer_free_bytes < required_local:
            raise ValueError(
                "local-buffer gate must cover the bounded cache plus upload/download probes"
            )
        if not self.artifacts_root.strip():
            raise ValueError("artifacts_root cannot be empty")

    def validate_production(self) -> None:
        """Reject smoke-scale changes on authoritative TPU/report paths."""

        self.validate()
        expected: dict[str, Any] = {
            "train_tokens": 50_000_000,
            "eval_tokens": 2_000_000,
            "shard_tokens": 131_072,
            "conditions": CONDITIONS,
            "bank_width": 1_024,
            "topk_features": 64,
            "seeds": (1, 2, 3),
            "effective_batch_tokens": 16_384,
            "microbatch_tokens": 4_096,
            "warmup_updates": 250,
            "learning_rate": 3e-4,
            "min_learning_rate": 3e-5,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "grad_clip": 1.0,
            "checkpoint_every_tokens": 10_000_000,
            "canary_tokens": 2_000_000,
            "health_gate_tokens": 5_000_000,
            "min_live_feature_fraction": 0.95,
            "min_control_free_bytes": 10_000_000_000,
            "min_local_buffer_free_bytes": 128 * 1024**3,
            "gcs_probe_bytes_per_process": 8 * 1024**3,
            "gcs_probe_block_bytes": 8 * 1024**2,
            "min_gcs_upload_bytes_per_second": 4_000_000_000.0,
            "min_gcs_download_bytes_per_second": 4_000_000_000.0,
            "capture_cache_max_bytes": 100 * 1024**3,
        }
        for name, required in expected.items():
            actual = getattr(self, name)
            if actual != required:
                raise ValueError(
                    f"authoritative production requires {name}={required!r}, got {actual!r}"
                )
        if self.planned_updates != 3_052 or self.final_padding_tokens != 3_968:
            raise ValueError(
                "production schedule must be 3,052 updates with 3,968 masked tokens"
            )

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Gemma4Config":
        if not isinstance(values, Mapping):
            raise TypeError("configuration must be a mapping")
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown configuration keys: {', '.join(unknown)}")
        return cls(**dict(values))


def load_config(path: str | Path) -> Gemma4Config:
    """Load JSON or YAML and reject unknown keys and invalid pins."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        try:
            import yaml  # Imported lazily to keep the core config lightweight.
        except ImportError as exc:  # pragma: no cover - exercised in minimal installs
            raise RuntimeError(
                "PyYAML is required to load a YAML configuration"
            ) from exc
        raw = yaml.safe_load(text)
    return Gemma4Config.from_mapping(raw)


def save_config(config: Gemma4Config, path: str | Path) -> None:
    """Atomically serialize a validated configuration as JSON or YAML."""

    config.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if destination.suffix.lower() == ".json":
        payload = json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n"
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised in minimal installs
            raise RuntimeError(
                "PyYAML is required to save a YAML configuration"
            ) from exc
        payload = yaml.safe_dump(config.as_dict(), sort_keys=False)
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)
