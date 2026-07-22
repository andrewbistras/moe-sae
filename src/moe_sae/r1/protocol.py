"""Versioned, canonical wire records for the DeepSeek-R1 control plane.

This module is deliberately independent of JAX, MaxText, and cloud clients.  It
defines the values that cross trust boundaries between a controller and
ephemeral executors.  Every record:

* is an immutable, slotted dataclass;
* carries an exact schema version in its JSON representation;
* rejects unknown fields and ambiguous integer values such as ``True``;
* has a deterministic canonical JSON encoding and SHA-256 digest.

Scientific, implementation, and deployment identities are separate on
purpose.  Moving an otherwise identical job to another TPU slice must not
change its scientific identity or logical work ID.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, ClassVar, Mapping, Self


_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{7,64}")
_CRC32C = re.compile(r"[A-Za-z0-9+/]{6}==")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]*")

# Public validation primitives for additive protocol versions.  Older record
# implementations retain their private spellings; new modules should use these
# names rather than importing underscored implementation details.
SHA256_PATTERN = _SHA256
IDENTIFIER_PATTERN = _IDENTIFIER


class ProtocolError(ValueError):
    """A control-plane record is malformed or violates its contract."""


def _reject_json_constant(value: str) -> None:
    raise ProtocolError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_json(value: str | bytes) -> Any:
    """Parse strict JSON, rejecting duplicate keys and non-finite numbers."""

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProtocolError("JSON must be UTF-8") from error
    if not isinstance(value, str):
        raise ProtocolError("JSON input must be str or bytes")
    try:
        parsed = json.loads(
            value,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except ProtocolError:
        raise
    except (TypeError, json.JSONDecodeError) as error:
        raise ProtocolError("invalid JSON") from error
    _validate_json_tree(parsed)
    return parsed


def _validate_json_tree(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ProtocolError(f"non-finite number at {path}")
        return
    if isinstance(value, list) or isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_json_tree(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError(f"non-string object key at {path}")
            _validate_json_tree(item, path=f"{path}.{key}")
        return
    raise ProtocolError(
        f"unsupported canonical JSON value at {path}: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the unique UTF-8 JSON representation of a record or JSON tree."""

    if isinstance(value, CanonicalRecord):
        value = value.as_dict()
    _validate_json_tree(value)
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ProtocolError("value cannot be encoded as canonical JSON") from error
    return encoded.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_object(
    value: Any,
    *,
    schema: str,
    fields: tuple[str, ...],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{schema} must be a JSON object")
    expected = {"schema", *fields}
    actual = set(value)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        details = []
        if unknown:
            details.append(f"unknown fields={unknown}")
        if missing:
            details.append(f"missing fields={missing}")
        raise ProtocolError(f"invalid {schema} fields: {'; '.join(details)}")
    if value["schema"] != schema:
        raise ProtocolError(
            f"schema mismatch: expected {schema!r}, got {value['schema']!r}"
        )
    return value


def _string(value: Any, name: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{name} must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ProtocolError(f"{name} has an invalid format")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ProtocolError(f"{name} must be an integer (booleans are forbidden)")
    if minimum is not None and value < minimum:
        raise ProtocolError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ProtocolError(f"{name} must be <= {maximum}")
    return value


def require_integer(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Validate an unambiguous JSON integer."""

    return _integer(value, name, minimum=minimum, maximum=maximum)


def require_string(
    value: Any,
    name: str,
    *,
    pattern: re.Pattern[str] | None = None,
) -> str:
    """Validate a non-empty JSON string, optionally against ``pattern``."""

    return _string(value, name, pattern=pattern)


def require_record_object(
    value: Any,
    *,
    schema: str,
    fields: tuple[str, ...],
) -> Mapping[str, Any]:
    """Validate the exact schema and field set of a canonical record."""

    return _strict_object(value, schema=schema, fields=fields)


def _number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ProtocolError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite")
    if strictly_positive and result <= 0.0:
        raise ProtocolError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ProtocolError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ProtocolError(f"{name} must be <= {maximum}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProtocolError(f"{name} must be a boolean")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProtocolError(f"{name} must be an array")
    result = tuple(_string(item, f"{name}[]") for item in value)
    if not result:
        raise ProtocolError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ProtocolError(f"{name} must not contain duplicates")
    return result


def _integer_tuple(value: Any, name: str, *, minimum: int = 0) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ProtocolError(f"{name} must be an array")
    result = tuple(_integer(item, f"{name}[]", minimum=minimum) for item in value)
    if not result:
        raise ProtocolError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ProtocolError(f"{name} must not contain duplicates")
    return result


class CanonicalRecord:
    """Mixin for strict immutable protocol records."""

    __slots__ = ()
    SCHEMA: ClassVar[str]

    def as_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        raise NotImplementedError

    @classmethod
    def from_json(cls, value: str | bytes) -> Self:
        parsed = parse_json(value)
        if not isinstance(parsed, Mapping):
            raise ProtocolError(f"{cls.SCHEMA} must be a JSON object")
        return cls.from_dict(parsed)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    def canonical_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class RegionalReplicaV1(CanonicalRecord):
    """One immutable, generation-pinned physical copy of logical content."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.regional-replica.v1"

    region: str
    uri: str
    generation: int
    size_bytes: int
    crc32c: str
    sha256: str

    def __post_init__(self) -> None:
        _string(self.region, "region", pattern=_IDENTIFIER)
        uri = _string(self.uri, "uri")
        if not uri.startswith("gs://") or uri == "gs://":
            raise ProtocolError("uri must be a non-empty gs:// object URI")
        _integer(self.generation, "generation", minimum=1)
        _integer(self.size_bytes, "size_bytes", minimum=1)
        _string(self.crc32c, "crc32c", pattern=_CRC32C)
        _string(self.sha256, "sha256", pattern=_SHA256)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "region": self.region,
            "uri": self.uri,
            "generation": self.generation,
            "size_bytes": self.size_bytes,
            "crc32c": self.crc32c,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _strict_object(
            value,
            schema=cls.SCHEMA,
            fields=("region", "uri", "generation", "size_bytes", "crc32c", "sha256"),
        )
        return cls(
            region=_string(obj["region"], "region", pattern=_IDENTIFIER),
            uri=_string(obj["uri"], "uri"),
            generation=_integer(obj["generation"], "generation", minimum=1),
            size_bytes=_integer(obj["size_bytes"], "size_bytes", minimum=1),
            crc32c=_string(obj["crc32c"], "crc32c", pattern=_CRC32C),
            sha256=_string(obj["sha256"], "sha256", pattern=_SHA256),
        )


@dataclass(frozen=True, slots=True)
class ContentRefV1(CanonicalRecord):
    """Logical content identity and its interchangeable regional replicas."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.content-ref.v1"

    sha256: str
    size_bytes: int
    media_type: str
    replicas: tuple[RegionalReplicaV1, ...]

    def __post_init__(self) -> None:
        _string(self.sha256, "sha256", pattern=_SHA256)
        _integer(self.size_bytes, "size_bytes", minimum=1)
        _string(self.media_type, "media_type")
        replicas = tuple(self.replicas)
        if not replicas or not all(
            isinstance(item, RegionalReplicaV1) for item in replicas
        ):
            raise ProtocolError("replicas must contain RegionalReplicaV1 records")
        uris = [item.uri for item in replicas]
        if len(set(uris)) != len(uris):
            raise ProtocolError("replica URIs must be unique")
        for replica in replicas:
            if replica.sha256 != self.sha256 or replica.size_bytes != self.size_bytes:
                raise ProtocolError(
                    "every replica must match the logical SHA-256 and size"
                )
        object.__setattr__(
            self,
            "replicas",
            tuple(
                sorted(
                    replicas, key=lambda item: (item.region, item.uri, item.generation)
                )
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "replicas": [item.as_dict() for item in self.replicas],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _strict_object(
            value,
            schema=cls.SCHEMA,
            fields=("sha256", "size_bytes", "media_type", "replicas"),
        )
        raw_replicas = obj["replicas"]
        if not isinstance(raw_replicas, list):
            raise ProtocolError("replicas must be an array")
        return cls(
            sha256=_string(obj["sha256"], "sha256", pattern=_SHA256),
            size_bytes=_integer(obj["size_bytes"], "size_bytes", minimum=1),
            media_type=_string(obj["media_type"], "media_type"),
            replicas=tuple(RegionalReplicaV1.from_dict(item) for item in raw_replicas),
        )


@dataclass(frozen=True, slots=True)
class ActivationContractV1(CanonicalRecord):
    """Exact scientific definition of the captured model quantity."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.activation-contract.v1"

    layer_one_indexed: int
    capture_site: str
    hidden_dim: int
    expert_count: int
    routed_experts_per_token: int
    sequence_length: int
    activation_dtype: str
    expert_id_dtype: str
    router_weight_dtype: str
    router_weight_sum: float
    router_selection_semantics: str

    def __post_init__(self) -> None:
        _integer(self.layer_one_indexed, "layer_one_indexed", minimum=1)
        _string(self.capture_site, "capture_site")
        _integer(self.hidden_dim, "hidden_dim", minimum=1)
        _integer(self.expert_count, "expert_count", minimum=1)
        _integer(
            self.routed_experts_per_token,
            "routed_experts_per_token",
            minimum=1,
            maximum=self.expert_count,
        )
        _integer(self.sequence_length, "sequence_length", minimum=1)
        _string(self.activation_dtype, "activation_dtype")
        _string(self.expert_id_dtype, "expert_id_dtype")
        _string(self.router_weight_dtype, "router_weight_dtype")
        _number(self.router_weight_sum, "router_weight_sum", strictly_positive=True)
        _string(self.router_selection_semantics, "router_selection_semantics")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "layer_one_indexed": self.layer_one_indexed,
            "capture_site": self.capture_site,
            "hidden_dim": self.hidden_dim,
            "expert_count": self.expert_count,
            "routed_experts_per_token": self.routed_experts_per_token,
            "sequence_length": self.sequence_length,
            "activation_dtype": self.activation_dtype,
            "expert_id_dtype": self.expert_id_dtype,
            "router_weight_dtype": self.router_weight_dtype,
            "router_weight_sum": float(self.router_weight_sum),
            "router_selection_semantics": self.router_selection_semantics,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "layer_one_indexed",
            "capture_site",
            "hidden_dim",
            "expert_count",
            "routed_experts_per_token",
            "sequence_length",
            "activation_dtype",
            "expert_id_dtype",
            "router_weight_dtype",
            "router_weight_sum",
            "router_selection_semantics",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            layer_one_indexed=_integer(
                obj["layer_one_indexed"], "layer_one_indexed", minimum=1
            ),
            capture_site=_string(obj["capture_site"], "capture_site"),
            hidden_dim=_integer(obj["hidden_dim"], "hidden_dim", minimum=1),
            expert_count=_integer(obj["expert_count"], "expert_count", minimum=1),
            routed_experts_per_token=_integer(
                obj["routed_experts_per_token"],
                "routed_experts_per_token",
                minimum=1,
            ),
            sequence_length=_integer(
                obj["sequence_length"], "sequence_length", minimum=1
            ),
            activation_dtype=_string(obj["activation_dtype"], "activation_dtype"),
            expert_id_dtype=_string(obj["expert_id_dtype"], "expert_id_dtype"),
            router_weight_dtype=_string(
                obj["router_weight_dtype"], "router_weight_dtype"
            ),
            router_weight_sum=_number(
                obj["router_weight_sum"], "router_weight_sum", strictly_positive=True
            ),
            router_selection_semantics=_string(
                obj["router_selection_semantics"], "router_selection_semantics"
            ),
        )


@dataclass(frozen=True, slots=True)
class SAEContractV1(CanonicalRecord):
    """Scientific SAE geometry and selection semantics."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.sae-contract.v1"

    conditions: tuple[str, ...]
    bank_count: int
    bank_width: int
    baseline_width: int
    full_width: int
    top_k_features: int
    per_token_topk: bool
    objective: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        conditions = _string_tuple(self.conditions, "conditions")
        if any(item not in {"A", "B", "F"} for item in conditions):
            raise ProtocolError("conditions may contain only A, B, and F")
        _integer(self.bank_count, "bank_count", minimum=1)
        _integer(self.bank_width, "bank_width", minimum=1)
        _integer(self.baseline_width, "baseline_width", minimum=1)
        _integer(self.full_width, "full_width", minimum=1)
        if self.full_width != self.bank_count * self.bank_width:
            raise ProtocolError("full_width must equal bank_count * bank_width")
        _integer(
            self.top_k_features,
            "top_k_features",
            minimum=1,
            maximum=min(self.baseline_width, self.full_width),
        )
        _boolean(self.per_token_topk, "per_token_topk")
        _string(self.objective, "objective")
        _integer_tuple(self.seeds, "seeds", minimum=0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "conditions": list(self.conditions),
            "bank_count": self.bank_count,
            "bank_width": self.bank_width,
            "baseline_width": self.baseline_width,
            "full_width": self.full_width,
            "top_k_features": self.top_k_features,
            "per_token_topk": self.per_token_topk,
            "objective": self.objective,
            "seeds": list(self.seeds),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "conditions",
            "bank_count",
            "bank_width",
            "baseline_width",
            "full_width",
            "top_k_features",
            "per_token_topk",
            "objective",
            "seeds",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            conditions=_string_tuple(obj["conditions"], "conditions"),
            bank_count=_integer(obj["bank_count"], "bank_count", minimum=1),
            bank_width=_integer(obj["bank_width"], "bank_width", minimum=1),
            baseline_width=_integer(obj["baseline_width"], "baseline_width", minimum=1),
            full_width=_integer(obj["full_width"], "full_width", minimum=1),
            top_k_features=_integer(obj["top_k_features"], "top_k_features", minimum=1),
            per_token_topk=_boolean(obj["per_token_topk"], "per_token_topk"),
            objective=_string(obj["objective"], "objective"),
            seeds=_integer_tuple(obj["seeds"], "seeds", minimum=0),
        )


@dataclass(frozen=True, slots=True)
class OptimizerContractV1(CanonicalRecord):
    """Training schedule whose changes alter the scientific experiment."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.optimizer-contract.v1"

    training_tokens: int
    updates: int
    effective_batch_tokens: int
    microbatch_tokens: int
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    gradient_clip_norm: float
    warmup_updates: int
    peak_learning_rate: float
    final_learning_rate: float

    def __post_init__(self) -> None:
        _integer(self.training_tokens, "training_tokens", minimum=1)
        _integer(self.updates, "updates", minimum=1)
        _integer(self.effective_batch_tokens, "effective_batch_tokens", minimum=1)
        _integer(self.microbatch_tokens, "microbatch_tokens", minimum=1)
        if self.effective_batch_tokens % self.microbatch_tokens:
            raise ProtocolError(
                "effective_batch_tokens must be divisible by microbatch_tokens"
            )
        _number(self.beta1, "beta1", minimum=0.0, maximum=1.0)
        _number(self.beta2, "beta2", minimum=0.0, maximum=1.0)
        _number(self.epsilon, "epsilon", strictly_positive=True)
        _number(self.weight_decay, "weight_decay", minimum=0.0)
        _number(self.gradient_clip_norm, "gradient_clip_norm", strictly_positive=True)
        _integer(self.warmup_updates, "warmup_updates", minimum=0)
        if self.warmup_updates >= self.updates:
            raise ProtocolError("warmup_updates must be smaller than updates")
        _number(self.peak_learning_rate, "peak_learning_rate", strictly_positive=True)
        _number(self.final_learning_rate, "final_learning_rate", strictly_positive=True)
        if self.final_learning_rate > self.peak_learning_rate:
            raise ProtocolError(
                "final_learning_rate must not exceed peak_learning_rate"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "training_tokens": self.training_tokens,
            "updates": self.updates,
            "effective_batch_tokens": self.effective_batch_tokens,
            "microbatch_tokens": self.microbatch_tokens,
            "beta1": float(self.beta1),
            "beta2": float(self.beta2),
            "epsilon": float(self.epsilon),
            "weight_decay": float(self.weight_decay),
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "warmup_updates": self.warmup_updates,
            "peak_learning_rate": float(self.peak_learning_rate),
            "final_learning_rate": float(self.final_learning_rate),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "training_tokens",
            "updates",
            "effective_batch_tokens",
            "microbatch_tokens",
            "beta1",
            "beta2",
            "epsilon",
            "weight_decay",
            "gradient_clip_norm",
            "warmup_updates",
            "peak_learning_rate",
            "final_learning_rate",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            training_tokens=_integer(
                obj["training_tokens"], "training_tokens", minimum=1
            ),
            updates=_integer(obj["updates"], "updates", minimum=1),
            effective_batch_tokens=_integer(
                obj["effective_batch_tokens"], "effective_batch_tokens", minimum=1
            ),
            microbatch_tokens=_integer(
                obj["microbatch_tokens"], "microbatch_tokens", minimum=1
            ),
            beta1=_number(obj["beta1"], "beta1", minimum=0.0, maximum=1.0),
            beta2=_number(obj["beta2"], "beta2", minimum=0.0, maximum=1.0),
            epsilon=_number(obj["epsilon"], "epsilon", strictly_positive=True),
            weight_decay=_number(obj["weight_decay"], "weight_decay", minimum=0.0),
            gradient_clip_norm=_number(
                obj["gradient_clip_norm"],
                "gradient_clip_norm",
                strictly_positive=True,
            ),
            warmup_updates=_integer(obj["warmup_updates"], "warmup_updates", minimum=0),
            peak_learning_rate=_number(
                obj["peak_learning_rate"],
                "peak_learning_rate",
                strictly_positive=True,
            ),
            final_learning_rate=_number(
                obj["final_learning_rate"],
                "final_learning_rate",
                strictly_positive=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class ScientificIdentityV1(CanonicalRecord):
    """Everything that can change the scientific meaning of the result."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.scientific-identity.v1"

    model_id: str
    model_revision: str
    model_tensor_inventory_sha256: str
    tokenizer_id: str
    tokenizer_revision: str
    corpus_id: str
    corpus_revision: str
    corpus_fingerprint_sha256: str
    train_split: str
    evaluation_split: str
    activation: ActivationContractV1
    sae: SAEContractV1
    optimizer: OptimizerContractV1
    report_contract_version: str

    def __post_init__(self) -> None:
        _string(self.model_id, "model_id")
        _string(self.model_revision, "model_revision")
        _string(
            self.model_tensor_inventory_sha256,
            "model_tensor_inventory_sha256",
            pattern=_SHA256,
        )
        _string(self.tokenizer_id, "tokenizer_id")
        _string(self.tokenizer_revision, "tokenizer_revision")
        _string(self.corpus_id, "corpus_id")
        _string(self.corpus_revision, "corpus_revision")
        _string(
            self.corpus_fingerprint_sha256,
            "corpus_fingerprint_sha256",
            pattern=_SHA256,
        )
        _string(self.train_split, "train_split")
        _string(self.evaluation_split, "evaluation_split")
        if not isinstance(self.activation, ActivationContractV1):
            raise ProtocolError("activation must be ActivationContractV1")
        if not isinstance(self.sae, SAEContractV1):
            raise ProtocolError("sae must be SAEContractV1")
        if not isinstance(self.optimizer, OptimizerContractV1):
            raise ProtocolError("optimizer must be OptimizerContractV1")
        if self.sae.bank_count != self.activation.expert_count:
            raise ProtocolError("SAE bank_count must equal activation expert_count")
        _string(self.report_contract_version, "report_contract_version")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_tensor_inventory_sha256": self.model_tensor_inventory_sha256,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "corpus_id": self.corpus_id,
            "corpus_revision": self.corpus_revision,
            "corpus_fingerprint_sha256": self.corpus_fingerprint_sha256,
            "train_split": self.train_split,
            "evaluation_split": self.evaluation_split,
            "activation": self.activation.as_dict(),
            "sae": self.sae.as_dict(),
            "optimizer": self.optimizer.as_dict(),
            "report_contract_version": self.report_contract_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "model_id",
            "model_revision",
            "model_tensor_inventory_sha256",
            "tokenizer_id",
            "tokenizer_revision",
            "corpus_id",
            "corpus_revision",
            "corpus_fingerprint_sha256",
            "train_split",
            "evaluation_split",
            "activation",
            "sae",
            "optimizer",
            "report_contract_version",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            model_id=_string(obj["model_id"], "model_id"),
            model_revision=_string(obj["model_revision"], "model_revision"),
            model_tensor_inventory_sha256=_string(
                obj["model_tensor_inventory_sha256"],
                "model_tensor_inventory_sha256",
                pattern=_SHA256,
            ),
            tokenizer_id=_string(obj["tokenizer_id"], "tokenizer_id"),
            tokenizer_revision=_string(obj["tokenizer_revision"], "tokenizer_revision"),
            corpus_id=_string(obj["corpus_id"], "corpus_id"),
            corpus_revision=_string(obj["corpus_revision"], "corpus_revision"),
            corpus_fingerprint_sha256=_string(
                obj["corpus_fingerprint_sha256"],
                "corpus_fingerprint_sha256",
                pattern=_SHA256,
            ),
            train_split=_string(obj["train_split"], "train_split"),
            evaluation_split=_string(obj["evaluation_split"], "evaluation_split"),
            activation=ActivationContractV1.from_dict(obj["activation"]),
            sae=SAEContractV1.from_dict(obj["sae"]),
            optimizer=OptimizerContractV1.from_dict(obj["optimizer"]),
            report_contract_version=_string(
                obj["report_contract_version"], "report_contract_version"
            ),
        )


@dataclass(frozen=True, slots=True)
class ImplementationIdentityV1(CanonicalRecord):
    """Pinned software/runtime implementation, including its image manifest."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.implementation-identity.v1"

    repository_commit: str
    maxtext_commit: str
    image_manifest_sha256: str
    python_version: str
    jax_version: str
    libtpu_version: str
    adapter_version: str
    converter_version: str

    def __post_init__(self) -> None:
        _string(self.repository_commit, "repository_commit", pattern=_REVISION)
        _string(self.maxtext_commit, "maxtext_commit", pattern=_REVISION)
        _string(
            self.image_manifest_sha256,
            "image_manifest_sha256",
            pattern=_SHA256,
        )
        _string(self.python_version, "python_version")
        _string(self.jax_version, "jax_version")
        _string(self.libtpu_version, "libtpu_version")
        _string(self.adapter_version, "adapter_version")
        _string(self.converter_version, "converter_version")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "repository_commit": self.repository_commit,
            "maxtext_commit": self.maxtext_commit,
            "image_manifest_sha256": self.image_manifest_sha256,
            "python_version": self.python_version,
            "jax_version": self.jax_version,
            "libtpu_version": self.libtpu_version,
            "adapter_version": self.adapter_version,
            "converter_version": self.converter_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "repository_commit",
            "maxtext_commit",
            "image_manifest_sha256",
            "python_version",
            "jax_version",
            "libtpu_version",
            "adapter_version",
            "converter_version",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            repository_commit=_string(
                obj["repository_commit"], "repository_commit", pattern=_REVISION
            ),
            maxtext_commit=_string(
                obj["maxtext_commit"], "maxtext_commit", pattern=_REVISION
            ),
            image_manifest_sha256=_string(
                obj["image_manifest_sha256"],
                "image_manifest_sha256",
                pattern=_SHA256,
            ),
            python_version=_string(obj["python_version"], "python_version"),
            jax_version=_string(obj["jax_version"], "jax_version"),
            libtpu_version=_string(obj["libtpu_version"], "libtpu_version"),
            adapter_version=_string(obj["adapter_version"], "adapter_version"),
            converter_version=_string(obj["converter_version"], "converter_version"),
        )


@dataclass(frozen=True, slots=True)
class DeploymentIdentityV1(CanonicalRecord):
    """Content links for one physical attempt; never a scientific work ID."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.deployment-identity.v1"

    implementation_identity_sha256: str
    deployment_envelope_sha256: str
    device_inventory_sha256: str
    attempt_id: str

    def __post_init__(self) -> None:
        _string(
            self.implementation_identity_sha256,
            "implementation_identity_sha256",
            pattern=_SHA256,
        )
        _string(
            self.deployment_envelope_sha256,
            "deployment_envelope_sha256",
            pattern=_SHA256,
        )
        _string(
            self.device_inventory_sha256,
            "device_inventory_sha256",
            pattern=_SHA256,
        )
        _string(self.attempt_id, "attempt_id", pattern=_IDENTIFIER)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "implementation_identity_sha256": self.implementation_identity_sha256,
            "deployment_envelope_sha256": self.deployment_envelope_sha256,
            "device_inventory_sha256": self.device_inventory_sha256,
            "attempt_id": self.attempt_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "implementation_identity_sha256",
            "deployment_envelope_sha256",
            "device_inventory_sha256",
            "attempt_id",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            implementation_identity_sha256=_string(
                obj["implementation_identity_sha256"],
                "implementation_identity_sha256",
                pattern=_SHA256,
            ),
            deployment_envelope_sha256=_string(
                obj["deployment_envelope_sha256"],
                "deployment_envelope_sha256",
                pattern=_SHA256,
            ),
            device_inventory_sha256=_string(
                obj["device_inventory_sha256"],
                "device_inventory_sha256",
                pattern=_SHA256,
            ),
            attempt_id=_string(obj["attempt_id"], "attempt_id", pattern=_IDENTIFIER),
        )


@dataclass(frozen=True, slots=True)
class HarvestRangeV1(CanonicalRecord):
    """A topology-neutral, immutable interval in a staged corpus."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.harvest-range.v1"

    scientific_identity_sha256: str
    corpus_fingerprint_sha256: str
    split: str
    record_start: int
    record_end: int
    scientific_token_start: int
    scientific_token_end: int

    def __post_init__(self) -> None:
        _string(
            self.scientific_identity_sha256,
            "scientific_identity_sha256",
            pattern=_SHA256,
        )
        _string(
            self.corpus_fingerprint_sha256,
            "corpus_fingerprint_sha256",
            pattern=_SHA256,
        )
        _string(self.split, "split")
        _integer(self.record_start, "record_start", minimum=0)
        _integer(self.record_end, "record_end", minimum=1)
        _integer(self.scientific_token_start, "scientific_token_start", minimum=0)
        _integer(self.scientific_token_end, "scientific_token_end", minimum=1)
        if self.record_end <= self.record_start:
            raise ProtocolError("record range must be non-empty")
        if self.scientific_token_end <= self.scientific_token_start:
            raise ProtocolError("scientific token range must be non-empty")

    @property
    def work_id(self) -> str:
        return f"harvest-{self.digest}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "scientific_identity_sha256": self.scientific_identity_sha256,
            "corpus_fingerprint_sha256": self.corpus_fingerprint_sha256,
            "split": self.split,
            "record_start": self.record_start,
            "record_end": self.record_end,
            "scientific_token_start": self.scientific_token_start,
            "scientific_token_end": self.scientific_token_end,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "scientific_identity_sha256",
            "corpus_fingerprint_sha256",
            "split",
            "record_start",
            "record_end",
            "scientific_token_start",
            "scientific_token_end",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            scientific_identity_sha256=_string(
                obj["scientific_identity_sha256"],
                "scientific_identity_sha256",
                pattern=_SHA256,
            ),
            corpus_fingerprint_sha256=_string(
                obj["corpus_fingerprint_sha256"],
                "corpus_fingerprint_sha256",
                pattern=_SHA256,
            ),
            split=_string(obj["split"], "split"),
            record_start=_integer(obj["record_start"], "record_start", minimum=0),
            record_end=_integer(obj["record_end"], "record_end", minimum=1),
            scientific_token_start=_integer(
                obj["scientific_token_start"],
                "scientific_token_start",
                minimum=0,
            ),
            scientific_token_end=_integer(
                obj["scientific_token_end"], "scientific_token_end", minimum=1
            ),
        )


@dataclass(frozen=True, slots=True)
class SAEJobV1(CanonicalRecord):
    """A topology-neutral condition/seed training job."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.sae-job.v1"

    scientific_identity_sha256: str
    capture_manifest_sha256: str
    activation_order_sha256: str
    condition: str
    seed: int

    def __post_init__(self) -> None:
        _string(
            self.scientific_identity_sha256,
            "scientific_identity_sha256",
            pattern=_SHA256,
        )
        _string(
            self.capture_manifest_sha256,
            "capture_manifest_sha256",
            pattern=_SHA256,
        )
        _string(
            self.activation_order_sha256,
            "activation_order_sha256",
            pattern=_SHA256,
        )
        if self.condition not in {"A", "B", "F"}:
            raise ProtocolError("condition must be A, B, or F")
        _integer(self.seed, "seed", minimum=0)

    @property
    def work_id(self) -> str:
        return f"sae-{self.digest}"

    @property
    def job_id(self) -> str:
        return self.work_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "scientific_identity_sha256": self.scientific_identity_sha256,
            "capture_manifest_sha256": self.capture_manifest_sha256,
            "activation_order_sha256": self.activation_order_sha256,
            "condition": self.condition,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "scientific_identity_sha256",
            "capture_manifest_sha256",
            "activation_order_sha256",
            "condition",
            "seed",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            scientific_identity_sha256=_string(
                obj["scientific_identity_sha256"],
                "scientific_identity_sha256",
                pattern=_SHA256,
            ),
            capture_manifest_sha256=_string(
                obj["capture_manifest_sha256"],
                "capture_manifest_sha256",
                pattern=_SHA256,
            ),
            activation_order_sha256=_string(
                obj["activation_order_sha256"],
                "activation_order_sha256",
                pattern=_SHA256,
            ),
            condition=_string(obj["condition"], "condition"),
            seed=_integer(obj["seed"], "seed", minimum=0),
        )


@dataclass(frozen=True, slots=True)
class AttemptLeaseV1(CanonicalRecord):
    """Fenced authority for one ephemeral executor to advance one work item."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.attempt-lease.v1"

    campaign_id: str
    work_id: str
    attempt_id: str
    fence: int
    executor_id: str
    deployment_identity_sha256: str
    acquired_at_unix_ms: int
    expires_at_unix_ms: int

    def __post_init__(self) -> None:
        _string(self.campaign_id, "campaign_id", pattern=_IDENTIFIER)
        _string(self.work_id, "work_id", pattern=_IDENTIFIER)
        _string(self.attempt_id, "attempt_id", pattern=_IDENTIFIER)
        _integer(self.fence, "fence", minimum=1)
        _string(self.executor_id, "executor_id", pattern=_IDENTIFIER)
        _string(
            self.deployment_identity_sha256,
            "deployment_identity_sha256",
            pattern=_SHA256,
        )
        _integer(self.acquired_at_unix_ms, "acquired_at_unix_ms", minimum=0)
        _integer(self.expires_at_unix_ms, "expires_at_unix_ms", minimum=1)
        if self.expires_at_unix_ms <= self.acquired_at_unix_ms:
            raise ProtocolError("lease expiry must be after acquisition")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "campaign_id": self.campaign_id,
            "work_id": self.work_id,
            "attempt_id": self.attempt_id,
            "fence": self.fence,
            "executor_id": self.executor_id,
            "deployment_identity_sha256": self.deployment_identity_sha256,
            "acquired_at_unix_ms": self.acquired_at_unix_ms,
            "expires_at_unix_ms": self.expires_at_unix_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "campaign_id",
            "work_id",
            "attempt_id",
            "fence",
            "executor_id",
            "deployment_identity_sha256",
            "acquired_at_unix_ms",
            "expires_at_unix_ms",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            campaign_id=_string(obj["campaign_id"], "campaign_id", pattern=_IDENTIFIER),
            work_id=_string(obj["work_id"], "work_id", pattern=_IDENTIFIER),
            attempt_id=_string(obj["attempt_id"], "attempt_id", pattern=_IDENTIFIER),
            fence=_integer(obj["fence"], "fence", minimum=1),
            executor_id=_string(obj["executor_id"], "executor_id", pattern=_IDENTIFIER),
            deployment_identity_sha256=_string(
                obj["deployment_identity_sha256"],
                "deployment_identity_sha256",
                pattern=_SHA256,
            ),
            acquired_at_unix_ms=_integer(
                obj["acquired_at_unix_ms"], "acquired_at_unix_ms", minimum=0
            ),
            expires_at_unix_ms=_integer(
                obj["expires_at_unix_ms"], "expires_at_unix_ms", minimum=1
            ),
        )


@dataclass(frozen=True, slots=True)
class ProgressV1(CanonicalRecord):
    """Monotonic numerical progress for the currently fenced attempt."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.progress.v1"

    work_id: str
    attempt_id: str
    fence: int
    sequence: int
    phase: str
    completed_units: int
    total_units: int
    updated_at_unix_ms: int

    def __post_init__(self) -> None:
        _string(self.work_id, "work_id", pattern=_IDENTIFIER)
        _string(self.attempt_id, "attempt_id", pattern=_IDENTIFIER)
        _integer(self.fence, "fence", minimum=1)
        _integer(self.sequence, "sequence", minimum=0)
        _string(self.phase, "phase", pattern=_IDENTIFIER)
        _integer(self.completed_units, "completed_units", minimum=0)
        _integer(self.total_units, "total_units", minimum=1)
        if self.completed_units > self.total_units:
            raise ProtocolError("completed_units must not exceed total_units")
        _integer(self.updated_at_unix_ms, "updated_at_unix_ms", minimum=0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "work_id": self.work_id,
            "attempt_id": self.attempt_id,
            "fence": self.fence,
            "sequence": self.sequence,
            "phase": self.phase,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
            "updated_at_unix_ms": self.updated_at_unix_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "work_id",
            "attempt_id",
            "fence",
            "sequence",
            "phase",
            "completed_units",
            "total_units",
            "updated_at_unix_ms",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            work_id=_string(obj["work_id"], "work_id", pattern=_IDENTIFIER),
            attempt_id=_string(obj["attempt_id"], "attempt_id", pattern=_IDENTIFIER),
            fence=_integer(obj["fence"], "fence", minimum=1),
            sequence=_integer(obj["sequence"], "sequence", minimum=0),
            phase=_string(obj["phase"], "phase", pattern=_IDENTIFIER),
            completed_units=_integer(
                obj["completed_units"], "completed_units", minimum=0
            ),
            total_units=_integer(obj["total_units"], "total_units", minimum=1),
            updated_at_unix_ms=_integer(
                obj["updated_at_unix_ms"], "updated_at_unix_ms", minimum=0
            ),
        )


@dataclass(frozen=True, slots=True)
class CostLedgerV1(CanonicalRecord):
    """Both accepted algorithmic cost and realized infrastructure overhead."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.cost-ledger.v1"

    successful_chip_seconds: float
    attempted_chip_seconds: float
    compilation_chip_seconds: float
    checkpoint_seconds: float
    queue_seconds: float
    egress_bytes: int
    replayed_units: int

    def __post_init__(self) -> None:
        successful = _number(
            self.successful_chip_seconds,
            "successful_chip_seconds",
            minimum=0.0,
        )
        attempted = _number(
            self.attempted_chip_seconds, "attempted_chip_seconds", minimum=0.0
        )
        if successful > attempted:
            raise ProtocolError(
                "successful_chip_seconds must not exceed attempted_chip_seconds"
            )
        _number(
            self.compilation_chip_seconds,
            "compilation_chip_seconds",
            minimum=0.0,
        )
        _number(self.checkpoint_seconds, "checkpoint_seconds", minimum=0.0)
        _number(self.queue_seconds, "queue_seconds", minimum=0.0)
        _integer(self.egress_bytes, "egress_bytes", minimum=0)
        _integer(self.replayed_units, "replayed_units", minimum=0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "successful_chip_seconds": float(self.successful_chip_seconds),
            "attempted_chip_seconds": float(self.attempted_chip_seconds),
            "compilation_chip_seconds": float(self.compilation_chip_seconds),
            "checkpoint_seconds": float(self.checkpoint_seconds),
            "queue_seconds": float(self.queue_seconds),
            "egress_bytes": self.egress_bytes,
            "replayed_units": self.replayed_units,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "successful_chip_seconds",
            "attempted_chip_seconds",
            "compilation_chip_seconds",
            "checkpoint_seconds",
            "queue_seconds",
            "egress_bytes",
            "replayed_units",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            successful_chip_seconds=_number(
                obj["successful_chip_seconds"],
                "successful_chip_seconds",
                minimum=0.0,
            ),
            attempted_chip_seconds=_number(
                obj["attempted_chip_seconds"],
                "attempted_chip_seconds",
                minimum=0.0,
            ),
            compilation_chip_seconds=_number(
                obj["compilation_chip_seconds"],
                "compilation_chip_seconds",
                minimum=0.0,
            ),
            checkpoint_seconds=_number(
                obj["checkpoint_seconds"], "checkpoint_seconds", minimum=0.0
            ),
            queue_seconds=_number(obj["queue_seconds"], "queue_seconds", minimum=0.0),
            egress_bytes=_integer(obj["egress_bytes"], "egress_bytes", minimum=0),
            replayed_units=_integer(obj["replayed_units"], "replayed_units", minimum=0),
        )


@dataclass(frozen=True, slots=True)
class AttemptReceiptV1(CanonicalRecord):
    """Immutable final claim made by one fenced attempt."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.attempt-receipt.v1"

    work_id: str
    attempt_id: str
    fence: int
    final_progress_sequence: int
    outcome: str
    outputs: tuple[ContentRefV1, ...]
    cost: CostLedgerV1
    termination_reason: str | None
    finished_at_unix_ms: int

    def __post_init__(self) -> None:
        _string(self.work_id, "work_id", pattern=_IDENTIFIER)
        _string(self.attempt_id, "attempt_id", pattern=_IDENTIFIER)
        _integer(self.fence, "fence", minimum=1)
        _integer(self.final_progress_sequence, "final_progress_sequence", minimum=0)
        if self.outcome not in {"SUCCEEDED", "PREEMPTED", "FAILED"}:
            raise ProtocolError("outcome must be SUCCEEDED, PREEMPTED, or FAILED")
        outputs = tuple(self.outputs)
        if not all(isinstance(item, ContentRefV1) for item in outputs):
            raise ProtocolError("outputs must contain ContentRefV1 records")
        if self.outcome == "SUCCEEDED" and not outputs:
            raise ProtocolError("successful receipt must contain at least one output")
        if len({item.sha256 for item in outputs}) != len(outputs):
            raise ProtocolError("receipt outputs must have distinct logical content")
        object.__setattr__(
            self, "outputs", tuple(sorted(outputs, key=lambda x: x.sha256))
        )
        if not isinstance(self.cost, CostLedgerV1):
            raise ProtocolError("cost must be CostLedgerV1")
        _optional_string(self.termination_reason, "termination_reason")
        if self.outcome != "SUCCEEDED" and self.termination_reason is None:
            raise ProtocolError("non-success receipt requires a termination_reason")
        _integer(self.finished_at_unix_ms, "finished_at_unix_ms", minimum=0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "work_id": self.work_id,
            "attempt_id": self.attempt_id,
            "fence": self.fence,
            "final_progress_sequence": self.final_progress_sequence,
            "outcome": self.outcome,
            "outputs": [item.as_dict() for item in self.outputs],
            "cost": self.cost.as_dict(),
            "termination_reason": self.termination_reason,
            "finished_at_unix_ms": self.finished_at_unix_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "work_id",
            "attempt_id",
            "fence",
            "final_progress_sequence",
            "outcome",
            "outputs",
            "cost",
            "termination_reason",
            "finished_at_unix_ms",
        )
        obj = _strict_object(value, schema=cls.SCHEMA, fields=fields)
        raw_outputs = obj["outputs"]
        if not isinstance(raw_outputs, list):
            raise ProtocolError("outputs must be an array")
        return cls(
            work_id=_string(obj["work_id"], "work_id", pattern=_IDENTIFIER),
            attempt_id=_string(obj["attempt_id"], "attempt_id", pattern=_IDENTIFIER),
            fence=_integer(obj["fence"], "fence", minimum=1),
            final_progress_sequence=_integer(
                obj["final_progress_sequence"],
                "final_progress_sequence",
                minimum=0,
            ),
            outcome=_string(obj["outcome"], "outcome"),
            outputs=tuple(ContentRefV1.from_dict(item) for item in raw_outputs),
            cost=CostLedgerV1.from_dict(obj["cost"]),
            termination_reason=_optional_string(
                obj["termination_reason"], "termination_reason"
            ),
            finished_at_unix_ms=_integer(
                obj["finished_at_unix_ms"], "finished_at_unix_ms", minimum=0
            ),
        )


__all__ = [
    "ActivationContractV1",
    "AttemptLeaseV1",
    "AttemptReceiptV1",
    "CanonicalRecord",
    "ContentRefV1",
    "CostLedgerV1",
    "DeploymentIdentityV1",
    "HarvestRangeV1",
    "ImplementationIdentityV1",
    "IDENTIFIER_PATTERN",
    "OptimizerContractV1",
    "ProgressV1",
    "ProtocolError",
    "RegionalReplicaV1",
    "SAEContractV1",
    "SAEJobV1",
    "SHA256_PATTERN",
    "ScientificIdentityV1",
    "canonical_json_bytes",
    "canonical_sha256",
    "parse_json",
    "require_integer",
    "require_record_object",
    "require_string",
]
