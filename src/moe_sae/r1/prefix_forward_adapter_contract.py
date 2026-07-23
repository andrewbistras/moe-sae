"""Fail-closed patch-plan authority for the first R1 forward prefix adapter.

This module deliberately implements no JAX kernel and mutates no MaxText
checkout.  It records the exact upstream source identities, B128/factor-two
geometry, packed-input semantics, target router semantics, execution boundary,
and three-output device ABI that a later implementation must satisfy.

The pinned MaxText BF16 batch-split path is useful machinery, but it is not the
requested prefix harvester today: it does not thread packed segment IDs into
attention, computes the target shared expert, returns a residual-updated layer
output, installs a custom backward, and has no target metadata ABI.  Those are
patch obligations here, never inferred implementation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
from typing import Any, ClassVar, NoReturn

from .maxtext_tap_patch import (
    MAXTEXT_BASE_COMMIT,
    PATCHED_MOE_SHA256,
    PATCH_SHA256,
    TARGET_RELATIVE_PATH,
)
from .prefix_batchsplit_contract import (
    PINNED_BATCH_SPLIT_FACTOR,
    PINNED_DEVICE_COUNT,
    PINNED_GLOBAL_BATCH_SIZE,
    PINNED_LAYOUT_LADDER,
    PINNED_PREFIX_BATCHSPLIT_CONTRACT,
    PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256,
    PINNED_SEQUENCE_LENGTH,
    PINNED_TARGET_LAYER_ZERO_INDEXED,
)
from .prefix_checkpoint_v2 import EXTERNAL_QUALIFICATION_REQUIRED
from .protocol import canonical_json_bytes


PINNED_HIDDEN_DIM = 7_168
PINNED_EXPERT_COUNT = 256
PINNED_EXPERTS_PER_TOKEN = 8
PINNED_ROUTING_GROUP_COUNT = 8
PINNED_SELECTED_ROUTING_GROUP_COUNT = 4
PINNED_GROUP_SCORE_TOP_K = 2
PINNED_ROUTED_SCALING_FACTOR = 2.5

_INPUT_BATCH_AXES = ("data", "fsdp", "expert")
_EXECUTION_EXCLUSIONS = (
    "target-shared-expert",
    "target-outer-residual-add",
    "suffix-layers-31-through-60-zero-indexed",
    "final-normalization",
    "lm-head-and-logits",
    "mtp-layer-61-zero-indexed",
    "backward-autodiff-and-custom-vjp",
    "forward-residual-host-offload",
)
_PARITY_GATES = tuple(
    gate
    for gate in EXTERNAL_QUALIFICATION_REQUIRED
    if gate
    in (
        "full-model-r1-logit-parity",
        "prefix-versus-full-routed-aggregate-parity",
    )
)
_UNMET_IMPLEMENTATION_GATES = (
    "sealed-batchsplit-adapter-patch-artifact",
    "packed-segment-aware-prefix-attention",
    "forward-only-prefix-kernel",
    "fresh-process-prefix-restore-binding",
    *_PARITY_GATES,
    "v6e-64-b128-compile-and-runtime-qualification",
)


class PrefixForwardAdapterContractError(ValueError):
    """The proposed adapter contract differs from the frozen first tranche."""


class PrefixForwardAdapterProductionUnavailable(RuntimeError):
    """This patch-plan-only tranche cannot authorize a production execution."""


def _require_string(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise PrefixForwardAdapterContractError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    value = _require_string(value, name)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PrefixForwardAdapterContractError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _require_positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise PrefixForwardAdapterContractError(
            f"{name} must be a positive integer (booleans are forbidden)"
        )
    return value


@dataclass(frozen=True, slots=True)
class MaxTextSourceIdentityV1:
    """One immutable upstream file on which the patch plan depends."""

    relative_path: str
    sha256: str
    size_bytes: int
    git_blob: str
    role: str

    def __post_init__(self) -> None:
        path = PurePosixPath(_require_string(self.relative_path, "relative_path"))
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise PrefixForwardAdapterContractError(
                "MaxText source identity path must be repository-relative"
            )
        _require_sha256(self.sha256, "sha256")
        _require_positive_integer(self.size_bytes, "size_bytes")
        git_blob = _require_string(self.git_blob, "git_blob")
        if len(git_blob) != 40 or any(
            character not in "0123456789abcdef" for character in git_blob
        ):
            raise PrefixForwardAdapterContractError(
                "git_blob must be a lowercase SHA-1 object ID"
            )
        _require_string(self.role, "role")

    def verify_payload(self, payload: bytes) -> None:
        """Reject bytes that are not the exact pinned upstream file."""

        if type(payload) is not bytes:
            raise PrefixForwardAdapterContractError(
                f"{self.relative_path} payload must be exact bytes"
            )
        if len(payload) != self.size_bytes:
            raise PrefixForwardAdapterContractError(
                f"{self.relative_path} size differs from pinned source"
            )
        actual = hashlib.sha256(payload).hexdigest()
        if actual != self.sha256:
            raise PrefixForwardAdapterContractError(
                f"{self.relative_path} SHA-256 differs from pinned source"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "git_blob": self.git_blob,
            "role": self.role,
        }


PINNED_MAXTEXT_SOURCE_IDENTITIES = (
    MaxTextSourceIdentityV1(
        relative_path="src/maxtext/layers/nnx_decoders.py",
        sha256="995758f6c81b4d0e3c026aac3753de0beb7f54498ea55d9d210003c3db84cf89",
        size_bytes=78_805,
        git_blob="0661bb8e76d6512903c88a837cbbc8e41bfe79af",
        role="scanned-decoder-batchsplit-dispatch",
    ),
    MaxTextSourceIdentityV1(
        relative_path="src/maxtext/models/deepseek.py",
        sha256="cd84afded72d5f5c68eeb4382d51c1e75598d95f63ac439cdb439534147d1347",
        size_bytes=22_620,
        git_blob="3ec5c32f1f51d687be72e906a33d1ce3a0d7599a",
        role="deepseek-layer-initialization-and-trace-shape",
    ),
    MaxTextSourceIdentityV1(
        relative_path="src/maxtext/models/deepseek_batchsplit.py",
        sha256="c74c018b8c24df92a7c9941cc5323405fecb9dcff001163a32c1cc0e1fd480df",
        size_bytes=95_724,
        git_blob="aba66b6df6845e740152f3660e2ed94c7cae37b4",
        role="bf16-pure-jax-batchsplit-forward-machinery",
    ),
    MaxTextSourceIdentityV1(
        relative_path="src/maxtext/configs/models/deepseek3-671b-batchsplit.yml",
        sha256="07f77ce265522bb925f937c079bcb0a0e2b58b07171f2db4af4402e554cf3cd1",
        size_bytes=2_786,
        git_blob="a24c5af9df77d040772d5d547588fb8cb4d6ce03",
        role="released-deepseek3-batchsplit-geometry",
    ),
)


@dataclass(frozen=True, slots=True)
class AdapterInputTensorV1:
    """One host-to-device tensor in the packed B128 input ABI."""

    name: str
    shape: tuple[int, int]
    dtype: str
    semantics: str

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or type(self.shape) is not tuple
            or len(self.shape) != 2
            or any(type(dimension) is not int for dimension in self.shape)
            or type(self.dtype) is not str
            or type(self.semantics) is not str
        ):
            raise PrefixForwardAdapterContractError(
                "packed input ABI fields must use exact scalar and tuple types"
            )
        expected = {
            "decoder_input_tokens": (
                (PINNED_GLOBAL_BATCH_SIZE, PINNED_SEQUENCE_LENGTH),
                "int32",
                "token-ids-with-padding-excluded-from-scientific-counts",
            ),
            "decoder_segment_ids": (
                (PINNED_GLOBAL_BATCH_SIZE, PINNED_SEQUENCE_LENGTH),
                "int32",
                "zero-is-padding-positive-ids-delimit-packed-segments",
            ),
            "decoder_positions": (
                (PINNED_GLOBAL_BATCH_SIZE, PINNED_SEQUENCE_LENGTH),
                "int32",
                "zero-based-position-reset-at-each-positive-segment",
            ),
        }.get(self.name)
        if expected is None or (self.shape, self.dtype, self.semantics) != expected:
            raise PrefixForwardAdapterContractError(
                f"{self.name!r} differs from the pinned packed input ABI"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "semantics": self.semantics,
        }


PINNED_ADAPTER_INPUTS = (
    AdapterInputTensorV1(
        "decoder_input_tokens",
        (PINNED_GLOBAL_BATCH_SIZE, PINNED_SEQUENCE_LENGTH),
        "int32",
        "token-ids-with-padding-excluded-from-scientific-counts",
    ),
    AdapterInputTensorV1(
        "decoder_segment_ids",
        (PINNED_GLOBAL_BATCH_SIZE, PINNED_SEQUENCE_LENGTH),
        "int32",
        "zero-is-padding-positive-ids-delimit-packed-segments",
    ),
    AdapterInputTensorV1(
        "decoder_positions",
        (PINNED_GLOBAL_BATCH_SIZE, PINNED_SEQUENCE_LENGTH),
        "int32",
        "zero-based-position-reset-at-each-positive-segment",
    ),
)


@dataclass(frozen=True, slots=True)
class PackedSegmentPolicyV1:
    """Packed-attention and batch-split semantics that must stay coupled."""

    batch_axis: int = 0
    sequence_axis: int = 1
    padding_segment_id: int = 0
    split_factor: int = PINNED_BATCH_SPLIT_FACTOR
    split_batch_only: bool = True
    split_metadata_with_tokens: bool = True
    preserve_sequence_axis: bool = True
    valid_token_predicate: str = "segment_id > 0"
    attention_predicate: str = (
        "query_segment_id == key_segment_id != 0 and key_position <= query_position"
    )
    current_upstream_threads_segment_ids: bool = False
    current_upstream_mask: str = "plain-causal-mask-without-packed-segment-boundaries"

    def __post_init__(self) -> None:
        expected = {
            "batch_axis": (0, int),
            "sequence_axis": (1, int),
            "padding_segment_id": (0, int),
            "split_factor": (PINNED_BATCH_SPLIT_FACTOR, int),
            "split_batch_only": (True, bool),
            "split_metadata_with_tokens": (True, bool),
            "preserve_sequence_axis": (True, bool),
            "valid_token_predicate": ("segment_id > 0", str),
            "attention_predicate": (
                "query_segment_id == key_segment_id != 0 and key_position <= query_position",
                str,
            ),
            "current_upstream_threads_segment_ids": (False, bool),
            "current_upstream_mask": (
                "plain-causal-mask-without-packed-segment-boundaries",
                str,
            ),
        }
        for field_name, (value, expected_type) in expected.items():
            actual = getattr(self, field_name)
            if type(actual) is not expected_type or actual != value:
                raise PrefixForwardAdapterContractError(
                    f"packed segment policy {field_name} changed"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_axis": self.batch_axis,
            "sequence_axis": self.sequence_axis,
            "padding_segment_id": self.padding_segment_id,
            "split_factor": self.split_factor,
            "split_batch_only": self.split_batch_only,
            "split_metadata_with_tokens": self.split_metadata_with_tokens,
            "preserve_sequence_axis": self.preserve_sequence_axis,
            "valid_token_predicate": self.valid_token_predicate,
            "attention_predicate": self.attention_predicate,
            "current_upstream_threads_segment_ids": (
                self.current_upstream_threads_segment_ids
            ),
            "current_upstream_mask": self.current_upstream_mask,
        }


@dataclass(frozen=True, slots=True)
class RouterCaptureSemanticsV1:
    """Exact target router computation and metadata identity."""

    hidden_input_dtype: str = "bfloat16"
    gate_kernel_storage_dtype: str = "bfloat16"
    correction_bias_storage_dtype: str = "float32"
    projection_compute_dtype: str = "float32"
    sigmoid_score_dtype: str = "float32"
    selection_score: str = "fp32-sigmoid-score-plus-fp32-correction-bias"
    hidden_dim: int = PINNED_HIDDEN_DIM
    expert_count: int = PINNED_EXPERT_COUNT
    routing_group_count: int = PINNED_ROUTING_GROUP_COUNT
    selected_routing_group_count: int = PINNED_SELECTED_ROUTING_GROUP_COUNT
    group_score_top_k: int = PINNED_GROUP_SCORE_TOP_K
    group_selection: str = "8-groups-top4-by-sum-of-top2"
    experts_per_token: int = PINNED_EXPERTS_PER_TOKEN
    weight_semantics: str = "selected-unbiased-fp32-sigmoid-normalized-times-2.5"
    weight_dtype: str = "float32"
    global_id_semantics: str = "global-id-before-expert-shard-local-remap"
    global_id_dtype: str = "int32"
    global_id_minimum: int = 0
    global_id_exclusive_maximum: int = PINNED_EXPERT_COUNT
    unique_ids_per_token: bool = True
    expected_weight_sum: float = PINNED_ROUTED_SCALING_FACTOR

    def __post_init__(self) -> None:
        expected = {
            "hidden_input_dtype": ("bfloat16", str),
            "gate_kernel_storage_dtype": ("bfloat16", str),
            "correction_bias_storage_dtype": ("float32", str),
            "projection_compute_dtype": ("float32", str),
            "sigmoid_score_dtype": ("float32", str),
            "selection_score": (
                "fp32-sigmoid-score-plus-fp32-correction-bias",
                str,
            ),
            "hidden_dim": (PINNED_HIDDEN_DIM, int),
            "expert_count": (PINNED_EXPERT_COUNT, int),
            "routing_group_count": (PINNED_ROUTING_GROUP_COUNT, int),
            "selected_routing_group_count": (
                PINNED_SELECTED_ROUTING_GROUP_COUNT,
                int,
            ),
            "group_score_top_k": (PINNED_GROUP_SCORE_TOP_K, int),
            "group_selection": ("8-groups-top4-by-sum-of-top2", str),
            "experts_per_token": (PINNED_EXPERTS_PER_TOKEN, int),
            "weight_semantics": (
                "selected-unbiased-fp32-sigmoid-normalized-times-2.5",
                str,
            ),
            "weight_dtype": ("float32", str),
            "global_id_semantics": (
                "global-id-before-expert-shard-local-remap",
                str,
            ),
            "global_id_dtype": ("int32", str),
            "global_id_minimum": (0, int),
            "global_id_exclusive_maximum": (PINNED_EXPERT_COUNT, int),
            "unique_ids_per_token": (True, bool),
            "expected_weight_sum": (PINNED_ROUTED_SCALING_FACTOR, float),
        }
        for field_name, (value, expected_type) in expected.items():
            actual = getattr(self, field_name)
            if type(actual) is not expected_type or actual != value:
                raise PrefixForwardAdapterContractError(
                    f"router capture semantic {field_name} changed"
                )

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ForwardStageV1:
    """One ordered portion of the only allowed prefix execution."""

    name: str
    layer_start_zero_indexed: int | None
    layer_stop_exclusive_zero_indexed: int | None
    semantics: str

    def __post_init__(self) -> None:
        _require_string(self.name, "stage.name")
        _require_string(self.semantics, "stage.semantics")
        for name in (
            "layer_start_zero_indexed",
            "layer_stop_exclusive_zero_indexed",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise PrefixForwardAdapterContractError(
                    f"{name} must be None or a non-negative integer"
                )
        if (self.layer_start_zero_indexed is None) != (
            self.layer_stop_exclusive_zero_indexed is None
        ):
            raise PrefixForwardAdapterContractError(
                "stage layer bounds must be both present or both absent"
            )
        if (
            self.layer_start_zero_indexed is not None
            and self.layer_start_zero_indexed >= self.layer_stop_exclusive_zero_indexed
        ):
            raise PrefixForwardAdapterContractError("stage layer range is empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer_start_zero_indexed": self.layer_start_zero_indexed,
            "layer_stop_exclusive_zero_indexed": (
                self.layer_stop_exclusive_zero_indexed
            ),
            "semantics": self.semantics,
        }


PINNED_FORWARD_STAGES = (
    ForwardStageV1(
        "token-embedding",
        None,
        None,
        "embed-packed-token-ids-once",
    ),
    ForwardStageV1(
        "completed-dense-prefix",
        0,
        3,
        "execute-attention-mlp-and-residual-for-all-three-dense-layers",
    ),
    ForwardStageV1(
        "completed-moe-prefix",
        3,
        30,
        "execute-attention-routed-plus-shared-experts-and-residual-for-27-layers",
    ),
    ForwardStageV1(
        "target-attention-and-branch-normalization",
        30,
        31,
        "execute-target-attention-residual-then-post-attention-rmsnorm",
    ),
    ForwardStageV1(
        "target-router-and-routed-experts",
        30,
        31,
        "emit-weighted-routed-aggregate-global-ids-and-fp32-weights-before-shared-expert-and-residual",
    ),
)


@dataclass(frozen=True, slots=True)
class AdapterPatchObligationV1:
    """A source-level change required before any kernel claim is possible."""

    ordinal: int
    source_relative_path: str
    symbol: str
    upstream_state: str
    required_state: str

    def __post_init__(self) -> None:
        _require_positive_integer(self.ordinal, "patch obligation ordinal")
        source_relative_path = _require_string(
            self.source_relative_path,
            "source_relative_path",
        )
        source_paths = {
            identity.relative_path for identity in PINNED_MAXTEXT_SOURCE_IDENTITIES
        }
        if source_relative_path not in source_paths:
            raise PrefixForwardAdapterContractError(
                "patch obligation references an unpinned MaxText source"
            )
        for field_name in ("symbol", "upstream_state", "required_state"):
            _require_string(getattr(self, field_name), field_name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "source_relative_path": self.source_relative_path,
            "symbol": self.symbol,
            "upstream_state": self.upstream_state,
            "required_state": self.required_state,
        }


PINNED_PATCH_OBLIGATIONS = (
    AdapterPatchObligationV1(
        1,
        "src/maxtext/layers/nnx_decoders.py",
        "Decoder.__call__ batch-split dispatch",
        "dispatches-all-58-moe-layers-and-omits-segment-ids-on-bf16-path",
        "dispatches-27-complete-moe-layers-plus-one-partial-target-and-threads-segment-ids",
    ),
    AdapterPatchObligationV1(
        2,
        "src/maxtext/models/deepseek_batchsplit.py",
        "scan_batch_split_layers/process_all_layers",
        "custom-vjp-full-moe-scan-with-forward-residual-host-offload",
        "forward-only-prefix-scan-with-no-custom-vjp-backward-or-host-offload",
    ),
    AdapterPatchObligationV1(
        3,
        "src/maxtext/models/deepseek_batchsplit.py",
        "init_splash_kernel/mla",
        "plain-causal-mask-does-not-consume-packed-segment-ids",
        "block-diagonal-packed-causal-mask-coupled-to-segment-ids-and-positions",
    ),
    AdapterPatchObligationV1(
        4,
        "src/maxtext/models/deepseek_batchsplit.py",
        "fetch_weights/shared_expert_and_route",
        "fetches-and-executes-shared-expert-before-routing-for-every-moe-layer",
        "completed-layers-retain-shared-expert-while-target-fetch-and-execution-omit-it",
    ),
    AdapterPatchObligationV1(
        5,
        "src/maxtext/models/deepseek_batchsplit.py",
        "expert_selection/route",
        "router-math-uses-ambient-dot-dtype-and-selected-ids-are-remapped-for-dispatch",
        "router-projection-and-scores-are-fp32-and-global-ids-are-captured-before-remap",
    ),
    AdapterPatchObligationV1(
        6,
        "src/maxtext/models/deepseek_batchsplit.py",
        "compute_linear/unroute_ubatch_fn",
        "unroute-adds-shared-expert-and-outer-residual-then-returns-layer-output",
        "target-unroute-emits-weighted-routed-aggregate-before-shared-expert-and-residual",
    ),
    AdapterPatchObligationV1(
        7,
        "src/maxtext/models/deepseek.py",
        "DeepSeekMoELayer batch-split trace branch",
        "trace-shape-api-returns-one-layer-output",
        "trace-shape-api-is-separated-from-static-three-output-prefix-adapter-entrypoint",
    ),
)


@dataclass(frozen=True, slots=True)
class ExistingTapReferenceV1:
    """Identity and scope boundary of the already-tested ordinary MoE tap."""

    target_relative_path: str = TARGET_RELATIVE_PATH.as_posix()
    patch_sha256: str = PATCH_SHA256
    patched_source_sha256: str = PATCHED_MOE_SHA256
    router_oracle: str = "moe_sae.r1.router_oracle.numpy_router_oracle"
    tap_parity_test: str = "tests/r1/test_maxtext_tap_patch.py"
    reusable_as_batchsplit_kernel: bool = False
    reason: str = "ordinary-routed-moe-tap-explicitly-rejects-use_batch_split_schedule"

    def __post_init__(self) -> None:
        target_relative_path = _require_string(
            self.target_relative_path,
            "ordinary tap target_relative_path",
        )
        _require_sha256(self.patch_sha256, "ordinary tap patch_sha256")
        _require_sha256(
            self.patched_source_sha256,
            "ordinary tap patched_source_sha256",
        )
        if target_relative_path != TARGET_RELATIVE_PATH.as_posix():
            raise PrefixForwardAdapterContractError("ordinary tap target changed")
        if self.patch_sha256 != PATCH_SHA256:
            raise PrefixForwardAdapterContractError(
                "ordinary tap patch identity changed"
            )
        if self.patched_source_sha256 != PATCHED_MOE_SHA256:
            raise PrefixForwardAdapterContractError(
                "ordinary tap patched source identity changed"
            )
        if self.reusable_as_batchsplit_kernel is not False:
            raise PrefixForwardAdapterContractError(
                "ordinary tap cannot be claimed as a batch-split kernel"
            )
        for field_name in ("router_oracle", "tap_parity_test", "reason"):
            _require_string(getattr(self, field_name), field_name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_relative_path": self.target_relative_path,
            "patch_sha256": self.patch_sha256,
            "patched_source_sha256": self.patched_source_sha256,
            "router_oracle": self.router_oracle,
            "tap_parity_test": self.tap_parity_test,
            "reusable_as_batchsplit_kernel": self.reusable_as_batchsplit_kernel,
            "reason": self.reason,
        }


PINNED_EXISTING_TAP_REFERENCE = ExistingTapReferenceV1()


@dataclass(frozen=True, slots=True)
class PrefixForwardAdapterContractV1:
    """The immutable first-tranche forward-only adapter patch plan."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.prefix-forward-adapter-contract.v1"

    maxtext_revision: str = MAXTEXT_BASE_COMMIT
    batchsplit_contract_sha256: str = PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256
    global_batch_size: int = PINNED_GLOBAL_BATCH_SIZE
    sequence_length: int = PINNED_SEQUENCE_LENGTH
    batch_split_factor: int = PINNED_BATCH_SPLIT_FACTOR
    device_count: int = PINNED_DEVICE_COUNT
    input_batch_axes: tuple[str, ...] = _INPUT_BATCH_AXES
    runtime_path: str = "bf16-pure-jax-batchsplit"
    quantization_enabled: bool = False
    source_identities: tuple[MaxTextSourceIdentityV1, ...] = (
        PINNED_MAXTEXT_SOURCE_IDENTITIES
    )
    inputs: tuple[AdapterInputTensorV1, ...] = PINNED_ADAPTER_INPUTS
    packed_segments: PackedSegmentPolicyV1 = PackedSegmentPolicyV1()
    router: RouterCaptureSemanticsV1 = RouterCaptureSemanticsV1()
    forward_stages: tuple[ForwardStageV1, ...] = PINNED_FORWARD_STAGES
    execution_exclusions: tuple[str, ...] = _EXECUTION_EXCLUSIONS
    patch_obligations: tuple[AdapterPatchObligationV1, ...] = PINNED_PATCH_OBLIGATIONS
    existing_tap_reference: ExistingTapReferenceV1 = PINNED_EXISTING_TAP_REFERENCE
    patch_artifact_available: bool = False
    packed_attention_available: bool = False
    forward_kernel_available: bool = False
    restore_bound: bool = False
    parity_evidence_bound: bool = False
    v6e_qualified: bool = False
    production_sealable: bool = False
    production_admission_enabled: bool = False

    def __post_init__(self) -> None:
        maxtext_revision = _require_string(
            self.maxtext_revision,
            "maxtext_revision",
        )
        batchsplit_contract_sha256 = _require_sha256(
            self.batchsplit_contract_sha256,
            "batchsplit_contract_sha256",
        )
        runtime_path = _require_string(self.runtime_path, "runtime_path")
        if maxtext_revision != MAXTEXT_BASE_COMMIT:
            raise PrefixForwardAdapterContractError("MaxText revision is not pinned")
        if batchsplit_contract_sha256 != PINNED_PREFIX_BATCHSPLIT_CONTRACT.digest:
            raise PrefixForwardAdapterContractError(
                "adapter is not bound to the pinned batch-split contract"
            )
        for field_name, expected in (
            ("global_batch_size", PINNED_GLOBAL_BATCH_SIZE),
            ("sequence_length", PINNED_SEQUENCE_LENGTH),
            ("batch_split_factor", PINNED_BATCH_SPLIT_FACTOR),
            ("device_count", PINNED_DEVICE_COUNT),
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value != expected:
                raise PrefixForwardAdapterContractError(
                    f"{field_name} must be exactly {expected}"
                )
        if self.global_batch_size // self.device_count != 2:
            raise PrefixForwardAdapterContractError(
                "B128 must place exactly two sequences per device"
            )
        if self.global_batch_size // self.device_count // self.batch_split_factor != 1:
            raise PrefixForwardAdapterContractError(
                "factor two must yield one sequence per split microbatch per device"
            )
        if (
            type(self.input_batch_axes) is not tuple
            or any(type(axis) is not str for axis in self.input_batch_axes)
            or self.input_batch_axes != _INPUT_BATCH_AXES
        ):
            raise PrefixForwardAdapterContractError(
                "input batch axes must be data×fsdp×expert"
            )
        if tuple(layout.device_count for layout in PINNED_LAYOUT_LADDER) != (
            PINNED_DEVICE_COUNT,
        ) * len(PINNED_LAYOUT_LADDER):
            raise PrefixForwardAdapterContractError("layout authority changed")
        if runtime_path != "bf16-pure-jax-batchsplit":
            raise PrefixForwardAdapterContractError(
                "runtime path must use the BF16 pure-JAX batch-split machinery"
            )
        if self.quantization_enabled is not False:
            raise PrefixForwardAdapterContractError(
                "quantization is outside the first forward adapter tranche"
            )
        if (
            type(self.source_identities) is not tuple
            or self.source_identities != PINNED_MAXTEXT_SOURCE_IDENTITIES
            or any(
                type(identity) is not MaxTextSourceIdentityV1
                for identity in self.source_identities
            )
        ):
            raise PrefixForwardAdapterContractError("MaxText source identities changed")
        if (
            type(self.inputs) is not tuple
            or self.inputs != PINNED_ADAPTER_INPUTS
            or any(type(item) is not AdapterInputTensorV1 for item in self.inputs)
        ):
            raise PrefixForwardAdapterContractError("packed input ABI changed")
        if type(self.packed_segments) is not PackedSegmentPolicyV1:
            raise PrefixForwardAdapterContractError(
                "packed segment policy type changed"
            )
        if type(self.router) is not RouterCaptureSemanticsV1:
            raise PrefixForwardAdapterContractError("router policy type changed")
        if (
            type(self.forward_stages) is not tuple
            or self.forward_stages != PINNED_FORWARD_STAGES
            or any(type(stage) is not ForwardStageV1 for stage in self.forward_stages)
        ):
            raise PrefixForwardAdapterContractError("forward stages changed")
        if self.forward_stages[-1].layer_stop_exclusive_zero_indexed != (
            PINNED_TARGET_LAYER_ZERO_INDEXED + 1
        ):
            raise PrefixForwardAdapterContractError("target layer boundary changed")
        if (
            type(self.execution_exclusions) is not tuple
            or any(type(item) is not str for item in self.execution_exclusions)
            or self.execution_exclusions != _EXECUTION_EXCLUSIONS
        ):
            raise PrefixForwardAdapterContractError("execution exclusions changed")
        if (
            type(self.patch_obligations) is not tuple
            or self.patch_obligations != PINNED_PATCH_OBLIGATIONS
            or tuple(item.ordinal for item in self.patch_obligations)
            != tuple(range(1, len(self.patch_obligations) + 1))
            or any(
                type(item) is not AdapterPatchObligationV1
                for item in self.patch_obligations
            )
        ):
            raise PrefixForwardAdapterContractError("patch obligations changed")
        if (
            type(self.existing_tap_reference) is not ExistingTapReferenceV1
            or self.existing_tap_reference != PINNED_EXISTING_TAP_REFERENCE
        ):
            raise PrefixForwardAdapterContractError("existing tap reference changed")
        for field_name in (
            "patch_artifact_available",
            "packed_attention_available",
            "forward_kernel_available",
            "restore_bound",
            "parity_evidence_bound",
            "v6e_qualified",
            "production_sealable",
            "production_admission_enabled",
        ):
            if getattr(self, field_name) is not False:
                raise PrefixForwardAdapterContractError(
                    f"{field_name} must remain false in the patch-plan tranche"
                )

    @property
    def device_output_abi(self) -> tuple[dict[str, Any], ...]:
        """Derive, rather than restate, the existing three-output device ABI."""

        outputs = PINNED_PREFIX_BATCHSPLIT_CONTRACT.capture_outputs
        if tuple(output.name for output in outputs) != (
            "routed_activation",
            "global_expert_ids",
            "scaled_router_weights",
        ):
            raise PrefixForwardAdapterContractError(
                "batch-split capture output ABI changed"
            )
        return tuple(
            {
                "name": output.name,
                "shape": list(output.shape),
                "device_dtype": output.device_dtype,
                "semantics": output.semantics,
            }
            for output in outputs
        )

    @property
    def unmet_implementation_gates(self) -> tuple[str, ...]:
        return _UNMET_IMPLEMENTATION_GATES

    def require_production_admission(self) -> NoReturn:
        """Fail unconditionally until a later, evidence-bearing authority exists."""

        raise PrefixForwardAdapterProductionUnavailable(
            "forward adapter production admission is unavailable; unmet gates: "
            + ", ".join(self.unmet_implementation_gates)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "maxtext_revision": self.maxtext_revision,
            "batchsplit_contract_sha256": self.batchsplit_contract_sha256,
            "global_batch_size": self.global_batch_size,
            "sequence_length": self.sequence_length,
            "batch_split_factor": self.batch_split_factor,
            "device_count": self.device_count,
            "input_batch_axes": list(self.input_batch_axes),
            "runtime_path": self.runtime_path,
            "quantization_enabled": self.quantization_enabled,
            "source_identities": [
                identity.as_dict() for identity in self.source_identities
            ],
            "inputs": [item.as_dict() for item in self.inputs],
            "packed_segments": self.packed_segments.as_dict(),
            "router": self.router.as_dict(),
            "forward_stages": [stage.as_dict() for stage in self.forward_stages],
            "execution_exclusions": list(self.execution_exclusions),
            "patch_obligations": [
                obligation.as_dict() for obligation in self.patch_obligations
            ],
            "existing_tap_reference": self.existing_tap_reference.as_dict(),
            "device_output_abi": list(self.device_output_abi),
            "patch_artifact_available": self.patch_artifact_available,
            "packed_attention_available": self.packed_attention_available,
            "forward_kernel_available": self.forward_kernel_available,
            "restore_bound": self.restore_bound,
            "parity_evidence_bound": self.parity_evidence_bound,
            "v6e_qualified": self.v6e_qualified,
            "unmet_implementation_gates": list(self.unmet_implementation_gates),
            "production_sealable": self.production_sealable,
            "production_admission_enabled": self.production_admission_enabled,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT = PrefixForwardAdapterContractV1()
PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT_SHA256 = (
    "d6a5668e2a789cd79f63272bd4c2f052030264ab6eba9a944259d672e46695f7"
)
if (
    PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT.digest
    != PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT_SHA256
):
    raise PrefixForwardAdapterContractError(
        "pinned forward adapter contract digest changed: "
        f"{PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT.digest}"
    )


__all__ = [
    "AdapterInputTensorV1",
    "AdapterPatchObligationV1",
    "ExistingTapReferenceV1",
    "ForwardStageV1",
    "MaxTextSourceIdentityV1",
    "PINNED_ADAPTER_INPUTS",
    "PINNED_EXISTING_TAP_REFERENCE",
    "PINNED_FORWARD_STAGES",
    "PINNED_MAXTEXT_SOURCE_IDENTITIES",
    "PINNED_PATCH_OBLIGATIONS",
    "PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT",
    "PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT_SHA256",
    "PackedSegmentPolicyV1",
    "PrefixForwardAdapterContractError",
    "PrefixForwardAdapterContractV1",
    "PrefixForwardAdapterProductionUnavailable",
    "RouterCaptureSemanticsV1",
]
