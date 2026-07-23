from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path, PurePosixPath

import pytest

from moe_sae.r1.maxtext_tap_patch import (
    MAXTEXT_BASE_COMMIT,
    PATCHED_MOE_SHA256,
    PATCH_SHA256,
)
from moe_sae.r1.prefix_batchsplit_contract import (
    PINNED_PREFIX_BATCHSPLIT_CONTRACT,
    PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256,
)
from moe_sae.r1.prefix_forward_adapter_contract import (
    AdapterInputTensorV1,
    AdapterPatchObligationV1,
    ExistingTapReferenceV1,
    MaxTextSourceIdentityV1,
    PINNED_EXISTING_TAP_REFERENCE,
    PINNED_MAXTEXT_SOURCE_IDENTITIES,
    PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT,
    PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT_SHA256,
    PackedSegmentPolicyV1,
    PrefixForwardAdapterContractError,
    PrefixForwardAdapterContractV1,
    PrefixForwardAdapterProductionUnavailable,
    RouterCaptureSemanticsV1,
)


class _EqualString(str):
    """Serialize attacker bytes while claiming equality with every string."""

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return hash("src/maxtext/models/deepseek_batchsplit.py")


class _EqualTuple(tuple):
    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


def _maxtext_checkout_candidates() -> list[Path]:
    configured = os.environ.get("MAXTEXT_C03_CHECKOUT")
    if configured:
        return [Path(configured).expanduser()]
    root = Path.home() / ".cache" / "uv" / "git-v0" / "checkouts"
    if not root.is_dir():
        return []
    return sorted(root.glob("*/c03f8bd3"))


def test_contract_freezes_b128_factor_two_and_derives_three_output_abi() -> None:
    contract = PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT
    assert contract == PrefixForwardAdapterContractV1()
    assert contract.maxtext_revision == MAXTEXT_BASE_COMMIT
    assert contract.digest == PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT_SHA256
    assert (
        contract.batchsplit_contract_sha256 == PINNED_PREFIX_BATCHSPLIT_CONTRACT_SHA256
    )
    assert (contract.global_batch_size, contract.sequence_length) == (128, 1_024)
    assert (contract.device_count, contract.batch_split_factor) == (64, 2)
    assert contract.global_batch_size // contract.device_count == 2
    assert (
        contract.global_batch_size
        // contract.device_count
        // contract.batch_split_factor
        == 1
    )
    assert contract.runtime_path == "bf16-pure-jax-batchsplit"
    assert contract.quantization_enabled is False
    assert contract.input_batch_axes == ("data", "fsdp", "expert")

    assert contract.device_output_abi == tuple(
        {
            "name": output.name,
            "shape": list(output.shape),
            "device_dtype": output.device_dtype,
            "semantics": output.semantics,
        }
        for output in PINNED_PREFIX_BATCHSPLIT_CONTRACT.capture_outputs
    )
    assert tuple(item["name"] for item in contract.device_output_abi) == (
        "routed_activation",
        "global_expert_ids",
        "scaled_router_weights",
    )
    assert tuple(item["device_dtype"] for item in contract.device_output_abi) == (
        "bfloat16",
        "int32",
        "float32",
    )
    with pytest.raises(FrozenInstanceError):
        contract.global_batch_size = 64  # type: ignore[misc]


def test_packed_input_metadata_is_explicit_and_must_split_with_tokens() -> None:
    contract = PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT
    assert tuple(item.name for item in contract.inputs) == (
        "decoder_input_tokens",
        "decoder_segment_ids",
        "decoder_positions",
    )
    assert all(item.shape == (128, 1_024) for item in contract.inputs)
    assert tuple(item.dtype for item in contract.inputs) == (
        "int32",
        "int32",
        "int32",
    )
    packed = contract.packed_segments
    assert packed.padding_segment_id == 0
    assert packed.split_factor == 2
    assert packed.split_batch_only is True
    assert packed.split_metadata_with_tokens is True
    assert packed.preserve_sequence_axis is True
    assert packed.valid_token_predicate == "segment_id > 0"
    assert packed.attention_predicate == (
        "query_segment_id == key_segment_id != 0 and key_position <= query_position"
    )
    assert packed.current_upstream_threads_segment_ids is False
    assert packed.current_upstream_mask == (
        "plain-causal-mask-without-packed-segment-boundaries"
    )

    with pytest.raises(PrefixForwardAdapterContractError, match="split_factor"):
        replace(packed, split_factor=1)
    with pytest.raises(PrefixForwardAdapterContractError, match="packed input ABI"):
        AdapterInputTensorV1(
            "decoder_segment_ids",
            (128, 1_024),
            "uint16",
            "zero-is-padding-positive-ids-delimit-packed-segments",
        )
    with pytest.raises(PrefixForwardAdapterContractError, match="packed segment"):
        replace(contract, packed_segments=replace(packed, preserve_sequence_axis=False))


def test_router_semantics_keep_fp32_math_and_global_ids_before_remap() -> None:
    router = PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT.router
    assert router.hidden_input_dtype == "bfloat16"
    assert router.gate_kernel_storage_dtype == "bfloat16"
    assert router.correction_bias_storage_dtype == "float32"
    assert router.projection_compute_dtype == "float32"
    assert router.sigmoid_score_dtype == "float32"
    assert router.selection_score == ("fp32-sigmoid-score-plus-fp32-correction-bias")
    assert (
        router.hidden_dim,
        router.expert_count,
        router.routing_group_count,
        router.selected_routing_group_count,
        router.group_score_top_k,
    ) == (7_168, 256, 8, 4, 2)
    assert router.group_selection == "8-groups-top4-by-sum-of-top2"
    assert router.experts_per_token == 8
    assert router.weight_semantics == (
        "selected-unbiased-fp32-sigmoid-normalized-times-2.5"
    )
    assert router.weight_dtype == "float32"
    assert router.global_id_semantics == ("global-id-before-expert-shard-local-remap")
    assert (router.global_id_dtype, router.global_id_minimum) == ("int32", 0)
    assert router.global_id_exclusive_maximum == 256
    assert router.unique_ids_per_token is True
    assert router.expected_weight_sum == 2.5

    with pytest.raises(PrefixForwardAdapterContractError, match="projection_compute"):
        replace(router, projection_compute_dtype="bfloat16")
    with pytest.raises(PrefixForwardAdapterContractError, match="global_id_semantics"):
        replace(
            router,
            global_id_semantics="expert-shard-local-id-after-remap",
        )
    with pytest.raises(PrefixForwardAdapterContractError, match="hidden_dim"):
        replace(router, hidden_dim=7_168.0)  # type: ignore[arg-type]
    with pytest.raises(PrefixForwardAdapterContractError, match="router policy"):
        replace(
            PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT,
            router=object(),  # type: ignore[arg-type]
        )


def test_execution_boundary_omits_only_target_shared_path_and_all_suffix_work() -> None:
    contract = PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT
    assert tuple(stage.name for stage in contract.forward_stages) == (
        "token-embedding",
        "completed-dense-prefix",
        "completed-moe-prefix",
        "target-attention-and-branch-normalization",
        "target-router-and-routed-experts",
    )
    completed_moe = contract.forward_stages[2]
    assert (
        completed_moe.layer_start_zero_indexed,
        completed_moe.layer_stop_exclusive_zero_indexed,
    ) == (3, 30)
    assert "routed-plus-shared-experts" in completed_moe.semantics
    target = contract.forward_stages[-1]
    assert (
        target.layer_start_zero_indexed,
        target.layer_stop_exclusive_zero_indexed,
    ) == (30, 31)
    assert "before-shared-expert-and-residual" in target.semantics
    assert contract.execution_exclusions == (
        "target-shared-expert",
        "target-outer-residual-add",
        "suffix-layers-31-through-60-zero-indexed",
        "final-normalization",
        "lm-head-and-logits",
        "mtp-layer-61-zero-indexed",
        "backward-autodiff-and-custom-vjp",
        "forward-residual-host-offload",
    )
    with pytest.raises(PrefixForwardAdapterContractError, match="exclusions"):
        replace(
            contract,
            execution_exclusions=contract.execution_exclusions[:-1],
        )


def test_patch_plan_records_every_observed_upstream_gap_without_kernel_claim() -> None:
    contract = PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT
    assert tuple(item.ordinal for item in contract.patch_obligations) == tuple(
        range(1, 8)
    )
    by_symbol = {item.symbol: item for item in contract.patch_obligations}
    dispatch = by_symbol["Decoder.__call__ batch-split dispatch"]
    assert "omits-segment-ids" in dispatch.upstream_state
    assert "threads-segment-ids" in dispatch.required_state
    scan = by_symbol["scan_batch_split_layers/process_all_layers"]
    assert "custom-vjp" in scan.upstream_state
    assert "no-custom-vjp-backward-or-host-offload" in scan.required_state
    attention = by_symbol["init_splash_kernel/mla"]
    assert "plain-causal-mask" in attention.upstream_state
    assert "block-diagonal-packed-causal-mask" in attention.required_state
    target = by_symbol["fetch_weights/shared_expert_and_route"]
    assert "executes-shared-expert-before-routing" in target.upstream_state
    assert "target-fetch-and-execution-omit-it" in target.required_state
    router = by_symbol["expert_selection/route"]
    assert "ambient-dot-dtype" in router.upstream_state
    assert "global-ids-are-captured-before-remap" in router.required_state
    output = by_symbol["compute_linear/unroute_ubatch_fn"]
    assert "adds-shared-expert-and-outer-residual" in output.upstream_state
    assert "weighted-routed-aggregate" in output.required_state

    assert contract.existing_tap_reference.patch_sha256 == PATCH_SHA256
    assert contract.existing_tap_reference.patched_source_sha256 == PATCHED_MOE_SHA256
    assert contract.existing_tap_reference.reusable_as_batchsplit_kernel is False
    assert "explicitly-rejects" in contract.existing_tap_reference.reason
    assert contract.existing_tap_reference == PINNED_EXISTING_TAP_REFERENCE
    with pytest.raises(PrefixForwardAdapterContractError, match="tap reference"):
        replace(
            contract,
            existing_tap_reference=ExistingTapReferenceV1(
                router_oracle="forged.router.oracle"
            ),
        )


def test_exact_pinned_maxtext_sources_match_contract_when_checkout_is_present() -> None:
    for checkout in _maxtext_checkout_candidates():
        if not checkout.is_dir():
            continue
        for identity in PINNED_MAXTEXT_SOURCE_IDENTITIES:
            identity.verify_payload(
                checkout.joinpath(
                    *PurePosixPath(identity.relative_path).parts
                ).read_bytes()
            )
        return
    pytest.skip("exact pinned MaxText checkout is absent")


def test_source_identity_verification_and_forgery_fail_closed() -> None:
    identity = PINNED_MAXTEXT_SOURCE_IDENTITIES[0]
    with pytest.raises(PrefixForwardAdapterContractError, match="exact bytes"):
        identity.verify_payload(bytearray(identity.size_bytes))  # type: ignore[arg-type]
    with pytest.raises(PrefixForwardAdapterContractError, match="size differs"):
        identity.verify_payload(b"")
    with pytest.raises(PrefixForwardAdapterContractError, match="repository-relative"):
        replace(identity, relative_path="../nnx_decoders.py")
    with pytest.raises(PrefixForwardAdapterContractError, match="SHA-256"):
        replace(identity, sha256="A" * 64)
    with pytest.raises(PrefixForwardAdapterContractError, match="source identities"):
        replace(
            PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT,
            source_identities=(
                replace(identity, role="forged-role"),
                *PINNED_MAXTEXT_SOURCE_IDENTITIES[1:],
            ),
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "patch_artifact_available",
        "packed_attention_available",
        "forward_kernel_available",
        "restore_bound",
        "parity_evidence_bound",
        "v6e_qualified",
        "production_sealable",
        "production_admission_enabled",
    ),
)
def test_no_boolean_can_forge_implementation_or_production(
    field_name: str,
) -> None:
    with pytest.raises(
        PrefixForwardAdapterContractError,
        match=f"{field_name} must remain false",
    ):
        replace(
            PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT,
            **{field_name: True},
        )


def test_production_admission_is_unconditionally_unavailable() -> None:
    contract = PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT
    assert contract.unmet_implementation_gates == (
        "sealed-batchsplit-adapter-patch-artifact",
        "packed-segment-aware-prefix-attention",
        "forward-only-prefix-kernel",
        "fresh-process-prefix-restore-binding",
        "full-model-r1-logit-parity",
        "prefix-versus-full-routed-aggregate-parity",
        "v6e-64-b128-compile-and-runtime-qualification",
    )
    with pytest.raises(
        PrefixForwardAdapterProductionUnavailable,
        match="production admission is unavailable.*packed-segment-aware",
    ):
        contract.require_production_admission()


def test_contract_serialization_is_stable_and_contains_no_implementation_claim() -> (
    None
):
    contract = PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT
    raw = contract.as_dict()
    assert raw["schema"] == "moe-sae.r1.prefix-forward-adapter-contract.v1"
    assert raw["maxtext_revision"] == MAXTEXT_BASE_COMMIT
    assert raw["device_output_abi"] == list(contract.device_output_abi)
    assert raw["patch_artifact_available"] is False
    assert raw["packed_attention_available"] is False
    assert raw["forward_kernel_available"] is False
    assert raw["restore_bound"] is False
    assert raw["parity_evidence_bound"] is False
    assert raw["v6e_qualified"] is False
    assert raw["production_sealable"] is False
    assert raw["production_admission_enabled"] is False
    assert contract.canonical_bytes() == contract.canonical_bytes()
    assert len(contract.digest) == 64


def test_nested_contract_records_reject_semantic_drift() -> None:
    with pytest.raises(PrefixForwardAdapterContractError, match="router capture"):
        RouterCaptureSemanticsV1(weight_dtype="bfloat16")
    with pytest.raises(PrefixForwardAdapterContractError, match="packed segment"):
        PackedSegmentPolicyV1(current_upstream_threads_segment_ids=True)
    with pytest.raises(PrefixForwardAdapterContractError, match="batch_axis"):
        PackedSegmentPolicyV1(batch_axis=False)  # type: ignore[arg-type]
    with pytest.raises(PrefixForwardAdapterContractError, match="MaxText revision"):
        replace(PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT, maxtext_revision="deadbeef")
    with pytest.raises(PrefixForwardAdapterContractError, match="global_batch_size"):
        replace(PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT, global_batch_size=64)
    with pytest.raises(PrefixForwardAdapterContractError, match="global_batch_size"):
        replace(
            PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT,
            global_batch_size=128.0,  # type: ignore[arg-type]
        )
    with pytest.raises(PrefixForwardAdapterContractError, match="runtime"):
        replace(
            PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT,
            runtime_path="deepseek-batchsplit-fp8",
        )
    with pytest.raises(PrefixForwardAdapterContractError, match="quantization"):
        replace(PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT, quantization_enabled=True)


def test_source_identity_type_is_exact() -> None:
    identity = PINNED_MAXTEXT_SOURCE_IDENTITIES[0]

    class Subclass(MaxTextSourceIdentityV1):
        pass

    forged = Subclass(
        identity.relative_path,
        identity.sha256,
        identity.size_bytes,
        identity.git_blob,
        identity.role,
    )
    with pytest.raises(PrefixForwardAdapterContractError, match="source identities"):
        replace(
            PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT,
            source_identities=(forged, *PINNED_MAXTEXT_SOURCE_IDENTITIES[1:]),
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("maxtext_revision", _EqualString("attacker-revision")),
        ("batchsplit_contract_sha256", _EqualString("0" * 64)),
        ("runtime_path", _EqualString("attacker-runtime")),
        ("input_batch_axes", _EqualTuple(("attacker-axis",))),
        (
            "execution_exclusions",
            (
                _EqualString("attacker-exclusion"),
                *PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT.execution_exclusions[1:],
            ),
        ),
    ),
)
def test_top_level_identity_fields_reject_equality_overriding_subclasses(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(PrefixForwardAdapterContractError):
        replace(
            PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT,
            **{field_name: value},
        )


def test_nested_path_identities_reject_equality_overriding_strings() -> None:
    with pytest.raises(PrefixForwardAdapterContractError, match="target_relative_path"):
        replace(
            PINNED_EXISTING_TAP_REFERENCE,
            target_relative_path=_EqualString("attacker/tap.py"),
        )
    obligation = PINNED_PREFIX_FORWARD_ADAPTER_CONTRACT.patch_obligations[1]
    with pytest.raises(PrefixForwardAdapterContractError, match="source_relative_path"):
        AdapterPatchObligationV1(
            obligation.ordinal,
            _EqualString("attacker/batchsplit.py"),
            obligation.symbol,
            obligation.upstream_state,
            obligation.required_state,
        )
