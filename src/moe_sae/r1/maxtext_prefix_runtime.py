"""Executable MaxText configuration seam for the DeepSeek-R1 prefix.

The numerical adapter lives as a patch against one pinned MaxText revision.
This module supplies the small repository-owned boundary around it:

* one exact B128×1024 forward-only configuration;
* the three 64-chip mesh candidates in benchmark order;
* fail-closed validation of the initialized MaxText configuration; and
* pruning of output-only NNX parameters before the 30 restored model leaves
  are bound; and
* one exact 44-leaf restore-to-compile path, including the 14 unscanned target
  leaves consumed by the patched block-31 adapter.

MaxText, JAX, and Flax are imported only by the two runtime functions.  The
controller and its tests therefore do not initialize a device runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from .prefix_batchsplit_contract import (
    PINNED_BATCH_SPLIT_FACTOR,
    PINNED_DEVICE_COUNT,
    PINNED_GLOBAL_BATCH_SIZE,
    PINNED_LAYOUT_LADDER,
    PINNED_SEQUENCE_LENGTH,
    PINNED_TARGET_LAYER_ZERO_INDEXED,
    build_pinned_prefix_performance_tree_plan,
)
from .jax_prefix_restore import (
    LeafSharding,
    RestoredPrefixParameters,
    restore_prefix_parameters,
)
from .prefix_restore_authority import PrefixDirectRestoreAuthorityV1


class MaxTextPrefixRuntimeError(RuntimeError):
    """The executable MaxText prefix boundary differs from its fixed profile."""


class PrefixInputBatch(NamedTuple):
    """Resident packed inputs for one static B128×1024 invocation."""

    decoder_input_tokens: Any
    decoder_positions: Any
    decoder_segment_ids: Any


@dataclass(frozen=True, slots=True)
class CompiledPrefix:
    """Compiled executable with resident parameter buffers kept as arguments."""

    executable: Any
    model: Any
    target_parameters: Any

    def __call__(self, batch: PrefixInputBatch) -> Any:
        if type(batch) is not PrefixInputBatch:
            raise MaxTextPrefixRuntimeError("prefix input has the wrong type")
        return self.executable(
            self.model,
            self.target_parameters,
            batch.decoder_input_tokens,
            batch.decoder_positions,
            batch.decoder_segment_ids,
        )


@dataclass(frozen=True, slots=True)
class PreparedPrefix:
    """One restored, bound, compiled prefix and its immutable restore identity."""

    compiled: CompiledPrefix
    restored: RestoredPrefixParameters

    def __post_init__(self) -> None:
        if (
            type(self.compiled) is not CompiledPrefix
            or type(self.restored) is not RestoredPrefixParameters
        ):
            raise MaxTextPrefixRuntimeError(
                "prepared prefix must contain exact compiled and restored state"
            )


@dataclass(frozen=True, slots=True)
class PrefixMeshCandidate:
    """One single-slice ICI mesh in the required benchmark order."""

    name: str
    fsdp: int
    expert: int

    def __post_init__(self) -> None:
        if self.name != f"EP{self.expert}/FSDP{self.fsdp}":
            raise MaxTextPrefixRuntimeError("mesh candidate name is not canonical")
        if self.fsdp * self.expert != PINNED_DEVICE_COUNT:
            raise MaxTextPrefixRuntimeError(
                "mesh candidate must consume exactly 64 devices"
            )


PREFIX_MESH_CANDIDATES = tuple(
    PrefixMeshCandidate(
        name=layout.name,
        fsdp=layout.fsdp_parallelism,
        expert=layout.expert_parallelism,
    )
    for layout in PINNED_LAYOUT_LADDER
)


@dataclass(frozen=True, slots=True)
class PrefixRuntimeSettings:
    """Inputs that may vary without changing the scientific program."""

    maxtext_base_yml: Path
    output_directory: Path
    mesh: PrefixMeshCandidate = PREFIX_MESH_CANDIDATES[0]
    run_name: str = "deepseek-r1-prefix-benchmark"

    def __post_init__(self) -> None:
        if not isinstance(self.maxtext_base_yml, Path):
            raise MaxTextPrefixRuntimeError("maxtext_base_yml must be a Path")
        if not isinstance(self.output_directory, Path):
            raise MaxTextPrefixRuntimeError("output_directory must be a Path")
        if self.mesh not in PREFIX_MESH_CANDIDATES:
            raise MaxTextPrefixRuntimeError("mesh is not in the fixed benchmark ladder")
        if (
            type(self.run_name) is not str
            or not self.run_name
            or any(character.isspace() for character in self.run_name)
        ):
            raise MaxTextPrefixRuntimeError(
                "run_name must be a non-empty token without whitespace"
            )

    def maxtext_overrides(self) -> dict[str, Any]:
        """Return the exact keyword overrides passed to ``pyconfig.initialize``."""

        return {
            "model_name": "deepseek3-671b-batchsplit",
            "override_model_config": True,
            "base_num_decoder_layers": PINNED_TARGET_LAYER_ZERO_INDEXED,
            "max_target_length": PINNED_SEQUENCE_LENGTH,
            "max_prefill_predict_length": PINNED_SEQUENCE_LENGTH,
            "per_device_batch_size": (
                PINNED_GLOBAL_BATCH_SIZE / PINNED_DEVICE_COUNT
            ),
            "batch_split_factor": PINNED_BATCH_SPLIT_FACTOR,
            "dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "enable_dropout": False,
            "dropout_rate": 0.0,
            "load_balance_loss_weight": 0.0,
            "routed_bias_update_rate": 0.0,
            "scan_layers": True,
            "pure_nnx": True,
            "pure_nnx_decoder": True,
            "logits_via_embedding": True,
            "use_batch_split_schedule": True,
            "pipeline_parallel_layers": 0,
            "use_multimodal": False,
            "use_audio": False,
            "engram_layers": [],
            "quantization": "",
            "use_qwix_quantization": False,
            "enable_checkpointing": False,
            "async_checkpointing": False,
            "enable_tensorboard": False,
            "enable_gcp_goodput_metrics": False,
            "enable_gcp_step_deviation_metrics": False,
            "use_pathways": False,
            "use_pathways_reshard": False,
            "dataset_type": "synthetic",
            "hardware": "tpu",
            "skip_jax_distributed_system": False,
            "ici_data_parallelism": 1,
            "ici_fsdp_parallelism": self.mesh.fsdp,
            "ici_expert_parallelism": self.mesh.expert,
            "ici_context_parallelism": 1,
            "ici_tensor_parallelism": 1,
            "ici_pipeline_parallelism": 1,
            "dcn_data_parallelism": 1,
            "dcn_fsdp_parallelism": 1,
            "dcn_expert_parallelism": 1,
            "dcn_context_parallelism": 1,
            "dcn_tensor_parallelism": 1,
            "dcn_pipeline_parallelism": 1,
            "run_name": self.run_name,
            "base_output_directory": str(self.output_directory),
        }


def initialize_maxtext_config(settings: PrefixRuntimeSettings) -> Any:
    """Initialize and validate the pinned MaxText runtime configuration."""

    if type(settings) is not PrefixRuntimeSettings:
        raise MaxTextPrefixRuntimeError("settings must be PrefixRuntimeSettings")
    if not settings.maxtext_base_yml.is_file():
        raise MaxTextPrefixRuntimeError("pinned MaxText base.yml is missing")
    try:
        from maxtext.configs import pyconfig
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise MaxTextPrefixRuntimeError("pinned MaxText is unavailable") from error
    config = pyconfig.initialize(
        ["moe-sae-r1-prefix", str(settings.maxtext_base_yml)],
        **settings.maxtext_overrides(),
    )
    validate_initialized_config(config, settings.mesh)
    return config


def assert_real_v6e64_runtime() -> None:
    """Reject emulation, a partial slice, or another TPU generation."""

    try:
        import jax
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise MaxTextPrefixRuntimeError("JAX is unavailable") from error
    devices = tuple(jax.devices())
    if (
        len(devices) != PINNED_DEVICE_COUNT
        or jax.process_count() != 16
        or jax.local_device_count() != 4
        or any(getattr(device, "platform", None) != "tpu" for device in devices)
        or any("v6" not in str(getattr(device, "device_kind", "")).lower() for device in devices)
    ):
        raise MaxTextPrefixRuntimeError(
            "qualification requires one real v6e-64 slice: 64 TPU devices, "
            "16 processes, and four local devices per process"
        )


def create_prefix_mesh(config: Any) -> Any:
    """Create the one 64-device MaxText mesh selected by the config."""

    try:
        import jax
        from maxtext.utils import maxtext_utils
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise MaxTextPrefixRuntimeError(
            "pinned JAX/MaxText runtime is unavailable"
        ) from error
    if jax.device_count() != PINNED_DEVICE_COUNT:
        raise MaxTextPrefixRuntimeError("prefix mesh requires exactly 64 devices")
    axes = tuple(getattr(config, "mesh_axes", ()))
    if axes != ("data", "fsdp", "expert", "context"):
        raise MaxTextPrefixRuntimeError(
            "DeepSeek batch-split mesh axes differ from the pinned four-axis order"
        )
    mesh = jax.sharding.Mesh(
        maxtext_utils.create_device_mesh(config),
        axes,
        axis_types=(jax.sharding.AxisType.Explicit,) * len(axes),
    )
    expected = {
        "data": 1,
        "fsdp": int(getattr(config, "ici_fsdp_parallelism", 0)),
        "expert": int(getattr(config, "ici_expert_parallelism", 0)),
        "context": 1,
    }
    if dict(mesh.shape) != expected:
        raise MaxTextPrefixRuntimeError(
            f"created mesh shape {dict(mesh.shape)!r} differs from {expected!r}"
        )
    return mesh


def create_synthetic_prefix_batch(mesh: Any) -> PrefixInputBatch:
    """Create full-valid packed inputs without replicating a global host array."""

    try:
        import jax
        import numpy as np
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise MaxTextPrefixRuntimeError("JAX/NumPy is unavailable") from error
    process_count = jax.process_count()
    if (
        jax.device_count() != PINNED_DEVICE_COUNT
        or type(process_count) is not int
        or process_count <= 0
        or PINNED_GLOBAL_BATCH_SIZE % process_count
    ):
        raise MaxTextPrefixRuntimeError(
            "global B128 cannot be partitioned over the active JAX processes"
        )
    local_batch = PINNED_GLOBAL_BATCH_SIZE // process_count
    local_shape = (local_batch, PINNED_SEQUENCE_LENGTH)
    global_shape = (PINNED_GLOBAL_BATCH_SIZE, PINNED_SEQUENCE_LENGTH)
    sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(("data", "fsdp", "expert"), None),
    )

    def global_array(local: Any) -> Any:
        return jax.make_array_from_process_local_data(
            sharding,
            local,
            global_shape=global_shape,
        )

    row = np.arange(local_batch, dtype=np.uint64)[:, None]
    column = np.arange(PINNED_SEQUENCE_LENGTH, dtype=np.uint64)[None, :]
    # Diverse, deterministic vocabulary rows are required for a balanced
    # synthetic router load.  Repeating this local pattern across processes is
    # harmless and avoids a control-plane broadcast.
    tokens = (
        1
        + ((row * PINNED_SEQUENCE_LENGTH + column) * 2_654_435_761)
        % (129_280 - 1)
    ).astype(np.int32)
    positions = np.broadcast_to(
        np.arange(PINNED_SEQUENCE_LENGTH, dtype=np.int32),
        local_shape,
    )
    # A positive, constant segment ID means every static position is useful
    # and belongs to one causal sequence.
    segments = np.ones(local_shape, dtype=np.int32)
    return PrefixInputBatch(
        global_array(tokens),
        global_array(positions),
        global_array(segments),
    )


def synchronize_prefix_output(output: Any) -> None:
    """Block on every leaf returned by the compiled prefix."""

    try:
        import jax
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise MaxTextPrefixRuntimeError("JAX is unavailable") from error
    jax.block_until_ready(output)


def _dtype_name(value: Any) -> str:
    try:
        return value.name
    except AttributeError:
        return str(value)


def validate_initialized_config(config: Any, mesh: PrefixMeshCandidate) -> None:
    """Reject a MaxText configuration that could execute a different program."""

    if mesh not in PREFIX_MESH_CANDIDATES:
        raise MaxTextPrefixRuntimeError("mesh is not in the benchmark ladder")
    expected = {
        "num_decoder_layers": PINNED_TARGET_LAYER_ZERO_INDEXED,
        "first_num_dense_layers": 3,
        "max_target_length": PINNED_SEQUENCE_LENGTH,
        "batch_split_factor": PINNED_BATCH_SPLIT_FACTOR,
        "scan_layers": True,
        "pure_nnx_decoder": True,
        "use_batch_split_schedule": True,
        "enable_dropout": False,
        "dropout_rate": 0.0,
        "load_balance_loss_weight": 0.0,
        "routed_bias_update_rate": 0.0,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "n_routing_groups": 8,
        "topk_routing_group": 4,
        "shared_experts": 1,
        "routed_scaling_factor": 2.5,
        "routed_score_func": "sigmoid",
        "routed_bias": True,
        "use_pathways": False,
        "using_pipeline_parallelism": False,
        "logits_via_embedding": True,
    }
    for name, wanted in expected.items():
        actual = getattr(config, name, None)
        if actual != wanted:
            raise MaxTextPrefixRuntimeError(
                f"initialized MaxText config requires {name}={wanted!r}; "
                f"received {actual!r}"
            )
    if _dtype_name(getattr(config, "dtype", None)) != "bfloat16":
        raise MaxTextPrefixRuntimeError("activation dtype must be bfloat16")
    if _dtype_name(getattr(config, "weight_dtype", None)) != "bfloat16":
        raise MaxTextPrefixRuntimeError("weight dtype must be bfloat16")
    if getattr(config, "quantization", None):
        raise MaxTextPrefixRuntimeError("prefix runtime must not enable quantization")
    if getattr(config, "use_qwix_quantization", False):
        raise MaxTextPrefixRuntimeError("prefix runtime must not enable Qwix")
    expected_parallelism = {
        "ici_data_parallelism": 1,
        "ici_fsdp_parallelism": mesh.fsdp,
        "ici_expert_parallelism": mesh.expert,
        "ici_context_parallelism": 1,
        "ici_tensor_parallelism": 1,
        "ici_pipeline_parallelism": 1,
    }
    for name, wanted in expected_parallelism.items():
        if getattr(config, name, None) != wanted:
            raise MaxTextPrefixRuntimeError(
                f"initialized MaxText config requires {name}={wanted}"
            )
    global_batch = getattr(config, "global_batch_size_to_load", None)
    if global_batch != PINNED_GLOBAL_BATCH_SIZE:
        raise MaxTextPrefixRuntimeError(
            f"global batch must be {PINNED_GLOBAL_BATCH_SIZE}; received "
            f"{global_batch!r}"
        )


def prune_output_only_parameters(model: Any) -> Any:
    """Remove exactly the two suffix/output parameter owners from an NNX model.

    ``logits_via_embedding=True`` means ``logits_dense`` should not exist, but
    it is rejected if present rather than silently carrying a large unused
    matrix.  The final decoder norm always exists upstream and is not part of
    the block-31 prefix.
    """

    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise MaxTextPrefixRuntimeError("MaxText model has no decoder")
    if hasattr(decoder, "logits_dense"):
        raise MaxTextPrefixRuntimeError(
            "MaxText model unexpectedly owns a logits projection"
        )
    if not hasattr(decoder, "decoder_norm"):
        raise MaxTextPrefixRuntimeError("MaxText model has no final decoder norm")
    del decoder.decoder_norm
    return model


def assert_pruned_parameter_count(model: Any) -> None:
    """Prove the pruned NNX graph owns embedding + 29 scanned leaves only."""

    try:
        from flax import nnx
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise MaxTextPrefixRuntimeError("Flax NNX is unavailable") from error
    _, parameters, _ = nnx.split(model, nnx.Param, ...)
    leaf_count = len(tuple(parameters.flat_state()))
    if leaf_count != 30:
        raise MaxTextPrefixRuntimeError(
            f"pruned MaxText prefix owns {leaf_count} Param leaves; expected 30"
        )


def _insert_nested(root: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = root
    for part in path[:-1]:
        existing = current.get(part)
        if existing is None:
            existing = {}
            current[part] = existing
        if type(existing) is not dict:
            raise MaxTextPrefixRuntimeError(
                "restored parameter paths collide while building a tree"
            )
        current = existing
    if path[-1] in current:
        raise MaxTextPrefixRuntimeError("restored parameter path is duplicated")
    current[path[-1]] = value


def split_restored_prefix_leaves(
    restored_by_leaf_key: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map the exact 44 restore leaves into the MaxText model/target trees.

    The first result is keyed by slash-separated NNX Param paths and contains
    embedding + scanned dense + scanned complete-MoE arrays (30 leaves).  The
    second is the nested 14-leaf Linen-style target tree consumed by the
    patched routed-only target.
    """

    if not isinstance(restored_by_leaf_key, Mapping):
        raise MaxTextPrefixRuntimeError("restored leaves must be a mapping")
    plan = build_pinned_prefix_performance_tree_plan()
    expected = {leaf.leaf_key: leaf for leaf in plan.leaves}
    if set(restored_by_leaf_key) != set(expected):
        missing = sorted(set(expected).difference(restored_by_leaf_key))
        extra = sorted(set(restored_by_leaf_key).difference(expected))
        raise MaxTextPrefixRuntimeError(
            f"restore must provide the exact 44 leaves; missing={missing!r} "
            f"extra={extra!r}"
        )

    model: dict[str, Any] = {}
    target: dict[str, Any] = {}
    for leaf in plan.leaves:
        value = restored_by_leaf_key[leaf.leaf_key]
        if tuple(getattr(value, "shape", ())) != leaf.logical_shape:
            raise MaxTextPrefixRuntimeError(
                f"{leaf.leaf_key} shape differs from the restore plan"
            )
        dtype = _dtype_name(getattr(value, "dtype", None))
        if dtype != leaf.dtype:
            raise MaxTextPrefixRuntimeError(
                f"{leaf.leaf_key} dtype {dtype!r} differs from {leaf.dtype!r}"
            )
        if leaf.section == "target-routed-only":
            path = tuple(
                leaf.leaf_key.removeprefix("target-routed-only-").split("-")
            )
            _insert_nested(target, path, value)
        else:
            path = leaf.leaf_key.removeprefix("params-").replace("-", "/")
            if path in model:
                raise MaxTextPrefixRuntimeError("model parameter path is duplicated")
            model[path] = value
    if len(model) != 30 or len(tuple(_walk_nested_leaves(target))) != 14:
        raise MaxTextPrefixRuntimeError(
            "restore did not produce the exact 30-model/14-target split"
        )
    return model, target


def _walk_nested_leaves(value: Mapping[str, Any]):
    for key in sorted(value):
        item = value[key]
        if isinstance(item, Mapping):
            yield from _walk_nested_leaves(item)
        else:
            yield item


def bind_pruned_model_parameters(
    abstract_model: Any,
    restored_model_by_path: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    """Bind the 30 restored arrays without materializing initialized weights.

    Returns ``(graphdef, bound_param_state, abstract_nonparam_state)``.  The
    caller must materialize the small RNG-only non-parameter state before
    compiling; large parameter initializers are never executed.
    """

    if not isinstance(restored_model_by_path, Mapping):
        raise MaxTextPrefixRuntimeError("restored model leaves must be a mapping")
    try:
        from flax import nnx
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise MaxTextPrefixRuntimeError("Flax NNX is unavailable") from error
    graphdef, parameters, nonparameters = nnx.split(
        abstract_model, nnx.Param, ...
    )
    template = {
        "/".join(str(part) for part in path): variable
        for path, variable in parameters.flat_state()
    }
    if set(template) != set(restored_model_by_path) or len(template) != 30:
        missing = sorted(set(template).difference(restored_model_by_path))
        extra = sorted(set(restored_model_by_path).difference(template))
        raise MaxTextPrefixRuntimeError(
            f"restored model tree differs from the pruned NNX graph; "
            f"missing={missing!r} extra={extra!r}"
        )

    def replace(path: tuple[Any, ...], variable: Any) -> Any:
        key = "/".join(str(part) for part in path)
        value = restored_model_by_path[key]
        template_value = variable.get_value()
        if tuple(value.shape) != tuple(template_value.shape):
            raise MaxTextPrefixRuntimeError(f"restored shape differs at {key}")
        wanted = _dtype_name(value.dtype)
        template_dtype = _dtype_name(template_value.dtype)
        if wanted != template_dtype:
            raise MaxTextPrefixRuntimeError(f"restored dtype differs at {key}")
        return variable.replace(value=value)

    return graphdef, parameters.map(replace), nonparameters


def create_abstract_pruned_model(config: Any, mesh: Any) -> Any:
    """Build graph structure and shapes without running 650 GB initializers."""

    try:
        import jax
        from flax import nnx
        from maxtext.common.common_types import MODEL_MODE_TRAIN
        from maxtext.utils import model_creation_utils
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise MaxTextPrefixRuntimeError(
            "pinned JAX/Flax/MaxText runtime is unavailable"
        ) from error
    create_model = model_creation_utils.get_nnx_create_model_fn(
        config,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
        rng_key=jax.random.key(0),
    )
    model = prune_output_only_parameters(nnx.eval_shape(create_model))
    assert_pruned_parameter_count(model)
    if not hasattr(model.decoder, "r1_forward_routed_prefix"):
        raise MaxTextPrefixRuntimeError(
            "MaxText does not contain the pinned routed-prefix adapter patch"
        )
    return model


_DENSE_PARTITION_SPECS = {
    "mlp-wi_0-kernel": ("fsdp", None, None),
    "mlp-wi_1-kernel": ("fsdp", None, None),
    "mlp-wo-kernel": (None, None, "fsdp"),
    "post_self_attention_layer_norm-scale": (None, None),
    "pre_self_attention_layer_norm-scale": (None, None),
    "self_attention-kv_norm-scale": (None, None),
    "self_attention-out-kernel": (None, None, None, "fsdp"),
    "self_attention-q_norm-scale": (None, None),
    "self_attention-wkv_a-kernel": ("fsdp", None, None),
    "self_attention-wkv_b-kernel": ("fsdp", None, None, None),
    "self_attention-wq_a-kernel": ("fsdp", None, None),
    "self_attention-wq_b-kernel": ("fsdp", None, None, None),
}
_MOE_PARTITION_SPECS = {
    "DeepSeekMoeBlock_0-MoeBlock_0-gate-bias": (None, None),
    "DeepSeekMoeBlock_0-MoeBlock_0-gate-kernel": ("fsdp", None, None),
    "DeepSeekMoeBlock_0-MoeBlock_0-wi_0": ("expert", None, "fsdp", None),
    "DeepSeekMoeBlock_0-MoeBlock_0-wi_1": ("expert", None, "fsdp", None),
    "DeepSeekMoeBlock_0-MoeBlock_0-wo": ("expert", None, None, "fsdp"),
    "DeepSeekMoeBlock_0-shared_experts-wi_0-kernel": ("fsdp", None, None),
    "DeepSeekMoeBlock_0-shared_experts-wi_1-kernel": ("fsdp", None, None),
    "DeepSeekMoeBlock_0-shared_experts-wo-kernel": (None, None, "fsdp"),
    "post_self_attention_layer_norm-scale": (None, None),
    "pre_self_attention_layer_norm-scale": (None, None),
    "self_attention-kv_norm-scale": (None, None),
    "self_attention-out-kernel": (None, None, None, "fsdp"),
    "self_attention-q_norm-scale": (None, None),
    "self_attention-wkv_a-kernel": ("fsdp", None, None),
    "self_attention-wkv_b-kernel": ("fsdp", None, None, None),
    "self_attention-wq_a-kernel": ("fsdp", None, None),
    "self_attention-wq_b-kernel": ("fsdp", None, None, None),
}


def _expected_partition_specs() -> dict[str, tuple[Any, ...]]:
    """Return the adapter's exact physical storage specs by 44-leaf key."""

    result: dict[str, tuple[Any, ...]] = {}
    plan = build_pinned_prefix_performance_tree_plan()
    completed_by_family = {
        leaf.family: leaf
        for leaf in plan.leaves
        if leaf.section == "scanned-completed-moe"
    }
    for leaf in plan.leaves:
        if leaf.section == "embedding":
            spec = (None, "fsdp")
        elif leaf.section == "scanned-dense":
            spec = _DENSE_PARTITION_SPECS.get(leaf.family)
        elif leaf.section == "scanned-completed-moe":
            spec = _MOE_PARTITION_SPECS.get(leaf.family)
        elif leaf.section == "target-routed-only":
            completed = completed_by_family.get(leaf.family)
            if completed is None or completed.scan_axis is None:
                raise MaxTextPrefixRuntimeError(
                    "target leaf has no completed-layer sharding analogue"
                )
            completed_spec = _MOE_PARTITION_SPECS.get(leaf.family)
            if completed_spec is None:
                raise MaxTextPrefixRuntimeError(
                    "target sharding family is not in the pinned adapter"
                )
            spec = (
                *completed_spec[: completed.scan_axis],
                *completed_spec[completed.scan_axis + 1 :],
            )
        else:  # pragma: no cover - frozen plan invariant
            raise MaxTextPrefixRuntimeError("unknown prefix restore section")
        if spec is None or len(spec) != len(leaf.logical_shape):
            raise MaxTextPrefixRuntimeError(
                f"physical partition spec is absent or rank-invalid for {leaf.leaf_key}"
            )
        result[leaf.leaf_key] = spec
    if len(result) != 44:
        raise MaxTextPrefixRuntimeError(
            "physical partition table must cover exactly 44 leaves"
        )
    return result


def create_prefix_restore_shardings(
    abstract_model: Any,
    config: Any,
    mesh: Any,
) -> tuple[LeafSharding, ...]:
    """Bind the exact adapter specs to a mesh and cross-check MaxText metadata.

    MaxText's abstract NNX parameters retain logical axis metadata but do not
    themselves carry a concrete ``NamedSharding``.  The first 30 specs are
    independently translated through MaxText's pinned logical-axis rules and
    compared with the adapter table.  Target specs remove only the completed
    scan axis, matching the patched target gather exactly.
    """

    try:
        import jax
        from flax import linen as nn
        from flax import nnx
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise MaxTextPrefixRuntimeError("JAX/Flax is unavailable") from error
    if not isinstance(mesh, jax.sharding.Mesh):
        raise MaxTextPrefixRuntimeError("restore shardings require a JAX mesh")
    _, parameters, _ = nnx.split(abstract_model, nnx.Param, ...)
    logical = nnx.get_partition_spec(parameters)
    physical = nn.logical_to_mesh_sharding(
        logical,
        mesh,
        getattr(config, "logical_axis_rules", ()),
    )
    physical_by_path: dict[str, Any] = {}
    for path, variable in physical.flat_state():
        key = "/".join(str(part) for part in path)
        if key in physical_by_path:
            raise MaxTextPrefixRuntimeError("MaxText sharding path is duplicated")
        physical_by_path[key] = variable.get_value()

    plan = build_pinned_prefix_performance_tree_plan()
    expected_specs = _expected_partition_specs()
    expected_model_paths = {
        leaf.leaf_key.removeprefix("params-").replace("-", "/")
        for leaf in plan.leaves
        if leaf.section != "target-routed-only"
    }
    if set(physical_by_path) != expected_model_paths or len(physical_by_path) != 30:
        raise MaxTextPrefixRuntimeError(
            "MaxText physical sharding tree differs from the exact 30 model leaves"
        )
    for path, sharding in physical_by_path.items():
        if (
            not isinstance(sharding, jax.sharding.NamedSharding)
            or sharding.mesh != mesh
        ):
            raise MaxTextPrefixRuntimeError(
                f"MaxText did not derive a mesh-bound NamedSharding at {path}"
            )
        leaf_key = "params-" + path.replace("/", "-")
        if tuple(sharding.spec) != expected_specs[leaf_key]:
            raise MaxTextPrefixRuntimeError(
                f"MaxText physical sharding differs from the adapter at {path}"
            )

    bindings: list[LeafSharding] = []
    for leaf in plan.leaves:
        sharding = jax.sharding.NamedSharding(
            mesh,
            jax.sharding.PartitionSpec(*expected_specs[leaf.leaf_key]),
        )
        try:
            shard_shape = sharding.shard_shape(leaf.logical_shape)
        except ValueError as error:
            raise MaxTextPrefixRuntimeError(
                f"{leaf.leaf_key} cannot be sharded on the selected mesh"
            ) from error
        if (
            len(shard_shape) != len(leaf.logical_shape)
            or any(
                logical % local
                for logical, local in zip(
                    leaf.logical_shape, shard_shape, strict=True
                )
            )
        ):
            raise MaxTextPrefixRuntimeError(
                f"{leaf.leaf_key} has a non-integral local shard"
            )
        bindings.append(LeafSharding(leaf.leaf_key, sharding))
    return tuple(bindings)


def create_device_synthetic_prefix_leaves(
    shardings: tuple[LeafSharding, ...],
    *,
    seed: int = 0,
) -> Mapping[str, Any]:
    """Allocate a balanced benchmark-only 44-leaf tree directly on devices.

    Giant expert matrices are runtime arguments filled with zero on-device;
    they therefore cannot be constant-folded out of the later prefix compile.
    Embeddings and router kernels are deterministic random values so routing
    exercises all experts instead of overflowing eight tie-selected experts.
    No host ever materializes a logical parameter tensor.
    """

    if (
        type(shardings) is not tuple
        or len(shardings) != 44
        or any(type(item) is not LeafSharding for item in shardings)
    ):
        raise MaxTextPrefixRuntimeError(
            "synthetic allocation requires the exact 44 leaf shardings"
        )
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise MaxTextPrefixRuntimeError("synthetic seed must be in [0, 2^31)")
    try:
        import jax
        import jax.numpy as jnp
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise MaxTextPrefixRuntimeError("JAX is unavailable") from error
    plan = build_pinned_prefix_performance_tree_plan()
    by_key = {item.leaf_key: item.sharding for item in shardings}
    expected = tuple(leaf.leaf_key for leaf in plan.leaves)
    if set(by_key) != set(expected):
        raise MaxTextPrefixRuntimeError(
            "synthetic shardings differ from the exact 44-leaf plan"
        )
    dtypes = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}
    output_shardings = {
        leaf.leaf_key: by_key[leaf.leaf_key] for leaf in plan.leaves
    }

    def allocate(key):
        arrays = {}
        for index, leaf in enumerate(plan.leaves):
            dtype = dtypes[leaf.dtype]
            if leaf.section == "embedding" or leaf.family.endswith("gate-kernel"):
                value = jax.random.normal(
                    jax.random.fold_in(key, index),
                    leaf.logical_shape,
                    dtype=dtype,
                )
                value = value * jnp.asarray(0.02, dtype=dtype)
            else:
                value = jnp.zeros(leaf.logical_shape, dtype=dtype)
            arrays[leaf.leaf_key] = value
        return arrays

    arrays = jax.jit(allocate, out_shardings=output_shardings)(
        jax.random.key(seed)
    )
    jax.block_until_ready(arrays)
    if tuple(arrays) != expected:
        raise MaxTextPrefixRuntimeError(
            "synthetic allocator changed canonical leaf order"
        )
    return arrays


def materialize_rng_state(nonparameters: Any) -> Any:
    """Materialize only the tiny abstract RNG state in a pruned NNX graph."""

    try:
        import jax
        import jax.numpy as jnp
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise MaxTextPrefixRuntimeError("JAX is unavailable") from error
    flat_state = getattr(nonparameters, "flat_state", None)
    map_state = getattr(nonparameters, "map", None)
    if not callable(flat_state) or not callable(map_state):
        raise MaxTextPrefixRuntimeError("non-parameter state is not an NNX State")
    entries = tuple(flat_state())
    if len(entries) != 48:
        raise MaxTextPrefixRuntimeError(
            f"pruned graph owns {len(entries)} non-parameter leaves; expected 48"
        )

    def materialize(path: tuple[Any, ...], variable: Any) -> Any:
        value = variable.get_value()
        shape = tuple(value.shape)
        dtype = value.dtype
        if jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            concrete = jnp.broadcast_to(jax.random.key(0), shape)
        else:
            concrete = jnp.zeros(shape, dtype=dtype)
        sharding = getattr(value, "sharding", None)
        if sharding is not None:
            concrete = jax.device_put(concrete, sharding)
        return variable.replace(value=concrete)

    return nonparameters.map(materialize)


def compile_prefix(
    graphdef: Any,
    parameters: Any,
    nonparameters: Any,
    target_parameters: Mapping[str, Any],
    example_batch: PrefixInputBatch,
) -> CompiledPrefix:
    """Lower and compile the patched MaxText prefix exactly once.

    Parameters remain dynamic executable arguments.  Capturing them as Python
    constants would embed hundreds of gigabytes into compilation artifacts.
    """

    if type(example_batch) is not PrefixInputBatch:
        raise MaxTextPrefixRuntimeError("example_batch has the wrong type")
    if not isinstance(target_parameters, Mapping):
        raise MaxTextPrefixRuntimeError("target parameters must be a mapping")
    try:
        from flax import nnx
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover - TPU image
        raise MaxTextPrefixRuntimeError("Flax NNX is unavailable") from error
    concrete_nonparameters = materialize_rng_state(nonparameters)
    model = nnx.merge(graphdef, parameters, concrete_nonparameters)

    def forward(
        runtime_model,
        target,
        decoder_input_tokens,
        decoder_positions,
        decoder_segment_ids,
    ):
        return runtime_model.decoder.r1_forward_routed_prefix(
            runtime_model.token_embedder,
            decoder_input_tokens,
            decoder_positions,
            decoder_segment_ids,
            target,
        )

    lowered = nnx.jit(forward).lower(
        model,
        target_parameters,
        example_batch.decoder_input_tokens,
        example_batch.decoder_positions,
        example_batch.decoder_segment_ids,
    )
    executable = lowered.compile()
    return CompiledPrefix(
        executable=executable,
        model=model,
        target_parameters=target_parameters,
    )


def restore_bind_compile_prefix(
    authority: PrefixDirectRestoreAuthorityV1,
    config: Any,
    mesh: Any,
    example_batch: PrefixInputBatch,
) -> PreparedPrefix:
    """Execute the complete 44-leaf restore, bind, and compile seam."""

    if type(authority) is not PrefixDirectRestoreAuthorityV1:
        raise MaxTextPrefixRuntimeError(
            "prefix preparation requires an exact opened restore authority"
        )
    if type(example_batch) is not PrefixInputBatch:
        raise MaxTextPrefixRuntimeError("example_batch has the wrong type")
    abstract_model = create_abstract_pruned_model(config, mesh)
    shardings = create_prefix_restore_shardings(abstract_model, config, mesh)
    restored = restore_prefix_parameters(authority, shardings)
    graphdef, parameters, nonparameters = bind_pruned_model_parameters(
        abstract_model,
        restored.model_by_path,
    )
    compiled = compile_prefix(
        graphdef,
        parameters,
        nonparameters,
        restored.target_params,
        example_batch,
    )
    return PreparedPrefix(compiled=compiled, restored=restored)


def create_compile_synthetic_prefix(
    config: Any,
    mesh: Any,
    example_batch: PrefixInputBatch,
    *,
    seed: int = 0,
) -> CompiledPrefix:
    """Create device-resident benchmark weights and compile the full prefix."""

    if type(example_batch) is not PrefixInputBatch:
        raise MaxTextPrefixRuntimeError("example_batch has the wrong type")
    abstract_model = create_abstract_pruned_model(config, mesh)
    shardings = create_prefix_restore_shardings(abstract_model, config, mesh)
    leaves = create_device_synthetic_prefix_leaves(shardings, seed=seed)
    model_by_path, target = split_restored_prefix_leaves(leaves)
    graphdef, parameters, nonparameters = bind_pruned_model_parameters(
        abstract_model,
        model_by_path,
    )
    return compile_prefix(
        graphdef,
        parameters,
        nonparameters,
        target,
        example_batch,
    )


__all__ = [
    "MaxTextPrefixRuntimeError",
    "PREFIX_MESH_CANDIDATES",
    "CompiledPrefix",
    "PrefixInputBatch",
    "PrefixMeshCandidate",
    "PrefixRuntimeSettings",
    "PreparedPrefix",
    "assert_pruned_parameter_count",
    "assert_real_v6e64_runtime",
    "bind_pruned_model_parameters",
    "compile_prefix",
    "create_compile_synthetic_prefix",
    "create_abstract_pruned_model",
    "create_device_synthetic_prefix_leaves",
    "create_prefix_mesh",
    "create_prefix_restore_shardings",
    "create_synthetic_prefix_batch",
    "initialize_maxtext_config",
    "materialize_rng_state",
    "prune_output_only_parameters",
    "restore_bind_compile_prefix",
    "split_restored_prefix_leaves",
    "synchronize_prefix_output",
    "validate_initialized_config",
]
