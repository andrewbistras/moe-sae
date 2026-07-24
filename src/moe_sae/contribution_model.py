"""Gemma 4 prefix adapter for Experiment-2 expert contributions.

The adapter reuses the already-qualified block-1--14 and target-attention
implementation from :mod:`moe_sae.model`.  Its only new model seam is the
opt-in MaxText ``RoutedMoE`` return pinned by ``patches/maxtext``:

``(aggregate, load_balance_loss, bias_updates, selected_outputs)``.

``selected_outputs`` is the eight raw selected expert outputs in native route
slot order, after Gemma's learned expert-output scale and before router
weighting.  The adapter recomputes only the small router gate, canonically
orders IDs, weights, and outputs together, and captures the raw shared-expert
output.  It deliberately does not reproduce MaxText's routing or expert
kernels.
"""

from __future__ import annotations

import hashlib
import inspect
from importlib import import_module
from pathlib import Path
import subprocess
from typing import Any, NamedTuple

from moe_sae.config import Gemma4Config
from moe_sae.contribution_artifacts import ROUTER_WEIGHT_SUM_ATOL
from moe_sae.contribution_config import (
    MAXTEXT_FORK_COMMIT,
    MAXTEXT_ROUTED_MOE_PATCHED_SHA256,
)
from moe_sae.model import (
    Gemma4PrefixAdapter,
    MaxTextUnavailableError,
    _array_value,
    _as_bfloat16,
    _first,
    _validate_input_shapes,
    _value,
)

__all__ = [
    "ContributionCaptureBatch",
    "ContributionInstrumentationEvidence",
    "Gemma4ContributionPrefixAdapter",
    "canonicalize_selected_experts",
]


class ContributionCaptureBatch(NamedTuple):
    """Device-side Experiment-2 capture rows.

    This is intentionally distinct from the NumPy wire object in
    :mod:`moe_sae.contribution_artifacts`.  A named tuple is a JAX pytree, so
    the compiled prefix can return it without registering process-global
    handlers.
    """

    expert_output: Any
    expert_ids: Any
    router_weights: Any
    shared_output: Any
    sequence_id: Any
    position: Any

    @property
    def rows(self) -> int:
        return int(self.expert_output.shape[0])


class ContributionInstrumentationEvidence(NamedTuple):
    """One compiled parity result plus the two outputs it compares."""

    equal: Any
    unmodified_full_block: Any
    instrumented_full_block: Any


class _NativeComponents(NamedTuple):
    """Target-block values before canonical route-slot reordering."""

    aggregate: Any
    expert_output: Any
    expert_ids: Any
    router_weights: Any
    shared_output: Any


def canonicalize_selected_experts(
    expert_output: Any,
    expert_ids: Any,
    router_weights: Any,
) -> tuple[Any, Any, Any, Any]:
    """Order slots by descending BF16 weight, then ascending expert ID.

    Returns ``(ordered_output, ordered_ids, ordered_weights, source_slots)``.
    One integer-key sort implements the lexicographic order used by the
    independent NumPy scientific oracle.  Router weights are strictly
    positive BF16 values, whose unsigned bit patterns are monotonic in value;
    inverting those bits therefore orders weight descending, while the low
    eight key bits order tied expert IDs ascending.  This avoids a second sort
    in the MFU-critical capture graph.

    Casting at this boundary is intentional.  The protocol calls the persisted
    BF16 router weight the native weight; ordering before this cast could give
    two numerically identical wire records different slot orders.
    """

    jnp = import_module("jax.numpy")
    output = _as_bfloat16(expert_output)
    ids = jnp.asarray(expert_ids)
    weights = _as_bfloat16(router_weights)

    if len(ids.shape) < 1:
        raise ValueError("expert_ids must include a route-slot dimension")
    if weights.shape != ids.shape:
        raise ValueError(
            f"router_weights must have shape {ids.shape}, got {weights.shape}"
        )
    if len(output.shape) != len(ids.shape) + 1:
        raise ValueError(
            "expert_output must have one hidden dimension after expert_ids, "
            f"got output {output.shape} and IDs {ids.shape}"
        )
    if output.shape[:-1] != ids.shape:
        raise ValueError(
            f"expert_output prefix must have shape {ids.shape}, got {output.shape}"
        )
    if not bool(jnp.issubdtype(ids.dtype, jnp.integer)):
        raise TypeError(f"expert_ids must use an integer dtype, got {ids.dtype}")
    if not bool(jnp.issubdtype(weights.dtype, jnp.floating)):
        raise TypeError(
            f"router_weights must use a floating dtype, got {weights.dtype}"
        )

    lax = import_module("jax.lax")
    weight_bits = lax.bitcast_convert_type(weights, jnp.uint16).astype(jnp.uint32)
    descending_weight = jnp.uint32(0xFFFF) - weight_bits
    sort_key = (descending_weight << jnp.uint32(8)) | ids.astype(jnp.uint32)
    source_slots = jnp.argsort(sort_key, axis=-1, stable=False)

    ordered_ids = jnp.take_along_axis(ids, source_slots, axis=-1).astype(jnp.uint8)
    ordered_weights = jnp.take_along_axis(weights, source_slots, axis=-1)
    ordered_output = jnp.take_along_axis(
        output,
        source_slots[..., None],
        axis=-2,
    )
    return ordered_output, ordered_ids, ordered_weights, source_slots


class Gemma4ContributionPrefixAdapter(Gemma4PrefixAdapter):
    """Capture block-15 routed components and the raw shared-expert output.

    Construction fails before compilation unless the target ``RoutedMoE``
    exposes the exact patched auxiliary keyword and its statically observable
    configuration is the qualified Gemma-4 sparse, dropless, TP1 path.
    Runtime arity and every static tensor shape are checked before a capture is
    allowed to escape.
    """

    def __init__(self, model: Any, config: Gemma4Config | None = None):
        super().__init__(model, config)
        target = self.layers[self.config.target_layer_index]
        self._validate_auxiliary_path(target.mlp.moe_block.routed_moe)

    @classmethod
    def from_maxtext(
        cls,
        model: Any,
        config: Gemma4Config | None = None,
        *,
        maxtext_root: str | Path,
    ) -> Gemma4ContributionPrefixAdapter:
        """Construct only from the clean, exact published Experiment-2 fork."""

        root = Path(maxtext_root).expanduser().resolve()
        try:
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise MaxTextUnavailableError(
                "cannot verify the Experiment-2 MaxText fork checkout"
            ) from error
        if head != MAXTEXT_FORK_COMMIT:
            raise MaxTextUnavailableError(
                "Experiment-2 MaxText checkout is not at the pinned fork commit "
                f"{MAXTEXT_FORK_COMMIT}"
            )
        if dirty:
            raise MaxTextUnavailableError(
                "Experiment-2 MaxText fork checkout has uncommitted changes"
            )
        try:
            maxtext = import_module("maxtext")
            import_module("jax")
        except (ImportError, ModuleNotFoundError) as error:
            raise MaxTextUnavailableError(
                "Experiment-2 capture requires the pinned MaxText fork and TPU "
                "dependencies"
            ) from error
        module_path = Path(maxtext.__file__).resolve()
        if not module_path.is_relative_to(root):
            raise MaxTextUnavailableError(
                "imported MaxText package does not come from maxtext_root"
            )
        return cls(model, config)

    @classmethod
    def from_installed_maxtext(
        cls,
        model: Any,
        config: Gemma4Config | None = None,
    ) -> Gemma4ContributionPrefixAdapter:
        """Construct from the frozen ``uv`` install used on TPU workers.

        Production workers install the published fork directly from
        ``uv.lock``; they do not need a second mutable Git checkout.  PEP-610
        provenance proves the exact commit, and a whole-file SHA-256 proves
        that the imported routed-MoE implementation is the manifest's patched
        post-image.
        """

        try:
            maxtext = import_module("maxtext")
            import_module("jax")
            from moe_sae.runtime import installed_maxtext_commit
        except (ImportError, ModuleNotFoundError) as error:
            raise MaxTextUnavailableError(
                "Experiment-2 capture requires the locked MaxText TPU install"
            ) from error
        observed = installed_maxtext_commit()
        if observed != MAXTEXT_FORK_COMMIT:
            raise MaxTextUnavailableError(
                "installed MaxText is not the pinned Experiment-2 fork commit "
                f"{MAXTEXT_FORK_COMMIT}"
            )
        package_root = Path(maxtext.__file__).resolve().parent
        routed_moe = package_root / "layers" / "moe.py"
        try:
            digest = hashlib.sha256(routed_moe.read_bytes()).hexdigest()
        except OSError as error:
            raise MaxTextUnavailableError(
                "cannot read the installed MaxText routed-MoE source"
            ) from error
        if digest != MAXTEXT_ROUTED_MOE_PATCHED_SHA256:
            raise MaxTextUnavailableError(
                "installed MaxText routed-MoE source differs from the "
                "manifest-verified fork post-image"
            )
        return cls(model, config)

    def __call__(
        self,
        token_ids: Any,
        decoder_positions: Any,
        decoder_segment_ids: Any,
        sequence_ids: Any,
        *,
        valid_token_indices: Any | None = None,
        model_mode: str = "train",
    ) -> ContributionCaptureBatch:
        """Run the exact prefix and return compacted contribution rows."""

        _validate_input_shapes(
            token_ids,
            decoder_positions,
            decoder_segment_ids,
            sequence_ids,
        )
        hidden = self._run_prefix(
            token_ids,
            decoder_positions,
            decoder_segment_ids,
            model_mode,
        )
        target = self.layers[self.config.target_layer_index]
        residual = self._target_attention_residual(
            target,
            hidden,
            decoder_positions,
            decoder_segment_ids,
            model_mode,
        )
        native = self._capture_native_components(target, residual)
        expert_output, expert_ids, router_weights, _ = canonicalize_selected_experts(
            native.expert_output,
            native.expert_ids,
            native.router_weights,
        )

        batch, sequence = token_ids.shape
        rows = batch * sequence
        flat_output = expert_output.reshape(
            (
                rows,
                self.config.experts_per_token,
                self.config.hidden_size,
            )
        )
        flat_ids = expert_ids.reshape((rows, self.config.experts_per_token))
        flat_weights = router_weights.reshape((rows, self.config.experts_per_token))
        flat_shared = _as_bfloat16(native.shared_output).reshape(
            (rows, self.config.hidden_size)
        )
        flat_positions = decoder_positions.reshape((rows,))
        if len(sequence_ids.shape) == 1:
            flat_sequence_ids = sequence_ids.repeat(sequence)
        else:
            flat_sequence_ids = sequence_ids.reshape((rows,))

        if valid_token_indices is not None:
            indices = valid_token_indices.reshape((-1,))
            flat_output = flat_output[indices]
            flat_ids = flat_ids[indices]
            flat_weights = flat_weights[indices]
            flat_shared = flat_shared[indices]
            flat_sequence_ids = flat_sequence_ids[indices]
            flat_positions = flat_positions[indices]

        if hasattr(flat_positions, "astype"):
            flat_positions = flat_positions.astype("uint16")
        return ContributionCaptureBatch(
            expert_output=flat_output,
            expert_ids=flat_ids,
            router_weights=flat_weights,
            shared_output=flat_shared,
            sequence_id=flat_sequence_ids,
            position=flat_positions,
        )

    def instrumentation_parity(
        self,
        token_ids: Any,
        decoder_positions: Any,
        decoder_segment_ids: Any,
        sequence_ids: Any,
        *,
        model_mode: str = "train",
    ) -> Any:
        """Return one compiled boolean covering every Experiment-2 seam."""

        return self.instrumentation_evidence(
            token_ids,
            decoder_positions,
            decoder_segment_ids,
            sequence_ids,
            model_mode=model_mode,
        ).equal

    def instrumentation_evidence(
        self,
        token_ids: Any,
        decoder_positions: Any,
        decoder_segment_ids: Any,
        sequence_ids: Any,
        *,
        model_mode: str = "train",
    ) -> ContributionInstrumentationEvidence:
        """Return parity plus both full-block outputs in a single execution.

        The check executes the auxiliary and unchanged routed paths on the same
        operands, independently recomposes the auxiliary output with the
        rerun gate weights using MaxText's FP32 accumulation expression, and
        carries the result through routed normalization, the independently
        captured shared branch, outer normalization, residual addition, and
        the layer scalar.  The final result must be array-equal to an
        unmodified full target-block invocation.
        """

        _validate_input_shapes(
            token_ids,
            decoder_positions,
            decoder_segment_ids,
            sequence_ids,
        )
        hidden = self._run_prefix(
            token_ids,
            decoder_positions,
            decoder_segment_ids,
            model_mode,
        )
        target = self.layers[self.config.target_layer_index]
        reference_full_block = _first(
            target(
                hidden,
                decoder_segment_ids,
                decoder_positions,
                True,
                model_mode,
                bidirectional_mask=None,
                kv_cache=None,
                attention_metadata=None,
            )
        )
        residual = self._target_attention_residual(
            target,
            hidden,
            decoder_positions,
            decoder_segment_ids,
            model_mode,
        )
        native = self._capture_native_components(target, residual)

        routed_inputs, gate_inputs = self._routed_operands(target.mlp, residual)
        unchanged_result = target.mlp.moe_block.routed_moe(
            routed_inputs,
            gate_inputs=gate_inputs,
            out_sharding=None,
        )
        if not isinstance(unchanged_result, tuple) or len(unchanged_result) != 3:
            raise TypeError(
                "unchanged MaxText RoutedMoE call must return exactly "
                "(aggregate, load_balance_loss, bias_updates)"
            )
        unchanged_aggregate = unchanged_result[0]
        self._validate_aggregate_shape(unchanged_aggregate, routed_inputs)

        recomposed_aggregate = self._recompose_native_aggregate(
            native.expert_output,
            native.router_weights,
            native.aggregate,
            target.mlp.moe_block.routed_moe,
        )
        jnp = import_module("jax.numpy")
        ordered_output, ordered_ids, ordered_weights, source_slots = (
            canonicalize_selected_experts(
                native.expert_output,
                native.expert_ids,
                native.router_weights,
            )
        )
        restored_slots = jnp.argsort(
            source_slots,
            axis=-1,
            stable=True,
        )
        restored_output = jnp.take_along_axis(
            ordered_output,
            restored_slots[..., None],
            axis=-2,
        )
        restored_ids = jnp.take_along_axis(
            ordered_ids,
            restored_slots,
            axis=-1,
        )
        restored_weights = jnp.take_along_axis(
            ordered_weights,
            restored_slots,
            axis=-1,
        )

        repeated_ids, repeated_weights = self._route(target.mlp, residual)
        repeated_weights = _as_bfloat16(repeated_weights)

        routed_branch = target.mlp.post_feedforward_layernorm_2(native.aggregate)
        shared_branch = target.mlp.post_feedforward_layernorm_1(native.shared_output)
        combined_mlp = routed_branch + shared_branch
        outer = combined_mlp
        if getattr(target, "post_ffw_norm", None) is not None:
            outer = target.post_ffw_norm(outer)
        reconstructed_full_block = (outer + residual) * _array_value(
            target.layer_scalar,
            dtype=getattr(target.config, "dtype", None),
        )

        ids_i32 = native.expert_ids.astype(jnp.int32)
        sorted_ids = jnp.sort(ids_i32, axis=-1)
        router_sums = native.router_weights.astype(jnp.float32).sum(axis=-1)
        checks = (
            jnp.array_equal(native.aggregate, unchanged_aggregate),
            jnp.array_equal(native.aggregate, recomposed_aggregate),
            jnp.array_equal(native.expert_output, restored_output),
            jnp.array_equal(ids_i32, restored_ids.astype(jnp.int32)),
            jnp.array_equal(
                native.router_weights,
                restored_weights,
            ),
            jnp.array_equal(ids_i32, repeated_ids.astype(jnp.int32)),
            jnp.array_equal(native.router_weights, repeated_weights),
            jnp.array_equal(reference_full_block, reconstructed_full_block),
            jnp.all(jnp.isfinite(native.expert_output)),
            jnp.all(jnp.isfinite(native.shared_output)),
            jnp.all(jnp.isfinite(native.router_weights)),
            jnp.all(native.router_weights > 0),
            jnp.all(ids_i32 >= 0),
            jnp.all(ids_i32 < self.config.num_routed_experts),
            jnp.all(jnp.diff(sorted_ids, axis=-1) > 0),
            jnp.all(jnp.abs(router_sums - 1.0) <= ROUTER_WEIGHT_SUM_ATOL),
        )
        return ContributionInstrumentationEvidence(
            equal=jnp.all(
                jnp.stack(tuple(jnp.asarray(check) for check in checks))
            ),
            unmodified_full_block=reference_full_block,
            instrumented_full_block=reconstructed_full_block,
        )

    def _capture_native_components(
        self,
        target: Any,
        residual: Any,
    ) -> _NativeComponents:
        routed_inputs, gate_inputs = self._routed_operands(target.mlp, residual)
        routed_moe = target.mlp.moe_block.routed_moe
        routed_result = routed_moe(
            routed_inputs,
            gate_inputs=gate_inputs,
            out_sharding=None,
            return_selected_outputs=True,
        )
        if not isinstance(routed_result, tuple) or len(routed_result) != 4:
            raise TypeError(
                "patched MaxText RoutedMoE call must return exactly "
                "(aggregate, load_balance_loss, bias_updates, selected_outputs)"
            )
        aggregate, _, _, selected_output = routed_result
        self._validate_aggregate_shape(aggregate, routed_inputs)
        expected_selected = residual.shape[:-1] + (
            self.config.experts_per_token,
            self.config.hidden_size,
        )
        if selected_output.shape != expected_selected:
            raise ValueError(
                "selected expert outputs must have shape "
                f"{expected_selected}, got {selected_output.shape}"
            )

        expert_ids, router_weights = self._route(target.mlp, residual)
        expected_route = residual.shape[:-1] + (self.config.experts_per_token,)
        if expert_ids.shape != expected_route:
            raise ValueError(
                f"expert IDs must have shape {expected_route}, got {expert_ids.shape}"
            )
        if router_weights.shape != expected_route:
            raise ValueError(
                "router weights must have shape "
                f"{expected_route}, got {router_weights.shape}"
            )

        jnp = import_module("jax.numpy")
        if not bool(jnp.issubdtype(expert_ids.dtype, jnp.integer)):
            raise TypeError(
                f"expert IDs must use an integer dtype, got {expert_ids.dtype}"
            )
        if not bool(jnp.issubdtype(router_weights.dtype, jnp.floating)):
            raise TypeError(
                f"router weights must use a floating dtype, got {router_weights.dtype}"
            )

        shared_inputs = target.pre_ffw_norm(residual)
        shared_output = target.mlp.moe_block.shared_experts(
            shared_inputs,
            intermediate_sharding=None,
            out_sharding=None,
        )
        if shared_output.shape != residual.shape:
            raise ValueError(
                f"raw shared output must have shape {residual.shape}, "
                f"got {shared_output.shape}"
            )
        return _NativeComponents(
            aggregate=aggregate,
            expert_output=_as_bfloat16(selected_output),
            expert_ids=expert_ids,
            router_weights=_as_bfloat16(router_weights),
            shared_output=_as_bfloat16(shared_output),
        )

    def _routed_operands(self, mlp: Any, residual: Any) -> tuple[Any, Any]:
        routed_inputs = mlp.pre_feedforward_layernorm_2(residual)
        if routed_inputs.shape != residual.shape:
            raise ValueError(
                f"routed inputs must preserve shape {residual.shape}, "
                f"got {routed_inputs.shape}"
            )
        gate_inputs = (
            mlp.gate_norm(residual)
            * (self.config.hidden_size**-0.5)
            * _value(mlp.pre_forward_scale_2)
        )
        if gate_inputs.shape != residual.shape:
            raise ValueError(
                f"router inputs must preserve shape {residual.shape}, "
                f"got {gate_inputs.shape}"
            )
        return routed_inputs, gate_inputs

    @staticmethod
    def _validate_aggregate_shape(aggregate: Any, routed_inputs: Any) -> None:
        if aggregate.shape != routed_inputs.shape:
            raise ValueError(
                f"routed aggregate must have shape {routed_inputs.shape}, "
                f"got {aggregate.shape}"
            )

    def _recompose_native_aggregate(
        self,
        selected_output: Any,
        router_weights: Any,
        aggregate: Any,
        routed_moe: Any,
    ) -> Any:
        """Mirror MaxText's qualified ``BKE,BK->BE`` FP32 weight sum."""

        jax = import_module("jax")
        jnp = import_module("jax.numpy")
        hidden = self.config.hidden_size
        top_k = self.config.experts_per_token
        flat_output = selected_output.astype(jnp.float32).reshape((-1, top_k, hidden))
        flat_weights = router_weights.astype(jnp.float32).reshape((-1, top_k))
        precision_name = getattr(routed_moe.config, "matmul_precision", "default")
        precision = jax.lax.Precision(precision_name)
        recomposed = jnp.einsum(
            "BKE,BK->BE",
            flat_output,
            flat_weights,
            precision=precision,
        )
        return recomposed.reshape(aggregate.shape).astype(aggregate.dtype)

    @staticmethod
    def _validate_auxiliary_path(routed_moe: Any) -> None:
        """Reject an unpatched or unqualified routed path before tracing."""

        try:
            signature = inspect.signature(type(routed_moe).__call__)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "cannot inspect target RoutedMoE auxiliary-return contract"
            ) from exc
        parameter = signature.parameters.get("return_selected_outputs")
        if parameter is None or parameter.default is not False:
            raise TypeError(
                "target RoutedMoE is not the pinned Experiment-2 patch: "
                "missing `return_selected_outputs: bool = False`"
            )

        config = getattr(routed_moe, "config", None)
        if config is None:
            raise TypeError("target RoutedMoE must expose its MaxText config")
        decoder_block = str(getattr(config, "decoder_block", "")).lower()
        if "gemma4" not in decoder_block:
            raise ValueError("selected-output capture is qualified only for Gemma 4")
        if getattr(config, "attention", None) == "vllm_rpa":
            raise ValueError(
                "selected-output capture is not qualified for fused-vLLM attention"
            )
        if getattr(config, "sparse_matmul", None) is not True:
            raise ValueError(
                "selected-output capture requires MaxText sparse_matmul=true"
            )
        if getattr(config, "use_ring_of_experts", None) is not False:
            raise ValueError(
                "selected-output capture is not qualified for ring of experts"
            )
        ragged_buffer_factor = getattr(config, "ragged_buffer_factor", None)
        if ragged_buffer_factor is None or ragged_buffer_factor > 0.0:
            raise ValueError(
                "selected-output capture requires a non-truncating "
                "ragged_buffer_factor <= 0"
            )
        if getattr(config, "float32_weight_sum", None) is not True:
            raise ValueError(
                "Experiment 2 requires float32_weight_sum=true for exact "
                "weighted-contribution recomposition"
            )
        matmul_precision = str(getattr(config, "matmul_precision", "")).lower()
        if not matmul_precision.endswith("default"):
            raise ValueError(
                "Experiment 2 freezes routed matmul_precision=default, got "
                f"{getattr(config, 'matmul_precision', None)!r}"
            )

        tensor_parallelism = getattr(
            routed_moe,
            "get_tensor_parallelism_size",
            None,
        )
        if not callable(tensor_parallelism) or tensor_parallelism() != 1:
            raise ValueError(
                "selected-output capture is qualified only for tensor parallelism 1"
            )
        if getattr(routed_moe, "is_hash_routing", None) is not False:
            raise ValueError(
                "selected-output capture is not qualified for hash routing"
            )

        quant = getattr(routed_moe, "quant", None)
        if quant is not None:
            quant_mode = getattr(quant, "quant_mode", None)
            if quant_mode is None:
                raise ValueError(
                    "cannot prove that routed quantization is outside serve mode"
                )
            if "serve" in str(quant_mode).lower():
                raise ValueError(
                    "selected-output capture is not qualified for serving quantization"
                )
