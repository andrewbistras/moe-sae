"""Gemma 4 prefix capture without a hard MaxText import dependency.

The production adapter is deliberately duck typed against the pinned MaxText
NNX model.  Importing :mod:`moe_sae.model` therefore works on laptops and in
unit tests that do not have MaxText or TPU libraries installed.  The explicit
``from_maxtext`` constructor performs dependency and checkout validation.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence
from urllib.parse import urlparse

import numpy as np

from moe_sae.config import MAXTEXT_COMMIT, Gemma4Config


class MaxTextUnavailableError(ImportError):
    """Raised when a production adapter is requested without MaxText/JAX."""


class CaptureBatch(NamedTuple):
    """Host-ready rows captured at the Gemma 4 routed-branch site.

    A named tuple is intentional: JAX treats it as a pytree without this
    package importing JAX or registering global pytree handlers.
    """

    activation: Any
    expert_ids: Any
    router_weights: Any
    sequence_id: Any
    position: Any

    @property
    def rows(self) -> int:
        return int(self.activation.shape[0])


@dataclass(frozen=True, slots=True)
class CaptureSummary:
    rows: int
    hidden_size: int
    experts_per_token: int
    min_expert_id: int
    max_expert_id: int
    max_router_sum_error: float


@dataclass(frozen=True, slots=True)
class CheckpointConversionSpec:
    """Secret-free description of a Hugging Face -> Orbax conversion."""

    maxtext_root: Path
    hf_model_path: str
    output_directory: str
    python_executable: str = sys.executable
    save_dtype: str = "bfloat16"
    lazy_load_tensors: bool = True

    @property
    def base_config(self) -> Path:
        return self.maxtext_root / "src" / "maxtext" / "configs" / "base.yml"

    def maxtext_overrides(self) -> dict[str, str]:
        return {
            "model_name": "gemma4-26b",
            "base_output_directory": self.output_directory,
            "use_multimodal": "false",
            "scan_layers": "false",
            "pure_nnx_decoder": "true",
            "hardware": "cpu",
            "skip_jax_distributed_system": "true",
        }


def verify_maxtext_checkout(maxtext_root: str | Path) -> str:
    """Return HEAD after verifying an exact, clean MaxText source pin."""

    root = Path(maxtext_root).expanduser().resolve()
    if not (root / "src" / "maxtext" / "configs" / "base.yml").is_file():
        raise FileNotFoundError(f"not a MaxText checkout: {root}")
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not determine MaxText commit at {root}") from exc
    actual = result.stdout.strip().lower()
    if actual != MAXTEXT_COMMIT:
        raise RuntimeError(
            f"MaxText must be at {MAXTEXT_COMMIT}; checkout is at {actual or '<unknown>'}"
        )
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not verify MaxText source cleanliness at {root}") from exc
    if status.strip():
        raise RuntimeError(
            "MaxText checkout must be clean before conversion/parity validation"
        )
    return actual


def maxtext_checkout_environment(
    maxtext_root: str | Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Force conversion subprocesses to import code from the verified checkout.

    The caller's credentials and ordinary dependency environment are retained,
    while Python package search is restricted to the checkout's ``src`` tree
    and test utilities.  This prevents an installed or user-site MaxText copy
    from satisfying a parity run attributed to different source code.
    """

    root = Path(maxtext_root).expanduser().resolve()
    verify_maxtext_checkout(root)
    result = dict(os.environ if environment is None else environment)
    result.pop("PYTHONHOME", None)
    result["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONSAFEPATH"] = "1"
    return result


def require_maxtext(maxtext_root: str | Path | None = None) -> tuple[Any, Any]:
    """Lazily import MaxText and JAX, optionally verifying the source checkout."""

    if maxtext_root is not None:
        verify_maxtext_checkout(maxtext_root)
    try:
        maxtext = import_module("maxtext")
        jax = import_module("jax")
    except (ImportError, ModuleNotFoundError) as exc:
        raise MaxTextUnavailableError(
            "Gemma 4 capture requires the pinned MaxText checkout and its TPU dependencies; "
            "install MaxText from commit " + MAXTEXT_COMMIT
        ) from exc
    return maxtext, jax


def checkpoint_conversion_spec(
    maxtext_root: str | Path,
    output_directory: str,
    *,
    hf_model_path: str | None = None,
    python_executable: str = sys.executable,
    config: Gemma4Config | None = None,
) -> CheckpointConversionSpec:
    """Build and validate a secret-free conversion specification."""

    cfg = config or Gemma4Config()
    cfg.validate()
    return CheckpointConversionSpec(
        maxtext_root=Path(maxtext_root),
        hf_model_path=hf_model_path or cfg.model_id,
        output_directory=output_directory,
        python_executable=python_executable,
    )


def build_checkpoint_conversion_command(
    spec: CheckpointConversionSpec,
) -> tuple[str, ...]:
    """Return argv for MaxText's official converter.

    The Hugging Face token is intentionally absent.  Current MaxText reads
    ``HF_TOKEN`` from the subprocess environment, which keeps credentials out
    of argv, logs, manifests, and shell history.
    """

    if not spec.base_config.is_file():
        raise FileNotFoundError(
            f"MaxText base config does not exist: {spec.base_config}"
        )
    overrides = tuple(
        f"{key}={value}" for key, value in spec.maxtext_overrides().items()
    )
    return (
        spec.python_executable,
        "-m",
        "maxtext.checkpoint_conversion.to_maxtext",
        str(spec.base_config),
        *overrides,
        f"--hf_model_path={spec.hf_model_path}",
        f"--save_dtype={spec.save_dtype}",
        f"--lazy_load_tensors={'true' if spec.lazy_load_tensors else 'false'}",
    )


def checkpoint_validation_command(
    spec: CheckpointConversionSpec,
    checkpoint_items_path: str,
    *,
    max_kl_div: float = 0.03,
) -> tuple[str, ...]:
    """Return argv for MaxText's Hugging Face logit-parity checker."""

    if not (0.0 < max_kl_div <= 0.03):
        raise ValueError("max_kl_div must be in (0, 0.03]")
    return (
        spec.python_executable,
        "-m",
        "tests.utils.forward_pass_logit_checker",
        str(spec.base_config),
        "tokenizer_path=google/gemma-4-26B-A4B",
        f"load_parameters_path={checkpoint_items_path}",
        "model_name=gemma4-26b",
        "use_multimodal=false",
        "scan_layers=false",
        "pure_nnx_decoder=true",
        "per_device_batch_size=1",
        "dtype=float32",
        "wi_tile_fwd_embed_dim=512",
        "wi_tile_fwd_mlp_dim=512",
        "wo_tile_fwd_embed_dim=512",
        "wo_tile_fwd_mlp_dim=512",
        f"--max_kl_div={max_kl_div}",
        "--run_hf_model=true",
        f"--hf_model_path={spec.hf_model_path}",
    )


def checkpoint_inventory_sha256(checkpoint_items_path: str) -> str:
    """Return a deterministic content-inventory digest for an Orbax checkpoint.

    Local checkpoints are hashed byte-for-byte.  For GCS checkpoints the digest
    covers every object's relative name, size, generation, and provider-reported
    CRC32C/MD5 checksum, avoiding a second multi-gigabyte download after MaxText's
    parity checker has already read the checkpoint.
    """

    if checkpoint_items_path.startswith("gs://"):
        parsed = urlparse(checkpoint_items_path)
        prefix = parsed.path.lstrip("/").rstrip("/")
        if not parsed.netloc or not prefix:
            raise ValueError(
                "GCS checkpoint path must include a bucket and object prefix"
            )
        try:
            storage = import_module("google.cloud.storage")
        except (ImportError, ModuleNotFoundError) as exc:
            raise MaxTextUnavailableError(
                "GCS checkpoint hashing requires google-cloud-storage in the conversion environment"
            ) from exc
        client = storage.Client()
        entries: list[dict[str, Any]] = []
        for blob in client.list_blobs(parsed.netloc, prefix=prefix + "/"):
            if blob.name.endswith("/"):
                continue
            relative = blob.name[len(prefix) :].lstrip("/")
            entries.append(
                {
                    "path": relative,
                    "size": int(blob.size),
                    "generation": str(blob.generation),
                    "crc32c": blob.crc32c,
                    "md5": blob.md5_hash,
                }
            )
        if not entries:
            raise FileNotFoundError(
                f"checkpoint prefix contains no objects: {checkpoint_items_path}"
            )
        payload = {
            "format": "orbax-gcs-content-inventory-v1",
            "objects": sorted(entries, key=lambda item: item["path"]),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    root = Path(checkpoint_items_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {root}")
    files = (
        [root]
        if root.is_file()
        else sorted(path for path in root.rglob("*") if path.is_file())
    )
    if not files:
        raise FileNotFoundError(f"checkpoint contains no files: {root}")
    digest = hashlib.sha256()
    digest.update(b"orbax-local-content-inventory-v1\0")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"checkpoint inventory refuses symbolic links: {path}")
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.digest())
    return digest.hexdigest()


class Gemma4PrefixAdapter:
    """Execute blocks 1--14, then capture block 15's routed branch.

    ``model`` must be an unscanned, pure-NNX MaxText model or a compatible fake.
    No MaxText symbol is imported by the ordinary constructor, which keeps
    offline validation and CPU tests lightweight.
    """

    def __init__(self, model: Any, config: Gemma4Config | None = None):
        self.model = model
        self.config = config or Gemma4Config()
        self.config.validate()
        self.decoder = _require_attribute(model, "decoder", "model")
        self.layers = _resolve_layers(self.decoder)
        validate_model_architecture(model, self.config, layers=self.layers)

    @classmethod
    def from_maxtext(
        cls,
        model: Any,
        config: Gemma4Config | None = None,
        *,
        maxtext_root: str | Path | None = None,
    ) -> "Gemma4PrefixAdapter":
        require_maxtext(maxtext_root)
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
    ) -> CaptureBatch:
        """Capture routed-branch rows.

        Inputs have shape ``[batch, sequence]``.  ``sequence_ids`` may be
        ``[batch]`` or ``[batch, sequence]``.  ``valid_token_indices`` indexes
        the flattened token dimension and lets the data pipeline remove BOS,
        EOS, and padding with a static-shape gather before device-to-host copy.
        """

        _validate_input_shapes(
            token_ids, decoder_positions, decoder_segment_ids, sequence_ids
        )
        hidden = self._run_prefix(
            token_ids,
            decoder_positions,
            decoder_segment_ids,
            model_mode,
        )

        target = self.layers[self.config.target_layer_index]
        activation, expert_ids, router_weights = self._capture_target(
            target,
            hidden,
            decoder_positions,
            decoder_segment_ids,
            model_mode,
        )

        batch, sequence = token_ids.shape
        flat_activation = activation.reshape(
            (batch * sequence, self.config.hidden_size)
        )
        flat_ids = expert_ids.reshape((batch * sequence, self.config.experts_per_token))
        flat_weights = router_weights.reshape(
            (batch * sequence, self.config.experts_per_token)
        )
        flat_positions = decoder_positions.reshape((batch * sequence,))
        if len(sequence_ids.shape) == 1:
            flat_sequence_ids = sequence_ids.repeat(sequence)
        else:
            flat_sequence_ids = sequence_ids.reshape((batch * sequence,))

        if valid_token_indices is not None:
            indices = valid_token_indices.reshape((-1,))
            flat_activation = flat_activation[indices]
            flat_ids = flat_ids[indices]
            flat_weights = flat_weights[indices]
            flat_positions = flat_positions[indices]
            flat_sequence_ids = flat_sequence_ids[indices]

        if hasattr(flat_ids, "astype"):
            flat_ids = flat_ids.astype("uint8")
        return CaptureBatch(
            activation=flat_activation,
            expert_ids=flat_ids,
            router_weights=flat_weights,
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
        """Prove that target instrumentation leaves the full block unchanged.

        The production benchmark calls this before timing.  It compares an
        untouched block-15 invocation with a manual recomposition built from
        the same attention residual, captured routed aggregate, shared expert,
        outer FFN normalization, residual addition, and layer scalar.  It also
        independently repeats the small router projection so the IDs and
        normalized weights persisted by capture must equal MaxText's routing
        primitives exactly.

        A JAX boolean scalar is returned so the check itself can be compiled
        under the same mesh and logical-axis contexts as harvesting.
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

        reference = _first(
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
        activation, expert_ids, router_weights = self._capture_routed(
            target.mlp, residual
        )

        shared_inputs = target.pre_ffw_norm(residual)
        shared_output = target.mlp.moe_block.shared_experts(
            shared_inputs,
            intermediate_sharding=None,
            out_sharding=None,
        )
        shared_output = target.mlp.post_feedforward_layernorm_1(shared_output)
        reconstructed = activation + shared_output
        if getattr(target, "post_ffw_norm", None) is not None:
            reconstructed = target.post_ffw_norm(reconstructed)
        reconstructed = (reconstructed + residual) * _array_value(
            target.layer_scalar,
            dtype=getattr(target.config, "dtype", None),
        )

        reference_ids, reference_weights = self._route(target.mlp, residual)
        reference_weights = _as_bfloat16(reference_weights)
        jnp = import_module("jax.numpy")
        return jnp.logical_and(
            jnp.array_equal(reference, reconstructed),
            jnp.logical_and(
                jnp.array_equal(expert_ids, reference_ids),
                jnp.array_equal(router_weights, reference_weights),
            ),
        )

    def _run_prefix(
        self,
        token_ids: Any,
        positions: Any,
        segment_ids: Any,
        model_mode: str,
    ) -> Any:
        hidden = self._embed(token_ids, positions, model_mode)
        for layer in self.layers[: self.config.target_layer_index]:
            hidden = _first(
                layer(
                    hidden,
                    segment_ids,
                    positions,
                    True,
                    model_mode,
                    bidirectional_mask=None,
                    kv_cache=None,
                    attention_metadata=None,
                )
            )
        return hidden

    def _embed(self, token_ids: Any, positions: Any, model_mode: str) -> Any:
        embed = getattr(self.decoder, "_apply_embedding", None)
        token_embedder = getattr(self.model, "token_embedder", None)
        if callable(embed) and token_embedder is not None:
            return embed(token_embedder, token_ids, positions, True, model_mode)
        fallback = getattr(self.model, "embed_tokens", None)
        if callable(fallback):
            return fallback(token_ids, positions)
        raise TypeError(
            "model must expose token_embedder + decoder._apply_embedding or embed_tokens"
        )

    def _capture_target(
        self,
        layer: Any,
        hidden: Any,
        positions: Any,
        segment_ids: Any,
        model_mode: str,
    ) -> tuple[Any, Any, Any]:
        residual = self._target_attention_residual(
            layer,
            hidden,
            positions,
            segment_ids,
            model_mode,
        )
        return self._capture_routed(layer.mlp, residual)

    def _target_attention_residual(
        self,
        layer: Any,
        hidden: Any,
        positions: Any,
        segment_ids: Any,
        model_mode: str,
    ) -> Any:
        # This is the attention half of Gemma4DecoderLayer.__call__.
        normalized = layer.pre_self_attention_norm(hidden)
        attention_result = layer.self_attention(
            normalized,
            normalized,
            positions,
            decoder_segment_ids=segment_ids,
            deterministic=True,
            model_mode=model_mode,
            bidirectional_mask=None,
            kv_cache=None,
            attention_metadata=None,
        )
        attention = _first(attention_result)
        if getattr(layer, "post_self_attention_norm", None) is not None:
            attention = layer.post_self_attention_norm(attention)
        return attention + hidden

    def _capture_routed(self, mlp: Any, residual: Any) -> tuple[Any, Any, Any]:
        """Run the routed branch and cast its storage boundary to BF16."""

        # Gemma4MoE applies this path independently from the shared expert.
        routed_inputs = mlp.pre_feedforward_layernorm_2(residual)
        unscaled_gate_inputs = mlp.gate_norm(residual)
        router_scale = _value(mlp.pre_forward_scale_2)
        gate_inputs = (
            unscaled_gate_inputs * (self.config.hidden_size**-0.5) * router_scale
        )

        routed_moe = mlp.moe_block.routed_moe
        routed_result = routed_moe(
            routed_inputs, gate_inputs=gate_inputs, out_sharding=None
        )
        routed_output = _first(routed_result)
        activation = _as_bfloat16(mlp.post_feedforward_layernorm_2(routed_output))

        # Re-run only the small router projection.  get_topk is the same method
        # used by the routed kernel, including Gemma 4 softmax + top-k renorm.
        expert_ids, router_weights = self._route(mlp, residual)
        return activation, expert_ids, _as_bfloat16(router_weights)

    def _route(self, mlp: Any, residual: Any) -> tuple[Any, Any]:
        unscaled_gate_inputs = mlp.gate_norm(residual)
        router_scale = _value(mlp.pre_forward_scale_2)
        gate_inputs = (
            unscaled_gate_inputs * (self.config.hidden_size**-0.5) * router_scale
        )
        routed_moe = mlp.moe_block.routed_moe
        logits_result = routed_moe.gate(gate_inputs)
        if not isinstance(logits_result, tuple) or len(logits_result) != 2:
            raise TypeError(
                "MaxText routed gate must return (gate_logits, pre_bias_logits)"
            )
        gate_logits, pre_bias_logits = logits_result
        router_weights, expert_ids = routed_moe.get_topk(
            gate_logits,
            pre_bias_logits,
            getattr(routed_moe, "rngs", None),
        )
        return expert_ids, router_weights


def validate_model_architecture(
    model: Any,
    config: Gemma4Config,
    *,
    layers: Sequence[Any] | None = None,
) -> None:
    """Fail before compilation if the checkpoint/model shape has drifted."""

    decoder = _require_attribute(model, "decoder", "model")
    resolved_layers = list(layers) if layers is not None else _resolve_layers(decoder)
    if len(resolved_layers) != config.num_decoder_layers:
        raise ValueError(
            f"expected {config.num_decoder_layers} unscanned decoder layers, found {len(resolved_layers)}"
        )
    model_config = getattr(model, "config", getattr(decoder, "config", None))
    if model_config is None:
        raise TypeError("model or decoder must expose a MaxText config")
    model_name = getattr(model_config, "model_name", None)
    if model_name != config.maxtext_model_name:
        raise ValueError(
            f"model_name must be {config.maxtext_model_name!r}, got {model_name!r}"
        )
    expected_aliases = (
        (("num_decoder_layers", "base_num_decoder_layers"), config.num_decoder_layers),
        (("emb_dim", "base_emb_dim", "hidden_size"), config.hidden_size),
        (("num_query_heads", "base_num_query_heads"), config.num_query_heads),
        (("num_kv_heads", "base_num_kv_heads"), config.num_kv_heads),
        (("global_num_kv_heads",), config.global_num_kv_heads),
        (("head_dim",), config.head_dim),
        (("global_head_dim",), config.global_head_dim),
        (("mlp_dim", "base_mlp_dim"), config.mlp_dim),
        (("moe_mlp_dim", "base_moe_mlp_dim"), config.moe_mlp_dim),
        (("num_experts",), config.num_routed_experts),
        (("num_experts_per_tok",), config.experts_per_token),
        (("shared_experts",), config.num_shared_experts),
        (("sliding_window_size",), config.sliding_window_size),
    )
    for aliases, expected in expected_aliases:
        actual = _first_attribute(model_config, aliases)
        if actual != expected:
            raise ValueError(f"model {aliases[0]} must be {expected}, got {actual!r}")
    for name, expected in (
        ("scan_layers", False),
        ("pure_nnx_decoder", True),
        ("use_multimodal", False),
        ("norm_topk_prob", True),
        ("use_random_routing", False),
        ("share_kv_projections", config.share_kv_projections),
        ("use_post_attn_norm", config.use_post_attn_norm),
        ("use_post_ffw_norm", config.use_post_ffw_norm),
        ("float32_gate_logits", config.float32_gate_logits),
    ):
        actual = getattr(model_config, name, None)
        if actual != expected:
            raise ValueError(f"model {name} must be {expected}, got {actual!r}")
    for name, expected in (("dropout_rate", 0.0), ("load_balance_loss_weight", 0.0)):
        actual = getattr(model_config, name, None)
        if actual != expected:
            raise ValueError(f"model {name} must be {expected}, got {actual!r}")
    for name, expected in (
        ("dtype", config.dtype),
        ("weight_dtype", config.weight_dtype),
    ):
        actual = getattr(model_config, name, None)
        if _dtype_name(actual) != expected:
            raise ValueError(f"model {name} must be {expected}, got {actual!r}")
    capacity_factor = getattr(model_config, "capacity_factor", None)
    if capacity_factor is None or capacity_factor > 0.0:
        raise ValueError(
            f"model capacity_factor must disable token dropping, got {capacity_factor!r}"
        )

    for index, layer in enumerate(resolved_layers):
        actual_attention = str(getattr(layer, "attention_type", "")).lower()
        expected_attention = "global" if (index + 1) % 6 == 0 else "local"
        if expected_attention not in actual_attention:
            raise ValueError(
                f"block {index + 1} must use {expected_attention} attention, "
                f"got {getattr(layer, 'attention_type', None)!r}"
            )
        layer_index = getattr(layer, "layer_idx", index)
        if int(layer_index) != index:
            raise ValueError(f"decoder layer index drift at block {index + 1}")

    target = resolved_layers[config.target_layer_index]
    if "local" not in str(getattr(target, "attention_type", "")).lower():
        raise ValueError("one-indexed block 15 must use local sliding attention")
    mlp = _require_attribute(target, "mlp", "target layer")
    for name in (
        "post_feedforward_layernorm_1",
        "pre_feedforward_layernorm_2",
        "gate_norm",
        "pre_forward_scale_2",
        "post_feedforward_layernorm_2",
        "moe_block",
    ):
        _require_attribute(mlp, name, "target Gemma4MoE")
    routed_moe = _require_attribute(
        mlp.moe_block, "routed_moe", "target RoutedAndSharedMoE"
    )
    if not callable(getattr(mlp.moe_block, "shared_experts", None)):
        raise TypeError("target routed/shared MoE lacks callable shared_experts")
    for name in ("gate", "get_topk"):
        if not callable(getattr(routed_moe, name, None)):
            raise TypeError(f"target routed MoE lacks callable {name}")


def validate_capture_batch(
    batch: CaptureBatch,
    config: Gemma4Config | None = None,
    *,
    require_storage_dtypes: bool = False,
) -> CaptureSummary:
    """Validate a host shard before it can be published in a manifest."""

    cfg = config or Gemma4Config()
    activation = np.asarray(batch.activation)
    ids = np.asarray(batch.expert_ids)
    weights = np.asarray(batch.router_weights)
    sequence_ids = np.asarray(batch.sequence_id)
    positions = np.asarray(batch.position)

    if activation.ndim != 2 or activation.shape[1] != cfg.hidden_size:
        raise ValueError(
            f"activation must have shape [N, {cfg.hidden_size}], got {activation.shape}"
        )
    rows = activation.shape[0]
    expected_route_shape = (rows, cfg.experts_per_token)
    if ids.shape != expected_route_shape:
        raise ValueError(
            f"expert_ids must have shape {expected_route_shape}, got {ids.shape}"
        )
    if weights.shape != expected_route_shape:
        raise ValueError(
            f"router_weights must have shape {expected_route_shape}, got {weights.shape}"
        )
    if sequence_ids.shape != (rows,) or positions.shape != (rows,):
        raise ValueError("sequence_id and position must each have shape [N]")
    if rows == 0:
        raise ValueError("capture batches cannot be empty")
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError("expert_ids must have an integer dtype")
    if not np.issubdtype(sequence_ids.dtype, np.integer) or not np.issubdtype(
        positions.dtype, np.integer
    ):
        raise TypeError("sequence_id and position must have integer dtypes")
    if not _all_finite_in_row_chunks(activation) or not _all_finite_in_row_chunks(
        weights
    ):
        raise ValueError("capture contains NaN or infinite values")
    if np.any(ids < 0) or np.any(ids >= cfg.num_routed_experts):
        raise ValueError("expert IDs must be in [0, 127]")
    if np.any(np.diff(np.sort(ids.astype(np.int16), axis=1), axis=1) == 0):
        raise ValueError("each token must route to eight unique experts")
    if np.any(weights <= 0.0):
        raise ValueError("router weights must be strictly positive")
    router_sums = weights.astype(np.float32).sum(axis=1)
    max_sum_error = float(np.max(np.abs(router_sums - 1.0)))
    if max_sum_error > 0.02:
        raise ValueError(
            f"router weights must sum to one (max error {max_sum_error:.6f})"
        )
    if np.any(positions < 0) or np.any(positions >= cfg.sequence_length):
        raise ValueError(f"positions must lie in [0, {cfg.sequence_length - 1}]")
    if require_storage_dtypes:
        if str(activation.dtype) != "bfloat16" or str(weights.dtype) != "bfloat16":
            raise TypeError(
                "activation and router_weights must be bfloat16 for storage"
            )
        if ids.dtype != np.dtype("uint8"):
            raise TypeError("expert_ids must be uint8 for storage")

    return CaptureSummary(
        rows=rows,
        hidden_size=cfg.hidden_size,
        experts_per_token=cfg.experts_per_token,
        min_expert_id=int(ids.min()),
        max_expert_id=int(ids.max()),
        max_router_sum_error=max_sum_error,
    )


def _resolve_layers(decoder: Any) -> list[Any]:
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise TypeError("pure NNX decoder must expose an unscanned layers collection")
    try:
        return [layers[index] for index in range(len(layers))]
    except (TypeError, KeyError, IndexError) as exc:
        raise TypeError(
            "decoder.layers must be an indexable unscanned collection"
        ) from exc


def _validate_input_shapes(
    token_ids: Any, positions: Any, segment_ids: Any, sequence_ids: Any
) -> None:
    if len(token_ids.shape) != 2:
        raise ValueError(
            f"token_ids must have shape [batch, sequence], got {token_ids.shape}"
        )
    if positions.shape != token_ids.shape or segment_ids.shape != token_ids.shape:
        raise ValueError(
            "decoder_positions and decoder_segment_ids must match token_ids"
        )
    if len(sequence_ids.shape) not in (1, 2):
        raise ValueError("sequence_ids must have shape [batch] or [batch, sequence]")
    if len(sequence_ids.shape) == 1 and sequence_ids.shape[0] != token_ids.shape[0]:
        raise ValueError(
            "one-dimensional sequence_ids must have one entry per batch row"
        )
    if len(sequence_ids.shape) == 2 and sequence_ids.shape != token_ids.shape:
        raise ValueError("two-dimensional sequence_ids must match token_ids")


def _first(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) else result


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _as_bfloat16(value: Any) -> Any:
    """Cast a JAX/NumPy-compatible array at the capture storage boundary."""

    jnp = import_module("jax.numpy")
    return value.astype(jnp.bfloat16)


def _array_value(value: Any, *, dtype: Any | None = None) -> Any:
    """Materialize an NNX parameter using the model's compute dtype."""

    jnp = import_module("jax.numpy")
    return jnp.asarray(_value(value), dtype=dtype)


def _require_attribute(value: Any, name: str, owner: str) -> Any:
    if not hasattr(value, name):
        raise TypeError(f"{owner} lacks required attribute {name}")
    return getattr(value, name)


def _first_attribute(value: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    raise TypeError(f"model config lacks all of: {', '.join(names)}")


def _dtype_name(value: Any) -> str:
    try:
        return str(np.dtype(value)).lower()
    except (TypeError, ValueError):
        text = str(value).lower()
        return "bfloat16" if "bfloat16" in text else text


def _all_finite_in_row_chunks(array: np.ndarray, chunk_rows: int = 4096) -> bool:
    """Check large BF16 activation shards without a shard-sized bool temporary."""

    for start in range(0, array.shape[0], chunk_rows):
        if not bool(np.all(np.isfinite(array[start : start + chunk_rows]))):
            return False
    return True
