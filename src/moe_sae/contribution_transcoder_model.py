"""Source-only Gemma prefix adapter for Experiment-3 transcoders.

The adapter executes blocks 1--14 and block-15 attention through its residual,
applies the exact routed-expert pre-FFN norm, compacts the scientific token
mask, and returns.  It never invokes the router, a routed expert, or the
shared expert.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import subprocess
from typing import Any, NamedTuple

from moe_sae.config import Gemma4Config
from moe_sae.contribution_config import MAXTEXT_FORK_COMMIT
from moe_sae.model import (
    Gemma4PrefixAdapter,
    MaxTextUnavailableError,
    _as_bfloat16,
    _validate_input_shapes,
)


__all__ = [
    "Gemma4RoutedInputPrefixAdapter",
    "RoutedInputCaptureBatch",
]


class RoutedInputCaptureBatch(NamedTuple):
    """Device-side routed-input rows; a JAX-compatible pytree."""

    routed_input: Any
    sequence_id: Any
    position: Any

    @property
    def rows(self) -> int:
        return int(self.routed_input.shape[0])


class Gemma4RoutedInputPrefixAdapter(Gemma4PrefixAdapter):
    """Capture only the exact input consumed by block-15 routed experts."""

    @classmethod
    def from_maxtext(
        cls,
        model: Any,
        config: Gemma4Config | None = None,
        *,
        maxtext_root: str | Path,
    ) -> "Gemma4RoutedInputPrefixAdapter":
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
                "cannot verify the Experiment-3 MaxText fork checkout"
            ) from error
        if head != MAXTEXT_FORK_COMMIT or dirty:
            raise MaxTextUnavailableError(
                "Experiment-3 capture requires the exact clean pinned MaxText fork"
            )
        try:
            maxtext = import_module("maxtext")
            import_module("jax")
        except (ImportError, ModuleNotFoundError) as error:
            raise MaxTextUnavailableError(
                "Experiment-3 capture requires the pinned MaxText/JAX environment"
            ) from error
        if not Path(maxtext.__file__).resolve().is_relative_to(root):
            raise MaxTextUnavailableError(
                "imported MaxText package does not come from maxtext_root"
            )
        return cls(model, config)

    @classmethod
    def from_installed_maxtext(
        cls,
        model: Any,
        config: Gemma4Config | None = None,
    ) -> "Gemma4RoutedInputPrefixAdapter":
        try:
            import_module("maxtext")
            import_module("jax")
            from moe_sae.runtime import installed_maxtext_commit
        except (ImportError, ModuleNotFoundError) as error:
            raise MaxTextUnavailableError(
                "Experiment-3 capture requires the locked MaxText TPU install"
            ) from error
        if installed_maxtext_commit() != MAXTEXT_FORK_COMMIT:
            raise MaxTextUnavailableError(
                "installed MaxText is not the pinned Experiment-2/3 fork commit"
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
    ) -> RoutedInputCaptureBatch:
        """Return compacted ``pre_feedforward_layernorm_2`` rows and stop."""

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
        routed_input = target.mlp.pre_feedforward_layernorm_2(residual)
        if routed_input.shape != residual.shape:
            raise ValueError(
                "routed pre-FFN norm must preserve shape "
                f"{residual.shape}, got {routed_input.shape}"
            )
        routed_input = _as_bfloat16(routed_input)

        batch, sequence = token_ids.shape
        rows = batch * sequence
        flat_input = routed_input.reshape((rows, self.config.hidden_size))
        flat_positions = decoder_positions.reshape((rows,))
        if len(sequence_ids.shape) == 1:
            flat_sequence_ids = sequence_ids.repeat(sequence)
        else:
            flat_sequence_ids = sequence_ids.reshape((rows,))

        if valid_token_indices is not None:
            indices = valid_token_indices.reshape((-1,))
            flat_input = flat_input[indices]
            flat_sequence_ids = flat_sequence_ids[indices]
            flat_positions = flat_positions[indices]

        if hasattr(flat_positions, "astype"):
            flat_positions = flat_positions.astype("uint16")
        return RoutedInputCaptureBatch(
            routed_input=flat_input,
            sequence_id=flat_sequence_ids,
            position=flat_positions,
        )
