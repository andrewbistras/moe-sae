"""SAE core (spec §5–§8, §11–§12): banks, BatchTopK engine, losses, optimizer.

This package is the single definition point for the `TrainBatch` contract
dataclass (docs/INTERFACES.md §"SAE training batch"); other modules import it
from `moe_sae.sae`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrainBatch:
    """One BatchTopK training batch (contract; all conditions consume this)."""

    c: torch.Tensor  # [B, 6, d] bf16 — gated contributions c_te, slot order
    experts: torch.Tensor  # [B, 6] int64 — routed expert ids (bank ids under A)
    gates: torch.Tensor  # [B, 6] float32
    token_type: torch.Tensor  # [B] uint8 — 0 text, 1 visual
    example_id: torch.Tensor  # [B] int32
    domain: torch.Tensor  # [B] uint8


# Re-exports (after TrainBatch so `from moe_sae.sae import TrainBatch` works
# from within these submodules during package import).
from moe_sae.sae.banks import GlobalSAE, RouterTiedBanks  # noqa: E402
from moe_sae.sae.batchtopk import ThetaEMA, ThetaEMAVector  # noqa: E402
from moe_sae.sae.conditions import (  # noqa: E402
    SAECondition,
    StepOutput,
    ThresholdedEncoding,
    build_condition,
    load_condition_from_run,
)
from moe_sae.sae.dead import DeadTracker  # noqa: E402
from moe_sae.sae.optim import SpecAdam  # noqa: E402

__all__ = [
    "TrainBatch",
    "RouterTiedBanks",
    "GlobalSAE",
    "ThetaEMA",
    "ThetaEMAVector",
    "DeadTracker",
    "SpecAdam",
    "SAECondition",
    "StepOutput",
    "ThresholdedEncoding",
    "build_condition",
    "load_condition_from_run",
]
