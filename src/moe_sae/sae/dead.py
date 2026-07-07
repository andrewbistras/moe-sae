"""Bank-relative dead-latent tracking (spec §8 frozen definition; contract
docs/INTERFACES.md §Deadness).

Latent j of bank e is *dead* iff it has not been selected by the main BatchTopK
within the most recent `window` tokens routed to bank e (bank-relative), AND
bank e has seen at least a full window (deadness is undefined — treated as
alive — before that). Only main-TopK selection resets a latent; aux-loss
selection never does (contract).

For global conditions (B/F) the single dictionary is one bank whose token
count advances by B per batch; the window stays 250,000 tokens routed to the
bank.
"""

from __future__ import annotations

import torch

from moe_sae.train.config import DEAD_WINDOW_BANK_TOKENS


class DeadTracker:
    """Per-bank token clocks + last-fire stamps, all state as int64 tensors."""

    def __init__(self, n_banks: int, f: int, window: int = DEAD_WINDOW_BANK_TOKENS):
        self.n_banks = n_banks
        self.f = f
        self.window = window
        self.tokens_seen = torch.zeros(n_banks, dtype=torch.int64)
        # Stamp of tokens_seen_e at latent j's last main-TopK selection.
        # Init 0 == "never fired": dead exactly when the bank clock passes the
        # window, consistent with the pre-window alive rule.
        self.last_fire = torch.zeros(n_banks, f, dtype=torch.int64)

    def update(self, routed_counts: torch.Tensor, fired_mask: torch.Tensor) -> None:
        """Advance bank clocks by this batch's routed loads, then stamp the
        latents selected by the *main* BatchTopK this batch.

        routed_counts: [n_banks] int — tokens routed to each bank this batch.
        fired_mask: [n_banks, f] bool — latents selected by main TopK.
        """
        assert routed_counts.shape == (self.n_banks,)
        assert fired_mask.shape == (self.n_banks, self.f)
        self.tokens_seen += routed_counts.to(torch.int64).cpu()
        stamp = self.tokens_seen.unsqueeze(1).expand_as(self.last_fire)
        self.last_fire = torch.where(fired_mask.cpu(), stamp, self.last_fire)

    def dead_mask(self) -> torch.Tensor:
        """[n_banks, f] bool. Dead iff age > window AND bank saw ≥ window."""
        age = self.tokens_seen.unsqueeze(1) - self.last_fire
        seen_full_window = self.tokens_seen.unsqueeze(1) >= self.window
        return (age > self.window) & seen_full_window

    def n_dead(self) -> int:
        return int(self.dead_mask().sum().item())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "tokens_seen": self.tokens_seen.clone(),
            "last_fire": self.last_fire.clone(),
            "window": torch.tensor(self.window, dtype=torch.int64),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        assert state["last_fire"].shape == (self.n_banks, self.f)
        self.tokens_seen = state["tokens_seen"].clone().to(torch.int64)
        self.last_fire = state["last_fire"].clone().to(torch.int64)
        self.window = int(state["window"].item())
