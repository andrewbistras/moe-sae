"""Run configuration. Frozen constants (spec §5–§11) are validated, not tunable.

Any value marked FROZEN below may not differ from the spec without a spec
version bump; `RunConfig.validate()` enforces this for scientific runs
(`pipeline=False`). The 1M pipeline-validation run (spec §10) uses
f=1024, k̄=32 and is exempt from the width/sparsity checks only.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from moe_sae.model_loading import NUM_ROUTED_EXPERTS, TOP_K_ROUTED

CONDITIONS = ("A", "B", "C", "D", "E", "F")

# FROZEN (spec §9, §6, §8, §11)
BATCH_TOKENS = 4096
THETA_EMA_DECAY = 0.999
DEAD_WINDOW_BANK_TOKENS = 250_000
AUX_TOKEN_BUDGET = 512
AUX_ALPHA = 1.0 / 32.0
ADAM_BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0
LR = 3e-4
LR_MIN = 3e-5
WARMUP_STEPS = 1000

# Discovery-scale frozen widths (spec §5, §12)
DISCOVERY_F = 4096
DISCOVERY_K_BAR = 64
PIPELINE_F = 1024
PIPELINE_K_BAR = 32

CONDITION_C_SALT = 0xC0FFEEC  # fixed, never seed-dependent (contract)


def global_width(condition: str, f: int) -> int:
    """Dictionary width per condition. A/C/D/E: per-bank f × 64 banks.
    B: compute-matched global (6f). F: capacity-matched global (64f)."""
    if condition in ("A", "C", "D", "E"):
        return f
    if condition == "B":
        return TOP_K_ROUTED * f
    if condition == "F":
        return NUM_ROUTED_EXPERTS * f
    raise ValueError(condition)


@dataclass
class RunConfig:
    run_name: str
    condition: str  # A..F
    seed: int
    f: int  # per-bank width for A/C/D/E; the *per-bank-equivalent* f for B/F
    k_bar: int
    total_tokens: int
    capture_root: str
    stats_path: str  # output of stats pass (mu_e etc.)
    out_root: str = "runs"
    pipeline: bool = False  # True only for the 1M plumbing run (spec §10)

    # FROZEN — present so they are echoed into run artifacts; validate() checks.
    batch_tokens: int = BATCH_TOKENS
    lr: float = LR
    lr_min: float = LR_MIN
    warmup_steps: int = WARMUP_STEPS
    adam_beta1: float = ADAM_BETAS[0]
    adam_beta2: float = ADAM_BETAS[1]
    grad_clip: float = GRAD_CLIP
    aux_alpha: float = AUX_ALPHA
    aux_token_budget: int = AUX_TOKEN_BUDGET
    dead_window: int = DEAD_WINDOW_BANK_TOKENS
    theta_ema_decay: float = THETA_EMA_DECAY

    # engineering knobs (non-scientific)
    device: str = "auto"
    log_every: int = 10
    ckpt_every_steps: int = 2000
    compute_dtype: str = "bfloat16"

    @property
    def data_seed(self) -> int:
        return 1000 + self.seed

    @property
    def n_steps(self) -> int:
        return self.total_tokens // self.batch_tokens

    @property
    def out_dir(self) -> Path:
        return Path(self.out_root) / self.run_name

    def validate(self) -> None:
        assert self.condition in CONDITIONS, self.condition
        assert self.batch_tokens == BATCH_TOKENS, "batch size is frozen (spec §9)"
        assert self.theta_ema_decay == THETA_EMA_DECAY, "θ EMA frozen (spec §6)"
        assert self.dead_window == DEAD_WINDOW_BANK_TOKENS, "dead window frozen (spec §8)"
        assert self.aux_token_budget == AUX_TOKEN_BUDGET, "aux budget frozen (spec §8)"
        assert abs(self.aux_alpha - AUX_ALPHA) < 1e-12, "aux α frozen (spec §8)"
        assert (self.adam_beta1, self.adam_beta2) == ADAM_BETAS, "Adam βs frozen (spec §11)"
        assert self.grad_clip == GRAD_CLIP, "grad clip frozen (spec §11)"
        assert self.lr == LR and self.lr_min == LR_MIN and self.warmup_steps == WARMUP_STEPS, (
            "LR schedule frozen (spec §11)"
        )
        if not self.pipeline:
            assert self.f == DISCOVERY_F and self.k_bar == DISCOVERY_K_BAR, (
                "scientific runs use f=4096, k̄=64 (spec §5/§6); "
                "set pipeline=True only for the 1M plumbing run"
            )

    # ---- serialization ----
    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(dataclasses.asdict(self), sort_keys=False))

    @staticmethod
    def from_yaml(path: str | Path) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return RunConfig(**raw)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)
