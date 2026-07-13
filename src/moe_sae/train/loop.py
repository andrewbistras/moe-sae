"""Training loop (spec §10–§11): schedule, step, logging, artifacts, resume.

Per step: `condition.training_step` → `loss.backward()` → `SpecAdam.step(lr)`
→ θ-EMA update → DeadTracker update. LR schedule: linear warmup over
`cfg.warmup_steps` then cosine decay lr → lr_min over the remaining steps
(spec §11).

Artifacts (contract §Run artifact layout): config.yaml echo, metrics.jsonl,
theta_trajectory.npy, checkpoints/{step}.safetensors, and final/
{theta_star.json, bank_loads.npy, dead_final.npy, checkpoint.safetensors}.

Ops additions (docs/OPS.md; all frozen-math-neutral — no training-step
number changes, no batch reordering):

- **Bitwise resume** (`resume=True`): restore the latest
  checkpoints/{k}.safetensors (params, θ-EMA, DeadTracker) plus its
  {k}.aux.npz sidecar (FP32 Adam moments + step count, cumulative
  bank_loads, θ trajectory — OPS §4: anything excluded from the checkpoint
  is a resume bug), then fast-forward the deterministic BatchComposer by k
  batches. The fast-forward is the ONLY stream interaction, so the resumed
  trajectory is bit-identical to an uninterrupted run.
- **metrics.jsonl truncation**: on resume, lines with step >= k are dropped
  before appending — a resumed run never duplicates a step.
- **Anomaly guard**: non-finite loss/grad_norm (or grad_norm above the
  optional ops-only `cfg.spike_threshold`) → emergency checkpoint +
  run-local logbook entry + RuntimeError. HALT, never skip: dropping or
  retrying a batch would desynchronize the frozen matched data order
  (spec §10; OPS §2 spike playbook).
- **MetricsSink / ops sidecar**: the canonical metrics.jsonl is written by
  MetricsSink in jsonl_only mode (byte-identical to the registered format);
  MFU + goodput numbers go to a separate ops_metrics.jsonl (+ TensorBoard
  under tb/ when available) so the registered artifact never changes.
- **A1 aux diagnostics (v0.1.2 addendum, committee-required)**: every log
  step records aux/primary loss ratio, mean aux-selected per token, fraction
  of aux-selected also main-selected (from StepOutput), and the aux/main
  gradient-norm ratio via a SECOND BACKWARD of the aux term only — run
  exclusively at log steps and costing roughly one extra backward there
  (~10% of a log step; `aux_grad_norm_ratio`). Cumulative per-bank
  aux-selection counts are accumulated across the run, persisted in each
  {step}.aux.npz sidecar (restored on resume) and finalized as
  final/aux_bank_counts.npy.

Decisions (recorded):
- Adam moments/step count live in {step}.aux.npz NEXT TO the .safetensors,
  not inside it: the checkpoint format is read by load_condition_from_run
  across the eval/analysis stack and stays untouched; the sidecar is
  additive ops state required by the OPS §4 bitwise-resume rule.
- resume=True on a completed run (final/checkpoint.safetensors exists) is a
  logged no-op, mirroring the capture driver ("a completed capture is a
  no-op"); durable state present WITHOUT resume is a RuntimeError, also
  mirroring capture.
- Anomaly checkpoints are named anomaly_{step}.safetensors so the numeric
  latest-checkpoint scan (and load_condition_from_run) never selects them.
  Loss is checked BEFORE backward, so a NaN-loss emergency checkpoint holds
  clean pre-update parameters; a NaN that first appears in the gradients is
  only detectable after `opt.step` (SpecAdam is frozen and cannot be
  intercepted mid-step), so that emergency checkpoint is forensic and the
  halt message directs auditors to the last periodic checkpoint.
- Anomaly/resume events go to a run-local logbook (out_dir/logs/LOGBOOK.md)
  so the record travels with the run artifacts.
- wall_time restarts at the resume moment; it is the one legitimately
  nondeterministic metrics field and is excluded from the bitwise contract
  (the OPS §4 drill compares metrics.jsonl minus wall_time).
- `spike_threshold` is read via getattr (default None = off): it is an
  ops-only knob, deliberately not a RunConfig field — the frozen config
  schema and its YAML echo stay exactly as registered.

`train` deliberately does NOT call `cfg.validate()`: validation of frozen
constants belongs to the entry-point script (scripts/run_train.py); tests
drive the loop with tiny non-frozen configs through the identical code path.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from moe_sae.sae import TrainBatch
from moe_sae.sae.conditions import SAECondition
from moe_sae.sae.optim import SpecAdam
from moe_sae.train.config import RunConfig
from moe_sae.util.manifest import logbook_append
from moe_sae.util.metrics import (
    GoodputTracker,
    MetricsSink,
    MFUMeter,
    flops_per_step_estimate,
    load_dispersion_stats,
)

OPS_METRICS_NAME = "ops_metrics.jsonl"
RUN_LOGBOOK = ("logs", "LOGBOOK.md")


def lr_at_step(step: int, cfg: RunConfig) -> float:
    """Spec §11: linear warmup to cfg.lr over warmup_steps, then cosine decay
    to cfg.lr_min at cfg.n_steps. `step` is zero-indexed."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    span = max(1, cfg.n_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / span)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# Resume plumbing (shared with moe_sae.tpu.train_jax)                          #
# --------------------------------------------------------------------------- #


def run_logbook_path(out_dir: str | Path) -> Path:
    """Run-local logbook (see module Decisions)."""
    return Path(out_dir).joinpath(*RUN_LOGBOOK)


def latest_checkpoint(ckpt_dir: str | Path) -> tuple[int, Path] | None:
    """Highest-step numeric checkpoints/{step}.safetensors, or None.
    Non-numeric stems (e.g. anomaly_{step}.safetensors) are ignored."""
    candidates: list[tuple[int, Path]] = []
    ckpt_dir = Path(ckpt_dir)
    if ckpt_dir.is_dir():
        for p in ckpt_dir.glob("*.safetensors"):
            try:
                candidates.append((int(p.stem), p))
            except ValueError:
                continue
    return max(candidates) if candidates else None


def aux_path_for(ckpt_path: str | Path) -> Path:
    """{step}.aux.npz sidecar next to a checkpoints/{step}.safetensors."""
    ckpt_path = Path(ckpt_path)
    return ckpt_path.with_name(f"{ckpt_path.stem}.aux.npz")


def resolve_resume(out_dir: str | Path, resume: bool) -> tuple[int, Path | None, bool]:
    """Resume decision for a run directory → (start_step, ckpt_path, complete).

    Mirrors the capture-driver semantics (moe_sae.capture.driver):
    durable state without resume=True is a hard error; resume on a completed
    run is a no-op (complete=True); resume with no durable checkpoint starts
    fresh from step 0.
    """
    out_dir = Path(out_dir)
    final_ckpt = out_dir / "final" / "checkpoint.safetensors"
    latest = latest_checkpoint(out_dir / "checkpoints")
    has_state = (
        final_ckpt.exists() or latest is not None or (out_dir / "metrics.jsonl").exists()
    )
    if not resume:
        if has_state:
            raise RuntimeError(
                f"{out_dir} already contains run state; pass resume=True "
                "(--resume) to continue it, or choose a fresh out_dir."
            )
        return 0, None, False
    if final_ckpt.exists():
        return 0, None, True  # completed run: resume is a no-op
    if latest is None:
        return 0, None, False  # nothing durable yet: fresh start
    return latest[0], latest[1], False


def restore_condition(condition: SAECondition, ckpt_path: str | Path) -> int:
    """Restore params + θ-EMA + DeadTracker from a checkpoint into an
    already-built condition (in place, so callers keep their reference and
    device placement). Returns the checkpoint's step."""
    restored, _, extras = SAECondition.load_checkpoint(ckpt_path)
    assert restored.condition == condition.condition, (
        restored.condition,
        condition.condition,
    )
    assert (restored.f, restored.k_bar) == (condition.f, condition.k_bar)
    with torch.no_grad():
        for name in ("W_enc", "b_enc", "D", "mu"):
            dst = getattr(condition.sae, name)
            src = getattr(restored.sae, name)
            assert dst.shape == src.shape, (name, dst.shape, src.shape)
            dst.copy_(src.to(dst.device))
    condition.theta_ema.load_state_dict(restored.theta_ema.state_dict())
    if condition.theta_ema_vec is not None and restored.theta_ema_vec is not None:
        condition.theta_ema_vec.load_state_dict(restored.theta_ema_vec.state_dict())
    condition.dead_tracker.load_state_dict(restored.dead_tracker.state_dict())
    return extras["step"]


def truncate_metrics_jsonl(path: str | Path, start_step: int) -> int:
    """Drop jsonl lines with step >= start_step (and any torn final line from
    a mid-write kill), preserving the surviving lines byte-for-byte. Returns
    the number of dropped lines."""
    p = Path(path)
    if not p.exists():
        return 0
    kept: list[str] = []
    dropped = 0
    for raw in p.read_text().splitlines(keepends=True):
        try:
            step = json.loads(raw)["step"]
        except (json.JSONDecodeError, KeyError, TypeError):
            dropped += 1  # torn write from the kill; the step will be re-run
            continue
        if step < start_step:
            kept.append(raw)
        else:
            dropped += 1
    p.write_text("".join(kept))
    return dropped


def save_aux(
    path: str | Path,
    bank_loads: torch.Tensor,
    theta_trajectory: list[float],
    extra_arrays: dict,
) -> None:
    """Write the {step}.aux.npz resume sidecar (see module Decisions)."""
    np.savez(
        str(path),
        bank_loads=bank_loads.detach().cpu().numpy(),
        theta_trajectory=np.asarray(theta_trajectory, dtype=np.float64),
        **extra_arrays,
    )


def aux_grad_norm_ratio(aux_term: torch.Tensor | None, params: list) -> float | None:
    """A1 diagnostic (ii): aux/main gradient-norm ratio via a second backward
    of the aux term ONLY (called by the loops at log steps; the extra backward
    costs roughly one more backward pass, ~10% of a log step's wall time —
    documented cost, committee-required).

    Call AFTER `loss.backward()` (with the graph retained): `p.grad` then holds
    g_total = g_main + g_aux; autograd.grad of the α-scaled aux term gives
    g_aux, and g_main = g_total − g_aux elementwise. Returns
    ‖g_aux‖ / ‖g_main‖ (None when the aux term carried no graph, i.e. nothing
    was dead and L_aux == 0 exactly)."""
    if aux_term is None or not aux_term.requires_grad:
        return None
    g_aux = torch.autograd.grad(aux_term, params, allow_unused=True)
    aux_sq = torch.zeros((), dtype=torch.float32)
    main_sq = torch.zeros((), dtype=torch.float32)
    for p, ga in zip(params, g_aux):
        gt = p.grad.float() if p.grad is not None else None
        ga32 = ga.float() if ga is not None else None
        if ga32 is not None:
            aux_sq += ga32.pow(2).sum().cpu()
        if gt is not None:
            gm = gt - ga32 if ga32 is not None else gt
            main_sq += gm.pow(2).sum().cpu()
    main_norm = float(main_sq.sqrt().item())
    aux_norm = float(aux_sq.sqrt().item())
    return aux_norm / max(main_norm, 1e-30)


def load_aux(ckpt_path: str | Path) -> "np.lib.npyio.NpzFile":
    """Load the aux sidecar for a checkpoint; loud error if absent (a
    checkpoint without its optimizer state cannot resume bitwise, OPS §4)."""
    p = aux_path_for(ckpt_path)
    if not p.exists():
        raise RuntimeError(
            f"checkpoint {ckpt_path} has no aux sidecar {p.name}: cannot "
            "resume bitwise without Adam moments/step count (docs/OPS.md §4)."
        )
    return np.load(p)


def _adam_state_arrays(opt: SpecAdam) -> dict:
    """FP32 Adam moments + step count as npz-ready arrays. Iterates
    `opt._params()` — the deterministic group order SpecAdam itself uses."""
    arrays: dict = {"adam_t": np.int64(opt.t)}
    for i, (p, _) in enumerate(opt._params()):
        st = opt.state[id(p)]
        arrays[f"adam_m_{i}"] = st["exp_avg"].detach().cpu().numpy()
        arrays[f"adam_v_{i}"] = st["exp_avg_sq"].detach().cpu().numpy()
    return arrays


def _load_adam_state(opt: SpecAdam, aux) -> None:
    opt.t = int(aux["adam_t"])
    with torch.no_grad():
        for i, (p, _) in enumerate(opt._params()):
            st = opt.state[id(p)]
            st["exp_avg"].copy_(torch.from_numpy(aux[f"adam_m_{i}"].copy()))
            st["exp_avg_sq"].copy_(torch.from_numpy(aux[f"adam_v_{i}"].copy()))


# --------------------------------------------------------------------------- #
# Anomaly guard (OPS §2: HALT, never skip)                                     #
# --------------------------------------------------------------------------- #


def halt_anomaly(
    out_dir: str | Path,
    step: int,
    reason: str,
    detail: dict,
    save_ckpt: Callable[[Path], None],
) -> None:
    """Emergency checkpoint + logbook entry + RuntimeError. Skipping or
    retrying the batch is never permitted (spec §10 matched data order)."""
    out_dir = Path(out_dir)
    ckpt = out_dir / "checkpoints" / f"anomaly_{step}.safetensors"
    save_ckpt(ckpt)
    logbook_append(
        "anomaly",
        {"reason": reason, "step": step, "checkpoint": str(ckpt), **detail},
        path=run_logbook_path(out_dir),
    )
    raise RuntimeError(
        f"anomaly halt at step {step}: {reason} {detail}. Emergency checkpoint "
        f"at {ckpt}. HALT-never-skip (docs/OPS.md §2) — audit, then resume "
        "with --resume from the last periodic checkpoint."
    )


def open_sinks(out_dir: str | Path) -> tuple[MetricsSink, MetricsSink]:
    """(canonical, ops) sinks: canonical is jsonl_only (byte-identical
    registered metrics.jsonl); ops points at ops_metrics.jsonl and mirrors to
    TensorBoard under out_dir/tb when available."""
    out_dir = Path(out_dir)
    canonical = MetricsSink(out_dir, jsonl_only=True)
    ops = MetricsSink(out_dir, enable_tensorboard=True)
    ops.metrics_path = out_dir / OPS_METRICS_NAME  # sidecar, not the registered file
    return canonical, ops


# --------------------------------------------------------------------------- #
# The loop                                                                     #
# --------------------------------------------------------------------------- #


def train(
    cfg: RunConfig,
    condition: SAECondition,
    batch_iter: Iterator[TrainBatch],
    out_dir: str | Path,
    *,
    resume: bool = False,
    peak_flops: float | None = None,
) -> None:
    out_dir = Path(out_dir)
    start_step, ckpt_path, complete = resolve_resume(out_dir, resume)
    if complete:
        logbook_append(
            "resume_noop",
            {"out_dir": str(out_dir), "note": "final checkpoint exists; run complete"},
            path=run_logbook_path(out_dir),
        )
        return
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "final").mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config.yaml")

    torch.manual_seed(cfg.seed)

    opt = SpecAdam(
        condition.optim_groups(),
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        clip=cfg.grad_clip,
    )
    theta_ema = condition.theta_ema
    dead_tracker = condition.dead_tracker

    bank_loads = None  # cumulative host-expert routed tokens, [64] int64
    theta_trajectory: list[float] = []
    # A1 diagnostic (v): cumulative per-bank aux-selection counts, persisted
    # in the checkpoint aux sidecar (and final/aux_bank_counts.npy).
    aux_bank_counts = torch.zeros(condition.n_dict_banks, dtype=torch.int64)
    gp = GoodputTracker()

    if ckpt_path is not None:
        k = restore_condition(condition, ckpt_path)
        assert k == start_step, (k, start_step)
        aux = load_aux(ckpt_path)
        _load_adam_state(opt, aux)
        bank_loads = torch.from_numpy(aux["bank_loads"].copy())
        theta_trajectory = [float(x) for x in aux["theta_trajectory"]]
        if "aux_bank_counts" in aux.files:
            aux_bank_counts = torch.from_numpy(aux["aux_bank_counts"].copy())
        with gp.data_wait():
            # Composer fast-forward: the ONLY stream interaction on resume —
            # consuming k batches reproduces the exact stream position.
            for _ in range(start_step):
                next(batch_iter)
        logbook_append(
            "resume",
            {"step": start_step, "checkpoint": str(ckpt_path)},
            path=run_logbook_path(out_dir),
        )
    if resume:
        truncate_metrics_jsonl(out_dir / "metrics.jsonl", start_step)
        truncate_metrics_jsonl(out_dir / OPS_METRICS_NAME, start_step)

    meter = None
    if peak_flops is not None:
        meter = MFUMeter(
            flops_per_step_estimate(
                cfg.condition, cfg.batch_tokens, cfg.f, condition.d, cfg.k_bar
            ),
            peak_flops,
        )
    sink, ops_sink = open_sinks(out_dir)
    t_start = time.time()
    spike = getattr(cfg, "spike_threshold", None)  # ops-only knob, default off

    n_steps = cfg.n_steps
    try:
        for step in range(start_step, n_steps):
            t_iter = time.perf_counter()
            with gp.data_wait():
                batch = next(batch_iter)
            is_log_step = step % cfg.log_every == 0 or step == n_steps - 1
            with gp.compute():
                out = condition.training_step(batch, step)
                loss_val = float(out.loss.item())
                if not math.isfinite(loss_val):
                    halt_anomaly(
                        out_dir, step, "non-finite loss", {"loss": loss_val},
                        lambda p: condition.save_checkpoint(p, theta_ema, dead_tracker, step),
                    )
                # A1 (ii): second backward of the aux term ONLY at log steps
                # (see aux_grad_norm_ratio; ~10% extra cost on log steps).
                need_ratio = is_log_step and out.aux_term is not None
                out.loss.backward(retain_graph=need_ratio)
                aux_grad_ratio = None
                if is_log_step:
                    params = [p for p, _ in opt._params()]
                    aux_grad_ratio = aux_grad_norm_ratio(
                        out.aux_term if need_ratio else None, params
                    )
                lr = lr_at_step(step, cfg)
                grad_norm = opt.step(lr)
                opt.zero_grad()
                if not math.isfinite(grad_norm) or (
                    spike is not None and grad_norm > spike
                ):
                    reason = (
                        "non-finite grad_norm"
                        if not math.isfinite(grad_norm)
                        else f"grad_norm spike > {spike}"
                    )
                    halt_anomaly(
                        out_dir, step, reason,
                        {"grad_norm": grad_norm, "loss": loss_val},
                        lambda p: condition.save_checkpoint(p, theta_ema, dead_tracker, step),
                    )

                theta_ema.update(out.theta_b)
                if out.theta_b_per_bank is not None and condition.theta_ema_vec is not None:
                    condition.theta_ema_vec.update(out.theta_b_per_bank)  # A4 (D)
                dead_tracker.update(out.dict_route_counts, out.fired_mask)
                theta_trajectory.append(out.theta_b)
                if bank_loads is None:
                    bank_loads = out.bank_route_counts.clone().to(torch.int64)
                else:
                    bank_loads += out.bank_route_counts.to(torch.int64)
                if out.aux_bank_counts is not None:
                    aux_bank_counts += out.aux_bank_counts.to(torch.int64)

            dt = max(time.perf_counter() - t_iter, 1e-9)
            ops_stats = meter.update(dt) if meter is not None else {"steps_per_sec": 1.0 / dt}
            if is_log_step:
                scalars = out.scalars()
                scalars["aux_grad_norm_ratio"] = aux_grad_ratio
                sink.log(
                    step,
                    {
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "theta_ema": theta_ema.value,
                        "wall_time": time.time() - t_start,
                        **scalars,
                    },
                )
                # A19: per-batch load-dispersion diagnostics (max bank load,
                # Gini, zero-token-bank fraction) from bank_route_counts;
                # condition D also logs its allocated-budget distribution.
                a19 = load_dispersion_stats(out.bank_route_counts.numpy())
                if condition.condition == "D":
                    from moe_sae.sae.batchtopk import largest_remainder_budgets

                    budgets = largest_remainder_budgets(
                        out.bank_route_counts.float(), cfg.k_bar * out.batch_tokens
                    )
                    b19 = load_dispersion_stats(budgets.numpy(), prefix="budget")
                    a19 = {**a19, "budget_max": b19["budget_max"], "budget_gini": b19["budget_gini"]}
                ops_sink.log(
                    step,
                    {
                        **ops_stats,
                        **gp.report(),
                        # A1 diagnostics mirrored into the ops sidecar
                        "aux_primary_ratio": scalars["aux_primary_ratio"],
                        "aux_grad_norm_ratio": aux_grad_ratio,
                        "aux_mean_selected_per_token": scalars["aux_mean_selected_per_token"],
                        "aux_frac_also_main": scalars["aux_frac_also_main"],
                        **a19,
                    },
                )

            if (step + 1) % cfg.ckpt_every_steps == 0 and (step + 1) < n_steps:
                with gp.checkpoint():
                    ckpt = out_dir / "checkpoints" / f"{step + 1}.safetensors"
                    condition.save_checkpoint(ckpt, theta_ema, dead_tracker, step + 1)
                    save_aux(
                        aux_path_for(ckpt), bank_loads, theta_trajectory,
                        {
                            **_adam_state_arrays(opt),
                            "aux_bank_counts": aux_bank_counts.numpy(),
                        },
                    )
    finally:
        sink.close()
        ops_sink.close()

    # θ* frozen at end of training (spec §6, changelog #6). Condition D
    # additionally freezes the per-bank vector θ*_e (A4); the scalar stays as
    # the log-only convention.
    theta_star_record = {
        "theta_star": theta_ema.value,
        "decay": theta_ema.decay,
        "n_steps": n_steps,
        "condition": condition.condition,
        "seed": cfg.seed,
    }
    if condition.theta_ema_vec is not None:
        theta_star_record = {
            "theta_star_per_bank": condition.theta_ema_vec.values_list(),
            **theta_star_record,
        }
    (out_dir / "final" / "theta_star.json").write_text(
        json.dumps(theta_star_record, indent=2)
    )
    np.save(out_dir / "theta_trajectory.npy", np.asarray(theta_trajectory, dtype=np.float64))
    np.save(out_dir / "final" / "bank_loads.npy", bank_loads.numpy())
    np.save(out_dir / "final" / "dead_final.npy", dead_tracker.dead_mask().numpy())
    # A1 diagnostic (v): cumulative per-bank aux pressure over the whole run.
    np.save(out_dir / "final" / "aux_bank_counts.npy", aux_bank_counts.numpy())
    condition.save_checkpoint(
        out_dir / "final" / "checkpoint.safetensors", theta_ema, dead_tracker, n_steps
    )
