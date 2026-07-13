"""Bitwise-resume drill + anomaly-halt guard for both training loops
(docs/OPS.md §4 resume-must-be-bitwise; §2 HALT-never-skip; spec §10 matched
data order).

The drill: train 6 steps uninterrupted vs train, get killed after step 3
(mid-interval, past the step-3 checkpoint — the metrics line for step 3 must
be truncated on resume), resume, run to 6. metrics.jsonl (minus wall_time,
the one legitimately nondeterministic field), final checkpoint tensors, θ*,
bank_loads, theta_trajectory, and the dead-tracker state must be EXACTLY
equal — no tolerances anywhere.

Decision (documented, mirrors test_train_loop): tiny non-frozen dims
(batch_tokens=32, f=8) with `RunConfig.validate()` deliberately skipped —
frozen-constant validation belongs to the entry-point scripts; the loops are
exercised through their real code paths. compute_dtype=float32 so the
checkpoint round-trip is exact on both backends. The JAX drill pins
capacity=256 ≥ the 192 candidate slots per batch, so the adaptive-capacity
ratchet can never fire mid-drill (capacity restore is itself covered by the
aux sidecar assertions).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from moe_sae.capture.writer import ShardWriter, iter_shards
from moe_sae.sae import TrainBatch
from moe_sae.sae.conditions import build_condition
from moe_sae.train.config import RunConfig
from moe_sae.train.loop import train

E, F_BANK, D_MODEL, B, K_BAR = 64, 8, 16, 32, 4
N_STEPS = 6
KILL_AT = 4  # steps 0..3 complete; checkpoint at 3; step-3 metrics truncated

# The registered metrics.jsonl key order (loop.py / train_jax.py line format;
# aux_* diagnostics appended per the v0.1.2 addendum A1 — committee-required).
METRICS_KEYS = [
    "step", "lr", "grad_norm", "theta_ema", "wall_time",
    "loss", "l_local", "l_aggregate", "l_aux", "theta_b",
    "l0_mean", "l0_visual", "l0_text", "n_dead_now", "k_selected",
    "aux_primary_ratio", "aux_mean_selected_per_token", "aux_frac_also_main",
    "aux_grad_norm_ratio",
]


class SimulatedKill(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Synthetic data (duck-types the BatchComposer contract; deterministic in     #
# its own Generator, so consuming k batches reproduces the stream position)   #
# --------------------------------------------------------------------------- #


def write_capture(root: Path, n_per_pool: int = 64, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for pool in ("visual", "text"):
        w = ShardWriter(root, pool, shard_size=48)
        experts = torch.stack(
            [torch.randperm(E, generator=g)[:6] for _ in range(n_per_pool)]
        )
        w.add(
            c=torch.randn(n_per_pool, 6, D_MODEL, generator=g),
            experts=experts,
            gates=torch.rand(n_per_pool, 6, generator=g),
            example_id=torch.arange(n_per_pool, dtype=torch.int32),
            domain=torch.zeros(n_per_pool, dtype=torch.uint8),
        )
        w.close()


def batch_iter_from_capture(root: Path, batch_tokens: int, seed: int):
    pools = {}
    for pool in ("visual", "text"):
        shards = list(iter_shards(root, pool))
        pools[pool] = {k: torch.cat([s[k] for s in shards], dim=0) for k in shards[0]}
    rng = torch.Generator().manual_seed(seed)
    half = batch_tokens // 2
    while True:
        parts = {"token_type": []}
        for k in ("c", "experts", "gates", "example_id", "domain"):
            parts[k] = []
        for pool, flag in (("visual", 1), ("text", 0)):
            data = pools[pool]
            n = data["domain"].shape[0]
            idx = torch.randint(0, n, (half,), generator=rng)
            for k in ("c", "experts", "gates", "example_id", "domain"):
                parts[k].append(data[k][idx])
            parts["token_type"].append(torch.full((half,), flag, dtype=torch.uint8))
        yield TrainBatch(
            c=torch.cat(parts["c"]),
            experts=torch.cat(parts["experts"]).to(torch.int64),
            gates=torch.cat(parts["gates"]),
            token_type=torch.cat(parts["token_type"]),
            example_id=torch.cat(parts["example_id"]),
            domain=torch.cat(parts["domain"]),
        )


def killing_iter(it, kill_at: int):
    """Yield batches; raise SimulatedKill when batch `kill_at` is requested
    (the killer picks the moment — the loop dies mid-interval)."""
    i = 0
    while True:
        if i == kill_at:
            raise SimulatedKill(f"simulated SIGKILL before batch {i}")
        yield next(it)
        i += 1


def tiny_config(condition: str, tmp_path: Path, run_name: str) -> RunConfig:
    return RunConfig(
        run_name=run_name,
        condition=condition,
        seed=1,
        f=F_BANK,
        k_bar=K_BAR,
        total_tokens=B * N_STEPS,
        capture_root=str(tmp_path / "captures"),
        stats_path="unused-in-test",
        out_root=str(tmp_path / "runs"),
        pipeline=True,
        batch_tokens=B,  # NOT frozen-valid; validate() intentionally skipped
        compute_dtype="float32",
        log_every=1,
        ckpt_every_steps=3,
    )


def make_condition(cfg: RunConfig):
    g = torch.Generator().manual_seed(7)
    mu_e = torch.randn(E, D_MODEL, generator=g) * 0.1
    mu_route = torch.randn(D_MODEL, generator=g) * 0.1
    return build_condition(cfg.condition, cfg, mu_e, mu_route, cfg.seed)


def fresh_batches(cfg: RunConfig):
    return batch_iter_from_capture(Path(cfg.capture_root), cfg.batch_tokens, cfg.data_seed)


# --------------------------------------------------------------------------- #
# Exact-equality comparison                                                    #
# --------------------------------------------------------------------------- #


def read_metrics_minus_wall_time(out_dir: Path) -> list[dict]:
    lines = [json.loads(x) for x in (out_dir / "metrics.jsonl").read_text().splitlines()]
    for ln in lines:
        ln.pop("wall_time")
    return lines


def assert_runs_exactly_equal(ref: Path, res: Path) -> None:
    # metrics.jsonl minus wall_time: EXACT equality, every step present once
    m_ref = read_metrics_minus_wall_time(ref)
    m_res = read_metrics_minus_wall_time(res)
    assert [ln["step"] for ln in m_res] == list(range(N_STEPS))
    assert m_ref == m_res

    # final checkpoint: every tensor (params, mu, θ state, dead state) exact
    t_ref = load_file(ref / "final" / "checkpoint.safetensors")
    t_res = load_file(res / "final" / "checkpoint.safetensors")
    assert set(t_ref) == set(t_res)
    for key in t_ref:
        assert torch.equal(t_ref[key], t_res[key]), f"final checkpoint {key} differs"

    # θ*, bank_loads, dead state, θ trajectory: exact
    assert (ref / "final" / "theta_star.json").read_text() == (
        res / "final" / "theta_star.json"
    ).read_text()
    for rel in (
        "final/bank_loads.npy",
        "final/dead_final.npy",
        "final/aux_bank_counts.npy",  # A1: cumulative aux pressure must resume
        "theta_trajectory.npy",
    ):
        a, b = np.load(ref / rel), np.load(res / rel)
        assert np.array_equal(a, b), f"{rel} differs"


# --------------------------------------------------------------------------- #
# The drill — torch loop                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("condition", ["A", "B"])  # routed banks + global SAE
def test_resume_bitwise_torch(condition, tmp_path):
    write_capture(tmp_path / "captures")

    cfg_ref = tiny_config(condition, tmp_path, f"ref_{condition}")
    train(cfg_ref, make_condition(cfg_ref), fresh_batches(cfg_ref), cfg_ref.out_dir)

    cfg = tiny_config(condition, tmp_path, f"int_{condition}")
    with pytest.raises(SimulatedKill):
        train(cfg, make_condition(cfg), killing_iter(fresh_batches(cfg), KILL_AT), cfg.out_dir)
    # killed past the step-3 checkpoint, with a step-3 metrics line on disk
    assert (cfg.out_dir / "checkpoints" / "3.safetensors").exists()
    assert (cfg.out_dir / "checkpoints" / "3.aux.npz").exists()
    assert read_metrics_minus_wall_time(cfg.out_dir)[-1]["step"] == KILL_AT - 1

    # resume: fresh loop over the same config/out_dir, fresh composer
    train(cfg, make_condition(cfg), fresh_batches(cfg), cfg.out_dir, resume=True)
    assert_runs_exactly_equal(cfg_ref.out_dir, cfg.out_dir)


# --------------------------------------------------------------------------- #
# The drill — jax loop                                                         #
# --------------------------------------------------------------------------- #


def test_resume_bitwise_jax(tmp_path):
    jax = pytest.importorskip("jax")
    from moe_sae.tpu.train_jax import train_jax

    device = jax.devices("cpu")[0]
    write_capture(tmp_path / "captures")

    cfg_ref = tiny_config("A", tmp_path, "jref_A")
    train_jax(
        cfg_ref, make_condition(cfg_ref), fresh_batches(cfg_ref), cfg_ref.out_dir,
        capacity=256, device=device,
    )

    cfg = tiny_config("A", tmp_path, "jint_A")
    with pytest.raises(SimulatedKill):
        train_jax(
            cfg, make_condition(cfg), killing_iter(fresh_batches(cfg), KILL_AT),
            cfg.out_dir, capacity=256, device=device,
        )
    aux = np.load(cfg.out_dir / "checkpoints" / "3.aux.npz")
    assert int(aux["capacity"]) == 256 and int(aux["adam_t"]) == 3

    train_jax(
        cfg, make_condition(cfg), fresh_batches(cfg), cfg.out_dir,
        capacity=None,  # must be restored from the aux sidecar
        device=device, resume=True,
    )
    assert_runs_exactly_equal(cfg_ref.out_dir, cfg.out_dir)


# --------------------------------------------------------------------------- #
# Resume-flag semantics (mirrors capture)                                      #
# --------------------------------------------------------------------------- #


def test_dirty_out_dir_without_resume_is_an_error(tmp_path):
    write_capture(tmp_path / "captures")
    cfg = tiny_config("A", tmp_path, "dirty")
    with pytest.raises(SimulatedKill):
        train(cfg, make_condition(cfg), killing_iter(fresh_batches(cfg), 2), cfg.out_dir)
    with pytest.raises(RuntimeError, match="--resume"):
        train(cfg, make_condition(cfg), fresh_batches(cfg), cfg.out_dir)


def test_resume_on_completed_run_is_noop(tmp_path):
    write_capture(tmp_path / "captures")
    cfg = tiny_config("A", tmp_path, "done")
    train(cfg, make_condition(cfg), fresh_batches(cfg), cfg.out_dir)
    before = (cfg.out_dir / "metrics.jsonl").read_bytes()
    final_before = (cfg.out_dir / "final" / "checkpoint.safetensors").read_bytes()

    train(cfg, make_condition(cfg), fresh_batches(cfg), cfg.out_dir, resume=True)

    assert (cfg.out_dir / "metrics.jsonl").read_bytes() == before
    assert (cfg.out_dir / "final" / "checkpoint.safetensors").read_bytes() == final_before
    assert "resume_noop" in (cfg.out_dir / "logs" / "LOGBOOK.md").read_text()


# --------------------------------------------------------------------------- #
# Anomaly guard: HALT, never skip                                              #
# --------------------------------------------------------------------------- #


def test_anomaly_nan_loss_halts_torch(tmp_path):
    write_capture(tmp_path / "captures")
    cfg = tiny_config("A", tmp_path, "nan")
    cond = make_condition(cfg)
    real_step = cond.training_step

    def poisoned(batch, step):
        out = real_step(batch, step)
        if step == 2:
            out.loss = torch.tensor(float("nan"))
        return out

    cond.training_step = poisoned
    with pytest.raises(RuntimeError, match="anomaly halt at step 2"):
        train(cfg, cond, fresh_batches(cfg), cfg.out_dir)

    assert (cfg.out_dir / "checkpoints" / "anomaly_2.safetensors").exists()
    logbook = (cfg.out_dir / "logs" / "LOGBOOK.md").read_text()
    assert "anomaly" in logbook and "non-finite loss" in logbook
    # steps 0..1 were healthy and logged; step 2 never produced a line
    assert read_metrics_minus_wall_time(cfg.out_dir)[-1]["step"] == 1


def test_spike_threshold_halts_torch(tmp_path):
    write_capture(tmp_path / "captures")
    cfg = tiny_config("A", tmp_path, "spike")
    cfg.spike_threshold = 0.0  # ops-only knob: any positive grad norm trips it
    with pytest.raises(RuntimeError, match="grad_norm spike"):
        train(cfg, make_condition(cfg), fresh_batches(cfg), cfg.out_dir)
    assert (cfg.out_dir / "checkpoints" / "anomaly_0.safetensors").exists()


def test_anomaly_nan_loss_halts_jax(tmp_path, monkeypatch):
    jax = pytest.importorskip("jax")
    from moe_sae.tpu.train_jax import JaxRun, train_jax

    write_capture(tmp_path / "captures")
    cfg = tiny_config("A", tmp_path, "jnan")
    real_step = JaxRun.training_step

    def poisoned(self, batch, lr, **kwargs):
        out, hrc = real_step(self, batch, lr, **kwargs)
        out["loss"] = float("nan")
        return out, hrc

    monkeypatch.setattr(JaxRun, "training_step", poisoned)
    with pytest.raises(RuntimeError, match="anomaly halt at step 0"):
        train_jax(
            cfg, make_condition(cfg), fresh_batches(cfg), cfg.out_dir,
            capacity=256, device=jax.devices("cpu")[0],
        )
    assert (cfg.out_dir / "checkpoints" / "anomaly_0.safetensors").exists()
    assert "non-finite loss" in (cfg.out_dir / "logs" / "LOGBOOK.md").read_text()


# --------------------------------------------------------------------------- #
# MetricsSink: canonical lines byte-identical to the registered format         #
# --------------------------------------------------------------------------- #


def test_metrics_lines_byte_identical_to_registered_format(tmp_path):
    """Every metrics.jsonl line must be exactly what the pre-MetricsSink loop
    wrote: `json.dumps({"step": step, <registered key order>}) + "\\n"`."""
    write_capture(tmp_path / "captures")
    cfg = tiny_config("A", tmp_path, "fmt")
    train(cfg, make_condition(cfg), fresh_batches(cfg), cfg.out_dir)

    raw = (cfg.out_dir / "metrics.jsonl").read_text().splitlines()
    assert len(raw) == N_STEPS  # log_every=1
    for line in raw:
        parsed = json.loads(line)
        assert list(parsed) == METRICS_KEYS
        assert line == json.dumps(parsed)  # byte-identical legacy json.dumps format
        # aux_grad_norm_ratio is null when no latent is dead (A1)
        assert all(
            parsed[k] is None or math.isfinite(parsed[k]) for k in METRICS_KEYS
        )

    # ops numbers live in the SEPARATE sidecar, never in metrics.jsonl
    ops = [
        json.loads(x)
        for x in (cfg.out_dir / "ops_metrics.jsonl").read_text().splitlines()
    ]
    assert [o["step"] for o in ops] == list(range(N_STEPS))
    assert all("load_max" not in json.loads(line) for line in raw)  # A19 is ops-only
    for o in ops:
        assert "steps_per_sec" in o and "compute_frac" in o and "total_seconds" in o
        # A19 load-dispersion diagnostics live in the ops sidecar
        assert "load_max" in o and "load_gini" in o and "load_zero_frac" in o
        assert o["load_max"] > 0 and 0.0 <= o["load_gini"] < 1.0
        assert 0.0 <= o["load_zero_frac"] <= 1.0
        assert "budget_max" not in o  # budget stats are condition-D-only


def test_ops_sidecar_condition_d_budget_stats(tmp_path):
    """A19: condition D additionally logs its allocated-budget distribution
    (max/Gini) in the ops sidecar; no training-math change."""
    write_capture(tmp_path / "captures")
    cfg = tiny_config("D", tmp_path, "dbudget")
    train(cfg, make_condition(cfg), fresh_batches(cfg), cfg.out_dir)
    ops = [
        json.loads(x)
        for x in (cfg.out_dir / "ops_metrics.jsonl").read_text().splitlines()
    ]
    assert ops
    for o in ops:
        assert o["budget_max"] > 0
        assert 0.0 <= o["budget_gini"] < 1.0
        assert "load_max" in o and "load_gini" in o and "load_zero_frac" in o
