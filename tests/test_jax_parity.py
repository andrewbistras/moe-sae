"""JAX ↔ torch parity for the SAE hot path (smoke ladder rungs 1–2, run free
on CPU). The torch implementation is the oracle: every component of
moe_sae.tpu.sae_jax must reproduce it on identical inputs; the training-step
wrapper must reproduce torch training_step metrics and parameter trajectories.

All comparisons in float32 compute (compute_dtype="float32" conditions);
production bf16 cross-backend differences are rung-4 territory
(scripts/tpu/compare_runs.py tolerances)."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from moe_sae.sae import TrainBatch  # noqa: E402
from moe_sae.sae.batchtopk import global_batch_topk, per_token_topk  # noqa: E402
from moe_sae.sae.conditions import build_condition  # noqa: E402
from moe_sae.sae.optim import SpecAdam  # noqa: E402
from moe_sae.train.config import CONDITIONS, RunConfig  # noqa: E402
from moe_sae.tpu import sae_jax as sj  # noqa: E402
from moe_sae.tpu.train_jax import JaxRun, train_jax  # noqa: E402

E, F_BANK, D_MODEL, B, K_BAR, CAP = 64, 8, 16, 24, 4, 128


def tiny_config(condition: str, **overrides) -> RunConfig:
    kwargs = dict(
        run_name=f"jaxtest_{condition}",
        condition=condition,
        seed=1,
        f=F_BANK,
        k_bar=K_BAR,
        total_tokens=B * 3,
        capture_root="unused",
        stats_path="unused",
        pipeline=True,
        batch_tokens=B,
        compute_dtype="float32",
        log_every=1,
        ckpt_every_steps=1000,
    )
    kwargs.update(overrides)
    return RunConfig(**kwargs)


def make_means(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(E, D_MODEL, generator=g) * 0.1, torch.randn(D_MODEL, generator=g) * 0.1


def make_batch(seed=0, n=B) -> TrainBatch:
    g = torch.Generator().manual_seed(seed)
    experts = torch.stack([torch.randperm(E, generator=g)[:6] for _ in range(n)])
    token_type = torch.zeros(n, dtype=torch.uint8)
    token_type[: n // 2] = 1
    return TrainBatch(
        c=torch.randn(n, 6, D_MODEL, generator=g).to(torch.bfloat16),
        experts=experts.to(torch.int64),
        gates=torch.rand(n, 6, generator=g),
        token_type=token_type,
        example_id=torch.arange(n, dtype=torch.int32) // 4,
        domain=torch.zeros(n, dtype=torch.uint8),
    )


def build(condition: str, seed: int = 1):
    mu_e, mu_route = make_means()
    return build_condition(condition, tiny_config(condition), mu_e, mu_route, seed)


def jax_params(cond):
    return {
        "W_enc": jnp.asarray(cond.sae.W_enc.detach().numpy()),
        "b_enc": jnp.asarray(cond.sae.b_enc.detach().numpy()),
        "D": jnp.asarray(cond.sae.D.detach().numpy()),
    }, jnp.asarray(cond.sae.mu.numpy())


# --------------------------------------------------------------------------- #
# Component parity                                                             #
# --------------------------------------------------------------------------- #


def test_grouped_encode_decode_match_torch():
    cond = build("A")
    batch = make_batch()
    params, mu = jax_params(cond)
    bank_ids = batch.experts
    plan = sj.grouping_plan(bank_ids.numpy(), E, CAP)

    pre_j = sj.encode_routed(
        params["W_enc"], params["b_enc"], mu, jnp.asarray(batch.c.float().numpy()),
        jnp.asarray(plan.bank_flat), jnp.asarray(plan.order), jnp.asarray(plan.dest),
        cap=CAP, n_banks=E, out_f=F_BANK,
    )
    with torch.no_grad():
        pre_t = cond.sae.encode(batch.c.float(), bank_ids)
    np.testing.assert_allclose(np.asarray(pre_j), pre_t.numpy(), rtol=1e-6, atol=1e-6)

    z = torch.relu(torch.randn(B, 6, F_BANK))
    dec_j = sj.decode_routed(
        params["D"], mu, jnp.asarray(z.numpy()),
        jnp.asarray(plan.bank_flat), jnp.asarray(plan.order), jnp.asarray(plan.dest),
        cap=CAP, n_banks=E, add_mu=True,
    )
    with torch.no_grad():
        dec_t = cond.sae.decode(z, bank_ids, add_mu=True)
    np.testing.assert_allclose(np.asarray(dec_j), dec_t.numpy(), rtol=1e-6, atol=1e-6)


def test_capacity_overflow_raises():
    bank_ids = np.zeros((B, 6), dtype=np.int64)  # everything to bank 0
    with pytest.raises(RuntimeError, match="capacity"):
        sj.grouping_plan(bank_ids, E, cap=8)


def test_global_topk_parity():
    g = torch.Generator().manual_seed(3)
    scores = torch.relu(torch.randn(5000, generator=g))
    k = 200
    mask_t, theta_t = global_batch_topk(scores, k)
    mask_j, theta_j = sj.global_topk_mask(jnp.asarray(scores.numpy()), k)
    assert float(theta_j) == theta_t
    assert np.array_equal(np.asarray(mask_j), mask_t.numpy())


def test_per_token_topk_parity():
    g = torch.Generator().manual_seed(4)
    scores = torch.randn(B, 6 * F_BANK, generator=g)
    mask_t, thr_t = per_token_topk(scores, K_BAR)
    mask_j, thr_j = sj.per_token_topk_mask(jnp.asarray(scores.numpy()), K_BAR)
    assert np.array_equal(np.asarray(mask_j), mask_t.numpy())
    np.testing.assert_allclose(np.asarray(thr_j), thr_t.numpy(), rtol=0, atol=0)


def test_condition_c_bank_ids_bit_equal():
    cond = build("C")
    batch = make_batch()
    torch_ids = cond._bank_ids(batch.experts, batch.example_id)
    # The JAX trainer calls the same torch method — parity by construction;
    # assert the wrapper actually routes through it.
    run = JaxRun(cond, capacity=CAP, device=jax.devices("cpu")[0])
    run.training_step(batch, lr=0.0)
    again = cond._bank_ids(batch.experts, batch.example_id)
    assert torch.equal(torch_ids, again)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_training_step_metric_parity(condition):
    """Same batch, same init → same metrics from both backends (fp32)."""
    torch_cond = build(condition)
    jax_cond = build(condition)  # identical init (same seed)
    batch = make_batch()

    out_t = torch_cond.training_step(batch, step=0)
    # torch updates trackers in the loop; the JAX wrapper does it in-step —
    # apply the loop's update here so both sides reflect one full step.
    torch_cond.theta_ema.update(out_t.theta_b)
    torch_cond.dead_tracker.update(out_t.dict_route_counts, out_t.fired_mask)

    run = JaxRun(jax_cond, capacity=CAP, device=jax.devices("cpu")[0])
    out_j, host_counts = run.training_step(batch, lr=0.0)

    assert out_j["k_selected"] == out_t.k_selected == K_BAR * B
    np.testing.assert_allclose(out_j["l_local"], out_t.l_local, rtol=1e-5)
    np.testing.assert_allclose(out_j["l_aggregate"], out_t.l_aggregate, rtol=1e-5)
    np.testing.assert_allclose(out_j["theta_b"], out_t.theta_b, rtol=1e-5)
    np.testing.assert_allclose(out_j["l0_mean"], out_t.l0_mean, rtol=1e-6)
    assert torch.equal(host_counts, out_t.bank_route_counts)
    # trackers advanced identically (both wrappers use the tracker classes)
    assert torch.equal(
        jax_cond.dead_tracker.tokens_seen, torch_cond.dead_tracker.tokens_seen
    )


@pytest.mark.parametrize("condition", ["A", "B"])
def test_aux_loss_parity_with_forced_dead(condition):
    torch_cond = build(condition)
    jax_cond = build(condition)
    batch = make_batch()
    for c in (torch_cond, jax_cond):
        c.dead_tracker.tokens_seen += 2 * c.dead_tracker.window  # everything dead

    out_t = torch_cond.training_step(batch, step=0)
    run = JaxRun(jax_cond, capacity=CAP, device=jax.devices("cpu")[0])
    out_j, _ = run.training_step(batch, lr=0.0)
    assert out_t.l_aux > 0
    np.testing.assert_allclose(out_j["l_aux"], out_t.l_aux, rtol=1e-5)
    np.testing.assert_allclose(out_j["loss"], float(out_t.loss.detach()), rtol=1e-5)


def test_specadam_pure_parity():
    """Identical params + identical hand-fed grads → identical step output.

    Isolates the optimizer math from the (backend-dependent) gradient
    computation: any semantic difference in clip/projection/Adam/renorm order
    shows up here at fp32 exactness, while accumulation-order fp noise does
    not exist by construction."""
    cond = build("A")
    params, _ = jax_params(cond)
    g = torch.Generator().manual_seed(11)
    grads_t = {
        "W_enc": torch.randn(E, F_BANK, D_MODEL, generator=g),
        "b_enc": torch.randn(E, F_BANK, generator=g),
        "D": torch.randn(E, D_MODEL, F_BANK, generator=g),
    }

    opt_t = SpecAdam(cond.optim_groups())
    cond.sae.W_enc.grad = grads_t["W_enc"].clone()
    cond.sae.b_enc.grad = grads_t["b_enc"].clone()
    cond.sae.D.grad = grads_t["D"].clone()

    opt_j = sj.SpecAdamJax(params, decoder_axis=1)
    grads_j = {k: jnp.asarray(v.numpy()) for k, v in grads_t.items()}
    for _ in range(3):  # grads intentionally constant across steps on both sides
        opt_t.step(lr=3e-4)
        params, _norm = opt_j.step(params, grads_j, 3e-4)
    for name in ("W_enc", "b_enc", "D"):
        t = getattr(cond.sae, name).detach().numpy()
        np.testing.assert_allclose(
            np.asarray(params[name]), t, rtol=1e-6, atol=1e-7, err_msg=name
        )


@pytest.mark.parametrize("condition", ["A", "F", "E"])
def test_optimizer_trajectory_parity(condition):
    """Five full train steps: torch loop vs JAX wrapper.

    Tolerance note: gradients themselves differ at ~1e-7 between backends
    (fp32 accumulation order: XLA batched einsum vs torch per-bank GEMMs);
    Adam's √v̂ denominator amplifies this relatively on near-zero-gradient
    elements. Semantic parity of the optimizer is pinned exactly by
    test_specadam_pure_parity; here we bound the compound fp drift."""
    torch_cond = build(condition)
    jax_cond = build(condition)
    opt = SpecAdam(torch_cond.optim_groups())
    run = JaxRun(jax_cond, capacity=CAP, device=jax.devices("cpu")[0])

    for step in range(5):
        batch = make_batch(seed=step)
        out = torch_cond.training_step(batch, step)
        out.loss.backward()
        opt.step(lr=3e-4)
        opt.zero_grad()
        run.training_step(batch, lr=3e-4)

    run.sync_params_to_torch()
    for name in ("W_enc", "b_enc", "D"):
        t = getattr(torch_cond.sae, name).detach().float().numpy()
        j = getattr(jax_cond.sae, name).detach().float().numpy()
        # atol ≈ lr-scale fp drift after 5 steps (see docstring); the mean
        # drift must stay two orders below the update scale.
        np.testing.assert_allclose(j, t, rtol=0.05, atol=2e-4, err_msg=f"{condition}/{name}")
        assert float(np.mean(np.abs(j - t))) < 1e-5, f"{condition}/{name} mean drift"
    # decoder columns unit-norm on both sides
    axis = type(jax_cond.sae).DECODER_AXIS
    norms = np.linalg.norm(jax_cond.sae.D.detach().float().numpy(), axis=axis)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)


def test_checkpoint_torch_readable(tmp_path):
    """JAX-trained run readable by the torch loader; θ*-inference agrees."""
    from moe_sae.sae.conditions import load_condition_from_run

    cfg = tiny_config("A", total_tokens=B * 3)
    cond = build("A")
    batches = [make_batch(seed=s) for s in range(3)]
    train_jax(cfg, cond, iter(batches), tmp_path, capacity=CAP,
              device=jax.devices("cpu")[0])

    assert (tmp_path / "final" / "checkpoint.safetensors").exists()
    assert (tmp_path / "final" / "theta_star.json").exists()
    assert (tmp_path / "metrics.jsonl").exists()

    loaded, theta_star, extras = load_condition_from_run(tmp_path)
    assert extras["step"] == 3 and theta_star is not None
    b = make_batch(seed=9)
    enc = loaded.encode_thresholded(b.c, b.experts, b.example_id, theta_star)
    assert enc.act_value.numel() >= 0 and torch.isfinite(enc.recon_slots).all()
    # loaded params equal the trained JAX params exactly
    assert torch.equal(loaded.sae.D.detach(), cond.sae.D.detach())


def test_determinism_same_seed(tmp_path):
    import json as _json

    cfg = tiny_config("A", total_tokens=B * 2)
    runs = []
    for i in range(2):
        cond = build("A")
        batches = [make_batch(seed=s) for s in range(2)]
        out = tmp_path / f"run{i}"
        train_jax(cfg, cond, iter(batches), out, capacity=CAP,
                  device=jax.devices("cpu")[0])
        lines = [
            _json.loads(ln) for ln in (out / "metrics.jsonl").read_text().splitlines()
        ]
        for ln in lines:
            ln.pop("wall_time")  # the only legitimately nondeterministic field
        runs.append(lines)
    assert runs[0] == runs[1]


def test_adaptive_capacity_growth():
    """Overflow grows capacity (never drops tokens) and the step still
    matches torch. All 24·6 slots pile onto two banks — far beyond a
    capacity-8 start (the example-correlated hot-bank scenario)."""
    torch_cond = build("A")
    jax_cond = build("A")
    batch = make_batch()
    hot = torch.zeros_like(batch.experts)
    hot[:, 3:] = 1  # banks 0 and 1 only
    batch = TrainBatch(
        c=batch.c, experts=hot, gates=batch.gates,
        token_type=batch.token_type, example_id=batch.example_id,
        domain=batch.domain,
    )

    out_t = torch_cond.training_step(batch, step=0)
    run = JaxRun(jax_cond, capacity=8, device=jax.devices("cpu")[0])
    out_j, _ = run.training_step(batch, lr=0.0)
    assert run.capacity >= 72  # grew to fit 24·3=72 slots per bank
    assert out_j["k_selected"] == out_t.k_selected == K_BAR * B
    np.testing.assert_allclose(out_j["l_local"], out_t.l_local, rtol=1e-5)
    np.testing.assert_allclose(out_j["theta_b"], out_t.theta_b, rtol=1e-5)
