"""SAECondition (spec §12): all six conditions on tiny dims — exact-K
selection, per-condition selection semantics, torch-seed-independent C
permutations, aux-loss wiring, θ*-inference, checkpoint round trip, and the
run-directory loader `load_condition_from_run`."""

import json

import numpy as np
import pytest
import torch

from moe_sae.sae import TrainBatch, load_condition_from_run
from moe_sae.sae.batchtopk import global_batch_topk, largest_remainder_budgets
from moe_sae.sae.conditions import SAECondition, build_condition
from moe_sae.train.config import CONDITION_C_SALT, CONDITIONS, RunConfig, global_width

E, F_BANK, D_MODEL, B, K_BAR = 64, 8, 16, 24, 4


def tiny_config(condition: str, **overrides) -> RunConfig:
    """Small-dim config for CPU tests. validate() is deliberately NOT called:
    frozen-constant validation belongs to scripts/run_train.py, and the frozen
    batch_tokens=4096 is far too large for unit tests (documented decision)."""
    kwargs = dict(
        run_name=f"test_{condition}",
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
    )
    kwargs.update(overrides)
    return RunConfig(**kwargs)


def make_means(seed=0):
    g = torch.Generator().manual_seed(seed)
    mu_e = torch.randn(E, D_MODEL, generator=g) * 0.1
    mu_route = torch.randn(D_MODEL, generator=g) * 0.1
    return mu_e, mu_route


def make_batch(seed=0, n=B) -> TrainBatch:
    g = torch.Generator().manual_seed(seed)
    experts = torch.stack([torch.randperm(E, generator=g)[:6] for _ in range(n)])
    token_type = torch.zeros(n, dtype=torch.uint8)
    token_type[: n // 2] = 1  # half visual, half text
    return TrainBatch(
        c=(torch.randn(n, 6, D_MODEL, generator=g)).to(torch.bfloat16),
        experts=experts.to(torch.int64),
        gates=torch.rand(n, 6, generator=g),
        token_type=token_type,
        example_id=torch.arange(n, dtype=torch.int32) // 4,
        domain=torch.zeros(n, dtype=torch.uint8),
    )


def build(condition: str, seed: int = 1) -> SAECondition:
    mu_e, mu_route = make_means()
    return build_condition(condition, tiny_config(condition), mu_e, mu_route, seed)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_training_step_shapes_and_exact_k(condition):
    cond = build(condition)
    out = cond.training_step(make_batch(), step=0)
    assert out.k_selected == K_BAR * B  # exact batch budget, every condition
    for name in ("l_local", "l_aggregate", "l_aux", "theta_b", "l0_mean"):
        v = getattr(out, name)
        assert np.isfinite(v), (name, v)
    assert out.loss.requires_grad
    out.loss.backward()
    for p in cond.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    # bookkeeping shapes
    assert out.bank_route_counts.shape == (E,)
    assert int(out.bank_route_counts.sum()) == 6 * B
    n_banks, f_dict = cond.n_dict_banks, cond.f_dict
    assert out.fired_mask.shape == (n_banks, f_dict)
    assert out.bank_fire_counts.shape == (n_banks,)
    assert int(out.bank_fire_counts.sum()) == K_BAR * B
    assert int(out.dict_route_counts.sum()) == (6 * B if not cond.is_global else B)
    # l0 can never exceed selections; z > 0 is a subset of selected
    assert out.l0_mean <= K_BAR * 6 * B  # trivially bounded, finite


@pytest.mark.parametrize("condition", ["B", "F"])
def test_global_width(condition):
    cond = build(condition)
    assert cond.sae.F == global_width(condition, F_BANK)
    assert cond.f_dict == (6 * F_BANK if condition == "B" else 64 * F_BANK)


def test_condition_a_theta_b_matches_naive():
    cond = build("A")
    batch = make_batch()
    out = cond.training_step(batch, step=0)
    with torch.no_grad():
        preacts = cond.sae.encode(batch.c.to(torch.float32), batch.experts)
    _, theta_naive = global_batch_topk(preacts.reshape(-1), K_BAR * B)
    assert out.theta_b == theta_naive


def test_condition_c_permutation_torch_seed_independent():
    batch = make_batch()
    torch.manual_seed(0)
    cond1 = build("C")
    ids1 = cond1._permuted_bank_ids(batch.experts, batch.example_id)
    torch.manual_seed(99999)
    cond2 = build("C")
    ids2 = cond2._permuted_bank_ids(batch.experts, batch.example_id)
    assert torch.equal(ids1, ids2)  # PCG64(salt ^ example_id) only, no torch RNG

    # matches the contract's stream directly
    for t in (0, 5, B - 1):
        eid = int(batch.example_id[t])
        rng = np.random.Generator(np.random.PCG64(CONDITION_C_SALT ^ eid))
        perm = rng.permutation(E)
        want = torch.from_numpy(perm[batch.experts[t].numpy()])
        assert torch.equal(ids1[t], want)


def test_condition_c_permutation_properties():
    cond = build("C")
    batch = make_batch()
    ids = cond._permuted_bank_ids(batch.experts, batch.example_id)
    # six distinct banks per token (bijection preserves distinctness)
    for t in range(B):
        assert len(set(ids[t].tolist())) == 6
    # same example → same permutation; different examples → different (w.h.p.)
    same_ex = batch.example_id[0] == batch.example_id[1]
    assert bool(same_ex)  # example_id = arange//4
    e_shared = torch.arange(6).reshape(1, 6).repeat(2, 1)
    two = cond._permuted_bank_ids(e_shared, batch.example_id[:2])
    assert torch.equal(two[0], two[1])
    other = cond._permuted_bank_ids(e_shared[:1], torch.tensor([1000], dtype=torch.int32))
    assert not torch.equal(two[0], other[0])


def test_condition_d_budgets_follow_loads():
    cond = build("D")
    batch = make_batch()
    out = cond.training_step(batch, step=0)
    loads = torch.bincount(batch.experts.reshape(-1), minlength=E)
    budgets = largest_remainder_budgets(loads.float(), K_BAR * B)
    assert torch.equal(out.bank_fire_counts, budgets)
    assert int((out.bank_fire_counts[loads == 0]).sum()) == 0


def test_condition_d_step_reports_per_bank_thetas():
    """A4: D's StepOutput carries this batch's per-bank thresholds; other
    conditions report None."""
    cond = build("D")
    batch = make_batch()
    out = cond.training_step(batch, step=0)
    assert out.theta_b_per_bank is not None
    assert out.theta_b_per_bank.shape == (E,)
    loads = torch.bincount(batch.experts.reshape(-1), minlength=E)
    budgets = largest_remainder_budgets(loads.float(), K_BAR * B)
    has_theta = ~torch.isnan(out.theta_b_per_bank)
    assert torch.equal(has_theta, budgets > 0)
    # the log-only scalar θ_b (kept as-is) is the min over per-bank thetas
    assert out.theta_b == pytest.approx(float(out.theta_b_per_bank[has_theta].min()))
    for other in ("A", "E"):
        assert build(other).training_step(batch, step=0).theta_b_per_bank is None


def test_condition_d_checkpoint_roundtrip_theta_vector(tmp_path):
    cond = build("D")
    batch = make_batch()
    out = cond.training_step(batch, step=0)
    cond.theta_ema_vec.update(out.theta_b_per_bank)
    path = tmp_path / "ckpt.safetensors"
    cond.save_checkpoint(path, cond.theta_ema, cond.dead_tracker, step=1)
    cond2, _, _ = SAECondition.load_checkpoint(path)
    a, b = cond.theta_ema_vec.value, cond2.theta_ema_vec.value
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    assert torch.equal(a[~torch.isnan(a)], b[~torch.isnan(b)])
    # non-D checkpoints keep the scalar-only format (backward compatible)
    condA = build("A")
    pA = tmp_path / "ckptA.safetensors"
    condA.save_checkpoint(pA, condA.theta_ema, condA.dead_tracker, step=1)
    from safetensors import safe_open

    with safe_open(str(pA), framework="pt") as fh:
        assert "theta_value_per_bank" not in fh.keys()


def test_condition_d_trains_and_freezes_theta_star_vector(tmp_path):
    """A4 end-to-end: D trains through the loop, freezes the θ*_e vector in
    theta_star.json, and load_condition_from_run hands the vector to eval."""
    from moe_sae.train.loop import train

    cfg = tiny_config("D", out_root=str(tmp_path))
    cond = build("D")
    batches = [make_batch(seed=s) for s in range(3)]
    train(cfg, cond, iter(batches), cfg.out_dir)

    raw = json.loads((cfg.out_dir / "final" / "theta_star.json").read_text())
    vec = raw["theta_star_per_bank"]
    assert isinstance(vec, list) and len(vec) == E
    assert raw["theta_star"] is not None  # scalar log-only value kept
    assert vec == cond.theta_ema_vec.values_list()
    assert any(v is not None for v in vec)

    cond2, theta_star, _ = load_condition_from_run(cfg.out_dir)
    assert isinstance(theta_star, torch.Tensor) and theta_star.shape == (E,)
    want = torch.tensor(
        [float("nan") if v is None else v for v in vec], dtype=torch.float32
    )
    assert torch.equal(torch.isnan(theta_star), torch.isnan(want))
    assert torch.allclose(theta_star[~torch.isnan(want)], want[~torch.isnan(want)])
    # inference accepts the vector
    b = make_batch(seed=9)
    enc = cond2.encode_thresholded(b.c, b.experts, b.example_id, theta_star)
    assert torch.isfinite(enc.recon_slots).all()


def test_condition_d_inference_uses_per_bank_thresholds():
    """A4: plant scale-divergent per-bank thresholds and assert the resulting
    activation pattern is exactly per-bank — and NOT producible by any global
    scalar θ*."""
    cond = build("D")
    batch = make_batch()
    theta = torch.full((E,), 1e9)  # high-θ banks: never fire
    low_banks = list(range(0, E, 2))
    for e in low_banks:
        theta[e] = 0.05
    theta_list = [None if e == 0 else float(theta[e]) for e in range(E)]  # bank 0: undefined
    enc = cond.encode_thresholded(batch.c, batch.experts, batch.example_id, theta_list)

    with torch.no_grad():
        a = cond.sae.encode(batch.c.float(), batch.experts).float()
    theta_eff = theta.clone()
    theta_eff[0] = float("nan")  # None → NaN → never fires
    dense = a * (a > theta_eff[batch.experts].unsqueeze(-1))
    tok, slot, lat = (dense > 0).nonzero(as_tuple=True)
    assert enc.act_value.numel() == tok.numel() > 0
    assert torch.allclose(enc.act_value, dense[tok, slot, lat])
    fired_banks = set(enc.act_bank.tolist())
    assert fired_banks <= set(low_banks) - {0}  # NaN bank 0 never fires
    # per-bank thresholding differs from ANY global scalar: some unfired
    # candidate's preact exceeds some fired candidate's value — impossible
    # under a single θ (scalar thresholding is monotone in the preact).
    unfired = a[(dense == 0) & (a > 0)]
    assert float(unfired.max()) > float(enc.act_value.min())


def test_condition_e_exact_k_per_token():
    cond = build("E")
    batch = make_batch()
    with torch.no_grad():
        preacts = cond.sae.encode(batch.c.to(torch.float32), batch.experts)
    from moe_sae.sae.batchtopk import per_token_topk

    mask, per_tok_theta = per_token_topk(preacts.reshape(B, 6 * F_BANK), K_BAR)
    assert torch.equal(mask.sum(dim=1), torch.full((B,), K_BAR))
    out = cond.training_step(batch, step=0)
    assert out.theta_b == pytest.approx(float(per_tok_theta.mean()))


@pytest.mark.parametrize("condition", ["A", "B"])
def test_aux_loss_zero_without_dead_and_active_with(condition):
    cond = build(condition)
    batch = make_batch()
    out = cond.training_step(batch, step=0)
    assert out.n_dead_now == 0
    assert out.l_aux == 0.0  # fresh tracker: nothing dead yet

    # Force everything dead: clocks past the window, never fired.
    cond.dead_tracker.tokens_seen += 2 * cond.dead_tracker.window
    out2 = cond.training_step(batch, step=1)
    assert out2.n_dead_now == cond.n_dict_banks * cond.f_dict
    assert out2.l_aux > 0.0 and np.isfinite(out2.l_aux)
    # loss = l_local + α·l_aux
    assert float(out2.loss.detach()) == pytest.approx(
        out2.l_local + cond.aux_alpha * out2.l_aux
    )
    # main selection identical: aux does not perturb the primary path
    assert out2.theta_b == out.theta_b and out2.l_local == pytest.approx(out.l_local)


@pytest.mark.parametrize("condition", ["A", "B"])
def test_aux_uses_raw_preacts_for_dead_latents(condition):
    """A1 committee pin (Reviewer F): aux-selected dead latents contribute
    their RAW (pre-ReLU) preact values — gradient must reach below-threshold
    latents to revive them. With every preact forced negative, ReLU'd aux
    values would be exactly zero (rhat == 0, l_aux == 1, zero gradient); raw
    values give a live gradient path into the encoder."""
    cond = build(condition)
    with torch.no_grad():
        cond.sae.b_enc.fill_(-50.0)  # every raw preact strongly negative
    cond.dead_tracker.tokens_seen += 2 * cond.dead_tracker.window  # all dead
    batch = make_batch()
    out = cond.training_step(batch, step=0)
    assert out.n_dead_now == cond.n_dict_banks * cond.f_dict
    assert out.aux_selected > 0  # negative preacts ARE selected (> -inf rule)
    assert out.l_aux > 0.0
    # ReLU'd aux values would give rhat == 0 => l_aux == Σ‖r‖²/Σ‖r‖² == 1.0
    assert out.l_aux != pytest.approx(1.0)
    out.loss.backward()
    # main path is fully dark (ReLU'd z == 0 everywhere): any encoder
    # gradient must have come through the RAW aux values.
    assert cond.sae.b_enc.grad is not None
    assert float(cond.sae.b_enc.grad.abs().sum()) > 0.0
    assert float(cond.sae.W_enc.grad.abs().sum()) > 0.0


def test_aux_diagnostics_counts_and_scalars():
    """A1 required diagnostics (Reviewer S): selection counts, overlap with
    main, per-bank pressure, and the graph-attached aux term."""
    cond = build("A")
    cond.dead_tracker.tokens_seen += 2 * cond.dead_tracker.window  # all dead
    batch = make_batch()
    out = cond.training_step(batch, step=0)
    # everything dead and budget (512) > 6f candidates: all 6f selected/token
    expect = 6 * F_BANK * B
    assert out.aux_selected == expect
    assert int(out.aux_bank_counts.sum()) == expect
    assert out.aux_bank_counts.shape == (cond.n_dict_banks,)
    # every main-selected candidate is dead here, so it is also aux-selected
    assert out.aux_main_overlap == out.k_selected
    sc = out.scalars()
    assert sc["aux_mean_selected_per_token"] == pytest.approx(expect / B)
    assert sc["aux_primary_ratio"] == pytest.approx(out.l_aux / out.l_local)
    assert sc["aux_frac_also_main"] == pytest.approx(out.k_selected / expect)
    assert out.aux_term is not None and out.aux_term.requires_grad
    assert float(out.aux_term.detach()) == pytest.approx(cond.aux_alpha * out.l_aux)
    # per-bank counts follow the routed visibility (only routed banks selected)
    loads = torch.bincount(batch.experts.reshape(-1), minlength=E)
    assert torch.equal(out.aux_bank_counts, loads * F_BANK)

    # nothing dead: diagnostics are all zero and no aux graph exists
    fresh = build("A")
    out0 = fresh.training_step(batch, step=0)
    assert out0.aux_selected == 0 and out0.aux_main_overlap == 0
    assert int(out0.aux_bank_counts.sum()) == 0
    assert out0.aux_term is None


def test_aux_grad_norm_ratio_second_backward():
    """A1 diagnostic (ii): the loop's second backward of the aux term yields
    ‖g_aux‖/‖g_main‖; None when nothing is dead."""
    from moe_sae.train.loop import aux_grad_norm_ratio

    cond = build("A")
    cond.dead_tracker.tokens_seen += 2 * cond.dead_tracker.window
    out = cond.training_step(make_batch(), step=0)
    out.loss.backward(retain_graph=True)
    params = list(cond.parameters())
    ratio = aux_grad_norm_ratio(out.aux_term, params)
    assert ratio is not None and ratio > 0.0 and np.isfinite(ratio)

    # reference: grads of α·l_aux and of l_local computed independently
    cond2 = build("A")
    cond2.dead_tracker.tokens_seen += 2 * cond2.dead_tracker.window
    out2 = cond2.training_step(make_batch(), step=0)
    g_aux = torch.autograd.grad(
        out2.aux_term, list(cond2.parameters()), retain_graph=True, allow_unused=True
    )
    g_tot = torch.autograd.grad(out2.loss, list(cond2.parameters()), allow_unused=True)
    aux_sq = sum(float(g.pow(2).sum()) for g in g_aux if g is not None)
    main_sq = sum(
        float(((gt if gt is not None else 0) - (ga if ga is not None else 0)).pow(2).sum())
        for gt, ga in zip(g_tot, g_aux)
        if gt is not None or ga is not None
    )
    want = aux_sq**0.5 / max(main_sq**0.5, 1e-30)
    assert ratio == pytest.approx(want, rel=1e-4)

    fresh = build("A")
    out0 = fresh.training_step(make_batch(), step=0)
    assert aux_grad_norm_ratio(out0.aux_term, list(fresh.parameters())) is None


def test_aux_respects_token_budget():
    cond = build("A")
    cond.aux_token_budget = 3  # tiny budget to make the cap bite
    cond.dead_tracker.tokens_seen += 2 * cond.dead_tracker.window
    batch = make_batch()
    scores = cond.sae.encode(batch.c.to(torch.float32), batch.experts).detach()
    dead_vis = cond.dead_tracker.dead_mask()[batch.experts]
    aux_mask = cond._aux_select(
        scores.reshape(B, 6 * F_BANK), dead_vis.reshape(B, 6 * F_BANK)
    )
    assert (aux_mask.sum(dim=1) <= 3).all()
    assert (aux_mask.sum(dim=1) == 3).all()  # all dead here, so budget binds


@pytest.mark.parametrize("condition", ["A", "C", "B"])
def test_encode_thresholded(condition):
    cond = build(condition)
    batch = make_batch()
    theta_star = 0.35
    out = cond.encode_thresholded(batch.c, batch.experts, batch.example_id, theta_star)
    assert (out.act_value > theta_star).all()  # strict θ* threshold
    assert int(out.l0.sum()) == out.act_value.numel()
    # naive reference: full dense encode + threshold
    with torch.no_grad():
        if cond.is_global:
            x = batch.c.float().sum(1)
            z = cond.sae.encode(x).float()
            assert (out.act_slot == 0).all()
            assert (out.act_bank == 0).all()
            assert out.recon_slots.shape == (batch.c.shape[0], 1, cond.d)
            bank_ids = None
        else:
            bank_ids = cond._bank_ids(batch.experts, batch.example_id)
            z = cond.sae.encode(batch.c.float(), bank_ids).float()
            assert out.recon_slots.shape == batch.c.shape
    dense = z * (z > theta_star)
    n_pos = int((dense > 0).sum())
    assert out.act_value.numel() == n_pos
    if not cond.is_global:
        vals = dense[out.act_token.long(), out.act_slot.long(), out.act_latent.long()]
        assert torch.allclose(out.act_value, vals)
        # act_bank is the assigned bank (permuted for C, host expert for A)
        assert torch.equal(
            out.act_bank, bank_ids[out.act_token.long(), out.act_slot.long()]
        )
        # recon = μ + Dz per slot, against a naive decode
        recon_naive = cond.sae.decode(
            dense.to(cond.compute_dtype), bank_ids, add_mu=True
        ).float()
        assert torch.allclose(out.recon_slots, recon_naive, atol=1e-3, rtol=1e-2)
        assert torch.equal(out.mu_slots, cond.sae.mu[bank_ids].float())


def test_checkpoint_roundtrip(tmp_path):
    cond = build("A")
    batch = make_batch()
    out = cond.training_step(batch, step=0)
    out.loss.backward()
    cond.theta_ema.update(out.theta_b)
    cond.dead_tracker.update(out.dict_route_counts, out.fired_mask)

    path = tmp_path / "ckpt.safetensors"
    cond.save_checkpoint(path, cond.theta_ema, cond.dead_tracker, step=1)
    cond2, theta_star, extras = SAECondition.load_checkpoint(path)

    assert cond2.condition == "A" and cond2.f == F_BANK and cond2.k_bar == K_BAR
    assert theta_star == pytest.approx(cond.theta_ema.value)
    assert extras["step"] == 1
    assert torch.equal(cond2.sae.W_enc.detach(), cond.sae.W_enc.detach())
    assert torch.equal(cond2.sae.D.detach(), cond.sae.D.detach())
    assert torch.equal(cond2.sae.mu, cond.sae.mu)
    assert torch.equal(cond2.dead_tracker.tokens_seen, cond.dead_tracker.tokens_seen)
    assert torch.equal(cond2.dead_tracker.last_fire, cond.dead_tracker.last_fire)
    # inference is bit-identical after reload
    a = cond.encode_thresholded(batch.c, batch.experts, batch.example_id, 0.2)
    b = cond2.encode_thresholded(batch.c, batch.experts, batch.example_id, 0.2)
    for k in ("act_token", "act_slot", "act_bank", "act_latent", "act_value",
              "l0", "recon_slots", "mu_slots"):
        assert torch.equal(getattr(a, k), getattr(b, k)), k


def test_load_condition_from_run_final_and_theta_json(tmp_path):
    """Regression: scripts called a nonexistent module-level
    conditions.load_checkpoint(run_dir, device=...); the run-level loader must
    exist, prefer final/, and treat theta_star.json as canonical over the
    checkpoint's stored EMA value."""
    cond = build("A")
    cond.theta_ema.update(0.5)  # checkpoint EMA differs from the json below
    run_dir = tmp_path / "run"
    cond.save_checkpoint(
        run_dir / "final" / "checkpoint.safetensors", cond.theta_ema, cond.dead_tracker, step=7
    )
    (run_dir / "final" / "theta_star.json").write_text(json.dumps({"theta_star": 0.25}))

    cond2, theta_star, extras = load_condition_from_run(run_dir, device="cpu")
    assert theta_star == 0.25  # json is canonical, beats the EMA's 0.5
    assert extras["step"] == 7
    assert torch.equal(cond2.sae.W_enc.detach(), cond.sae.W_enc.detach())

    # bare-float theta_star.json is also accepted
    (run_dir / "final" / "theta_star.json").write_text("0.125")
    _, theta_star, _ = load_condition_from_run(run_dir)
    assert theta_star == 0.125


def test_load_condition_from_run_highest_step_and_theta_fallback(tmp_path):
    cond = build("A")
    run_dir = tmp_path / "run"
    cond.theta_ema.update(0.3)
    cond.save_checkpoint(
        run_dir / "checkpoints" / "900.safetensors", cond.theta_ema, cond.dead_tracker, step=900
    )
    with torch.no_grad():
        cond.sae.W_enc.add_(1.0)  # distinguish the later checkpoint
    cond.theta_ema.update(0.9)
    cond.save_checkpoint(
        run_dir / "checkpoints" / "1000.safetensors", cond.theta_ema, cond.dead_tracker, step=1000
    )
    cond2, theta_star, extras = load_condition_from_run(run_dir)
    assert extras["step"] == 1000  # numeric order ("900" > "1000" lexically)
    assert torch.equal(cond2.sae.W_enc.detach(), cond.sae.W_enc.detach())
    # no final/theta_star.json → fall back to the checkpoint's EMA value
    assert theta_star == pytest.approx(cond.theta_ema.value)


def test_load_condition_from_run_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        load_condition_from_run(tmp_path / "empty_run")

    # checkpoint present but θ never initialized and no theta_star.json
    cond = build("B")
    run_dir = tmp_path / "run"
    cond.save_checkpoint(
        run_dir / "final" / "checkpoint.safetensors", cond.theta_ema, cond.dead_tracker, step=0
    )
    with pytest.raises(ValueError, match="theta_star"):
        load_condition_from_run(run_dir)


def test_checkpoint_none_theta(tmp_path):
    cond = build("B")
    path = tmp_path / "ckpt.safetensors"
    cond.save_checkpoint(path, cond.theta_ema, cond.dead_tracker, step=0)
    _, theta_star, _ = SAECondition.load_checkpoint(path)
    assert theta_star is None  # EMA never initialized


def test_decoder_columns_view():
    assert build("A").decoder_columns().shape == (E, D_MODEL, F_BANK)
    assert build("F").decoder_columns().shape == (D_MODEL, 64 * F_BANK)


def test_l0_by_token_type_partition():
    cond = build("A")
    batch = make_batch()
    out = cond.training_step(batch, step=0)
    # mean of the two halves reconstructs the overall mean (equal halves)
    assert out.l0_mean == pytest.approx(0.5 * (out.l0_visual + out.l0_text))
