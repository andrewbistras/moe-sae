"""DeadTracker (spec §8 frozen deadness; contract §Deadness): bank-relative
windows, pre-window alive rule, low-load banks, main-TopK-only resets,
state round trip."""

import torch

from moe_sae.sae.dead import DeadTracker

E, F, W = 3, 4, 100


def counts(*per_bank):
    return torch.tensor(per_bank, dtype=torch.int64)


def no_fires():
    return torch.zeros(E, F, dtype=torch.bool)


def test_alive_before_full_window():
    """Deadness is undefined (treated as alive) until a bank has seen a full
    window, even for latents that never fired."""
    tr = DeadTracker(E, F, window=W)
    tr.update(counts(W - 1, W, 0), no_fires())
    dead = tr.dead_mask()
    assert not dead[0].any()  # 99 < window
    assert not dead[1].any()  # == window: age == window, not > window
    assert not dead[2].any()  # bank never saw a token
    tr.update(counts(2, 1, 0), no_fires())
    dead = tr.dead_mask()
    assert dead[0].all()  # 101 > window, never fired
    assert dead[1].all()  # 101 > window
    assert not dead[2].any()


def test_fire_resets_bank_relative_clock():
    tr = DeadTracker(E, F, window=W)
    fired = no_fires()
    fired[0, 1] = True
    tr.update(counts(60, 0, 0), fired)  # latent (0,1) fires at clock 60
    tr.update(counts(90, 0, 0), no_fires())  # clock 150
    dead = tr.dead_mask()
    assert dead[0, 0] and dead[0, 2] and dead[0, 3]  # never fired, 150 > 100
    assert not dead[0, 1]  # age 90 <= window
    tr.update(counts(11, 0, 0), no_fires())  # clock 161, age 101 > window
    assert tr.dead_mask()[0, 1]


def test_low_load_bank_uses_own_clock():
    """A low-load bank's latents stay alive while a high-load bank's die —
    the window is bank-relative (spec changelog #3)."""
    tr = DeadTracker(2, F, window=W)
    for _ in range(50):
        tr.update(torch.tensor([40, 1]), torch.zeros(2, F, dtype=torch.bool))
    dead = tr.dead_mask()
    assert dead[0].all()  # 2000 tokens seen, never fired
    assert not dead[1].any()  # only 50 tokens seen — pre-window, alive


def test_n_dead_and_state_roundtrip():
    tr = DeadTracker(E, F, window=W)
    fired = no_fires()
    fired[1, 0] = True
    tr.update(counts(150, 150, 10), fired)
    assert tr.n_dead() == int(tr.dead_mask().sum())
    assert tr.n_dead() == 2 * F - 1  # banks 0,1 past window; (1,0) fired at 150

    state = tr.state_dict()
    tr2 = DeadTracker(E, F, window=1)  # deliberately wrong window
    tr2.load_state_dict(state)
    assert tr2.window == W
    assert torch.equal(tr2.tokens_seen, tr.tokens_seen)
    assert torch.equal(tr2.last_fire, tr.last_fire)
    assert torch.equal(tr2.dead_mask(), tr.dead_mask())
    # state is a copy, not a view
    state["tokens_seen"][0] = 12345
    assert tr2.tokens_seen[0] != 12345


def test_fire_stamp_uses_post_advance_clock():
    """Firing in the same batch as the tokens counts as 'fired now'."""
    tr = DeadTracker(1, 1, window=W)
    fired = torch.ones(1, 1, dtype=torch.bool)
    tr.update(torch.tensor([W + 50]), fired)
    assert not tr.dead_mask().any()  # fired at clock W+50, age 0
