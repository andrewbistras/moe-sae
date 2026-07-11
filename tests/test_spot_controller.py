"""Spot-fleet controller (scripts/tpu/spot_controller.py): queue CAS
semantics, lease lifecycle, and a 3-controller / 8-ticket simulation with a
flaky executor. Pure ops tooling — no model, no training math; everything
runs locally on CPU in a tmp_path queue with injected executors.

Decision (documented): timing constants below (lease ~0.6 s, heartbeat 50 ms,
executor sleeps 10-20 ms) are chosen so that a heartbeat would have to miss
~12 consecutive renewal windows for a live run's lease to lapse — the
simulation asserts *exactly one* successful execution per ticket, which is
precisely the property the heartbeat exists to provide, so the margins are
deliberately wide to keep the assertion meaningful rather than flaky."""

import dataclasses
import importlib.util
import json
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _import_script(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / "tpu" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass field-type resolution needs this
    spec.loader.exec_module(mod)
    return mod


sc = _import_script("spot_controller")


def make_ticket(name, tmp_path, priority=0, max_attempts=50):
    return sc.Ticket(
        run_name=name,
        priority=priority,
        max_attempts=max_attempts,
        config={
            "config": f"configs/{name}.yaml",
            "out_dir": str(tmp_path / "runs" / name),
            "trainer": "jax",
        },
    )


def write_artifact(ticket):
    final = Path(ticket.config["out_dir"]) / "final"
    final.mkdir(parents=True, exist_ok=True)
    (final / "theta_star.json").write_text(json.dumps({"theta_star": 0.5}))


class RecordingExecutor(sc.Executor):
    """Succeeds instantly; records execution order and per-ticket overlap."""

    def __init__(self):
        self.lock = threading.Lock()
        self.order = []
        self.inflight = Counter()
        self.max_inflight = Counter()
        self.successes = Counter()

    def _enter(self, name):
        with self.lock:
            self.order.append(name)
            self.inflight[name] += 1
            self.max_inflight[name] = max(self.max_inflight[name], self.inflight[name])

    def _exit(self, name):
        with self.lock:
            self.inflight[name] -= 1

    def execute(self, ticket):
        self._enter(ticket.run_name)
        try:
            write_artifact(ticket)
            with self.lock:
                self.successes[ticket.run_name] += 1
            return 0
        finally:
            self._exit(ticket.run_name)


class SimulatedPreemption(RuntimeError):
    """Stands in for the slice dying under the executor mid-run."""


class FlakyExecutor(RecordingExecutor):
    """Fails (nonzero rc) or preempts (raises) a configurable fraction of
    attempts. `scripted` forces per-ticket outcome sequences (consumed first,
    then the seeded RNG decides), making 'a preemption happened' assertable
    without depending on RNG internals across Python versions."""

    def __init__(self, p_fail=0.2, p_preempt=0.25, seed=0, scripted=None,
                 hopeless=(), duration=0.01):
        super().__init__()
        import random

        self.rng = random.Random(seed)
        self.p_fail, self.p_preempt = p_fail, p_preempt
        self.scripted = {k: list(v) for k, v in (scripted or {}).items()}
        self.hopeless = set(hopeless)
        self.duration = duration
        self.preemptions = Counter()

    def _outcome(self, name):
        if name in self.hopeless:
            return "fail"
        with self.lock:
            script = self.scripted.get(name)
            if script:
                return script.pop(0)
            roll = self.rng.random()
        if roll < self.p_preempt:
            return "preempt"
        if roll < self.p_preempt + self.p_fail:
            return "fail"
        return "ok"

    def execute(self, ticket):
        name = ticket.run_name
        self._enter(name)
        try:
            time.sleep(self.duration)
            outcome = self._outcome(name)
            if outcome == "preempt":
                with self.lock:
                    self.preemptions[name] += 1
                raise SimulatedPreemption(name)
            if outcome == "fail":
                return 1
            write_artifact(ticket)
            with self.lock:
                self.successes[name] += 1
            return 0
        finally:
            self._exit(name)


def make_controller(queue, executor, worker, **kw):
    kw.setdefault("lease_seconds", 0.6)
    kw.setdefault("heartbeat_seconds", 0.05)
    kw.setdefault("poll_seconds", 0.01)
    kw.setdefault("backoff_base", 0.02)
    kw.setdefault("backoff_cap", 0.1)
    kw.setdefault("orphan_age_seconds", 5.0)
    return sc.Controller(queue, executor, worker=worker, **kw)


def states(queue):
    return {name: queue.read(name)[0] for name in queue.names()}


# --------------------------------------------------------------------------- #
# The multi-controller simulation                                              #
# --------------------------------------------------------------------------- #


def test_multi_controller_simulation(tmp_path):
    """3 controllers, 8 tickets, ~45% flaky attempts: every recoverable
    ticket ends done exactly once with no concurrent double-execution;
    a forced preemption is recovered via lease expiry; the hopeless ticket
    fails terminally after exactly max_attempts."""
    queue = sc.LocalQueue(tmp_path / "queue")
    names = [f"discovery_sim_{i}" for i in range(8)]
    hopeless = names[7]
    forced = names[3]
    for i, name in enumerate(names):
        queue.create(make_ticket(
            name, tmp_path, priority=i % 3,
            max_attempts=3 if name == hopeless else 50,
        ))

    executor = FlakyExecutor(
        scripted={forced: ["preempt", "preempt", "ok"]}, hopeless=[hopeless]
    )
    controllers = [make_controller(queue, executor, f"w{i}") for i in range(3)]
    threads = [
        threading.Thread(target=c.run, kwargs={"drain": True}, name=f"ctl-{i}")
        for i, c in enumerate(controllers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)
    hung = [t.name for t in threads if t.is_alive()]
    for c in controllers:
        c.stop()
    assert not hung, f"controllers never drained: {hung}"

    final = states(queue)
    # Every recoverable ticket ends done; hopeless ends failed.
    for name in names:
        want = "failed" if name == hopeless else "done"
        assert final[name].state == want, (name, final[name])
    # Done exactly once: exactly one *successful* execution per ticket …
    for name in names:
        assert executor.successes[name] == (0 if name == hopeless else 1), name
        assert sc.default_artifact_check(final[name]) == (name != hopeless)
    # … and never two controllers inside the same ticket concurrently.
    assert max(executor.max_inflight.values()) == 1, executor.max_inflight
    # Preempted ticket was actually preempted, then recovered to done, with
    # each preemption charged to `attempt` (lease-expiry recovery path).
    assert executor.preemptions[forced] >= 2
    assert final[forced].attempt >= 2
    # Failed-after-max respected exactly.
    assert final[hopeless].attempt == 3
    # A ticket abandoned by preemption leaves nothing orphaned in .swap/.
    queue.recover_orphans(max_age_seconds=0.0)
    assert states(queue) == final


# --------------------------------------------------------------------------- #
# Queue CAS + lease unit tests                                                 #
# --------------------------------------------------------------------------- #


def test_claim_cas_is_mutually_exclusive(tmp_path):
    queue = sc.LocalQueue(tmp_path / "q")
    queue.create(make_ticket("r", tmp_path))
    ticket, version = queue.read("r")
    a = dataclasses.replace(ticket, state="running", worker="w0")
    b = dataclasses.replace(ticket, state="running", worker="w1")
    wins = [queue.try_swap("r", version, a), queue.try_swap("r", version, b)]
    assert sum(w is not None for w in wins) == 1
    ticket2, version2 = queue.read("r")
    assert ticket2.worker == ("w0" if wins[0] is not None else "w1")
    assert version2 == ticket.revision + 1


def test_create_refuses_duplicates(tmp_path):
    queue = sc.LocalQueue(tmp_path / "q")
    queue.create(make_ticket("r", tmp_path))
    with pytest.raises(FileExistsError):
        queue.create(make_ticket("r", tmp_path))


def test_ticket_json_roundtrip(tmp_path):
    t = make_ticket("r", tmp_path, priority=7)
    t.state, t.attempt, t.lease_expiry, t.worker = "running", 2, 123.5, "w0"
    assert sc.Ticket.from_json(t.to_json()) == t


def test_orphan_recovery(tmp_path):
    """A controller that dies mid-swap leaves the ticket in .swap/; the
    janitor restores it once stale, and skips it while fresh."""
    queue = sc.LocalQueue(tmp_path / "q")
    queue.create(make_ticket("r", tmp_path))
    canon = queue.root / "r.json"
    orphan = queue.swap / "r.deadbeef.json"
    canon.rename(orphan)
    queue.recover_orphans(max_age_seconds=60.0)  # fresh: not touched
    assert not canon.exists()
    queue.recover_orphans(max_age_seconds=0.0)  # stale: restored
    assert canon.exists() and queue.read("r")[0].run_name == "r"
    assert not orphan.exists()


def test_lease_expiry_requeues_with_attempt_and_backoff(tmp_path):
    queue = sc.LocalQueue(tmp_path / "q")
    ticket = make_ticket("r", tmp_path)
    ticket.state, ticket.worker, ticket.lease_expiry = "running", "dead", time.time() - 1
    queue.create(ticket)
    ctl = make_controller(queue, RecordingExecutor(), "w0",
                          backoff_base=100.0, backoff_cap=10000.0)
    before = time.time()
    ctl._sweep_expired(time.time())
    after, _ = queue.read("r")
    assert after.state == "pending" and after.worker is None
    assert after.attempt == 1
    assert after.not_before >= before + 100.0  # backoff_base * 2**0
    # A second expiry doubles the backoff.
    after.state, after.lease_expiry = "running", time.time() - 1
    _, v = queue.read("r")
    queue.try_swap("r", v, after)
    ctl._sweep_expired(time.time())
    again, _ = queue.read("r")
    assert again.attempt == 2 and again.not_before >= time.time() + 150.0


def test_lease_expiry_respects_max_attempts(tmp_path):
    queue = sc.LocalQueue(tmp_path / "q")
    ticket = make_ticket("r", tmp_path, max_attempts=1)
    ticket.state, ticket.lease_expiry = "running", time.time() - 1
    queue.create(ticket)
    make_controller(queue, RecordingExecutor(), "w0")._sweep_expired(time.time())
    assert queue.read("r")[0].state == "failed"


def test_backoff_schedule(tmp_path):
    ctl = make_controller(sc.LocalQueue(tmp_path / "q"), RecordingExecutor(), "w0",
                          backoff_base=2.0, backoff_cap=10.0)
    assert [ctl.backoff(a) for a in (1, 2, 3, 4, 9)] == [2.0, 4.0, 8.0, 10.0, 10.0]


# --------------------------------------------------------------------------- #
# Controller behavior units                                                    #
# --------------------------------------------------------------------------- #


def test_priority_claim_order(tmp_path):
    queue = sc.LocalQueue(tmp_path / "q")
    for name, prio in (("lo", 0), ("hi", 5), ("mid", 2)):
        queue.create(make_ticket(name, tmp_path, priority=prio))
    executor = RecordingExecutor()
    make_controller(queue, executor, "w0").run(drain=True)
    assert executor.order == ["hi", "mid", "lo"]
    assert all(t.state == "done" for t in states(queue).values())


def test_rc_zero_without_artifact_is_a_failure(tmp_path):
    """Success is the θ* artifact (spec §20.4), not the returncode."""

    class LyingExecutor(RecordingExecutor):
        def execute(self, ticket):
            return 0  # claims success, writes nothing

    queue = sc.LocalQueue(tmp_path / "q")
    queue.create(make_ticket("r", tmp_path, max_attempts=2))
    make_controller(queue, LyingExecutor(), "w0").run(drain=True)
    final = queue.read("r")[0]
    assert final.state == "failed" and final.attempt == 2
    assert "theta_star.json missing" in final.note


def test_failed_after_max_attempts_with_backoff_gaps(tmp_path):
    class AlwaysFail(RecordingExecutor):
        def __init__(self):
            super().__init__()
            self.stamps = []

        def execute(self, ticket):
            self.stamps.append(time.monotonic())
            return 1

    queue = sc.LocalQueue(tmp_path / "q")
    queue.create(make_ticket("r", tmp_path, max_attempts=3))
    executor = AlwaysFail()
    make_controller(queue, executor, "w0", backoff_base=0.05, backoff_cap=10.0).run(drain=True)
    final = queue.read("r")[0]
    assert final.state == "failed" and final.attempt == 3
    assert len(executor.stamps) == 3
    gap1 = executor.stamps[1] - executor.stamps[0]
    gap2 = executor.stamps[2] - executor.stamps[1]
    assert gap1 >= 0.05 and gap2 >= 0.10  # exponential: base, 2*base


def test_graceful_stop_releases_lease_without_charging(tmp_path):
    """SIGTERM path: stop() terminates the child; the ticket returns to
    pending with attempt unchanged, ready for any other controller."""

    class Blocking(RecordingExecutor):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.killed = threading.Event()

        def execute(self, ticket):
            self.started.set()
            assert self.killed.wait(timeout=30)
            return 143  # SIGTERM'd child

        def terminate(self):
            self.killed.set()

    queue = sc.LocalQueue(tmp_path / "q")
    queue.create(make_ticket("r", tmp_path))
    executor = Blocking()
    ctl = make_controller(queue, executor, "w0")
    thread = threading.Thread(target=ctl.run)
    thread.start()
    assert executor.started.wait(timeout=30)
    ctl.stop()
    thread.join(timeout=30)
    assert not thread.is_alive()
    final = queue.read("r")[0]
    assert final.state == "pending"
    assert final.attempt == 0 and final.worker is None
    assert final.note == "released on shutdown"


def test_heartbeat_keeps_long_run_leased(tmp_path):
    """A run several leases long is never swept from under its owner."""

    class Slow(RecordingExecutor):
        def execute(self, ticket):
            self._enter(ticket.run_name)
            try:
                time.sleep(0.5)
                write_artifact(ticket)
                return 0
            finally:
                self._exit(ticket.run_name)

    queue = sc.LocalQueue(tmp_path / "q")
    queue.create(make_ticket("r", tmp_path))
    executor = Slow()
    controllers = [
        make_controller(queue, executor, f"w{i}", lease_seconds=0.1,
                        heartbeat_seconds=0.02)
        for i in range(2)
    ]
    threads = [threading.Thread(target=c.run, kwargs={"drain": True}) for c in controllers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert queue.read("r")[0].state == "done"
    assert executor.max_inflight["r"] == 1  # the second controller never stole it


# --------------------------------------------------------------------------- #
# Executors                                                                    #
# --------------------------------------------------------------------------- #


def test_local_executor_command_shape(tmp_path, monkeypatch):
    recorded = {}

    class FakeProc:
        def wait(self):
            return 0

        def terminate(self):
            pass

    def fake_popen(cmd, cwd=None, env=None):
        recorded.update(cmd=cmd, cwd=cwd, env=env)
        return FakeProc()

    monkeypatch.setattr(sc.subprocess, "Popen", fake_popen)
    executor = sc.LocalExecutor(tmp_path)
    ticket = make_ticket("r", tmp_path)
    ticket.config["env"] = {"TPU_VISIBLE_CHIPS": "2"}
    ticket.config["extra_args"] = ["--device-index", "0"]
    assert executor.execute(ticket) == 0
    assert recorded["cmd"] == [
        "uv", "run", "--no-sync", "python", "scripts/run_train_jax.py",
        "--config", "configs/r.yaml", "--device-index", "0",
    ]
    assert recorded["cwd"] == tmp_path
    assert recorded["env"]["TPU_VISIBLE_CHIPS"] == "2"

    ticket.config["trainer"] = "torch"
    executor.execute(ticket)
    assert recorded["cmd"][4] == "scripts/run_train.py"


def test_remote_executor_is_a_documented_stub():
    executor = sc.RemoteSliceExecutor(zone="us-east5-b")
    with pytest.raises(NotImplementedError, match="FLEET"):
        executor.execute(sc.Ticket(run_name="r", config={}))


def test_gcs_queue_command_shapes():
    """GcsQueue is exercised only for command construction: CAS must pin the
    generation (--if-generation-match) and create must use generation 0."""
    calls = []

    def runner(args, **kw):
        calls.append((args, kw.get("input")))
        stdout = {"describe": "42\n", "cat": sc.Ticket(run_name="r", config={}).to_json()}
        verb = args[3] if args[2] == "objects" else args[2]
        return subprocess.CompletedProcess(args, 0, stdout=stdout.get(verb, ""), stderr="")

    queue = sc.GcsQueue("gs://bucket/fleet/queue/", runner=runner)
    ticket = sc.Ticket(run_name="r", config={})
    queue.create(ticket)
    assert calls[-1][0][:4] == ["gcloud", "storage", "cp", "-"]
    assert "--if-generation-match=0" in calls[-1][0]
    assert calls[-1][0][4] == "gs://bucket/fleet/queue/r.json"

    got = queue.read("r")
    assert got is not None and got[1] == "42"

    assert queue.try_swap("r", "42", ticket) == "42"
    swap_call = calls[-2]  # last call is the post-write describe
    assert "--if-generation-match=42" in swap_call[0]
    assert json.loads(swap_call[1])["revision"] == ticket.revision
