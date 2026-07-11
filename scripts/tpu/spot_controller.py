#!/usr/bin/env python
"""Spot-fleet controller: a run queue + claim/lease loop for many training
runs across multiple preemptible TPU slices (ops tooling for spec §20.4; see
docs/FLEET.md for lifecycle, lease semantics, and the v6e-64 runbook notes).

    uv run python scripts/tpu/spot_controller.py submit --run-name discovery_A_seed1 \\
        --config configs/discovery_A_seed1.yaml --trainer jax --priority 1
    uv run python scripts/tpu/spot_controller.py run --queue queue/ [--drain]
    uv run python scripts/tpu/spot_controller.py status --queue queue/

This is OPS tooling only: it launches the existing frozen trainers
(scripts/run_train_jax.py / scripts/run_train.py) with a ticket's config and
never touches training math, data order, selection, or metrics formats. A
retried run restarts the identical frozen batch sequence (spec §10), so a
retry is scientifically identical to a first attempt — only compute is lost.

Decisions (ops, no scientific weight):
- Coordination is queue-only. N controllers on N slices share nothing but the
  queue directory; there is no leader, no RPC, no controller registry. A
  controller that dies (spot preemption) simply stops heartbeating and its
  ticket's lease lapses; any surviving controller recovers it.
- Mutual exclusion is compare-and-swap on an opaque version token. LocalQueue
  implements CAS by atomically renaming the ticket file into a private .swap/
  holding dir (only one os.rename on the same source succeeds), checking the
  embedded `revision` counter, and renaming back. GcsQueue maps the same
  interface onto GCS object generations via `gcloud storage cp
  --if-generation-match` (the only atomic primitive gcloud exposes); it is a
  thin subprocess wrapper with an injectable runner and is not used by tests.
- Failure vs preemption: a nonzero returncode from the executor is a
  *failure* → attempt += 1, exponential backoff, requeue (max_attempts →
  failed). An *exception* from the executor — or the controller dying
  outright — is preemption-shaped: the ticket is abandoned untouched and
  recovered by lease expiry, which also charges attempt += 1 (per task
  contract: failure and preemption both increment).
- Graceful shutdown (SIGTERM/SIGINT, installed by the CLI only) terminates
  the child and releases the lease back to pending *without* charging an
  attempt: an operator draining a slice is neither a failure nor a
  preemption.
- Success requires the artifact, not just returncode 0: the ticket's
  out_dir must contain final/theta_star.json (the θ*-freeze artifact every
  trainer writes last, spec §20.4/§6-freeze). rc==0 without it is a failure.
- Stdlib only, so the controller runs on a bare TPU VM before any project
  deps are installed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple, Union

log = logging.getLogger("spot_controller")

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"
STATES = (PENDING, RUNNING, DONE, FAILED)

Version = Union[int, str]  # LocalQueue: revision int; GcsQueue: generation str


# --------------------------------------------------------------------------- #
# Ticket                                                                       #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Ticket:
    """One run. Serialized as <run_name>.json in the queue directory.

    `config` is opaque to the queue; the executors read:
      config["config"]     — RunConfig YAML path (required)
      config["out_dir"]    — run output dir, artifact-checked (required)
      config["trainer"]    — "jax" (run_train_jax.py) | "torch" (run_train.py)
      config["extra_args"] — optional list of extra CLI args
      config["env"]        — optional {name: value} env overrides
                             (e.g. TPU_VISIBLE_CHIPS)
    """

    run_name: str
    config: dict
    priority: int = 0
    state: str = PENDING
    attempt: int = 0
    max_attempts: int = 5
    lease_expiry: float = 0.0
    worker: Optional[str] = None
    not_before: float = 0.0
    revision: int = 0
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Ticket":
        raw = json.loads(text)
        return cls(**{f.name: raw[f.name] for f in dataclasses.fields(cls) if f.name in raw})


def default_artifact_check(ticket: Ticket) -> bool:
    """§20.4 completion artifact: the frozen θ* the trainer writes last."""
    return (Path(ticket.config["out_dir"]) / "final" / "theta_star.json").exists()


# --------------------------------------------------------------------------- #
# Queue backends                                                               #
# --------------------------------------------------------------------------- #


class QueueBackend:
    """Compare-and-swap ticket store. All controller coordination goes
    through these four operations; see docs/FLEET.md."""

    def create(self, ticket: Ticket) -> None:
        raise NotImplementedError

    def names(self) -> list[str]:
        raise NotImplementedError

    def read(self, name: str) -> Optional[Tuple[Ticket, Version]]:
        raise NotImplementedError

    def try_swap(self, name: str, version: Version, new: Ticket) -> Optional[Version]:
        """Replace ticket `name` iff its current version matches. Returns the
        new version token on success, None on any contention/absence."""
        raise NotImplementedError

    def recover_orphans(self, max_age_seconds: float = 30.0) -> None:
        """Repair artifacts of a controller that died mid-swap (no-op where
        CAS is server-side, e.g. GCS)."""


class LocalQueue(QueueBackend):
    """Directory of <run_name>.json tickets; CAS via atomic os.rename.

    Swap protocol: rename canonical → .swap/<name>.<nonce>.json (atomic; at
    most one renamer of the same inode wins), verify `revision`, write the
    replacement to a temp file, rename temp over the held file, rename the
    held file back to canonical. Readers only ever see complete files. A
    crash mid-swap leaves the ticket in .swap/; recover_orphans() renames
    stale holds back after max_age_seconds.
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.swap = self.root / ".swap"
        self.swap.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def create(self, ticket: Ticket) -> None:
        path = self._path(ticket.run_name)
        if path.exists():
            raise FileExistsError(f"ticket already queued: {path}")
        tmp = self.swap / f"{ticket.run_name}.{uuid.uuid4().hex}.create"
        tmp.write_text(ticket.to_json())
        os.rename(tmp, path)

    def names(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def read(self, name: str) -> Optional[Tuple[Ticket, Version]]:
        try:
            ticket = Ticket.from_json(self._path(name).read_text())
        except FileNotFoundError:
            return None  # mid-swap elsewhere, or gone
        return ticket, ticket.revision

    def try_swap(self, name: str, version: Version, new: Ticket) -> Optional[Version]:
        canon = self._path(name)
        hold = self.swap / f"{name}.{uuid.uuid4().hex}.json"
        try:
            os.rename(canon, hold)  # atomic claim of the file itself
        except FileNotFoundError:
            return None
        current = Ticket.from_json(hold.read_text())
        if current.revision != version:
            os.rename(hold, canon)  # lost the CAS; put it back untouched
            return None
        new.revision = current.revision + 1
        tmp = self.swap / f"{name}.{uuid.uuid4().hex}.next"
        tmp.write_text(new.to_json())
        os.rename(tmp, hold)  # hold now has complete new content
        os.rename(hold, canon)
        return new.revision

    def recover_orphans(self, max_age_seconds: float = 30.0) -> None:
        now = time.time()
        for orphan in self.swap.glob("*.json"):
            try:
                if now - orphan.stat().st_mtime < max_age_seconds:
                    continue
                name = orphan.name.rsplit(".", 2)[0]  # <name>.<nonce>.json
                canon = self._path(name)
                if canon.exists():
                    orphan.unlink()  # canonical was restored elsewhere
                else:
                    os.rename(orphan, canon)
                    log.warning("recovered orphaned ticket %s", name)
            except FileNotFoundError:
                pass  # another controller's janitor got there first


class GcsQueue(QueueBackend):
    """Same interface on gs://bucket/prefix via `gcloud storage` subprocess.

    Version token = the object *generation*; CAS = `cp --if-generation-match`
    (create uses generation 0 == only-if-absent). Reads describe first, cat
    second: if the object turns over in between, the stale generation makes
    the next swap fail, which is safe. try_swap re-describes to learn the new
    generation; if a competitor writes in that window the returned token is
    stale and later swaps fail — again safe (treated as a lost lease).
    `runner` is injectable for tests; no gcloud is invoked at import time.
    """

    def __init__(self, url: str, runner: Callable[..., subprocess.CompletedProcess] = None):
        self.url = url.rstrip("/")
        self.runner = runner or (
            lambda args, **kw: subprocess.run(args, capture_output=True, text=True, **kw)
        )

    def _obj(self, name: str) -> str:
        return f"{self.url}/{name}.json"

    def _generation(self, name: str) -> Optional[str]:
        res = self.runner(
            ["gcloud", "storage", "objects", "describe", self._obj(name),
             "--format=value(generation)"]
        )
        return res.stdout.strip() if res.returncode == 0 else None

    def create(self, ticket: Ticket) -> None:
        res = self.runner(
            ["gcloud", "storage", "cp", "-", self._obj(ticket.run_name),
             "--if-generation-match=0"],
            input=ticket.to_json(),
        )
        if res.returncode != 0:
            raise FileExistsError(f"ticket already queued: {self._obj(ticket.run_name)}")

    def names(self) -> list[str]:
        res = self.runner(["gcloud", "storage", "ls", f"{self.url}/*.json"])
        if res.returncode != 0:
            return []
        return sorted(line.rsplit("/", 1)[-1][: -len(".json")]
                      for line in res.stdout.split() if line.endswith(".json"))

    def read(self, name: str) -> Optional[Tuple[Ticket, Version]]:
        gen = self._generation(name)
        if gen is None:
            return None
        res = self.runner(["gcloud", "storage", "cat", self._obj(name)])
        if res.returncode != 0:
            return None
        return Ticket.from_json(res.stdout), gen

    def try_swap(self, name: str, version: Version, new: Ticket) -> Optional[Version]:
        new.revision += 1  # informational under GCS; generation is the token
        res = self.runner(
            ["gcloud", "storage", "cp", "-", self._obj(name),
             f"--if-generation-match={version}"],
            input=new.to_json(),
        )
        return self._generation(name) if res.returncode == 0 else None


# --------------------------------------------------------------------------- #
# Executors                                                                    #
# --------------------------------------------------------------------------- #


class Executor:
    """execute(ticket) -> returncode. 0 = success (still artifact-checked).
    Nonzero = failure. Raising = preemption-shaped sudden death: the
    controller abandons the ticket and lease expiry recovers it.
    Optional terminate() is called on graceful controller shutdown."""

    def execute(self, ticket: Ticket) -> int:
        raise NotImplementedError

    def terminate(self) -> None:  # pragma: no cover - optional hook
        pass


class LocalExecutor(Executor):
    """Runs the frozen trainer as a subprocess on this host/slice.

    trainer "jax" → scripts/run_train_jax.py, "torch" → scripts/run_train.py;
    args are `--config <ticket.config['config']>` plus config["extra_args"].
    The controller passes nothing else — every scientific knob lives in the
    frozen RunConfig YAML, which the trainer itself validates.
    """

    SCRIPTS = {"jax": "scripts/run_train_jax.py", "torch": "scripts/run_train.py"}

    def __init__(self, repo_root: Union[str, Path],
                 python: Iterable[str] = ("uv", "run", "--no-sync", "python")):
        self.repo_root = Path(repo_root)
        self.python = list(python)
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def execute(self, ticket: Ticket) -> int:
        script = self.SCRIPTS[ticket.config.get("trainer", "torch")]
        cmd = self.python + [script, "--config", ticket.config["config"]]
        cmd += list(ticket.config.get("extra_args", []))
        env = {**os.environ, "PYTHONUNBUFFERED": "1", **ticket.config.get("env", {})}
        log.info("[%s] exec: %s", ticket.run_name, shlex.join(cmd))
        with self._lock:
            self._proc = subprocess.Popen(cmd, cwd=self.repo_root, env=env)
        try:
            return self._proc.wait()
        finally:
            with self._lock:
                self._proc = None

    def terminate(self) -> None:
        with self._lock:
            if self._proc is not None:
                self._proc.terminate()


class RemoteSliceExecutor(Executor):
    """SKETCH ONLY of the v6e-64 multi-host path — intentionally a stub.

    The real implementation (docs/FLEET.md, "The v6e-64 multi-host path")
    would, per execute(ticket):

      1. Ensure capacity:
         gcloud compute tpus queued-resources create <name> --spot \\
             --accelerator-type=v6e-64 --zone=<zone> --node-id=<name> ...
         and poll `describe` until ACTIVE (spot: may be SUSPENDED/preempted
         at any time — that is exactly what the lease mechanism absorbs).
      2. Launch one process per host:
         gcloud compute tpus tpu-vm ssh <name> --worker=all \\
             --command='cd ~/moe-sae && uv run --no-sync python \\
                 scripts/run_train_jax.py --config <ticket yaml>'
         (per-host process groups; jax.distributed handles the rest).
      3. Block on the ssh processes; map their exit status to a returncode;
         on slice preemption, raise so the controller abandons the ticket
         and lease expiry requeues it.

    Do not implement gcloud logic here until the v6e-64 regime is real; the
    provisioning stanzas live in scripts/tpu/00_provision.sh.
    """

    def __init__(self, zone: str, accelerator_type: str = "v6e-64"):
        self.zone = zone
        self.accelerator_type = accelerator_type

    def execute(self, ticket: Ticket) -> int:
        raise NotImplementedError(
            "RemoteSliceExecutor is a documented stub — see docs/FLEET.md "
            "('The v6e-64 multi-host path') for the runbook it sketches."
        )


# --------------------------------------------------------------------------- #
# Controller                                                                   #
# --------------------------------------------------------------------------- #


class _Heartbeat(threading.Thread):
    """Renews the running ticket's lease every `interval` seconds via CAS.
    If a renewal loses the CAS the lease is gone (another controller expired
    and re-claimed the ticket); `lost` is set and renewal stops."""

    def __init__(self, queue: QueueBackend, ticket: Ticket, version: Version,
                 lease_seconds: float, interval: float):
        super().__init__(daemon=True, name=f"heartbeat-{ticket.run_name}")
        self.queue, self.lease_seconds, self.interval = queue, lease_seconds, interval
        self.ticket, self.version = ticket, version
        self.lost = False
        self._halt = threading.Event()  # named to avoid Thread._stop()

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            renewed = dataclasses.replace(self.ticket, lease_expiry=time.time() + self.lease_seconds)
            new_version = self.queue.try_swap(self.ticket.run_name, self.version, renewed)
            if new_version is None:
                self.lost = True
                log.warning("[%s] lost lease during run", self.ticket.run_name)
                return
            self.ticket, self.version = renewed, new_version

    def stop(self) -> None:
        self._halt.set()
        self.join()


class Controller:
    """Claim → execute → settle loop. Safe to run N-way across slices with
    no coordination beyond the shared queue (docs/FLEET.md)."""

    def __init__(
        self,
        queue: QueueBackend,
        executor: Executor,
        worker: Optional[str] = None,
        lease_seconds: float = 300.0,
        heartbeat_seconds: Optional[float] = None,
        poll_seconds: float = 5.0,
        backoff_base: float = 60.0,
        backoff_cap: float = 3600.0,
        orphan_age_seconds: float = 30.0,
        artifact_check: Callable[[Ticket], bool] = default_artifact_check,
    ):
        self.queue = queue
        self.executor = executor
        self.worker = worker or f"{socket.gethostname()}:{os.getpid()}"
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds or lease_seconds / 3.0
        self.poll_seconds = poll_seconds
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.orphan_age_seconds = orphan_age_seconds
        self.artifact_check = artifact_check
        self._stop = threading.Event()

    # -- lifecycle helpers -------------------------------------------------- #

    def backoff(self, attempt: int) -> float:
        """Delay before retry number `attempt`+1 (attempt = completed tries)."""
        return min(self.backoff_cap, self.backoff_base * 2.0 ** max(0, attempt - 1))

    def _settle_failure(self, ticket: Ticket, version: Version, note: str) -> None:
        """attempt += 1; requeue with exponential backoff, or fail terminally."""
        attempt = ticket.attempt + 1
        if attempt >= ticket.max_attempts:
            new = dataclasses.replace(ticket, state=FAILED, attempt=attempt,
                                      worker=None, lease_expiry=0.0, note=note)
        else:
            new = dataclasses.replace(
                ticket, state=PENDING, attempt=attempt, worker=None, lease_expiry=0.0,
                not_before=time.time() + self.backoff(attempt), note=note,
            )
        if self.queue.try_swap(ticket.run_name, version, new) is None:
            log.warning("[%s] settle(%s) lost CAS; leaving to owner", ticket.run_name, note)
        else:
            log.info("[%s] %s → %s (attempt %d/%d)", ticket.run_name, note,
                     new.state, attempt, ticket.max_attempts)

    def _sweep_expired(self, now: float) -> None:
        """Return preempted runs (running, lease lapsed) to the queue."""
        for name in self.queue.names():
            got = self.queue.read(name)
            if got is None:
                continue
            ticket, version = got
            if ticket.state == RUNNING and ticket.lease_expiry <= now:
                self._settle_failure(ticket, version, "lease expired (preemption)")

    def _claim_one(self, now: float) -> Optional[Tuple[Ticket, Version]]:
        candidates = []
        for name in self.queue.names():
            got = self.queue.read(name)
            if got is None:
                continue
            ticket, version = got
            if ticket.state == PENDING and ticket.not_before <= now:
                candidates.append((-ticket.priority, ticket.attempt, name, ticket, version))
        for _, _, _, ticket, version in sorted(candidates, key=lambda c: c[:3]):
            claimed = dataclasses.replace(
                ticket, state=RUNNING, worker=self.worker,
                lease_expiry=time.time() + self.lease_seconds,
            )
            new_version = self.queue.try_swap(ticket.run_name, version, claimed)
            if new_version is not None:
                return claimed, new_version
        return None

    def _drained(self) -> bool:
        for name in self.queue.names():
            got = self.queue.read(name)
            if got is not None and got[0].state not in (DONE, FAILED):
                return False
        return True

    # -- execute one claimed ticket ----------------------------------------- #

    def _execute(self, ticket: Ticket, version: Version) -> None:
        hb = _Heartbeat(self.queue, ticket, version, self.lease_seconds, self.heartbeat_seconds)
        hb.start()
        rc: Optional[int] = None
        try:
            rc = self.executor.execute(ticket)
        except Exception:  # preemption-shaped: abandon; lease expiry recovers
            log.exception("[%s] executor died; abandoning to lease expiry", ticket.run_name)
        finally:
            hb.stop()
        ticket, version = hb.ticket, hb.version
        if hb.lost or rc is None:
            return  # not ours anymore / abandoned — never touch the ticket
        if rc == 0 and self.artifact_check(ticket):
            done = dataclasses.replace(ticket, state=DONE, worker=None,
                                       lease_expiry=0.0, note="")
            if self.queue.try_swap(ticket.run_name, version, done) is None:
                log.warning("[%s] done-CAS lost (lease raced); dropping result",
                            ticket.run_name)
            else:
                log.info("[%s] done", ticket.run_name)
        elif self._stop.is_set():
            # Graceful shutdown killed the child: release, don't charge.
            released = dataclasses.replace(ticket, state=PENDING, worker=None,
                                           lease_expiry=0.0, note="released on shutdown")
            if self.queue.try_swap(ticket.run_name, version, released) is None:
                log.warning("[%s] release lost CAS", ticket.run_name)
            else:
                log.info("[%s] lease released on shutdown", ticket.run_name)
        else:
            note = f"rc={rc}" if rc != 0 else "rc=0 but final/theta_star.json missing"
            self._settle_failure(ticket, version, note)

    # -- main loop ----------------------------------------------------------- #

    def run(self, drain: bool = False) -> None:
        """Loop until stop(); with drain=True, also return once every ticket
        is terminal (done/failed)."""
        log.info("controller %s up (lease=%.0fs)", self.worker, self.lease_seconds)
        while not self._stop.is_set():
            self.queue.recover_orphans(self.orphan_age_seconds)
            now = time.time()
            self._sweep_expired(now)
            claimed = self._claim_one(now)
            if claimed is not None:
                self._execute(*claimed)
                continue  # re-poll immediately; more work may be eligible
            if drain and self._drained():
                log.info("controller %s: queue drained", self.worker)
                return
            self._stop.wait(self.poll_seconds)

    def stop(self) -> None:
        self._stop.set()
        self.executor.terminate()


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def make_queue(url: str) -> QueueBackend:
    return GcsQueue(url) if url.startswith("gs://") else LocalQueue(url)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("submit", help="enqueue one run ticket")
    p.add_argument("--queue", required=True, help="queue dir or gs:// prefix")
    p.add_argument("--run-name", required=True)
    p.add_argument("--config", required=True, help="RunConfig YAML path")
    p.add_argument("--out-dir", default=None, help="default: runs/<run-name>")
    p.add_argument("--trainer", choices=("jax", "torch"), default="jax")
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=5)

    p = sub.add_parser("run", help="run a controller loop on this slice")
    p.add_argument("--queue", required=True)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--worker", default=None)
    p.add_argument("--lease-seconds", type=float, default=300.0)
    p.add_argument("--poll-seconds", type=float, default=5.0)
    p.add_argument("--backoff-base", type=float, default=60.0)
    p.add_argument("--drain", action="store_true",
                   help="exit once every ticket is done/failed")

    p = sub.add_parser("status", help="print the queue")
    p.add_argument("--queue", required=True)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    queue = make_queue(args.queue)

    if args.cmd == "submit":
        out_dir = args.out_dir or str(Path("runs") / args.run_name)
        queue.create(Ticket(
            run_name=args.run_name, priority=args.priority, max_attempts=args.max_attempts,
            config={"config": args.config, "out_dir": out_dir, "trainer": args.trainer},
        ))
        print(f"queued {args.run_name}")
        return 0

    if args.cmd == "status":
        for name in queue.names():
            got = queue.read(name)
            if got is None:
                continue
            t, _ = got
            lease = ""
            if t.state == RUNNING:
                lease = f" worker={t.worker} lease_in={t.lease_expiry - time.time():+.0f}s"
            print(f"{t.state:>8}  {name}  attempt={t.attempt}/{t.max_attempts}"
                  f" prio={t.priority}{lease}  {t.note}")
        return 0

    controller = Controller(
        queue, LocalExecutor(args.repo_root), worker=args.worker,
        lease_seconds=args.lease_seconds, poll_seconds=args.poll_seconds,
        backoff_base=args.backoff_base,
    )
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: controller.stop())
    controller.run(drain=args.drain)
    return 0


if __name__ == "__main__":
    sys.exit(main())
