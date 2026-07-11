# Fleet — running many frozen-spec runs on preemptible TPU slices

Operational design for the coming regime: dozens of runs (spec §20.4-style
suites, replications, the block-10 follow-up) spread across multiple
preemptible slices (spot v6e-64s in several regions, TRC-funded). Code:
`scripts/tpu/spot_controller.py`; tests: `tests/test_spot_controller.py`.
This is ops tooling only — the controller launches the frozen trainers
unchanged and cannot alter a training-step number or a batch order.

## Ticket lifecycle

One JSON ticket per run in a shared queue (local dir or `gs://` prefix).

```
                          submit
                            │
                            v
                       ┌─────────┐   claim (CAS: state→running,
        backoff over   │ pending │──── worker, lease_expiry) ────┐
        (not_before)   └─────────┘                               v
                          ^  ^  ^                           ┌─────────┐
    rc!=0 / rc==0-but-no- │  │  │                           │ running │──heartbeat──┐
    artifact: attempt+1,  │  │  └── released on shutdown ── └─────────┘  (renew     │
    backoff = base·2^     │  │      (SIGTERM: attempt NOT     │  │  │     lease)    │
    (attempt-1) ──────────┘  │      charged)                  │  │  └───────<───────┘
                             │                                │  │
    lease expired (slice     │                                │  │ rc==0 AND
    preempted / controller ──┘                                │  │ final/theta_star.json
    died): attempt+1, backoff                                 │  │ exists
                                                              │  v
                   attempt >= max_attempts                    │ ┌──────┐
                   (from either path)                         │ │ done │
                            │                                 │ └──────┘
                            v                                 │
                       ┌────────┐                             │
                       │ failed │<────────────────────────────┘
                       └────────┘
```

Success is the **artifact**, not the exit code: a run is `done` only when
`<out_dir>/final/theta_star.json` exists — the θ*-freeze file every trainer
writes last (spec §20.4, changelog #6). `rc==0` without it is a failure.

## Lease semantics

Every mutation is a **compare-and-swap** on a version token: an embedded
`revision` counter guarded by atomic `os.rename` for the local queue, the
object *generation* guarded by `gcloud storage cp --if-generation-match` for
GCS. There are no partial updates and no lock files.

- **Claim** — CAS `pending → running` with `worker` and
  `lease_expiry = now + lease_seconds` (default 300 s). Exactly one
  contender's CAS succeeds.
- **Heartbeat** — the owner renews the lease every `lease_seconds/3` via CAS.
  A heartbeat that loses its CAS means the lease was expired and re-claimed
  elsewhere; the owner marks itself `lost` and never touches the ticket
  again (its eventual result is dropped). This is what makes double-`done`
  impossible even under races.
- **Expiry** — any controller that sees `running` with `lease_expiry <= now`
  CASes it back to `pending` with `attempt += 1` and exponential backoff.
  This is the *only* preemption-recovery mechanism: a preempted slice does
  nothing on its way out.
- **Skew** — leases compare wall clocks across machines; keep
  `lease_seconds` (minutes) two orders of magnitude above realistic NTP skew
  (sub-second). Do not tune the lease down.
- **Crash mid-swap (local queue only)** — the ticket file sits in
  `queue/.swap/`; every controller's loop restores stale holds
  (`recover_orphans`). GCS CAS is server-side, so no orphan window exists.

## N controllers, N slices, zero coordination

Each slice runs one `spot_controller.py run` per schedulable unit (one per
chip today, one per slice for multi-host runs — chip pinning via the
ticket's `env: {TPU_VISIBLE_CHIPS: …}`). Controllers share **only the
queue**: no leader, no registry, no RPC, no heartbeat channel besides the
lease field itself. Any controller can claim, expire, or fail any ticket;
adding capacity is starting another controller, losing capacity is silent
(its leases lapse). Priorities are advisory (`-priority, attempt, name`
claim order) — starvation-free because backoff, not priority, gates retries.

Because data order is frozen (spec §10), a retry restarts the *identical*
batch sequence: retries are scientifically indistinguishable from first
attempts, and the queue never needs to know how far a dead run got.

## The v6e-64 multi-host path

`RemoteSliceExecutor` is a deliberate stub (raises `NotImplementedError`,
pointing here). What changes when one run spans a 16-host v6e-64:

- **Per-host process groups** — `LocalExecutor`'s single subprocess becomes
  `gcloud compute tpus tpu-vm ssh <node> --worker=all --command='… uv run
  --no-sync python scripts/run_train_jax.py --config …'`: one process per
  host, `jax.distributed.initialize()` forming the group. The controller
  still holds **one** lease for the whole slice-run; ssh exit statuses fold
  into one returncode, and a slice preemption kills the ssh sessions →
  exception → abandon → lease expiry requeues.
- **Orbax adoption trigger** — the current checkpoint writer is a
  single-process safetensors dump; it is correct as long as one process owns
  all parameters. The first run whose SAE state is *sharded across hosts* is
  the trigger to adopt Orbax async multi-host checkpointing. Do not adopt
  earlier: format churn with no consumer.
- **Multi-tier checkpoints** — spot slices die without notice, so
  checkpoints must be (1) frequent-and-local (host NVMe, every few minutes,
  survives nothing but is cheap), (2) periodic-to-GCS (survives the slice),
  (3) final artifacts to GCS (`final/`, the only tier the queue inspects).
  Mid-run checkpoints are only worth their disk once **stream resume**
  exists (the batch composer must be able to fast-forward to step *n* of the
  frozen order — cheap, since order is deterministic); until then retries
  restart from step 0 and tiers (1)/(2) stay disabled, as today.

## TRC operational notes

- **Queued-resource preemption behavior** — spot/preemptible TRC capacity
  can be deleted at any moment with no terminal notice; the queued-resource
  then reports `SUSPENDED`/deleted and must be re-created (re-`create` is a
  new queue position, possibly hours→days in a tight region). Treat
  "slice exists" as a cache, never as state: the run queue is the state.
  Spread controllers across the several regions TRC grants; the queue makes
  cross-region failover automatic.
- **Why every run must be resume-capable** — expected slice lifetime under
  spot is shorter than a discovery run; without restartability the fleet
  makes *negative* progress (attempts burn quota without converging).
  Today "resume" = full restart on the frozen data order (correct, wasteful);
  the economic crossover to true checkpoint-resume comes when
  (mean-time-to-preemption) < (run length) × (acceptable waste) — at that
  point implement stream fast-forward + tier (2) checkpoints before scaling
  the fleet further.
- Provisioning stanzas (bucket colocation, queued-resource create, spot
  flags) live in `scripts/tpu/00_provision.sh`; per-worker launch patterns
  in `scripts/tpu/03_discovery_suite.sh`.
