# Gemma 10M horizon supervisor

`scripts/tpu/run_horizon.py` is the cloud-resident control loop for the
learning-horizon run. It does not contain training, evaluation, capture, or
report logic. Every effect is an argv from one immutable YAML/JSON launch
plan, executed without a shell.

The v1 plan is deliberately exact:

- one train and one eval target-capture task;
- one train and one eval transcoder-overlay task;
- at least two region mirrors;
- exactly SAE and transcoder A/R/B/F training trajectories;
- every trajectory evaluated at 2M, 5M, and 10M tokens; and
- one report for each scientific model.

Every train command must contain `{attempt_id}` and `{segment_seconds}`.
`segment.seconds` cannot exceed 3,600. A segment may continue only when its
declared checkpoint pointer changes; a completed trajectory is recognized
only from its declared immutable completion file. Numerics, identity/hash,
schema, range, overflow, and token-drop failures are terminal. Only classified
Spot preemption, resource unavailability, and SSH transport loss are retried.

Private v6e resources use `queued_spot` or `direct_spot`. The controller
derives a new TPU name from the globally increasing attempt ID, passes both
`--spot` and `--internal-ips`, validates a four-endpoint `v6e-16`, and polls
only exact `describe` operations. It never lists a TPU collection or a GCS
prefix to infer state.

Validate before installing the service:

```bash
uv run python scripts/tpu/run_horizon.py validate \
  --plan /mnt/moe-sae-control/horizon-10m/launch-plan.yaml
```

For an ordinary detached launch:

```bash
nohup uv run python scripts/tpu/run_horizon.py run \
  --plan /mnt/moe-sae-control/horizon-10m/launch-plan.yaml \
  >/mnt/moe-sae-control/horizon-10m/controller.log 2>&1 &
```

For systemd, install `deploy/systemd/moe-sae-horizon.service`, then create
`/etc/moe-sae/horizon.env`:

```text
HORIZON_REPO=/home/tpuops/src/moe-sae
HORIZON_PYTHON=/home/tpuops/src/moe-sae/.venv/bin/python
HORIZON_PLAN=/mnt/moe-sae-control/horizon-10m/launch-plan.yaml
TPU_SSH_USER=abistras
```

The supplied unit runs as `tpuops` (UID/GID 2001), matching ownership of the
shared control tree. Its TPU transports explicitly target the existing
`abistras` Linux account on each worker via `TPU_SSH_USER`; this prevents the
controller account name from silently creating/using a different remote user.
Keep the checkout, virtual environment, gcloud configuration, and SSH key
material readable by `tpuops`.

Use `KillMode=control-group`, as in the supplied unit. The state root must be a
small POSIX control volume, not GCS. Attempt intents/results, completion
witnesses, failures, TPU incarnation bindings, releases, and the final record
are all create-only and self-hashed. Run status is read-only:

```bash
uv run python scripts/tpu/run_horizon.py status \
  --plan /mnt/moe-sae-control/horizon-10m/launch-plan.yaml
```

## Task object contract

Each task contains `id`, `kind`, `depends_on`, an argv `command`, optional
argv-array `prepare_commands`, non-secret literal `environment`, an immutable
`completion`, a kind-specific `role`, and optional `resource_id`. Training
also contains:

```yaml
segment:
  seconds: 3500
  max_segments: 16
  boundary_exit_codes: [0, 124, 137, 143]
  checkpoint:
    path: /mnt/control/checkpoints/latest.json
    format: json
    expected: {}
```

Completion JSON checks use dotted top-level paths in `expected`; set
`self_hash_field` when the producer publishes a canonical self-hash. The
following placeholders are supported in argv values:
`{attempt_id}`, `{attempt_number}`, `{campaign_id}`, `{plan_sha256}`,
`{segment_seconds}`, `{task_id}`, `{resource_id}`, `{tpu_name}`,
`{tpu_zone}`, and `{queued_resource}`.
