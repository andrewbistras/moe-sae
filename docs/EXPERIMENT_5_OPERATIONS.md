# Experiment 5: Ready-Three Operations

This is the launch procedure for the `ready-three-v1` execution profile. It
runs the complete, unchanged scientific matrix—24 training trajectories and
18 held-out evaluations—on the three TPU slices that are already healthy.
Four historical Europe resource rows remain in the immutable plan for audit
continuity, but receive no tasks and are never probed by the controller.

This profile performs no qualification run and creates no TPU.

## Grant and cost boundary

The active fleet is:

| Resource ID | TPU | Allocation used | Role |
|---|---|---|---|
| `slice-sae-a` | `gemma4-horizon-v5e-usc-a` | existing `us-central1-a` spot v5e grant | SAE and transcoder seed 0 |
| `slice-sae-b` | `gemma4-horizon-harvest-0728` | existing `us-central2-b` on-demand v4 grant | pooled seeds 1–2 |
| `slice-transcoder-b` | `gemma4-horizon-v4-spot-trans-b` | existing `us-central2-b` spot v4 grant | pooled seeds 1–2 |

Cloud Quotas is not evidence of TRC funding. Do not create an additional TPU
merely because the quota page shows headroom; the award email is the TPU
allowlist. TPU usage in this table is grant-funded. The campaign's USD 50
cash ceiling covers the controller, disks, Filestore, GCS, operations, and
networking.

The seed assignment keeps every H/F comparison on one accelerator generation:
seed 0 is v5e; seeds 1 and 2 are v4.

## Strict order

1. Commit and deploy one clean final SHA to the controller and all three TPU
   slices.
2. Prove that no workload is running on those slices.
3. Empty and remount each worker's `/dev/shm` at 180 GiB.
4. Attest and probe all three TPU incarnations before hydrating any cache.
5. Generate the create-only evidence override.
6. Generate and validate the create-only campaign plan.
7. Start the controller as a durable systemd user unit.
8. Verify that the first three attempts have been admitted, then monitor the
   immutable status and journal.

Changing code after step 1 invalidates the evidence. Commit and redeploy, then
repeat steps 3–8 with a new commit-scoped evidence identity.

## Deploy the exact final commit

The dependency lock and TPU image are unchanged, so this campaign needs an
exact Git-bundle update, not another OS/bootstrap cycle. From the clean
workstation checkout:

```bash
export GCP_PROJECT='neural-dragnet'
export CONTROLLER_ZONE='us-central2-b'
export PROJECT_COMMIT="$(git rev-parse HEAD)"
export SHORT_COMMIT="${PROJECT_COMMIT:0:12}"
export DEPLOY_BUNDLE="/tmp/moe-sae-$SHORT_COMMIT.bundle"

test -z "$(git status --porcelain=v1 --untracked-files=all)"
git bundle create "$DEPLOY_BUNDLE" HEAD
git bundle verify "$DEPLOY_BUNDLE"

gcloud compute scp \
  --project="$GCP_PROJECT" \
  --zone="$CONTROLLER_ZONE" \
  "$DEPLOY_BUNDLE" \
  "gemma4-horizon-controller:/tmp/moe-sae-$SHORT_COMMIT.bundle"

gcloud compute ssh gemma4-horizon-controller \
  --project="$GCP_PROJECT" \
  --zone="$CONTROLLER_ZONE" \
  --command="set -eu
    sudo install -d -m 0700 -o tpuops -g tpuops /home/tpuops/.cache/moe-sae-deploy/$SHORT_COMMIT
    sudo install -m 0600 -o tpuops -g tpuops /tmp/moe-sae-$SHORT_COMMIT.bundle /home/tpuops/.cache/moe-sae-deploy/$SHORT_COMMIT/repository.bundle
    sudo -u tpuops test -z \"\$(sudo -u tpuops git -C /home/tpuops/src/moe-sae status --porcelain=v1 --untracked-files=all)\"
    sudo -u tpuops git -C /home/tpuops/src/moe-sae fetch /home/tpuops/.cache/moe-sae-deploy/$SHORT_COMMIT/repository.bundle HEAD
    sudo -u tpuops git -C /home/tpuops/src/moe-sae checkout --detach FETCH_HEAD
    test \"\$(sudo -u tpuops git -C /home/tpuops/src/moe-sae rev-parse HEAD)\" = '$PROJECT_COMMIT'
    sudo -u tpuops test -z \"\$(sudo -u tpuops git -C /home/tpuops/src/moe-sae status --porcelain=v1 --untracked-files=all)\""
```

Then SSH to the controller, become `tpuops`, start Bash, and copy that same
bundle to every worker:

```bash
sudo -iu tpuops
exec bash -l
test -n "$BASH_VERSION"

export GCP_PROJECT='neural-dragnet'
export PROJECT_COMMIT="$(git -C /home/tpuops/src/moe-sae rev-parse HEAD)"
export SHORT_COMMIT="${PROJECT_COMMIT:0:12}"
export DEPLOY_BUNDLE="/home/tpuops/.cache/moe-sae-deploy/$SHORT_COMMIT/repository.bundle"

for spec in \
  'gemma4-horizon-v5e-usc-a us-central1-a' \
  'gemma4-horizon-harvest-0728 us-central2-b' \
  'gemma4-horizon-v4-spot-trans-b us-central2-b'
do
  read -r tpu_name tpu_zone <<< "$spec"
  remote_bundle="/tmp/moe-sae-$SHORT_COMMIT.bundle"

  gcloud compute tpus tpu-vm scp "$DEPLOY_BUNDLE" \
    "abistras@$tpu_name:$remote_bundle" \
    --project="$GCP_PROJECT" \
    --zone="$tpu_zone" \
    --worker=all \
    --internal-ip \
    --quiet

  gcloud compute tpus tpu-vm ssh "abistras@$tpu_name" \
    --project="$GCP_PROJECT" \
    --zone="$tpu_zone" \
    --worker=all \
    --internal-ip \
    --quiet \
    --command="set -eu
      test -z \"\$(git -C /home/abistras/moe-sae status --porcelain=v1 --untracked-files=all)\"
      git -C /home/abistras/moe-sae fetch $remote_bundle HEAD
      git -C /home/abistras/moe-sae checkout --detach FETCH_HEAD
      test \"\$(git -C /home/abistras/moe-sae rev-parse HEAD)\" = '$PROJECT_COMMIT'
      test -z \"\$(git -C /home/abistras/moe-sae status --porcelain=v1 --untracked-files=all)\""
done
```

## Controller variables

Continue on `gemma4-horizon-controller` as `tpuops` in Bash:

```bash
test -n "$BASH_VERSION"
export GCP_PROJECT='neural-dragnet'
export EXECUTION_PROFILE='ready-three-v1'
export CONTROLLER_REPO='/home/tpuops/src/moe-sae'
export CONTROLLER_PYTHON="$CONTROLLER_REPO/.venv/bin/python-direct"
export R8_PLAN='/mnt/moe-sae-control/horizon-10m/contracts/direct-campaign-2f778bb-r8.yaml'
export TPU_REPO='/home/abistras/moe-sae'
export REMOTE_UV='/home/abistras/.local/bin/uv'
export TPU_PROD='/home/abistras/moe-sae/artifacts/gemma4_v1/production'
export CACHE_ROOT='/dev/shm/moe-sae-local/horizon-10m/cache'

export PROJECT_COMMIT="$(git -C "$CONTROLLER_REPO" rev-parse HEAD)"
export SHORT_COMMIT="${PROJECT_COMMIT:0:12}"
export EVIDENCE_ATTEMPT_ID="${EVIDENCE_ATTEMPT_ID:-a1}"
export EVIDENCE_RUN_ID="gemma4-experiment5-$EXECUTION_PROFILE-$SHORT_COMMIT"

test "${#PROJECT_COMMIT}" -eq 40
[[ "$EVIDENCE_ATTEMPT_ID" =~ ^[a-z][a-z0-9-]{0,15}$ ]]
test -z "$(git -C "$CONTROLLER_REPO" status --porcelain=v1 --untracked-files=all)"
test -x "$CONTROLLER_PYTHON"
```

The same command must print the same SHA on all four workers of all three
slices:

```bash
for spec in \
  'gemma4-horizon-v5e-usc-a us-central1-a' \
  'gemma4-horizon-harvest-0728 us-central2-b' \
  'gemma4-horizon-v4-spot-trans-b us-central2-b'
do
  read -r tpu_name tpu_zone <<< "$spec"
  gcloud compute tpus tpu-vm ssh "abistras@$tpu_name" \
    --project="$GCP_PROJECT" \
    --zone="$tpu_zone" \
    --worker=all \
    --internal-ip \
    --quiet \
    --command="set -eu
      test \"\$(git -C $TPU_REPO rev-parse HEAD)\" = '$PROJECT_COMMIT'
      test -z \"\$(git -C $TPU_REPO status --porcelain=v1 --untracked-files=all)\"
      printf '%s %s\n' \"\$(hostname -s)\" \"\$(git -C $TPU_REPO rev-parse HEAD)\""
done
```

Every printed SHA must equal `PROJECT_COMMIT`. Stop if any checkout is dirty.

## Empty-cache capacity preparation

The evidence probe requires at least 176 GiB free on every worker. A resident
capture correctly makes it fail, so clear only the bounded, reproducible cache
and remount `/dev/shm` before probing. The authoritative captures remain in
GCS and the regional mirror.

First verify there is no active Experiment 5 or horizon worker:

```bash
if systemctl --user list-units --state=running --no-legend --plain \
  'moe-sae-experiment5-*' | grep -q .; then
  echo 'an Experiment 5 controller unit is still running' >&2
  exit 1
fi

for spec in \
  'gemma4-horizon-v5e-usc-a us-central1-a' \
  'gemma4-horizon-harvest-0728 us-central2-b' \
  'gemma4-horizon-v4-spot-trans-b us-central2-b'
do
  read -r tpu_name tpu_zone <<< "$spec"
  gcloud compute tpus tpu-vm ssh "abistras@$tpu_name" \
    --project="$GCP_PROJECT" \
    --zone="$tpu_zone" \
    --worker=all \
    --internal-ip \
    --quiet \
    --command='! pgrep -af "[t]rain_contribution.py|[c]ontribution_training_worker|[e]xperiment5_evaluation_worker|[e]valuate_experiment5|[r]un_experiment5_campaign|[p]robe_controller_identity"'
done
```

Then clear and remount all workers:

```bash
for spec in \
  'gemma4-horizon-v5e-usc-a us-central1-a' \
  'gemma4-horizon-harvest-0728 us-central2-b' \
  'gemma4-horizon-v4-spot-trans-b us-central2-b'
do
  read -r tpu_name tpu_zone <<< "$spec"
  gcloud compute tpus tpu-vm ssh "abistras@$tpu_name" \
    --project="$GCP_PROJECT" \
    --zone="$tpu_zone" \
    --worker=all \
    --internal-ip \
    --quiet \
    --command="set -eu
      sudo rm -rf -- $CACHE_ROOT
      sudo mount -o remount,size=180G /dev/shm
      sudo install -d -m 0750 -o \$(id -un) -g \$(id -gn) $CACHE_ROOT
      test \"\$(find $CACHE_ROOT -mindepth 1 -maxdepth 1 -print -quit)\" = ''
      findmnt -rn -o TARGET,SOURCE,FSTYPE,OPTIONS --target $CACHE_ROOT
      df -B1 --output=avail $CACHE_ROOT"
done
```

Do not hydrate a cache between this step and evidence generation.

## Attest and probe the three active slices

This helper freezes the exact TPU description and then collectively proves
the controller-to-JAX mapping and empty-cache capacity:

```bash
attest_and_probe() {
  local resource_id="$1"
  local tpu_name="$2"
  local tpu_zone="$3"
  local accelerator_type="$4"
  local provisioning_model="$5"

  local assignment_id="experiment5-$EXECUTION_PROFILE-$SHORT_COMMIT-$EVIDENCE_ATTEMPT_ID-$resource_id"
  local probe_id="probe-$assignment_id"
  local assignment_root="$TPU_PROD/horizon-10m/topology/$tpu_name/assignments/$assignment_id"
  local probe_root="$assignment_root/probes/$probe_id"

  gcloud compute tpus tpu-vm ssh "abistras@$tpu_name" \
    --project="$GCP_PROJECT" \
    --zone="$tpu_zone" \
    --worker=0 \
    --internal-ip \
    --quiet \
    --command="$REMOTE_UV run --frozen --extra tpu python $TPU_REPO/scripts/tpu/attest_accelerator.py \
      --run-id $EVIDENCE_RUN_ID \
      --scientific-model hybrid \
      --condition all \
      --project $GCP_PROJECT \
      --tpu-name $tpu_name \
      --zone $tpu_zone \
      --accelerator-type $accelerator_type \
      --provisioning-model $provisioning_model \
      --output-root $assignment_root"

  "$CONTROLLER_PYTHON" "$CONTROLLER_REPO/scripts/tpu/probe_controller_identity.py" \
    --project "$GCP_PROJECT" \
    --tpu-name "$tpu_name" \
    --zone "$tpu_zone" \
    --project-commit "$PROJECT_COMMIT" \
    --project-root "$TPU_REPO" \
    --remote-uv "$REMOTE_UV" \
    --contract "$assignment_root/accelerator-contract.json" \
    --probe-id "$probe_id" \
    --probe-root "$probe_root" \
    --cache-root "$CACHE_ROOT" \
    --required-lane-bytes 159000000000 \
    --required-free-bytes 188978561024 \
    --ssh-user abistras \
    --internal-ip
}

attest_and_probe \
  slice-sae-a \
  gemma4-horizon-v5e-usc-a \
  us-central1-a \
  v5litepod-16 \
  SPOT

attest_and_probe \
  slice-sae-b \
  gemma4-horizon-harvest-0728 \
  us-central2-b \
  v4-32 \
  ON_DEMAND

attest_and_probe \
  slice-transcoder-b \
  gemma4-horizon-v4-spot-trans-b \
  us-central2-b \
  v4-32 \
  SPOT
```

These artifacts are create-only. Never overwrite partial or prior evidence.
After a failed probe, increment `EVIDENCE_ATTEMPT_ID` (for example, `a2`) and
rerun the complete three-resource evidence sequence. `EVIDENCE_RUN_ID` remains
commit-bound as required by the validator.

## Generate the evidence override

Set the exact controller-visible probe roots:

```bash
export SAE_A_PROBE="/mnt/moe-sae-usc1/horizon-10m/topology/gemma4-horizon-v5e-usc-a/assignments/experiment5-$EXECUTION_PROFILE-$SHORT_COMMIT-$EVIDENCE_ATTEMPT_ID-slice-sae-a/probes/probe-experiment5-$EXECUTION_PROFILE-$SHORT_COMMIT-$EVIDENCE_ATTEMPT_ID-slice-sae-a"
export SAE_B_PROBE="/mnt/moe-sae-control/horizon-10m/topology/gemma4-horizon-harvest-0728/assignments/experiment5-$EXECUTION_PROFILE-$SHORT_COMMIT-$EVIDENCE_ATTEMPT_ID-slice-sae-b/probes/probe-experiment5-$EXECUTION_PROFILE-$SHORT_COMMIT-$EVIDENCE_ATTEMPT_ID-slice-sae-b"
export TRANS_B_PROBE="/mnt/moe-sae-control/horizon-10m/topology/gemma4-horizon-v4-spot-trans-b/assignments/experiment5-$EXECUTION_PROFILE-$SHORT_COMMIT-$EVIDENCE_ATTEMPT_ID-slice-transcoder-b/probes/probe-experiment5-$EXECUTION_PROFILE-$SHORT_COMMIT-$EVIDENCE_ATTEMPT_ID-slice-transcoder-b"

export CONTRACT_ROOT='/mnt/moe-sae-control/experiment5/contracts'
export OVERRIDE="$CONTRACT_ROOT/evidence-$EXECUTION_PROFILE-$SHORT_COMMIT-$EVIDENCE_ATTEMPT_ID.yaml"

"$CONTROLLER_PYTHON" "$CONTROLLER_REPO/scripts/tpu/build_experiment5_evidence_overrides.py" \
  --r8-plan "$R8_PLAN" \
  --project-commit "$PROJECT_COMMIT" \
  --execution-profile "$EXECUTION_PROFILE" \
  --probe-root "slice-sae-a=$SAE_A_PROBE" \
  --probe-root "slice-sae-b=$SAE_B_PROBE" \
  --probe-root "slice-transcoder-b=$TRANS_B_PROBE" \
  --output "$OVERRIDE"
```

The generator rejects missing, extra, stale-commit, wrong-incarnation,
wrong-topology, wrong-provisioning, or wrong-cache evidence. It derives all
hashes and refuses to overwrite an existing output.

## Generate and validate the campaign plan

Set `ALREADY_SPENT_USD` explicitly from the Experiment 5 ledger at launch,
including any bounded cleanup exposure. There is no built-in placeholder:

```bash
: "${ALREADY_SPENT_USD:?export ALREADY_SPENT_USD from the current Experiment 5 ledger}"
export PLAN="$CONTRACT_ROOT/campaign-$EXECUTION_PROFILE-$SHORT_COMMIT-$EVIDENCE_ATTEMPT_ID.yaml"

"$CONTROLLER_PYTHON" -m moe_sae.experiment5_plan_builder \
  --r8-plan "$R8_PLAN" \
  --project-commit "$PROJECT_COMMIT" \
  --execution-profile "$EXECUTION_PROFILE" \
  --resource-evidence-overrides "$OVERRIDE" \
  --already-spent-usd "$ALREADY_SPENT_USD" \
  --output "$PLAN"

"$CONTROLLER_PYTHON" "$CONTROLLER_REPO/scripts/tpu/run_experiment5_campaign.py" \
  validate \
  --plan "$PLAN"

sha256sum "$OVERRIDE" "$PLAN"
```

Validation must report exactly seven resource rows, 24 trajectories, and 18
evaluations. The controller will create workers only for the three active
resource IDs.

## Durable launch

Confirm user lingering once:

```bash
loginctl show-user "$(id -un)" -p Linger --value
```

It must print `yes`. Then launch a transient user service. It survives SSH
disconnects, writes to the journal, retries a controller-process failure
three times, and leaves all scientific state in the immutable campaign root:

```bash
export LAUNCH_ATTEMPT_ID="${LAUNCH_ATTEMPT_ID:-a1}"
[[ "$LAUNCH_ATTEMPT_ID" =~ ^[a-z][a-z0-9-]{0,15}$ ]]
export CAMPAIGN_UNIT="moe-sae-experiment5-$SHORT_COMMIT-$LAUNCH_ATTEMPT_ID"

if systemctl --user status "$CAMPAIGN_UNIT" >/dev/null 2>&1; then
  echo "unit already exists; increment LAUNCH_ATTEMPT_ID" >&2
  exit 1
fi

systemd-run --user \
  --unit="$CAMPAIGN_UNIT" \
  --description="Gemma Experiment 5 ready-three $SHORT_COMMIT" \
  --property=Type=exec \
  --property=Restart=on-failure \
  --property=RestartSec=30s \
  --property=StartLimitIntervalSec=3600 \
  --property=StartLimitBurst=3 \
  "$CONTROLLER_PYTHON" \
  "$CONTROLLER_REPO/scripts/tpu/run_experiment5_campaign.py" \
  run \
  --plan "$PLAN"
```

Immediately verify:

```bash
systemctl --user status "$CAMPAIGN_UNIT" --no-pager
journalctl --user -u "$CAMPAIGN_UNIT" -n 100 --no-pager

"$CONTROLLER_PYTHON" "$CONTROLLER_REPO/scripts/tpu/run_experiment5_campaign.py" \
  status \
  --plan "$PLAN"
```

Do not disconnect until the journal shows admitted attempts on all three
active slices or a clear fail-closed error.

## Monitoring and completion

Use:

```bash
journalctl --user \
  -u "$CAMPAIGN_UNIT" \
  -f
```

and, in another shell:

```bash
watch -n 60 \
  "$CONTROLLER_PYTHON $CONTROLLER_REPO/scripts/tpu/run_experiment5_campaign.py status --plan $PLAN"
```

The status includes completed training/evaluation counts, the durable campaign
clock, conservative infrastructure accrual, watchdog projection, attempt
ledger, and any budget stop.

Completion requires:

- all 24 final training records;
- all 18 paired held-out evaluation records;
- the sealed Experiment 5 report;
- no missing or duplicate panel/seed/condition/milestone cells; and
- preserved per-run MFU, tokens/s, chip-seconds, wall time, exact L0, and dead
  rate.

After sealing and copying the final report, stop the transient unit if it has
not exited, then shut down the controller and delete any explicitly disposable
storage identified in the cost ledger. Never delete the sealed GCS captures,
checkpoints, evidence, plan, controller state, or report.
