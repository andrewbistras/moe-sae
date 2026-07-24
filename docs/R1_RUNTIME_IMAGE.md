# DeepSeek-R1 v6e runtime image

This directory is the isolated, reproducible **build input** for the R1 TPU
runtime. It is not a qualified image and it cannot mint production evidence.
The shared repository project remains the sealed Gemma environment and still
pins MaxText at `48c7da3e28e8f57fe36ddf09d00eec53072a2ea0`.

## Static identity

`deploy/r1/runtime/` is a separate Python project and lock with these critical
pins:

| component | identity |
|---|---|
| Python | `3.12.11` |
| uv | `0.11.14` |
| MaxText base | `c03f8bd307d18049fa814fc8c104bd9ec8c292aa` |
| MaxText patches | `f5b4f810…`, `d78349f8…`, then `c1df8ec6…` |
| JAX / JAXLIB | `0.10.2` / `0.10.2` |
| libtpu | `0.0.43` |
| Packer / Google Compute plugin | `1.15.4` / `1.2.7` |
| target userspace | TPU-optimized Ubuntu `22.04`, `x86_64` |
| base resolver input | `projects/ubuntu-os-accelerator-images/global/images/family/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` |

The lock contains the complete upstream MaxText TPU extra. The bake then
applies all three content-bound R1 patches to a separate clean checkout and replaces
the pristine install with a deterministic, non-editable wheel. It builds that
wheel twice on the Linux baker with the base commit time as
`SOURCE_DATE_EPOCH`; unequal bytes abort the bake. The retained wheel,
canonical patch attestation, installed RECORD, and four patched source hashes
are measured together. Torch is absent. Every PyPI artifact in the lock has a
SHA-256, MaxText is an exact 40-character Git revision, and the lock has no
alternate index.

The uv artifact is the official
`uv-x86_64-unknown-linux-gnu.tar.gz` release asset with archive SHA-256
`f3b623eb0e6141a7053d571d59a0bdc341e0f238ea8f5f0b4815ddbec9a2a296`;
the extracted `uv` executable must hash to
`b5cbc3a3f35debad0b4770811efd190bcf460b654114d6a3f71e0ce298468e5d`.

`deploy/r1/bundle.toml` binds the byte hash and executable mode of the isolated
project/lock, deterministic-context authority, Packer template, build-only
provisioner, image measurer, systemd unit, and entrypoint. Its validator always
returns `production_qualified=false` and requires the explicit `UNBUILT` state.

Run the offline checks from repository root:

```bash
uv lock --check --project deploy/r1/runtime
uv run python - <<'PY'
from moe_sae.r1.runtime_bundle import validate_runtime_bundle

identity = validate_runtime_bundle(".")
print(identity.digest)
print(identity.as_dict())
PY
```

The conversion planner accepts only this isolated lock:

```python
from moe_sae.r1.checkpoint_conversion import attest_r1_runtime_lock

attest_r1_runtime_lock("deploy/r1/runtime")  # accepted R1 pin
attest_r1_runtime_lock(".")                  # rejects Gemma pin
```

## Deterministic context

`context-files.json` is an exact 35-file allowlist. It includes the isolated
runtime project, all three MaxText patches, the patch/restore/benchmark closure, the
R1 protocol/control/MFU modules needed by the worker, the fixed entrypoint,
service, and image measurement inputs. It excludes the root project and lock,
Gemma configuration/modules/CLI, old TPU bootstrap scripts, tests, docs, Git
metadata, ignored artifacts, and credentials.

After these files are committed, build the payload from the exact commit—not
from the checkout:

```bash
export R1_COMMIT="$(git rev-parse HEAD)"
uv run python deploy/r1/build_context.py \
  --repository . \
  --commit "$R1_COMMIT" \
  --output /tmp/moe-sae-r1-runtime.tar
```

The public builder accepts only the fixed production allowlist. The requested
commit must equal `HEAD`; the working-tree bundle must pass semantic validation;
and every bundle authority plus every context input must be byte/mode-identical
to that commit. Unrelated dirty files remain irrelevant. The builder then reads
every archive byte and executable mode with `git ls-tree`/`git show`,
rejects symlinks, submodules, LFS pointers, unsafe paths, duplicate/case-folding
paths, oversized content, and token-shaped secrets, then emits a deterministic
USTAR archive with sorted entries, zero uid/gid/mtime, and a canonical internal
manifest. Dirty, untracked, ignored, current-user, clock, cwd, and umask state
do not affect the archive.

## Image bake contract

Google documents the certified v6e/v5p image line as TPU-optimized Ubuntu
22.04 (kernel 6.8) in `ubuntu-os-accelerator-images`; the TPU-optimized Ubuntu
24.04 line is for TPU7x and is not this runtime's base. See the
[TPU OS images matrix](https://docs.cloud.google.com/tpu/docs/tpu-os-images).
The family above is only a controller resolver input. The Packer template
accepts one canonical, dated base-image selfLink from that exact project,
never an image family or independently supplied project/name. It derives the
selected project and image name from that selfLink. Before a real build, the
trusted operator/controller must resolve the family once to an exact image and
numeric GCE image ID, record the family-resolution API receipt, and create a
canonical `ImageRecipeV1`. The receipt must validate the API-reported family,
TPU license, `READY` status, and `X86_64` architecture; the image name alone is
not certification. Both exact values enter the cleared build environment and
must equal the recipe. Packer never consumes the family alias. The final
publication/API receipt still proves which base image Compute Engine actually
used. The recipe also binds the committed context tar, isolated
project/lock, MaxText revision, Python/uv/Packer versions, template,
provisioner, and service unit. Packer receives the canonical recipe and exact
runtime, recipe, and executed-provisioner hashes. The provisioner itself pins,
downloads, verifies, and extracts the sole official uv member; it rejects any
disagreement before installing the service.

The provisioner runs through `sudo env -i` with a fixed PATH and an enumerated
environment. It refuses a base image without its bounded shell/tool/CA surface;
it does not consult a moving apt repository. uv resolves only the frozen,
hashed lock while baking. The provisioner fetches only the exact MaxText
commit, applies all three hash-bound patches, builds twice without build isolation
using pinned Hatch tools, and installs from a permanent root-owned wheel. The
bake runs native `systemd-analyze verify` on the
installed unit; the Python substring/line scans are supplementary guardrails,
not claims of HCL or systemd syntax validity. Before extracting, a base-Python
verifier requires the exact context member allowlist, regular-file types,
normalized paths, modes, ownership/timestamps, bounded sizes, canonical
manifest, and manifest/content hashes. Extraction then ignores archive owner
and permission restoration. The project currently has no Cloud NAT. Its
short-lived Packer VM therefore
receives an ephemeral external IP solely for build-time egress to the exact
uv, Git, and locked PyPI inputs. Packer connects to it over the default VPC's
internal address, and the VM is deleted after the image is created. This avoids
a persistent NAT gateway and its hourly/data-processing charges.

At boot, the image performs no apt/pip/uv/Git mutation and no reboot. It has
one root-owned environment under `/opt/moe-sae-r1`, one fixed unprivileged
`moe_sae` user, and a fixed systemd argv:

```text
/opt/moe-sae-r1/runtime/bin/python -I -m moe_sae.r1.runtime_entrypoint \
  --metadata-key=moe-sae-r1-launch-ref
```

There is no shell command, `EnvironmentFile`, arbitrary metadata argv, baked
human credential, or SSH-owned lifetime. The service can write only its
state/cache/tmp/run paths and `/dev/shm`; it deliberately does not use
`PrivateDevices`, because that could hide `/dev/vfio/*` from libtpu.

The bake emits a diagnostic observation containing exact OS/dpkg hashes,
Python executable and shared-library hashes, uv hash, every installed Python
distribution's version and RECORD hash, and the size/SHA-256 of every actual
installed distribution file after verifying its RECORD entry. It imports the
runtime entrypoint, proves its resolved path remains inside the sealed venv,
and requires its bytes to equal the committed source. It also binds the local
wheel `direct_url.json`, retained wheel and patch attestations, installed
hashes for the four patched MaxText files, packaged `base.yml`, five critical
benchmark/restore modules, context/recipe/project/lock/unit identities, and
`production_qualified=false`. This observation is input to the existing
`ImageManifestV1`; it is not itself that manifest.

## Deliberate stop boundary

The image now contains the executable benchmark and patched MaxText, but
`runtime_entrypoint.py` currently validates the fixed uid/user, non-interactive
lifetime, six minimum environment values, metadata-key name, and absence of human/file-backed
credential variables, then exits before metadata, JAX, checkpoint, or control
I/O. This is safe only because it never imports JAX. Before production admission,
the runtime must enforce an exact allowlist for JAX/libtpu/XLA, Python/loader,
proxy, cache, and related environment inputs; the present six-value check is
not sufficient for numerical execution. The following gates remain:

1. A controller-side Compute API receipt must prove source image, exact instance
   template, MIG membership, workload policy, service account, accelerator, and
   zone. It must bind the documented v6e family lookup to the resolved numeric
   base-image ID/selfLink and API fields above. The existing host
   self-attestation cannot prove its own placement.
2. A bake-and-boot attestation must establish that the resolved image contains
   and starts every v6e-required kernel, udev, VFIO, and TPU service
   prerequisite. Static knowledge of the image family does not prove this.
3. A generation/hash-pinned launch reference and controller challenge must be
   implemented without allowing metadata to become shell or argv.
4. Exactly 16 TPU VMs/JAX processes must initialize one 64-device JAX
   topology, agree collectively on the launch/environment/image fingerprint,
   and produce a complete `DeviceInventoryV1` before scientific input opens.
   The benchmark runner now performs the required explicit distributed-JAX
   initialization before any backend query; MaxText skips it when checkpointing
   is disabled.
5. A final SSH-disconnect/logind-duration test must prove systemd owns the
   process and `/dev/shm` sentinel.
6. Checkpoint restore/parity, v6e kernel qualification, and the preregistered
   >=51% useful-MFU gate must pass.

The managed-Python download artifact and the Packer executable/plugin binaries
are pinned by version but not yet by committed artifact digests. Their observed
bytes and a trusted builder-toolchain authority remain image-qualification
requirements; static bundle validation does not close that gap.

No build should be labeled qualified until all six pass.

## Deployment protocol boundary

The service is one-shot (`Restart=no`). Once any host enters distributed JAX,
a failed/preempted attempt is replaced by a fresh whole MIG and launch fence;
systemd never replays one worker under a stale launch.

Both deployment protocols now require the Compute MIG-compatible Spot
termination action `STOP`. Compute rejects `DELETE` for Spot VMs in a managed
instance group even though the standalone Spot guide recommends it. A stopped
member is never restarted or reused: `DO_NOTHING` prevents per-VM repair, and
the controller requires a fresh whole 16-VM MIG, launch reference, and fence
for every replacement attempt. This preserves terminal whole-slice semantics
without encoding an API-invalid template. A successor attempt may move regions
without changing the image, runtime, or scientific identity. The V2 records
are still a protocol oracle: the cloud API adapter and a documented live
create/preempt/recreate test remain required before production admission. See
the
[official TPU Spot VM guide](https://docs.cloud.google.com/tpu/docs/spot-compute).
