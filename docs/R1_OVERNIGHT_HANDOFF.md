# R1 overnight handoff

This branch now has one concrete route from regional checkpoint bytes to a
compiled DeepSeek-R1 block-31 prefix, plus a checkpoint-free synthetic v6e-64
benchmark. It is not yet a production launch.

## Global delivery and recovery

Use the existing US-CENTRAL2 bucket only for small controller heads, identities,
and receipts. Bulk checkpoint, capture, and training-checkpoint objects remain
in a regional bucket beside the slice that consumes them.

- Prefer a v6e-64 in `us-east1-d` or `europe-west4-a`.
- Stage the 649.674 GB prefix checkpoint once into the selected primary region.
- Create a second regional replica only after another region is actually
  selected. Never move a capture to chase compute; move the next attempt to a
  region that already owns the capture.
- Keep regional buckets Standard, with soft delete and object versioning off.
  Content addressing and sealed manifests provide immutability.
- Poll the tiny global control head from one controller. The 16 workers receive
  the admitted attempt through their slice rendezvous; they do not each poll
  GCS.
- Cache exact generation-pinned objects on each VM. One verified remote fill is
  followed by local `pread` calls, with no LIST, discovery, or silent refetch.

`RegionScheduler.claim` admits only a complete v6e-64 candidate whose region
already contains every sealed bulk input. The claim advances
`attempt_epoch = source_fence + 1` through one bounded CAS and is then
cross-bound to `AttemptAuthorityV2`. A preempted or dead attempt cannot publish
under a later fence. A replacement slice resumes only from the last fully
sealed checkpoint/cursor and receives a new attempt ID.

The v5e-64 allocations are fallback candidates only after numerical and
performance qualification. The v4-32 allocations cannot execute the frozen
B128/64-device science program and should be used for auxiliary conversion,
validation, or Gemma work—not silently substituted into R1 measurements.

## MaxText and restore state

Apply the source series only to a clean checkout at
`c03f8bd307d18049fa814fc8c104bd9ec8c292aa`:

```zsh
python - <<'PY'
from moe_sae.r1.maxtext_patch_series import apply_pinned_maxtext_patch_series

print(
    apply_pinned_maxtext_patch_series(
        "/absolute/path/to/maxtext",
        patch_root="/absolute/path/to/moe-sae/patches/r1/maxtext",
    ).as_dict()
)
PY
```

The series binds both patch hashes and the final hashes of `moe.py`,
`nnx_decoders.py`, and `deepseek_prefix.py`. The runtime then:

1. constructs an explicit EP/FSDP mesh;
2. creates and prunes the abstract NNX prefix graph;
3. translates MaxText logical axes and verifies all 44 physical shardings;
4. restores exact callback-backed arrays, or creates benchmark-only device
   arrays;
5. binds 30 model leaves and passes 14 target leaves separately;
6. compiles the dynamic model with `nnx.jit`; and
7. returns BF16 routed activations, int32 global top-8 IDs, and FP32 scaled
   weights.

CPU tracing passes through the complete adapter and stops only when it reaches
the TPU-only Splash/Pallas lowering. The decisive evidence is therefore a real
v6e-64 run.

## First v6e-64 command

Run each mesh independently on a fresh complete slice. The command allocates
the full logical parameter footprint on-device but reads no checkpoint and
writes only one small local JSON result:

```zsh
python -m moe_sae.r1.prefix_benchmark_runner \
  --maxtext-base-yml /opt/moe-sae-r1/runtime/lib/python3.12/site-packages/maxtext/configs/base.yml \
  --output-directory /var/tmp/moe-sae-r1/benchmark \
  --mesh EP16/FSDP4 \
  --seed 0 \
  --result-json /var/tmp/moe-sae-r1/benchmark/ep16-fsdp4.json
```

Repeat with `EP32/FSDP2` and `EP8/FSDP8`. Each run compiles once, validates the
output/router ABI, completes five warmups, and measures 500 device-synchronized
steps. For the frozen full B128×1024 workload, 51% MFU requires an aggregate
step time no greater than 163.551 ms. Allocation, compile, output validation,
and warmups are reported separately.

For the first image qualification, run
`scripts/r1/qualify_v6e_mig.sh` on the persistent controller. The script is
deliberately narrower than the production launch protocol: it observes one
already-created bulk MIG, requires exactly 16 running members and an `ACTIVE`
8×8 topology, then starts the fixed command above on all private workers under
systemd. It cannot create, repair, resize, or delete TPU resources. Its bounded
journals, statuses, and sole process-zero JSON remain on the controller disk,
so a failed compile or preemption produces no GCS upload bill. Override only
its named environment inputs when qualifying a later immutable template or
mesh:

```zsh
sudo env \
  PROJECT=neural-dragnet \
  ZONE=us-east1-d \
  MIG=r1-v6e-bench-27e25e5-ue1d-a1 \
  EXPECTED_TEMPLATE=r1-v6e-spot-stop-27e25e5-us-east1-v1 \
  ATTEMPT=ue1d-a1-ep16-fsdp4 \
  MESH=EP16/FSDP4 \
  /usr/local/sbin/qualify_v6e_mig.sh
```

The Compute API rejects Spot `DELETE` for members of a managed instance group,
despite the standalone TPU Spot guide recommending it. The executable template
therefore uses `STOP` plus MIG `DO_NOTHING`. A stopped member is never resumed:
the controller treats it as whole-slice loss and requires a fresh MIG, launch
reference, and fence.

Do not stage the real checkpoint until this ladder identifies a viable layout.
Afterward, run the same winner through the generation-pinned 44-leaf restore,
record cache remote/local byte counters, perform full-prefix parity, and only
then begin activation capture.

## Remaining hard gates

- Build/publish the updated immutable image and connect its deliberately
  stopped systemd boundary to one generation/hash-pinned, one-shot launch.
- Compile and run on real v6e-64, inspect HLO and parameter input/output aliasing,
  and measure all three layouts.
- Prove no routed token is dropped at the actual synthetic and real-data loads.
- Reach at least 51% useful-prefix MFU; keep a 55% synthetic target to leave
  room for capture instrumentation.
- Convert and restore released R1 bytes, pass full/prefix parity, then connect
  the existing bounded capture writer and sealed-manifest path.
