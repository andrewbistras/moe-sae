# DeepSeek-R1 MaxText routed-MoE prefix tap

## Status and boundary

This tranche adds a sealed source patch and its fail-closed application utility.
It does **not** wire the patch into the runtime bundle, container image, decoder,
prefix runner, or harvesting controller. The artifact intentionally lives under
`patches/r1/`, not `deploy/r1/`, so the existing closed runtime-bundle input tree
does not change before that integration is reviewed.

The patch modifies only `src/maxtext/layers/moe.py` in this exact MaxText base:

| Identity | Sealed value |
|---|---|
| MaxText commit | `c03f8bd307d18049fa814fc8c104bd9ec8c292aa` |
| Base `moe.py` Git blob | `7737734b09c8be5dc332f9b29748ff93b7c96cff` |
| Base `moe.py` SHA-256 | `b1a56ed4f74c7a072e944381f04fa0f23541dcdf73e9f10e298c4010a487c94d` |
| Patched `moe.py` Git blob | `c84c56b28bdd31b2e16001d5028db47de2b274ae` |
| Patched `moe.py` SHA-256 | `f6fe4ace7538f542747d20b3167e3424219aed4562552bd5a51b178d4f0cf527` |
| Patch SHA-256 | `f5b4f810dd981add4f13baad91d5811cf632ec32d17266b67b84644783ae850b` |

The patch artifact is
`patches/r1/maxtext/0001-routed-moe-prefix-tap.patch`. These identities are
also constants in `moe_sae.r1.maxtext_tap_patch`; a mismatch is an error, not a
warning or a best-effort application.

## Tap contract

`RoutedMoE` gains one static constructor argument:

```python
capture_router_metadata: bool = False
```

With the default `False`, the public result remains exactly the existing
three-tuple:

```text
(routed_aggregate, load_balance_loss, routed_bias_updates)
```

With `True`, the same native invocation returns an explicit five-tuple:

```text
(
  routed_aggregate,
  load_balance_loss,
  routed_bias_updates,
  global_top_k_expert_ids,   # int32 [batch, sequence, top_k]
  scaled_router_weights,     # float32 [batch, sequence, top_k]
)
```

The aggregate is not reconstructed by a second implementation. The IDs are the
original global IDs emitted by the existing `get_topk` invocation before any
roll, sort, shard-local remap, or grouped-matmul dispatch transform. The weights
are the same normalized and routed-scaling-factor-multiplied FP32 values used by
native unpermute. The output casts make the metadata ABI explicit.

The patch also stores DeepSeek 3/4 expert-selection correction bias in FP32 so
the checkpoint leaf is not rounded at restore. Storage dtype alone is not
sufficient: `GateLogit.__call__` casts the bias, inputs, and kernel to its
configured compute dtype. Tap-on therefore requires
`float32_gate_logits=True`. The exact pinned DeepSeek default is `False`, and
tap-on rejects that default because a non-BF16-representable correction bias can
change global router IDs near a selection boundary. Tap-off remains unchanged.

### A2A, ring, and token chunks

- The all-to-all path carries the untouched logical-batch IDs and weights beside
  native dispatch metadata; it does not alter the all-to-all payload.
- The ring path retains global IDs through its shard-mapped route. After the
  expert-axis batch gather, it returns the logical input-batch slice belonging
  to that expert shard while leaving gathered weights intact for native
  unpermute.
- Ring token chunks collect metadata per sequence chunk and concatenate on the
  sequence axis in original order.
- Tap-off keeps the original three output shardings. Tap-on statically appends
  two batch/sequence-sharded, top-k-replicated output specifications.

There is no router recomputation and no NNX `Intermediate`/`sow` side channel.
This matters both for exactness and for avoiding an extra router projection in
the prefix HLO.

## Supported and rejected modes

Tap-on currently supports the native sparse top-k routed-MoE path, including
the all-to-all and ring strategies. `num_moe_token_chunks > 1` is supported and
tested for the ring path. This is distinct from MoE *embedding* chunks:
`num_moe_emb_chunks > 0` is rejected.

Construction fails immediately when tap-on is combined with any path whose
metadata semantics have not been proved:

- `float32_gate_logits=False` (including the pinned DeepSeek default);
- dense MoE (`sparse_matmul=False`);
- the `vllm_rpa` fused-MoE path;
- hash routing;
- random routing;
- MoE embedding chunks;
- the batch-split schedule; or
- any positive `ragged_buffer_factor`, because pinned MaxText may truncate
  expert contributions while still retaining the pre-truncation route IDs; or
- quantization.

These checks apply only to tap-on. Tap-off preserves the existing MaxText mode
surface.

## Applying and attesting

Apply to an absolute, exclusively owned copied/image-build MaxText source root,
never to the shared uv Git cache:

```python
from moe_sae.r1.maxtext_tap_patch import apply_maxtext_tap_patch

attestation = apply_maxtext_tap_patch("/workspace/maxtext")
print(attestation.as_dict())
```

Application is idempotent for the one accepted patched hash. The utility:

1. walks every absolute directory component with dirfd-relative
   `O_DIRECTORY|O_NOFOLLOW` opens and rejects a symlinked root or parent;
2. opens the patch with `O_NOFOLLOW`, requires mode `0644` and exactly one hard
   link, bounds it to its exact byte size, verifies the full open-file and path
   metadata remain stable, and checks the sealed hash;
3. accepts only the exact base or exact patched `moe.py` hash;
4. takes a nonblocking cooperative exclusive lock on the opened target parent;
5. parses the pinned one-file unified diff and checks every context line in
   memory; blank context markers are normalized so the patch file is itself
   whitespace-clean when staged;
6. verifies the complete patched result hash before touching the target;
7. creates, writes, and `fsync`s a dirfd-relative same-directory temporary;
8. rechecks the source and every opened directory identity immediately before
   replacement; and
9. uses dirfd-relative `os.replace`, directory-`fsync`, and post-replacement
   directory-chain validation.

Use an expected state when a stage must distinguish an unmodified checkout from
an image-ready patched tree:

```python
from moe_sae.r1.maxtext_tap_patch import attest_maxtext_tree

attest_maxtext_tree("/workspace/maxtext", expected_state="patched")
```

Source drift, patch drift, oversize input, noncanonical mode, hardlink alias,
symlinked path component, path replacement during a read, cooperative lock
contention, parent replacement, and atomic replacement failure all stop the
operation. Sealed target reads also require mode `0644` and one link. Reads are
bounded and use open/fstat/read/post-fstat plus a final full path-stat
comparison. The attestation deliberately contains content identities rather
than a machine-specific absolute path.

The lock is cooperative, not a kernel-enforced ban on every other writer. No
portable check-then-rename sequence can guarantee protection from an arbitrary
noncooperating process that has write access and races after the final check.
Application therefore requires exclusive/trusted ownership of the isolated
build tree for its duration. Dirfds prevent a replaced ancestor from redirecting
the mutation, and chain revalidation detects observed parent replacement; they
do not broaden this guarantee to hostile writers.

The returned `expected_upstream_commit` is the commit against which the sealed
`moe.py` blob was authored; this utility does **not** claim to attest every file
in the checkout or independently verify Git cleanliness. Image construction
must compose this target-file receipt with the isolated runtime lock/tree
receipt before making a whole-source-tree claim.

## Verification

The fast tests use a local exact c03 checkout when one is available:

```zsh
pytest -q tests/r1/test_maxtext_tap_patch.py
```

Set `MAXTEXT_C03_CHECKOUT` when the checkout is outside the uv cache. The test
fails if an explicitly configured checkout is wrong; it honestly skips
checkout-dependent cases when no external exact source is available.

The CPU numerical probe additionally needs a Python environment containing the
pinned MaxText runtime dependencies. When those are not installed in the test
environment, it skips with an explicit reason. It can be selected directly:

```zsh
export MAXTEXT_C03_CHECKOUT=/path/to/maxtext-c03-checkout
export MAXTEXT_C03_PYTHON=/path/to/compatible/python
pytest -q tests/r1/test_maxtext_tap_patch.py
```

That probe copies the exact source tree, applies the sealed patch, and checks:

- tap-off returns three values and tap-on returns five;
- tap-on and tap-off routed aggregates are bitwise identical;
- the correction bias is FP32 while the tiny gate kernel remains BF16, using a
  deliberately non-BF16-representable bias near a selection boundary for which
  BF16 rounding changes the router IDs;
- the FP32-compute tap matches the oracle, tap-on rejects
  `float32_gate_logits=False`, and that default remains valid for tap-off;
- global IDs and scaled FP32 weights match the independent NumPy DeepSeek
  router oracle;
- the single-device A2A control path executes numerically, and tap-on adds no
  `stablehlo.dot_general` over tap-off, ruling out a second router projection
  in this fixture;
- a separate forced-eight-device CPU subprocess runs EP8 ring routing with
  `num_moe_token_chunks=2`, exercises all-gather and reduce-scatter, checks
  tap-off/tap-on aggregate identity, verifies every per-shard logical-batch
  metadata slice, and matches the full concatenated result to the oracle;
- the same forced EP8 mesh lowers tap-off and tap-on A2A HLO with the expected
  ragged-all-to-all custom calls and no extra router dot. XLA:CPU cannot execute
  the multi-device `ragged-all-to-all` opcode, so multi-chip A2A numerical
  execution remains a TPU acceptance gate; and
- positive drop-capable `ragged_buffer_factor` and the other unsupported modes
  fail at construction.

Unit tests separately cover sealed hashes, deterministic application,
idempotence, source/patch drift, source compilation, bounded reads, canonical
mode, hardlink/target-symlink/root-symlink/parent-symlink rejection,
cooperative-lock contention, file and parent replacement races, and injected
atomic-replace failure with temporary-file cleanup.

## Exact semantic limitations

- The receipt attests the patch artifact and one `moe.py` target, not a clean
  MaxText tree. `expected_upstream_commit` names the source commit for that
  sealed blob; a separate runtime-lock/tree receipt must verify the checkout.
- The flag exists only on direct `RoutedMoE` construction. It is not threaded
  through `RoutedAndSharedMoE`, a decoder block, or a prefix executable, so the
  normal model call path cannot request the five-tuple yet.
- The separate `prefix_forward_adapter_contract` pins the batch-split source
  identities and the exact B128 forward-only repair obligations. It does not
  make this ordinary-MoE patch reusable by the batch-split kernel, and it
  deliberately carries no patch-artifact or execution claim.
- Tap-off preserves the three-value API and original output shard specs. The
  patched DeepSeek tap-off program is not promised to have byte-identical whole
  HLO to the unpatched source because correction bias intentionally changes from
  BF16 storage to FP32. The tests prove no extra router dot between tap-off and
  tap-on within the patched source.
- EP8 ring execution is tested with synthetic tiny weights on one host. EP8 A2A
  is lowered and inspected, while numerical multi-device A2A must wait for TPU
  because XLA:CPU reports its `ragged-all-to-all` opcode as unimplemented.
  Neither result proves v6e kernel choice, multi-host/DCN behavior, MFU, or real
  checkpoint parity.
- Ring `num_moe_token_chunks=2` is numerically tested; the implementation
  supports greater static token-chunk counts under MaxText's divisibility rules.
  `num_moe_emb_chunks>0` remains explicitly unsupported.
- The patch exposes native router results but does not itself assert uniqueness,
  range, no-drop, or per-token weight-sum invariants on every production token.
  The prefix preflight and capture canary must enforce those properties.
- Changing correction-bias storage changes the expected parameter-tree dtype.
  Checkpoint conversion and restore parity must explicitly prove the FP32 bias
  leaf before this patch is used with released R1 weights.

## Gates still required before production harvesting

This patch is necessary but not a complete prefix harvester. A later explicit
integration must still:

1. pin and attest the MaxText source and patch in the image build;
2. thread the static flag only into the selected target routed-MoE block;
3. stop after the required routed aggregate site without running downstream
   shared experts or decoder blocks;
4. expose the five-tuple through the prefix executable and writer schema;
5. prove full-prefix parity with the unmodified decoder at the target site;
6. compile both tap states on the intended v6e mesh and compare relevant HLO,
   sharding, collectives, and routed-GEMM count (the CPU EP8 test covers ring
   control-flow and ownership, not v6e kernel lowering/performance);
7. validate multi-chip A2A (which is compile-only on the local CPU backend) and
   ring global-ID ownership, no drops, dtype, shape, and oracle samples on real
   checkpoint weights; and
8. run the performance and writer-backpressure canaries before an unattended
   harvest.

Until those gates land, this artifact should not be described as image-wired or
production-enabled.
