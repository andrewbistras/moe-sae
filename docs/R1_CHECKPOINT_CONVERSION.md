# DeepSeek-R1-0528 checkpoint conversion contract

**Status:** source-backed design audit, bounded reference execution core,
verified local safetensor reader, diagnostic local immutable output store,
generation-pinned GCS byte store and conversion publisher, and a
topology-neutral logical/regional checkpoint protocol, 2026-07-23. No weight
payload was downloaded during this work. Safetensor
headers, the Hugging Face index and configuration, and the pinned MaxText
source were inspected.

This document defines the only checkpoint conversion that may authorize the
one-indexed block-31 activation prefix. It is intentionally fail closed. A
converter completing without an exception is not evidence that the resulting
checkpoint is scientifically equivalent to the released model.

The active implementation has deliberately separate layers:

- [`checkpoint_inventory.py`](../src/moe_sae/r1/checkpoint_inventory.py) is the
  sole authority for the exact source inventory, routed-prefix selection,
  FP8/scale pairing, and canonical HF-to-MaxText map.
- [`checkpoint_conversion.py`](../src/moe_sae/r1/checkpoint_conversion.py)
  derives deterministic, shard-ordered chunks from that authority and executes
  bounded identity, transpose/reshape, and blockwise FP8-to-BF16 transforms
  against pluggable verified-source and create-only-sink protocols. Synthetic
  tests cover layout parity, boundary blocks, FP32 multiply-before-BF16
  rounding, bitwise BF16/FP32 passthrough, cross-shard scales, crash/resume,
  conflicting receipts, and pre-I/O memory rejection.
- [`safetensor_source.py`](../src/moe_sae/r1/safetensor_source.py) maps a
  controller-prestaged directory to the source protocol. It pins regular-file
  identities, hashes each complete blob before payload access, revalidates the
  strict header against the full authenticated shard inventory, performs
  bounded FP8/BF16/FP32 region reads, and fails on observed mutation. Its
  end-to-end test executes the real conversion core and verifies resume plus
  independent BF16 and FP8 numerical oracles.
- [`prefix_checkpoint.py`](../src/moe_sae/r1/prefix_checkpoint.py) is a
  diagnostic local implementation of the output protocol. It publishes
  content-derived chunk files and exact bindings without replacement, recovers
  every tested instruction-boundary publication state, and seals one stable
  plan-bound storage manifest only after every exact object validates. Source
  observations accumulate under plan-derived paths and seal separately, so a
  reused-first retry cannot make the storage manifest order-dependent. Its
  bounded reader reconstructs exact flat tensor regions and exposes complete
  source-observation evidence as an explicit stronger gate.
- [`gcs_immutable_byte_store.py`](../src/moe_sae/r1/gcs_immutable_byte_store.py)
  is the first production storage primitive. It creates content-addressed or
  caller-derived exact objects with `if_generation_match=0`, strict custom
  metadata, decimal-string generation receipts, server CRC32C, and local
  whole-object SHA-256/CRC32C verification through generation- and
  metageneration-pinned bounded streams. The binary `objects/sha256` and JSON
  `records/sha256` namespaces are reserved and must agree exactly with the
  receipt SHA-256 and canonical suffix. It accepts exact immutable `bytes` or
  a synchronous bounded `memoryview`; every mutable view is first copied into
  at most 64 MiB of immutable bytes and SHA-256 checked before a canonical
  object path is opened. Upload and verification use application-owned slices
  of at most 8 MiB. It has no listing or delete API and cannot itself seal a
  checkpoint. These strengthened semantics use storage/store identity v2, so
  an object carrying the earlier v1 store identity is incompatible.
- [`gcs_prefix_checkpoint_sink.py`](../src/moe_sae/r1/gcs_prefix_checkpoint_sink.py)
  adapts that primitive to the conversion executor. It reattests the admitted
  store identity and bounds around every operation, resumes only through an
  exact plan/chunk binding path, verifies the bound payload generation, then
  publishes payload, binding, logical manifest, and regional envelope in that
  order. It neither writes a mutable latest pointer nor performs controller
  CAS, and its returned publication remains `production_sealable=false`.
- [`prefix_checkpoint_v2.py`](../src/moe_sae/r1/prefix_checkpoint_v2.py)
  separates one topology-neutral logical checkpoint identity from its
  generation-pinned regional replicas. It binds the 21,930 selected source
  tensors to 510 packed MaxText destination arrays and 22,108 immutable
  payload leaves, publishes per-leaf source/physical bindings, and opens a
  regional replica with exactly two bounded metadata reads before lazily
  fetching only touched binding and payload objects. Its manifest and regional
  envelope are each capped at 32 MiB and remain explicitly
  `production_sealable=false`. The bounded GCS sink/publisher is wired to this
  protocol; controller CAS and qualification are not.

The production GCS writer role for this primitive must contain only bucket
metadata read plus object create/get permissions. It must not include object
list, update, or delete. For objects above the pinned client's 8 MiB multipart
limit, the primitive supplies the precomputed CRC32C for server finalization
validation but sets the client checksum mode to `None`; this avoids the pinned
client's unqualified `DataCorruption` delete path. Multipart objects retain
client CRC32C validation. Denying delete remains defense in depth. Any
incompatible object already occupying an exact path is an operator incident,
never an overwrite request.

The constructor rejects a bucket with a default CMEK because GCS does not
expose the required server CRC32C receipt for CMEK objects. It attests region
and default encryption once and assumes trusted operators do not update bucket
configuration during the store's lifetime; the writer identity itself has no
bucket-update authority. Client injection is test-only, produces a separate
unqualified store identity, and can never claim distributed production
qualification. Only an internally constructed exact 3.13.0 client does so.

The chunk identity binds the exact conversion-module hash, the measured NumPy
and `ml_dtypes` native-extension and distribution-record hashes, and an
explicit externally attested source-adapter implementation hash.
`ml_dtypes==0.5.4` is a direct dependency rather than an accidental transitive
JAX dependency.

This is not an Orbax converter and produces no production seal. The local
source adapter proves bytes only after a trusted controller has prestaged
them; it does not authenticate the download or object-store generation. The
local output store uses POSIX inode and fsync semantics and explicitly marks
its path-derived identity as local-only. It is a diagnostic proof of the sink
contract, not a GCS, TensorStore, Orbax, or multi-host implementation. There is
no controller/GCS prestager, fresh-process MaxText restore, independent real-R1
FP8 hash oracle, full-model logit oracle, or prefix/full parity result yet. The
GCS sink has only deterministic/fault-injected local evidence, not a real-bucket
throughput and chaos qualification. The only production sealing function fails
unconditionally until those gates exist.

The memory limits are statically checked bounds on the core's numerical array
working set under synchronous source/sink contracts. The local source and
output adapters additionally enforce their declared stream-buffer limits.
The GCS sink takes an already-resident conversion `memoryview`. Its default
explicit staging budget is 72 MiB: one complete 64 MiB immutable snapshot plus
one 8 MiB application upload/verification slice. The conversion planner charges
that complete bound on every chunk. The pinned topology remains 22,108 chunks;
its maximum explicit resident estimate is 306,716,544 bytes under these
defaults. This is not a claim about allocator overhead or opaque copies inside
the pinned HTTP client; total RSS remains a measured qualification gate.
Full-scale throughput and scratch behavior remain unqualified. Controller
integration and the later JAX/Orbax mapper remain
unimplemented obligations. The shared project environment also still resolves
MaxText to the Gemma commit; it must not be used for R1.
The isolated project at `deploy/r1/runtime/` now pins the MaxText revision below
plus exact Python/JAX/JAXLIB/libtpu versions. The pinned plan constructor
measures both of its files and accepts that lock while continuing to reject the
shared root lock. This closes the static lock-identity gap only: the image is
unbuilt, the worker admission path is deliberately fail-closed, and
authenticated prestaging, a real-bucket-qualified checkpoint publication,
restore, or parity run still cannot seal.

## Immutable source identity

| item | immutable identity |
|---|---|
| Hugging Face repository | `deepseek-ai/DeepSeek-R1-0528` |
| Hugging Face revision | `4236a6af538feda4548eca9ab308586007567f52` |
| MaxText revision | `c03f8bd307d18049fa814fc8c104bd9ec8c292aa` |
| `config.json` SHA-256 | `79ddea672a62e95d3f0be27be434375538e1988975971cffcf87c96c1de84a65` |
| `model.safetensors.index.json` SHA-256 | `a2c5d238ab1e8301a117336e9797e0d3551cb8f6dc9dc787318b23d8f54c9248` |
| `modeling_deepseek.py` SHA-256 | `eb6d535742061cf9da7a4232ecfe2f089d5ce20080afc5d2181e9d70ec4af8a4` |

Primary sources:

- [R1 configuration at the pinned revision](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/4236a6af538feda4548eca9ab308586007567f52/config.json)
- [R1 safetensor index at the pinned revision](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/4236a6af538feda4548eca9ab308586007567f52/model.safetensors.index.json)
- [R1 model implementation at the pinned revision](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/4236a6af538feda4548eca9ab308586007567f52/modeling_deepseek.py)
- [MaxText DeepSeek configuration at the pinned revision](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/src/maxtext/configs/models/deepseek3-671b.yml)
- [MaxText batch-split configuration at the pinned revision](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/src/maxtext/configs/models/deepseek3-671b-batchsplit.yml)

The released configuration has 61 main transformer blocks, hidden width
7,168, three initial dense blocks, 256 routed experts, top-8 routing, one
shared expert, eight routing groups, four selected groups, and one additional
MTP block. It declares `torch_dtype=bfloat16`, but that field does **not**
describe the physical weight payload: the released matrices are predominantly
blockwise FP8 E4M3 with FP32 inverse scales of block size 128 by 128.

## Full released snapshot inventory

The exact pinned index contains 91,991 tensor entries in 163 safetensor shards.
Header inspection gives:

| source dtype | tensors | elements | payload bytes |
|---|---:|---:|---:|
| FP8 E4M3 | 45,808 | 680,571,043,840 | 680,571,043,840 |
| BF16 | 316 | 3,918,786,560 | 7,837,573,120 |
| FP32 | 45,867 | 41,555,600 | 166,222,400 |
| total | 91,991 | 684,531,386,000 | 688,574,839,360 |

The physical shard blobs total 688,586,727,753 bytes including safetensor
headers. The index's `metadata.total_size` is 1,369,062,772,000 bytes; it is
not the physical download size and must not be used as one.

Of the FP32 tensors, 45,808 are inverse-scale tensors. The remaining 59 are
router correction biases. There is an exact one-to-one FP8/scale bijection in
the pinned snapshot: no FP8 tensor lacks a scale and no scale is orphaned.

The eventual conversion attestation must record the Hugging Face LFS SHA-256
and size for every downloaded shard, then independently hash the received
bytes. Repository revision and prose inventory are not substitutes for that
payload receipt. As cross-checks, shard 1 is 5,234,139,343 bytes with LFS
SHA-256 `3c7ce6cd03018d584e3f52543d1263aeec16945b071fc8d7bceccd6e658b120a`;
shard 78 is 1,747,417,474 bytes with LFS SHA-256
`2346499580e45575f22c8042ae6b0ac78332c38a306589519e9d87cee4cc28ab`.

## Exact block-31 prefix inventory

One-indexed block 31 is Hugging Face `model.layers.30`. The required prefix is:

1. `model.embed_tokens.weight`;
2. every tensor in `model.layers.0` through `model.layers.29`;
3. from `model.layers.30`, its input normalization, complete self-attention,
   post-attention normalization, router gate and correction bias, and all 256
   routed experts;
4. nothing else.

In particular, the target block's shared expert, main layers 31 through 60,
the final normalization, LM head, and MTP layer 61 are excluded. The selected
prefix has this exact logical inventory:

| component | logical tensors | parameters | logical dtypes |
|---|---:|---:|---|
| token embedding | 1 | 926,679,040 | 1 BF16 |
| complete dense layers 0--2 | 36 | 1,750,450,176 | 24 FP8, 12 BF16 |
| complete MoE layers 3--29 | 21,114 | 310,696,729,344 | 20,952 FP8, 135 BF16, 27 FP32 |
| target layer-30 routed experts | 768 | 11,274,289,152 | 768 FP8 |
| target attention and input norm | 8 | 187,114,496 | 5 FP8, 3 BF16 |
| target post-attention norm | 1 | 7,168 | 1 BF16 |
| target router | 2 | 1,835,264 | 1 BF16, 1 FP32 |
| total | 21,930 | 324,837,104,640 | 21,749 FP8, 153 BF16, 28 FP32 |

Those logical tensors require 21,749 FP32 inverse-scale tensors, for 43,679
source entries total. They occur in source shards 1 through 78. The exact
storage arithmetic is:

| quantity | bytes | decimal size |
|---|---:|---:|
| required tensor payload, including scales | 325,894,763,808 | 325.895 GB |
| complete source shard blobs 1--78 | 325,944,454,407 | 325.944 GB |
| converted tensor payload, all logical tensors BF16 | 649,674,209,280 | 649.674 GB |
| converted payload with 28 router biases retained FP32 | 649,674,223,616 | 649.674 GB |

The 21,930 rows above count Hugging Face logical source tensors, not restored
MaxText arrays. Expert slices with a common destination key pack into 510
unique MaxText destination tensors. Under the pinned 64 MiB output-chunk
limit, those 510 arrays contain exactly 22,108 ordered immutable payload
leaves. The distinction is part of the V2 manifest contract: it records
43,679 source entries, 21,930 source logical tensors, 510 destination tensors,
and 22,108 chunks separately.

The final object-store footprint will be slightly larger because of Orbax or
TensorStore metadata and chunking. On a 64-chip slice the mixed tensor payload
averages 10.151 GB per chip under perfect partitioning. That arithmetic is not
an HBM fit proof.

## Source-to-MaxText tensor map

For unscanned MaxText, HF layers 0--2 map to
`decoder.dense_layers_0..2`; HF layers 3--60 map to
`decoder.moe_layers_0..57`. The target HF layer 30 is therefore
`decoder.moe_layers_27`.

| Hugging Face source | source shape | MaxText logical shape |
|---|---:|---:|
| `model.embed_tokens.weight` | `[129280, 7168]` | `[129280, 7168]` |
| `input_layernorm.weight` | `[7168]` | `[7168]` |
| `post_attention_layernorm.weight` | `[7168]` | `[7168]` |
| `self_attn.q_a_proj.weight` | `[1536, 7168]` | `[7168, 1536]` |
| `self_attn.q_a_layernorm.weight` | `[1536]` | `[1536]` |
| `self_attn.q_b_proj.weight` | `[24576, 1536]` | `[1536, 128, 192]` |
| `self_attn.kv_a_proj_with_mqa.weight` | `[576, 7168]` | `[7168, 576]` |
| `self_attn.kv_a_layernorm.weight` | `[512]` | `[512]` |
| `self_attn.kv_b_proj.weight` | `[32768, 512]` | `[512, 128, 256]` |
| `self_attn.o_proj.weight` | `[7168, 16384]` | `[128, 128, 7168]` |
| dense `gate_proj` or `up_proj` | `[18432, 7168]` | `[7168, 18432]` |
| dense `down_proj` | `[7168, 18432]` | `[18432, 7168]` |
| `mlp.gate.weight` | `[256, 7168]` | `[7168, 256]` |
| `mlp.gate.e_score_correction_bias` | `[256]` | `[256]`, FP32 |
| all routed `gate_proj` or `up_proj` | 256 x `[2048, 7168]` | `[256, 7168, 2048]` |
| all routed `down_proj` | 256 x `[7168, 2048]` | `[256, 2048, 7168]` |
| shared `gate_proj` or `up_proj` | `[2048, 7168]` | `[7168, 2048]` |
| shared `down_proj` | `[7168, 2048]` | `[2048, 7168]` |

The authoritative MaxText mappings and reshape hooks are in
[`param_mapping.py`](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/src/maxtext/checkpoint_conversion/utils/param_mapping.py#L1610-L1774),
with expected source shapes in
[`hf_shape.py`](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/src/maxtext/checkpoint_conversion/utils/hf_shape.py#L428-L587).

MaxText defaults to `scan_layers=true` and `param_scan_axis=1`. A complete
full-model scanned MoE tensor therefore has, for example, routed gate/up shape
`[256, 58, 7168, 2048]`, routed down shape
`[256, 58, 2048, 7168]`, and router shape `[7168, 58, 256]`.

The stock scanned mapping cannot express the required partial target: each
scanned leaf spans all 58 MoE blocks. The correctness oracle should first use
an unscanned, explicitly prefix-shaped tree. The performance tree may scan the
three complete dense blocks and 27 complete preceding MoE blocks, but block 31
must be a separate routed-only target module. Both trees require their own
complete destination inventory and parity proof.

## Upstream support verdict

MaxText states that R1 is supported because it shares the DeepSeek-V3
architecture. That statement is accurate at the architecture and tensor-map
level. It does not provide an accepted direct conversion for the exact raw
R1-0528 snapshot.

### Standalone two-stage path

The documented first stage,
[`deepseek_fp8_to_bf16.py`](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/src/maxtext/checkpoint_conversion/standalone_scripts/deepseek_fp8_to_bf16.py#L40-L165),
implements the intended blockwise FP32 multiplication and BF16 cast. It is not
fail closed:

- a missing scale prints a warning and preserves the raw FP8 tensor;
- the output index replaces the source metadata with an empty object;
- it creates a complete intermediate BF16 Hugging Face checkpoint;
- it does not produce a source/destination inventory or content attestation.

Both subsequent DeepSeek converters explicitly reject keys ending in
`_scale_inv`: [scanned rejection](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/src/maxtext/checkpoint_conversion/standalone_scripts/convert_deepseek_family_ckpt.py#L329-L337)
and [unscanned rejection](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/src/maxtext/checkpoint_conversion/standalone_scripts/convert_deepseek_family_unscanned_ckpt.py#L74-L108).
They also materialize large state in RAM and cast tensors to
`torch.float16`, rather than preserving BF16 and the FP32 router bias; see the
[scanned conversion](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/src/maxtext/checkpoint_conversion/standalone_scripts/convert_deepseek_family_ckpt.py#L414-L428)
and [unscanned conversion](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/src/maxtext/checkpoint_conversion/standalone_scripts/convert_deepseek_family_unscanned_ckpt.py#L130-L145).

### Generic converter

The generic converter's DeepSeek map has the correct architecture-level
coverage, scanned/unscanned hooks, lazy shard loading, explicit revision, and
`save_dtype` interface. It is still unsafe on the raw snapshot:

- its default `deepseek3-671b` Hugging Face ID is DeepSeek-V3, so R1 requires
  both the explicit repository and exact revision;
- it maps the FP8 weight key but not `_scale_inv`, and performs no blockwise
  dequantization;
- lazy serialization materializes with `np.array(value)` without requiring
  the declared save dtype, as shown in
  [`to_maxtext.py`](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/src/maxtext/checkpoint_conversion/to_maxtext.py#L212-L292);
- source-map validation proves MaxText destination coverage, not FP8-scale
  consumption or exact source coverage.

There is no R1-0528-specific conversion or logit end-to-end test at this
commit. The DeepSeek mapping unit tests only assert that representative keys
exist. The V3 daily test uses a separately stored, old V3 Orbax checkpoint and
a permissive one-prompt logit comparison; it does not chain the released FP8
conversion into parity. See the [V3 conversion script](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/tests/end_to_end/tpu/deepseek/v3-671b/1_test_deepseek.sh#L27-L42)
and [V3 logit test](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/tests/end_to_end/tpu/deepseek/v3-671b/2_test_deepseek.sh#L30-L49).

The pinned MaxText commit is consequently an **architecture source pin**, not
a completed conversion attestation or an accepted runtime. It becomes the
accepted runtime only after the conversion and parity gates below pass.

## Fail-closed streaming conversion

The converter should go directly from the 78 required source shards into the
prefix checkpoint. It must not create a full 1.37 TB BF16 Hugging Face copy.
A separate full-model Orbax checkpoint may be produced with the same audited
engine for the logit oracle, but it is a distinct validated output, not an
intermediate on the prefix conversion path. The prefix and full outputs must
derive independently from the same immutable source tensors and transforms.

### 1. Inventory before payload conversion

1. Require the exact repository revision and the three file hashes above.
2. Fetch the exact-revision per-shard LFS SHA-256 and size list.
3. Parse every safetensor header and require the expected 91,991-entry full
   inventory before selecting the prefix.
4. Derive the required source set from the selection rule above; do not infer
   progress by shard number or existing output filenames.
5. Require exact source key, shape, and dtype matches.
6. Require each FP8 tensor's scale at `<weight-name>_scale_inv`, dtype FP32,
   and shape `[ceil(M / 128), ceil(N / 128)]`.
7. Reject missing scales, orphan scales, duplicate keys, unexpected dtypes,
   target-shared or suffix leakage, and old-schema output.

### 2. Convert and transform once

For each immutable source key:

- validate the containing shard bytes before opening it;
- for FP8, multiply `weight.float32 * scale.float32` independently for every
  128 by 128 block, then cast exactly once to BF16;
- copy BF16 tensors without an intermediate FP16 representation;
- retain every router correction bias as FP32;
- apply the audited transpose/reshape directly into its destination slice;
- emit no dense latent or per-expert activation artifact;
- write chunked destination arrays so a full `[256, ...]` expert stack is
  never required in host RAM.

The reference core rejects non-finite inverse scales. It does not currently
invent a positivity gate: scale positivity remains to be established by the
real 21,749-scale payload scan, while the released transform itself is FP32
multiplication.

Progress is journaled by an immutable identity containing the source revision,
source shard hash, source key, transform version, destination logical key, and
destination slice. A resume may reuse a slice only after verifying that exact
identity and output hash.

The production source adapter must keep local scratch to one validated source
shard; the core rejects any declared source blob or numerical chunk above its
plan limits before reading it. Durable output is written under a unique attempt
prefix. No mutable `latest` pointer is published during conversion.

### 3. Publish logical content and regional placement separately

The V2 logical manifest contains only topology-neutral facts: exact source,
MaxText, inventory, plan, conversion-engine, numeric-codec, source-adapter, and
runtime-lock identities; every destination key, shape, dtype, ordered chunk
interval, and content hash; and the exact 43,679 / 21,930 / 510 / 22,108
cardinalities above. Its SHA-256 is computed over its canonical bytes; no
self-hash is serialized into those bytes.

Regional placement is separate. Each chunk binding joins one logical output
receipt and its plan-derived source lineage to one generation-pinned GCS
payload receipt. A regional envelope then joins the exact logical-manifest
receipt to the ordered set of exact binding-record receipts for one store and
region. Bucket, URI, generation, metageneration, CRC32C, host, and region never
enter the logical manifest, so a byte-identical cross-region copy retains the
same logical checkpoint ID while receiving a different regional-envelope ID.

The only accepted crash order is:

1. create or verify each content-addressed payload;
2. create or verify its plan/chunk-derived binding record;
3. create or verify the content-addressed logical manifest;
4. create or verify the content-addressed regional envelope last;
5. publish the exact envelope receipt through the controller's fenced CAS.

An interrupted earlier stage may leave unreachable immutable bytes, never an
apparently complete checkpoint. Opening requires both an exact envelope
receipt and an independently expected logical-manifest SHA-256. It performs no
listing and accepts no mutable `latest` path.

V2 proves storage structure only. Both records hard-code
`production_sealable=false`; a later qualification record must join complete
source-observation evidence, fresh-process restore, Orbax/TensorStore
interoperability, an independent numerical oracle, full-model and routed-prefix
parity, and real throughput/RSS measurements without rewriting the logical
checkpoint identity.

## Router and capture semantics

The pinned Hugging Face router
[`MoEGate.forward`](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/4236a6af538feda4548eca9ab308586007567f52/modeling_deepseek.py#L393-L473):

1. casts hidden states and the BF16 gate matrix to FP32 for the projection;
2. applies sigmoid;
3. adds the FP32 correction bias for group and expert selection only;
4. selects four of eight groups using the sum of each group's top two scores;
5. selects eight experts from those groups;
6. gathers the corresponding **unbiased** sigmoid scores;
7. normalizes them and multiplies by 2.5.

The target is the weighted routed aggregate returned by
[`moe_infer`](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/4236a6af538feda4548eca9ab308586007567f52/modeling_deepseek.py#L522-L608),
before the shared expert and before the decoder residual at
[`DeepseekV3DecoderLayer.forward`](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/4236a6af538feda4548eca9ab308586007567f52/modeling_deepseek.py#L1197-L1217).

MaxText's ordinary routed MoE implements the same group/top-k structure and
can request FP32 gate logits. The correction bias must nevertheless remain an
FP32 parameter rather than being rounded by the converter.

The optimized batch-split path needs an explicit correctness repair before it
can be used:

- [`fetch_weights`](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/src/maxtext/models/deepseek_batchsplit.py#L46-L84)
  currently casts the gate matrix and bias to the global dtype;
- [`expert_selection`](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/src/maxtext/models/deepseek_batchsplit.py#L1897-L1923)
  does not force the router projection to FP32;
- routing converts global expert IDs into shard-local IDs;
- [`unroute_ubatch_fn`](https://github.com/AI-Hypercomputer/maxtext/blob/c03f8bd307d18049fa814fc8c104bd9ec8c292aa/src/maxtext/models/deepseek_batchsplit.py#L2330-L2348)
  immediately combines routed, shared, and residual outputs.

The native target path must emit global IDs and scaled weights before local-ID
remapping, unroute the weighted routed result, skip target shared-expert
computation, and stop before residual addition. Hugging Face uses unsorted
TopK whereas JAX returns ordered candidates; comparisons and durable captures
therefore canonicalize each token as expert-ID-to-weight pairs rather than
treating slot order as semantics.

## Required parity and acceptance gates

### Conversion integrity

- Every selected destination key has exactly one source derivation.
- Every BF16 passthrough tensor and FP32 bias is bitwise identical after the
  layout transform is inverted.
- Every FP8 tensor has a deterministic converted BF16 hash. An independent
  implementation checks boundary blocks and deterministic samples against
  `source.float32 * scale.float32` followed by BF16 rounding.
- Exact destination key, shape, dtype, element, and byte totals match this
  document.
- A fresh process restores the sealed checkpoint on the intended 64-chip mesh
  with no missing, default-initialized, coerced, or unused required leaf.

### Architecture and router parity

- With real layer-30 weights and frozen inputs, the pinned Hugging Face and
  MaxText implementations produce identical expert-ID sets for every token.
- Emitted IDs are eight unique integers in `[0, 255]`; weights are finite,
  nonnegative, correspond to the same IDs, and sum to 2.5.
- Group restriction, correction-bias selection, unbiased emitted weights, and
  no-token-drop behavior are tested independently.
- The ordinary and batch-split MaxText paths agree on IDs, weights, and routed
  aggregate within predeclared BF16 numerical tolerances. A tolerance is never
  relaxed after seeing a production discrepancy.

### Prefix and model parity

- The routed-only target adapter and an unmodified full block produce the same
  routed aggregate from the same block input.
- Instrumentation enabled and disabled produce identical scientific outputs.
- The prefix checkpoint and full checkpoint agree at the block-31 attention
  residual, normalized MoE input, IDs, weights, and routed aggregate on a
  frozen validation corpus.
- A multi-prompt, multi-position R1 Hugging Face/MaxText logit oracle at the
  exact revisions passes `max_kl_div <= 0.03`. The existing upstream V3
  `max_kl_div=0.1`, four-token test is not sufficient evidence.
- Compiled HLO and the restored parameter inventory prove that target shared
  experts, layers 32--61, final normalization, LM head, MTP, and logits do not
  execute in the production prefix.

Only after all gates seal may the prefix checkpoint identity enter image,
benchmark, capture, or campaign manifests. A failed conversion remains an
immutable diagnostic attempt; it is never repaired in place or relabeled as a
qualified checkpoint.
