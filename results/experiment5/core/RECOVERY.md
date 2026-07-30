# Core Experiment 5 evidence recovery

Recovered from the existing `gemma4-control` and `gemma4-horizon-usc1`
Filestores on 2026-07-30. The authoritative generation is the report-recovery
bundle `gemma4-experiment5-report-recovery-e0abd4a1af5b-a1`, not an earlier
report attempt.

## Authoritative sealed outputs

| File | SHA-256 |
|---|---|
| `sealed/EXPERIMENT_5.md` | `954a8a39921f5a38000b5149ba040bddd8615933e070e0f9e9e4e4713a65ab0f` |
| `sealed/experiment5-result.json` | `f498bef6b3b7cdfccce9ef8178d730ba4a14402b520f3089054bb1d539970c8c` |
| `sealed/heldout-basis-audit.json` | `d4c6e31553daceb2fc4c5049f24fb1a30aded97995b68f312e08f23d6781c132` |
| `sealed/recovery-envelope.json` | `a79489c9e4124785d05c8e9703c5423ba7c852adb84fae2ae1fcd21c86ebed93` |
| `sealed/recovery-handoff.json` | `1d3bfa54421d1be4731ba177a3f6c6f1cdaa90dd904dcc119c511e53607df06e` |
| `sealed/final.json` | `eabdde4093396cf7be374876c0abbee34615f24d0ee51a02c1b2cec54aaf5c31` |

The scientific report's canonical self-hash is
`572633c5db9808f4544daf9d751f714f8cde4845d25a86ad8a1a4941dd883570`.

## Independent verification

The controller checkout was temporarily detached at the exact reporting
revision `e0abd4a1af5ba352a5b1af84cbf591ecc6d3faec`. The following verifier reopened
the immutable plan, all 18 scientific results, all 18 paired-score sidecars,
the evaluation barrier, the budget ledger, and every sealed output:

```text
uv run python scripts/tpu/run_experiment5_report_recovery.py verify
  --plan /mnt/moe-sae-control/experiment5/report-recovery-v1/e0abd4a1af5b-a1/contracts/report-recovery.yaml
```

It returned `status: recovery_report_sealed`, report hash
`572633c5db9808f4544daf9d751f714f8cde4845d25a86ad8a1a4941dd883570`,
basis-audit hash
`32b72d3b0fa06990dbe53e2576841bd65fee31a46b482dc0a7a818d6a7a464a2`,
and maximum aggregate SST relative drift
`1.1468397747854292e-10`, below its `1e-9` tolerance. The checkout was restored
cleanly to `600e4522f7e2efed6c0c644fac1030b88102269a`.

This directory preserves:

- the six sealed recovery outputs;
- all 18 `paired-scores.json` sidecars and 18 matching scientific results;
- the authoritative evaluation and report-recovery plans;
- the final campaign and evidence plans;
- the training/evaluation completion records and budget ledger.

The copied core evidence comprised 49 files and 2,468,209 bytes before this
recovery note was added.

## Transfer and controller audit

- Controller: `gemma4-horizon-controller`, `n2-standard-4`,
  `neural-dragnet/us-central2-b`.
- Start requested: `2026-07-30T01:19:28Z`.
- Compute start timestamp: `2026-07-30T01:19:35.359Z`.
- Recovery handoff time: `2026-07-30T01:31:11Z`.
- The controller was deliberately left running for the immediately following
  closure campaign; shutdown was transferred to that operator.
- Existing Filestores were mounted with `-o ro,nolock`; explicit write probes
  failed as expected.
- `gcloud compute ssh` executed the rebuild/verifier and created one temporary
  compact archive.
- `gcloud compute scp` transferred that archive to the operator workstation.
- Archive: 71 files, 770,570 compressed bytes, SHA-256
  `3b3b2ebd3a7484f8037b9d3c988d36dadb84b84ce3e3d2cf85102b9ab6dec90c`.
- The local archive digest matched the controller digest.
- Before these notes were added, the sorted per-file SHA-256 listing for both
  recovered directories had aggregate digest
  `af29b35a7740aa3d4769b0b39e3d6bb8e53d8e9a502c1a695235b12f9c00b331`.
- No TPU was created or started, no new bulk copy was made, no GCS object was
  written, and no cloud resource was deleted.
