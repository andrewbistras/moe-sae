# Experiment 4 evidence recovery

Recovered from the existing `gemma4-control`, `gemma4-horizon-usc1`, and
`gemma4-horizon-ew4` Filestores on 2026-07-30. No TPU was started and no cloud
object was written.

## Authoritative sealed outputs

| File | SHA-256 |
|---|---|
| `sealed/experiment4-result.json` | `a0f97335fd2c36c81e2a9ef5975f547066b1b2084882ae461e5f4284ebda06ac` |
| `sealed/EXPERIMENT_4.md` | `9b0193d7ee8778bd9c8b6577aaf05306d00204ddaba950fd6bacaf068c97f7f1` |

The report's canonical self-hash is
`18908abf1c22512d5f4dc9180ea52c28f57953c0046193f4e8957f9c72cdcb88`.

## Independent verification

The report was rebuilt into a fresh local directory on the controller using:

```text
uv run python -m moe_sae.horizon_curve_report
  --sae-result <us-central1 2M/5M/10M scientific results>
  --transcoder-result <europe-west4 2M/5M/10M scientific results>
  --run-contract <the eight A/R/B/F SAE/transcoder contracts>
  --output-root /tmp/moe-sae-recovery/experiment4-rebuild-20260730T0127Z
```

Both generated files were compared byte-for-byte with the sealed files using
`cmp`. They were identical, and their generated SHA-256 digests matched the
table above.

This directory preserves:

- the two sealed report files;
- all six source `scientific-result.json` files;
- the three source transcoder Markdown reports;
- all eight run contracts;
- the final SAE/transcoder configs and direct campaign plan.

The copied evidence comprised 22 files and 174,305 bytes before this recovery
note was added.
