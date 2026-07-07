#!/bin/bash
# TPU smoke-test ladder — run rungs IN ORDER on the slice; each rung falsifies
# one assumption and costs minutes. Stop at the first failure.
#
# Rungs 1-2 also run locally (CPU) for free: they are the parity tests shipped
# in tests/test_jax_parity.py / test_jax_multidevice.py.
set -euo pipefail
cd ~/moe-sae

echo "=== rung 1: numerics parity (fp32, tiny dims) ==="
uv run --no-sync pytest tests/test_jax_parity.py -x -q

echo "=== rung 2: multi-device orchestration (real chips this time) ==="
uv run --no-sync pytest tests/test_jax_multidevice.py -x -q

echo "=== rung 3: input pipeline throughput ==="
# cold GCS stream vs /dev/shm cache; prints GB/s and steps/s at both A and F
# shapes. Targets: >= 1.5 GB/s/host cold (aggregate ~6 GB/s), cache >= 10 GB/s.
uv run --no-sync python scripts/tpu/bench_input.py \
  --gcs "$BUCKET/captures/pipeline_1m" --cache /dev/shm/moe_sae/pipeline_1m

echo "=== rung 4: pipeline-scale cross-backend replication (1M tokens, f=1024) ==="
# Retrains condition A from the SAME shards and data order as the local MPS
# run; then compares metrics trajectories + theta_star (tolerance: few %).
uv run --no-sync python scripts/run_train_jax.py --config configs/pipeline_1m_A.yaml
uv run --no-sync python scripts/tpu/compare_runs.py \
  runs/pipeline_1m_A_jax runs/pipeline_1m_A  # local run pulled from bucket

echo "=== rung 5: A-vs-C sanity-gate replication on TPU ==="
uv run --no-sync python scripts/run_train_jax.py --config configs/pipeline_1m_C.yaml
# eval with the UNCHANGED torch eval stack (checkpoint format is compatible):
uv run --no-sync python scripts/run_eval.py --run runs/pipeline_1m_A_jax \
  --capture /dev/shm/moe_sae/pipeline_1m --split dev --dump dumps/jax_A/dev --device cpu
uv run --no-sync python scripts/run_eval.py --run runs/pipeline_1m_C_jax \
  --capture /dev/shm/moe_sae/pipeline_1m --split dev --dump dumps/jax_C/dev --device cpu
uv run --no-sync python -c "
from moe_sae.eval.sanity_gate import compare_sanity
v = compare_sanity('runs/pipeline_1m_A_jax', 'runs/pipeline_1m_C_jax',
                   'reports/sanity_gate_tpu.json')
print('TPU sanity gate:', 'PASS' if v['passed'] else 'FAIL', v['ratio_a_over_c'])
"
echo "=== ladder complete — compare reports/sanity_gate_tpu.json vs the local verdict ==="
