#!/bin/bash
# The 14-run discovery suite (§20 step 4) on a v4-32: one run per chip,
# F-runs given dedicated chips (they are the 10.7× stragglers). All runs read
# the same shard cache; the spec's matched-data-order requirement (§10) means
# every run consumes the identical batch sequence — I/O is paid once.
#
# Run assignment (16 chips / 4 hosts, worker w chip c):
#   w0: A1 A2 A3 C1    w1: C2 C3 B1 B2    w2: B3 D1 E1 F1    w3: F2 F3 (+2 idle)
#
# Launch (per worker, chips pinned via TPU_VISIBLE_CHIPS):
#   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=$W \
#     --command="cd ~/moe-sae && bash scripts/tpu/03_discovery_suite.sh $W"
set -euo pipefail
cd ~/moe-sae
W="${1:?worker index 0-3}"

declare -a ASSIGN
case "$W" in
  0) ASSIGN=(discovery_A_seed1 discovery_A_seed2 discovery_A_seed3 discovery_C_seed1) ;;
  1) ASSIGN=(discovery_C_seed2 discovery_C_seed3 discovery_B_seed1 discovery_B_seed2) ;;
  2) ASSIGN=(discovery_B_seed3 discovery_D_seed1 discovery_E_seed1 discovery_F_seed1) ;;
  3) ASSIGN=(discovery_F_seed2 discovery_F_seed3) ;;
esac

for i in "${!ASSIGN[@]}"; do
  cfg="${ASSIGN[$i]}"
  TPU_VISIBLE_CHIPS="$i" PYTHONUNBUFFERED=1 \
    nohup uv run --no-sync python scripts/run_train_jax.py \
      --config "configs/${cfg}.yaml" < /dev/null \
      >> "logs/tpu_${cfg}.log" 2>&1 &
  echo "launched $cfg on worker $W chip $i (pid $!)"
done
wait
echo "worker $W: all runs complete"
# afterwards: rsync runs/ to the bucket; θ*-freeze per run is in final/;
# eval + analyses run anywhere (torch stack reads JAX checkpoints directly).
