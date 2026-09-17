#!/bin/zsh
# R1 ladder (2026-08-15): inject replica + R1a (df-seeded) + R1b (zero init).
# Sequential on the GTX 1650. Logs to logs/, checkpoints to the repo root.
set -e
cd "$(dirname "$0")/.."
DATA=data/sintetico_N256_seq512_win1000_seed4245.dat
ARGS=(--data "$DATA" --dim 512 --ffn 1024 --blocks 5 --seq 512 --val 0.1 --seed 7 --gate-beta 0.0 --log 250 --epochs 2)
BASE=(EVA_GPU=1 EVA_ALPHA_TRACE=250)

echo "=== [1/3] inject replica (r2) ==="
env EVA_READ_WIN=257 EVA_READ_DF_N=256 EVA_READ_INJECT=1 "${BASE[@]}" \
  target/release/eva_llm_v0 train "${ARGS[@]}" --out 16m5b_N256_T1_win257_inject_r2.weights \
  > logs/train_N256_win257_inject_r2.log 2>&1
echo "inject_r2 done: $(tail -2 logs/train_N256_win257_inject_r2.log | head -1)"

echo "=== [2/3] R1a (windowed read, no q*g, df-seeded) ==="
env EVA_READ_WIN=257 EVA_READ_DF_N=256 EVA_READ_R1=1 EVA_WREAD_TRACE=50 "${BASE[@]}" \
  target/release/eva_llm_v0 train "${ARGS[@]}" --out 16m5b_N256_T1_win257_r1a.weights \
  > logs/train_N256_win257_r1a.log 2>&1
echo "r1a done: $(tail -2 logs/train_N256_win257_r1a.log | head -1)"

echo "=== [3/3] R1b (windowed read, no q*g, zero init) ==="
env EVA_READ_WIN=257 EVA_READ_R1=1 EVA_WREAD_TRACE=50 "${BASE[@]}" \
  target/release/eva_llm_v0 train "${ARGS[@]}" --out 16m5b_N256_T1_win257_r1b.weights \
  > logs/train_N256_win257_r1b.log 2>&1
echo "r1b done: $(tail -2 logs/train_N256_win257_r1b.log | head -1)"

echo "=== ALL THREE TRAINING RUNS FINISHED ==="
