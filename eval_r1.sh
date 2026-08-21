#!/bin/zsh
# Eval the R1 ladder checkpoints with the SAME env they were trained with.
cd "$(dirname "$0")"
DATA=data/sintetico_N256_seq512_win1000_seed4245.dat
BIN=target/release/examples/mask_eval

echo "=== inject r2 (needs INJECT env) ==="
env EVA_READ_WIN=257 EVA_READ_DF_N=256 EVA_READ_INJECT=1 EVA_GPU=1 \
  "$BIN" 16m5b_N256_T1_win257_inject_r2.weights "$DATA" 256 \
  | rg "RECURRENT|recurrent top-1|MODEL"

echo "=== R1a (needs R1 + DF_N envs) ==="
env EVA_READ_WIN=257 EVA_READ_DF_N=256 EVA_READ_R1=1 EVA_GPU=1 \
  "$BIN" 16m5b_N256_T1_win257_r1a.weights "$DATA" 256 \
  | rg "RECURRENT|recurrent top-1|MODEL"

echo "=== R1b (needs R1 env, no DF_N) ==="
env EVA_READ_WIN=257 EVA_READ_R1=1 EVA_GPU=1 \
  "$BIN" 16m5b_N256_T1_win257_r1b.weights "$DATA" 256 \
  | rg "RECURRENT|recurrent top-1|MODEL"
