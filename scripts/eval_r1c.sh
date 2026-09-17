#!/bin/zsh
# Eval the R1c structured-inverse checkpoint (needs R1C + DF_N envs).
cd "$(dirname "$0")/.."
env EVA_READ_WIN=257 EVA_READ_DF_N=256 EVA_READ_R1C=1 EVA_GPU=1 \
  target/release/examples/mask_eval 16m5b_N256_T1_win257_r1c.weights \
  data/sintetico_N256_seq512_win1000_seed4245.dat 256 \
  | rg "RECURRENT|recurrent top-1|MODEL"
