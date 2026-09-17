#!/bin/bash
# CONDITION B2.1 (structured init from the finite difference + separate LR).
# On top of condition B (EVA_READ_WIN=K=N+1) we add:
#   EVA_READ_DF_N=N   -> w[N-1]=1/beta, w[N]=-alpha_c/beta (readout starts
#                        expressing the recovered write at t-N+1)
#   EVA_READ_LR=10    -> LR x10 ONLY for the wread parameters
# Recipe otherwise identical to the base. K=1 (no-window control) == base.
cd "$(dirname "$0")/.."
N256(){
  EVA_GPU=1 EVA_ALPHA_TRACE=250 EVA_READ_WIN=257 EVA_READ_DF_N=256 EVA_READ_LR=10 \
  target/release/eva_llm_v0 train --data data/sintetico_N256_seq512_win1000_seed4245.dat \
  --dim 512 --ffn 1024 --blocks 5 --seq 512 --val 0.1 --seed 7 --gate-beta 0.0 --log 250 \
  --epochs 2 --out 16m5b_N256_T1_win257_df256_lr10.weights \
  > logs/train_N256_win257_df256.log 2>&1; echo N256_EXIT=$? >> logs/train_N256_win257_df256.log
}
N128(){
  EVA_GPU=1 EVA_ALPHA_TRACE=250 EVA_READ_WIN=129 EVA_READ_DF_N=128 EVA_READ_LR=10 \
  target/release/eva_llm_v0 train --data data/sintetico_N128_seq512_win1000_seed4244.dat \
  --dim 512 --ffn 1024 --blocks 5 --seq 512 --val 0.1 --seed 7 --gate-beta 0.0 --log 250 \
  --epochs 2 --out 16m5b_N128_T1_win129_df128_lr10.weights \
  > logs/train_N128_win129_df128.log 2>&1; echo N128_EXIT=$? >> logs/train_N128_win129_df128.log
}
N32(){
  EVA_GPU=1 EVA_ALPHA_TRACE=250 EVA_READ_WIN=33 EVA_READ_DF_N=32 EVA_READ_LR=10 \
  target/release/eva_llm_v0 train --data data/sintetico_N32_seq512_win1000_seed4243.dat \
  --dim 512 --ffn 1024 --blocks 5 --seq 512 --val 0.1 --seed 7 --gate-beta 0.0 --log 250 \
  --epochs 2 --out 16m5b_N32_T1_win33_df32_lr10.weights \
  > logs/train_N32_win33_df32.log 2>&1; echo N32_EXIT=$? >> logs/train_N32_win33_df32.log
}
"$1"
