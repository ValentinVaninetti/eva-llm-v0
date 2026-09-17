#!/bin/bash
cd "$(dirname "$0")/.."
# CONDITION B (windowed readout), recipe identical to the base but EVA_READ_WIN=K=N+1
EVA_GPU=1 EVA_ALPHA_TRACE=250 EVA_READ_WIN=257 target/release/eva_llm_v0 train --data data/sintetico_N256_seq512_win1000_seed4245.dat --dim 512 --ffn 1024 --blocks 5 --seq 512 --val 0.1 --seed 7 --gate-beta 0.0 --log 250 --epochs 2 --out 16m5b_N256_T1_win257.weights > logs/train_N256_win257.log 2>&1; echo N256_WIN_EXIT=$? >> logs/train_N256_win257.log
EVA_GPU=1 EVA_ALPHA_TRACE=250 EVA_READ_WIN=129 target/release/eva_llm_v0 train --data data/sintetico_N128_seq512_win1000_seed4244.dat --dim 512 --ffn 1024 --blocks 5 --seq 512 --val 0.1 --seed 7 --gate-beta 0.0 --log 250 --epochs 2 --out 16m5b_N128_T1_win129.weights > logs/train_N128_win129.log 2>&1; echo N128_WIN_EXIT=$? >> logs/train_N128_win129.log
EVA_GPU=1 EVA_ALPHA_TRACE=250 EVA_READ_WIN=33 target/release/eva_llm_v0 train --data data/sintetico_N32_seq512_win1000_seed4243.dat --dim 512 --ffn 1024 --blocks 5 --seq 512 --val 0.1 --seed 7 --gate-beta 0.0 --log 250 --epochs 2 --out 16m5b_N32_T1_win33.weights > logs/train_N32_win33.log 2>&1; echo N32_WIN_EXIT=$? >> logs/train_N32_win33.log
