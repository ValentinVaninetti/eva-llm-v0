#!/bin/bash
cd "$(dirname "$0")/.."
EVA_READ_WIN=257 EVA_GPU=1 target/release/examples/mask_eval 16m5b_N256_T1_win257.weights data/sintetico_N256_seq512_win1000_seed4245.dat 256 2>&1 | tee logs/eval_N256_win257.log
EVA_READ_WIN=129 EVA_GPU=1 target/release/examples/mask_eval 16m5b_N128_T1_win129.weights data/sintetico_N128_seq512_win1000_seed4244.dat 128 2>&1 | tee logs/eval_N128_win129.log
EVA_READ_WIN=33 EVA_GPU=1 target/release/examples/mask_eval 16m5b_N32_T1_win33.weights data/sintetico_N32_seq512_win1000_seed4243.dat 32 2>&1 | tee logs/eval_N32_win33.log
