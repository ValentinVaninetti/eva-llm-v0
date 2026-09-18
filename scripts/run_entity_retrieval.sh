#!/bin/zsh
# Entity retrieval benchmark on the Quijote (2026-08-15).
# 4 combos: {T1, T13} x {independent, carry}, each val+train.
cd "$(dirname "$0")/.."
export EVA_GPU=1
run() { # $1=label $2=weights $3=temp $4=carry
  local label=$1 w=$2 t=$3 carry=$4
  local envs=(EVA_ALPHA_TEMP=$t EVA_ENTITY_TRAIN=1)
  if [[ -n "$carry" ]]; then envs+=(EVA_ENTITY_CARRY=1); fi
  echo "=== $label ($(date +%T)) ===" | tee -a logs/entity_retrieval_chain.log
  env "${envs[@]}" target/release/examples/entity_retrieval "$w" data/quijote.txt \
    > "logs/entity_retrieval_$label.log" 2>&1
  echo "done $label ($(date +%T))" | tee -a logs/entity_retrieval_chain.log
}
run quijote_T1_indep 16m5b_quijote_T1.weights 1.0 ""
run quijote_T13_indep 16m5b_quijote_T13.weights 1.3 ""
run quijote_T1_carry 16m5b_quijote_T1.weights 1.0 carry
run quijote_T13_carry 16m5b_quijote_T13.weights 1.3 carry
echo "ALL DONE ($(date +%T))" | tee -a logs/entity_retrieval_chain.log
