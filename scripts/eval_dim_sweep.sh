#!/bin/bash
# The four cells of the channels-per-key sweep.
#
# Claim under test: what governs the binding is NOT the number of
# keys but the CHANNELS PER KEY (dim/keys). If true, cells with the
# same ratio must behave alike even at different dim
# y distinta cantidad de claves.
#
#   dim 512 / k=2  -> 256 can/clave -> 99,65%  (ya medido)
#   dim 512 / k=4  -> 128 can/clave -> ?       (run de esta noche)
#   dim 512 / k=8  ->  64 can/clave -> 23,42%  (ya medido)
#   dim 256 / k=2  -> 128 can/clave -> ?       must match 512/k=4
#   dim 256 / k=4  ->  64 can/clave -> ?       must match 512/k=8
#
# Runs locally: evaluation only, not a comparable run. EVA_GPU=1 measured on
# 9 s against 32 s per 30 windows; forward only, changes no number.
set -e
cd "$(dirname "$0")/.."

ev () {
  local ckpt=$1 data=$2 label=$3
  if [ ! -f "$ckpt" ]; then echo "MISSING $ckpt -- fetching it from the remote"; scp "${REMOTE:?set REMOTE=user@host}:~/evallm_check/$ckpt" . ; fi
  echo
  echo "############ $label ############"
  EVA_GPU=1 ./target/release/examples/eval_associative "$ckpt" "$data" 300 2>&1 \
    | tee "logs/eval_${label// /_}.log" \
    | grep -A6 "what the top-1 prediction IS"
}

K4=data/assoc_k4ctl_seq512_win22500_seed42.dat
K2=data/assoc_k2_seq512_win22500_seed42.dat

ev 16m5b_assoc_k4ctl_huge_clock.weights  "$K4" "dim512_k4"
ev 16m5b_assoc_k4ctl_d256_clock.weights  "$K4" "dim256_k4"
ev 16m5b_assoc_k2_d256_clock.weights     "$K2" "dim256_k2"

echo
echo "=== already-measured references (same binary, same corpus) ==="
echo "  dim 512 / k=2 : 99,65% correcto /  0,00% otra clave"
echo "  dim 512 / k=8 : 23,42% correcto / 76,49% otra clave"
