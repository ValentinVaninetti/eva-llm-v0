#!/bin/bash
# Las cuatro celdas del barrido de canales por clave.
#
# La cuenta bajo prueba: lo que gobierna el vinculo NO es la cantidad de
# claves sino los CANALES POR CLAVE (dim/claves). Si es cierta, las celdas
# con el mismo cociente tienen que dar parecido, aunque tengan distinto dim
# y distinta cantidad de claves.
#
#   dim 512 / k=2  -> 256 can/clave -> 99,65%  (ya medido)
#   dim 512 / k=4  -> 128 can/clave -> ?       (corrida de esta noche)
#   dim 512 / k=8  ->  64 can/clave -> 23,42%  (ya medido)
#   dim 256 / k=2  -> 128 can/clave -> ?       tiene que dar como 512/k=4
#   dim 256 / k=4  ->  64 can/clave -> ?       tiene que dar como 512/k=8
#
# Corre en Cristina: evaluacion, no corrida comparable. EVA_GPU=1 medido en
# 9 s contra 32 s por 30 ventanas; es solo forward, no cambia ningun numero.
set -e
cd "$(dirname "$0")"

ev () {
  local ckpt=$1 data=$2 label=$3
  if [ ! -f "$ckpt" ]; then echo "FALTA $ckpt -- lo traigo de martha"; scp "martha:~/evallm_check/$ckpt" . ; fi
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
echo "=== referencias ya medidas (mismo binario, mismo corpus) ==="
echo "  dim 512 / k=2 : 99,65% correcto /  0,00% otra clave"
echo "  dim 512 / k=8 : 23,42% correcto / 76,49% otra clave"
