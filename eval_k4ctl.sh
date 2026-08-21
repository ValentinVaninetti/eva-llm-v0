#!/bin/bash
# Evaluacion del barrido de canales por clave (k=4, corpus grande).
# Corre en Cristina: es evaluacion, no una corrida comparable.
#
# Espera POR PID, nunca por patron: pgrep/pkill -f se encuentran a si mismos
# dentro del bucle que espera (nos costo 20 horas una vez, y una sesion ssh).
set -e
cd "$(dirname "$0")"
PID=${1:-122674}
CKPT=16m5b_assoc_k4ctl_huge_clock.weights
DATA=data/assoc_k4ctl_seq512_win22500_seed42.dat
OUT=logs/eval_assoc_k4ctl.log

echo "esperando el PID $PID en martha..."
while timeout 30 ssh martha "kill -0 $PID 2>/dev/null"; do sleep 120; done
echo "termino. traigo el checkpoint..."
scp "martha:~/evallm_check/$CKPT" .
scp "martha:~/evallm_check/train_assoc_k4ctl_huge_clock.log" logs/
scp "martha:~/evallm_check/temps_k4ctl.log" logs/

# 300 ventanas por seccion ya dan decenas de miles de posiciones de consulta;
# las 22.5k completas tardan mas de una hora y no agregan precision util.
# EVA_GPU=1 medido: 9 s contra 32 s por 30 ventanas, 3,5x. Aca no cambia
# ningun resultado (es solo forward), asi que no hay razon para no usarlo.
EVA_GPU=1 ./target/release/examples/eval_associative "$CKPT" "$DATA" 300 2>&1 | tee "$OUT"
echo
echo "=== para comparar, lo ya medido ==="
echo "  k=2 (dim 512): 99,65% correcto / 0,00% otra clave"
echo "  k=8 (dim 512): 23,42% correcto / 76,49% otra clave"
echo "  la cuenta del reparto de canales predice que k=4 caiga EN EL MEDIO,"
echo "  y que dim=1024/k=8 (la corrida siguiente) se parezca a este k=4."
