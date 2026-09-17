#!/bin/bash
# Evaluatestion of the channels-per-key sweep (k=4, large corpus).
# Runs on the local box: this is evaluation, not a comparable run.
#
# Wait BY PID, never by pattern: pgrep/pkill -f match themselves inside
# the waiting loop (cost us 20 hours once, and a live ssh session).
set -e
cd "$(dirname "$0")/.."
PID=${1:-122674}
CKPT=16m5b_assoc_k4ctl_huge_clock.weights
DATA=data/assoc_k4ctl_seq512_win22500_seed42.dat
OUT=logs/eval_assoc_k4ctl.log

echo "esperando el PID $PID en martha..."
while timeout 30 ssh "${REMOTE:-martha}" "kill -0 $PID 2>/dev/null"; do sleep 120; done
echo "termino. traigo el checkpoint..."
scp "martha:~/evallm_check/$CKPT" .
scp "martha:~/evallm_check/train_assoc_k4ctl_huge_clock.log" logs/
scp "martha:~/evallm_check/temps_k4ctl.log" logs/

# 300 windows per section already give tens of thousands of query positions;
# the full 22.5k take over an hour and add no useful precision.
# EVA_GPU=1 measured: 9 s against 32 s per 30 windows, 3.5x. It changes no
# result here (forward only), so there is no reason not to use it.
EVA_GPU=1 ./target/release/examples/eval_associative "$CKPT" "$DATA" 300 2>&1 | tee "$OUT"
echo
echo "=== para comparar, lo ya medido ==="
echo "  k=2 (dim 512): 99,65% correcto / 0,00% otra clave"
echo "  k=8 (dim 512): 23,42% correcto / 76,49% otra clave"
echo "  la cuenta del reparto de canales predice que k=4 caiga EN EL MEDIO,"
echo "  y que dim=1024/k=8 (la run siguiente) se parezca a este k=4."
