#!/bin/bash
# Evalua el barrido de 12 INICIALIZACIONES (orden de datos fijo, --seed 7).
# Mide D en el paso 2500 y 5000: la tasa binaria + la curva que pidio GPT
# (progreso subcritico vs dos regimenes separados) y el chequeo que pidio
# Dante (si la bifurcacion se corre en el tiempo al variar init, se ve aca).
cd "$(dirname "$0")"
D=data/assoc_k2_seq512_win22500_seed42.dat
printf "%-8s  %-12s  %-12s  %s\n" "init" "D@2500" "D@5000" "veredicto"
printf "%-8s  %-12s  %-12s  %s\n" "----" "------" "------" "---------"
n=0; tot=0
for I in 101 102 103 104 105 106 107 108 109 110 111 112; do
  vals=""
  for S in 2500 5000; do
    f="snaps_init/init_i${I}_step${S}.weights"
    [ -f "$f" ] || timeout 200 scp -q "martha:~/evallm_check/snaps_init/init_i${I}_step${S}.weights" snaps_init/ 2>/dev/null
    if [ -f "$f" ]; then
      d=$(EVA_GPU=1 ./target/release/examples/query_swap "$f" "$D" 40 3 2>/dev/null | grep -oE "directional shift [+-][0-9.]+" | grep -oE "[+-][0-9.]+")
      vals="$vals ${d:-?}"
    else
      vals="$vals (falta)"
    fi
  done
  d2500=$(echo $vals | awk '{print $1}'); d5000=$(echo $vals | awk '{print $2}')
  ver="ciego"
  if awk -v x="$d5000" 'BEGIN{exit !(x>0.05)}' 2>/dev/null; then ver="LEE"; n=$((n+1)); fi
  tot=$((tot+1))
  printf "%-8s  %-12s  %-12s  %s\n" "$I" "$d2500" "$d5000" "$ver"
done
echo
echo "  INIT (orden de datos FIJO):  $n de $tot"
echo "  referencia -- orden de datos variable, init fija: 2 de 17 (11,8%)"
