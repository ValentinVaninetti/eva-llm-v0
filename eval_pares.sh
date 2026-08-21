#!/bin/bash
# Evalua todos los pares base/nogate que haya terminado la cola corta.
# Trae los checkpoints que falten y saca el veredicto de cada uno.
cd "$(dirname "$0")"
mkdir -p snaps
DATA=data/assoc_k2_seq512_win22500_seed42.dat
printf "%-6s  %-22s  %-22s\n" "seed" "BASE" "NO-GATE"
printf "%-6s  %-22s  %-22s\n" "----" "----------------------" "----------------------"
nb=0; nn=0; tb=0; tn=0
for S in 12 13 14 15 16 17 18 19 20 21 22 23; do
  linea=""
  for arm in base nogate; do
    f="snaps/corto_${arm}_s$S.weights"
    if [ ! -f "$f" ]; then
      timeout 120 scp -q "martha:~/evallm_check/corto_${arm}_s$S.weights" snaps/ 2>/dev/null
      [ -f "$f" ] || { linea="$linea  $(printf '%-22s' '(falta)')"; continue; }
    fi
    d=$(EVA_GPU=1 ./target/release/examples/query_swap "$f" "$DATA" 60 3 2>/dev/null | grep -oE "directional shift [+-][0-9.]+" | grep -oE "[+-][0-9.]+")
    if [ -z "$d" ]; then linea="$linea  $(printf '%-22s' '-')"; continue; fi
    # Umbral: los ciegos medidos dan |d| < 0.01; los videntes, > 0.19.
    # No hay nada en el medio en ninguna corrida hasta ahora.
    if awk -v x="$d" 'BEGIN{exit !(x>0.05)}'; then
      v="LEE  ($d)"
      [ "$arm" = base ] && nb=$((nb+1)) || nn=$((nn+1))
    else
      v="ciego ($d)"
    fi
    [ "$arm" = base ] && tb=$((tb+1)) || tn=$((tn+1))
    linea="$linea  $(printf '%-22s' "$v")"
  done
  [ -n "$linea" ] && printf "%-6s%s\n" "$S" "$linea"
done
echo
echo "  BASE   (corridas cortas): $nb de $tb"
echo "  NOGATE (corridas cortas): $nn de $tn"
echo "  + base historica (corridas completas, seeds 7-11): 1 de 5"
