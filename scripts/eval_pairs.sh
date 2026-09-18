#!/bin/bash
# Evaluatestes every base/nogate pair the short queue has finished.
# Fetches any missing checkpoints and reports a verdict for each.
cd "$(dirname "$0")/.."
mkdir -p snaps
DATA=data/assoc_k2_seq512_win22500_seed42.dat
printf "%-6s  %-22s  %-22s\n" "seed" "BASE" "NO-GATE"
printf "%-6s  %-22s  %-22s\n" "----" "----------------------" "----------------------"
nb=0; nn=0; tb=0; tn=0
for S in 12 13 14 15 16 17 18 19 20 21 22 23; do
  line=""
  for arm in base nogate; do
    f="snaps/corto_${arm}_s$S.weights"
    if [ ! -f "$f" ]; then
      timeout 120 scp -q "${REMOTE:?set REMOTE=user@host}:~/evallm_check/corto_${arm}_s$S.weights" snaps/ 2>/dev/null
      [ -f "$f" ] || { line="$line  $(printf '%-22s' '(missing)')"; continue; }
    fi
    d=$(EVA_GPU=1 ./target/release/examples/query_swap "$f" "$DATA" 60 3 2>/dev/null | grep -oE "directional shift [+-][0-9.]+" | grep -oE "[+-][0-9.]+")
    if [ -z "$d" ]; then line="$line  $(printf '%-22s' '-')"; continue; fi
    # Threshold: measured blind arms give |d| < 0.01; sighted ones, > 0.19.
    # Nothing has landed in between in any run so far.
    if awk -v x="$d" 'BEGIN{exit !(x>0.05)}'; then
      v="LEE  ($d)"
      [ "$arm" = base ] && nb=$((nb+1)) || nn=$((nn+1))
    else
      v="ciego ($d)"
    fi
    [ "$arm" = base ] && tb=$((tb+1)) || tn=$((tn+1))
    line="$line  $(printf '%-22s' "$v")"
  done
  [ -n "$line" ] && printf "%-6s%s\n" "$S" "$line"
done
echo
echo "  BASE   (runs cortas): $nb de $tb"
echo "  NOGATE (runs cortas): $nn de $tn"
echo "  + base historica (runs completas, seeds 7-11): 1 de 5"
