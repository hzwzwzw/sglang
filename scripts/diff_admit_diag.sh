#!/usr/bin/env bash
# Diff PP admit-time decision logs across ranks to find first divergence.
#
# Usage:
#   bash diff_admit_diag.sh [logdir]              # default: /tmp
#   bash diff_admit_diag.sh /tmp 1782111380968     # focus on a specific crash
#
# Workflow:
# 1. Run the server with SGLANG_PP_ADMIT_DIAG=1 set.
# 2. After a crash, this script extracts ADMIT_DECISION events per rank,
#    compares them, and prints the first divergent rid.
# 3. For deeper debugging, grep the rid across all rank logs.

set -e

DIR="${1:-/tmp}"
TS="${2:-}"

cd "$DIR"

LOGS=$(ls sglang_pp_admit_pp*_tp0.log 2>/dev/null | sort)
if [ -z "$LOGS" ]; then
  echo "No pp_admit logs found in $DIR. Set SGLANG_PP_ADMIT_DIAG=1 and re-run."
  exit 1
fi

echo "=== ranks with logs ==="
for f in $LOGS; do
  rank=$(echo "$f" | grep -oE 'pp[0-9]+')
  n_admit=$(grep -c "^ADMIT_DECISION" "$f" || echo 0)
  n_apply=$(grep -c "^PHASE_B_IN" "$f" || echo 0)
  n_insert=$(grep -c "^TREE_INSERT" "$f" || echo 0)
  echo "  $rank: admits=$n_admit apply=$n_apply tree_inserts=$n_insert"
done

echo ""
echo "=== first divergent rid (compare admit decisions) ==="
PP0=$(ls sglang_pp_admit_pp0_tp0.log 2>/dev/null)
PP1=$(ls sglang_pp_admit_pp1_tp0.log 2>/dev/null)
if [ -n "$PP0" ] && [ -n "$PP1" ]; then
  diff \
    <(grep "^ADMIT_DECISION" "$PP0" | awk '{for(i=1;i<=NF;i++) if($i~/^rid=|^local_len=|^agreed_len=|^action=/) printf "%s ", $i; print ""}') \
    <(grep "^ADMIT_DECISION" "$PP1" | awk '{for(i=1;i<=NF;i++) if($i~/^rid=|^local_len=|^agreed_len=|^action=/) printf "%s ", $i; print ""}') \
    | head -10
else
  echo "Need pp0 and pp1 logs to diff. Got: $PP0 / $PP1"
fi

echo ""
echo "=== sample agreed-vs-local divergence per rank ==="
for f in $LOGS; do
  rank=$(echo "$f" | grep -oE 'pp[0-9]+')
  echo "--- $rank ---"
  # Print events where local_len != agreed_len (truncation actually applied).
  grep "^ADMIT_DECISION" "$f" \
    | awk '{
        local_len=""; agreed_len="";
        for(i=1;i<=NF;i++) {
          if($i ~ /^local_len=/) local_len=substr($i,11);
          if($i ~ /^agreed_len=/) agreed_len=substr($i,12);
        }
        if(local_len != "" && agreed_len != "" && local_len != agreed_len)
          print $0
      }' | head -5
done

echo ""
echo "=== suggested next steps ==="
echo "Pick a divergent rid (e.g. 98e04b91a88f) from above and run:"
echo "  for f in $DIR/sglang_pp_admit_pp*_tp0.log; do"
echo '    echo "=== $f ==="; grep "rid=98e04b91a88f" "$f"; done'
echo ""
echo "Find the source: same token_hash on different ranks should give"
echo "the same TREE_INSERT result_len. If they don't match, walk back"
echo "the inserts ordered by step= field."
