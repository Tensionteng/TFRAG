#!/bin/bash
# Health check across every machine running a campaign.
#
# Prints one line per host plus any NEW failure since the last run, so a broken
# config surfaces in minutes instead of at the end of a multi-hour sweep. Failures
# are the thing to watch: a run that dies writes no record, so the cell silently
# vanishes from the results table rather than showing up as an error.
#
# Usage
#   bash scripts/craft_rebuttal/monitor.sh              # one pass
#   watch -n 300 bash scripts/craft_rebuttal/monitor.sh # or from cron/loop
set -u

HOSTS=${HOSTS:-"gpu6 gpu7 gpu8"}
STATE=${STATE:-/tmp/craft_monitor_state}
mkdir -p "$STATE"

echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
TOTAL_FAIL=0
for H in $HOSTS; do
  OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$H" '
    cd ~/code/TFRAG 2>/dev/null || cd /home/tengshiyuan/code/TFRAG 2>/dev/null || exit 9
    N=$(ls runs/*.json 2>/dev/null | wc -l)
    A=$(pgrep -cf "[r]un[.]py")
    L=$(ls -t /tmp/b*.log 2>/dev/null | head -1)
    # grep -c prints 0 and exits 1 when there is no match; `|| echo 0` would then
    # append a second line and shift every field of the summary.
    FAILS=$(grep -c FAILED "$L" 2>/dev/null); FAILS=${FAILS:-0}
    DONE=$(grep -c "all workers finished" "$L" 2>/dev/null); DONE=${DONE:-0}
    DISK=$(df -h . | tail -1 | awk "{print \$4}")
    echo "$N|$A|$FAILS|$DONE|$DISK|$(basename ${L:-none})"
    # newest distinct error lines, for triage
    grep -hoE "(RuntimeError|ValueError|ModuleNotFoundError|TypeError|AssertionError|FileNotFoundError): .*" logs/*.log 2>/dev/null \
      | sort | uniq -c | sort -rn | head -3
  ' 2>/dev/null) || { echo "  $H  UNREACHABLE"; continue; }

  HEAD=$(echo "$OUT" | head -1)
  IFS='|' read -r N A FAILS DONE DISK LOG <<< "$HEAD"
  ERRS=$(echo "$OUT" | tail -n +2)

  STATUS="running"
  [ "${A:-0}" = "0" ] && STATUS="IDLE"
  [ "${DONE:-0}" != "0" ] && STATUS="finished"
  printf "  %-6s %-9s records=%-5s active=%-4s failed=%-4s disk=%-6s log=%s\n" \
    "$H" "$STATUS" "$N" "$A" "$FAILS" "$DISK" "$LOG"

  # Flag disk pressure before it kills runs. CLEAN=1 also reclaims checkpoints of
  # runs that already produced a record -- those are finished and never reopened,
  # unlike the directories of in-flight runs, which are left alone.
  case "$DISK" in
    *K|[0-9]M|[0-9][0-9]M|[0-9][0-9][0-9]M)
      echo "      !! LOW DISK: $DISK"
      if [ "${CLEAN:-0}" = "1" ]; then
        ssh -o BatchMode=yes "$H" '
          cd ~/code/TFRAG 2>/dev/null || cd /home/tengshiyuan/code/TFRAG
          n=0
          for d in checkpoints/*/; do
            s=$(basename "$d")
            [ -f "runs/$s.json" ] && { rm -rf "$d"; n=$((n+1)); }
          done
          rm -rf test_results/*
          echo "      reclaimed $n completed checkpoint dirs, now $(df -h . | tail -1 | awk "{print \$4}") free"
        ' 2>/dev/null
      else
        echo "      run with CLEAN=1 to reclaim completed-run checkpoints"
      fi
      ;;
  esac

  if [ -n "$ERRS" ]; then
    PREV="$STATE/$H.errs"
    if ! diff -q <(echo "$ERRS") "$PREV" >/dev/null 2>&1; then
      echo "$ERRS" | sed 's/^/      NEW ERROR: /'
      echo "$ERRS" > "$PREV"
    fi
  fi
  TOTAL_FAIL=$((TOTAL_FAIL + ${FAILS:-0}))
done
echo "  total failures across hosts: $TOTAL_FAIL"
