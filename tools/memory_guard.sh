#!/bin/bash
# usage: memory_guard.sh LIMIT_MB LOGFILE cmd args...   runs cmd, kills it if its RSS (with children) exceeds LIMIT_MB,
# or if free system memory falls below 1500 MB. Peak RSS is written to the log.
LIM=$1; LOG=$2; shift 2
"$@" >> "$LOG" 2>&1 &
PID=$!; PEAK=0
while kill -0 $PID 2>/dev/null; do
  RSS=$(ps -o rss= --ppid $PID -p $PID 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}')
  AVAIL=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
  if [ "$RSS" -gt "$LIM" ] || [ "$AVAIL" -lt 1500 ]; then
    echo "GUARD: killed at RSS ${RSS} MB (available ${AVAIL} MB)" >> "$LOG"; kill $PID; pkill -P $PID; break
  fi
  sleep 3
done
wait $PID; echo "GUARD: exit $? peak RSS ${PEAK} MB" >> "$LOG"
