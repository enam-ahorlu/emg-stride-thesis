#!/usr/bin/env bash
# s1_memory_watchdog.sh
# Babysits run_s1_classical.sh. Paging (hard page-faults to disk) is ~1000x slower
# than RAM, so if free RAM collapses or sustained hard-fault I/O appears, the cheapest
# recovery is: kill the whole pipeline and relaunch with --resume (per-subject
# checkpoints mean we lose at most the in-flight subject).
#
# Trip conditions (any):
#   - available RAM < MIN_AVAIL_MB
#   - hard page-faults/sec > MAX_HPF for two consecutive reads 5s apart
#   - the train_classical python private bytes > MAX_PRIV_MB
#
# Exits when the run log shows "S1 CLASSICAL END", or after MAX_RESTARTS interventions
# (then it stops touching things and just logs that manual attention is needed).
set -u
cd "$(dirname "$0")"

LOG=_run_logs/s1_classical.log
WLOG=_run_logs/s1_watchdog.log
# Proactive: restart while there is still comfortable headroom, so the box never
# actually thrashes. A fresh process starts ~1 GB; each cycle gets ~10+ subjects
# before tripping, then a ~20s relaunch via --resume. Cheap by design.
POLL=45
MIN_AVAIL_MB=1800
MAX_HPF=500
MAX_PRIV_MB=8000
MAX_RESTARTS=40
mkdir -p _run_logs
echo "==== watchdog start $(date -u +%FT%TZ)  poll=${POLL}s minAvail=${MIN_AVAIL_MB} maxHPF=${MAX_HPF} maxPriv=${MAX_PRIV_MB} ====" | tee -a "$WLOG"

restarts=0
missing=0

read_stats() {  # -> "AVAIL HPF PYWS PYPRIV PYPID"
  powershell -NoProfile -Command '
    $m = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $p = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like "*train_classical_loso*" } | Select-Object -First 1
    $ws = 0; $pv = 0; $id = 0
    if ($p) { $ws = [int]($p.WorkingSetSize/1MB); $pv = [int]($p.PrivatePageCount/1MB); $id = $p.ProcessId }
    "{0} {1} {2} {3} {4}" -f $m.AvailableMBytes, $m.PagesInputPersec, $ws, $pv, $id
  ' 2>/dev/null | tr -d "\r"
}

kill_pipeline() {
  powershell -NoProfile -Command '
    Get-CimInstance Win32_Process | Where-Object {
      ($_.Name -eq "python.exe" -and $_.CommandLine -like "*train_classical_loso*") -or
      ($_.Name -eq "bash.exe"  -and $_.CommandLine -like "*run_s1_classical.sh*" -and $_.CommandLine -notlike "*watchdog*" -and $_.CommandLine -notlike "*powershell*")
    } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {} }
  ' >/dev/null 2>&1
  sleep 10
}

relaunch() {
  nohup bash run_s1_classical.sh >> _run_logs/s1_classical.nohup 2>&1 &
  disown 2>/dev/null || true
  sleep 20
}

while true; do
  if grep -q "S1 CLASSICAL END" "$LOG" 2>/dev/null; then
    echo "[$(date -u +%FT%TZ)] run log shows S1 CLASSICAL END -- watchdog exiting" | tee -a "$WLOG"
    exit 0
  fi

  stats=$(read_stats)
  avail=$(echo "$stats" | awk '{print $1}')
  hpf=$(echo   "$stats" | awk '{print $2}')
  pyws=$(echo  "$stats" | awk '{print $3}')
  pypriv=$(echo "$stats" | awk '{print $4}')
  pypid=$(echo "$stats" | awk '{print $5}')
  : "${avail:=0}" "${hpf:=0}" "${pyws:=0}" "${pypriv:=0}" "${pypid:=0}"
  echo "[$(date -u +%FT%TZ)] avail=${avail}MB hpf/s=${hpf} pyWS=${pyws}MB pyPriv=${pypriv}MB pid=${pypid} restarts=${restarts}" >> "$WLOG"

  # crash detection: no python for 4 consecutive polls AND the run log has not
  # advanced (no new "Fitting 5 folds" lines) AND no END marker
  if [ "$pypid" = "0" ]; then
    missing=$((missing+1))
    prog_now=$(grep -c "Fitting 5 folds" "$LOG" 2>/dev/null || echo 0)
    if [ "$missing" -ge 4 ] && [ "${prog_now:-0}" = "${prog_last:-x}" ]; then
      echo "[$(date -u +%FT%TZ)] no python for 4 polls, log not advancing (${prog_now} fits) -- treating as crash, relaunching" | tee -a "$WLOG"
      if [ "$restarts" -lt "$MAX_RESTARTS" ]; then kill_pipeline; relaunch; restarts=$((restarts+1)); fi
      missing=0
    fi
    prog_last="${prog_now:-0}"
    sleep "$POLL"; continue
  fi
  missing=0
  prog_last=$(grep -c "Fitting 5 folds" "$LOG" 2>/dev/null || echo 0)

  trip=""
  [ "$avail" -lt "$MIN_AVAIL_MB" ] && trip="avail ${avail}MB < ${MIN_AVAIL_MB}MB"
  [ "$pypriv" -gt "$MAX_PRIV_MB" ] && trip="pyPriv ${pypriv}MB > ${MAX_PRIV_MB}MB"
  if [ -z "$trip" ] && [ "$hpf" -gt "$MAX_HPF" ]; then
    sleep 5
    hpf2=$(read_stats | awk '{print $2}'); : "${hpf2:=0}"
    [ "$hpf2" -gt "$MAX_HPF" ] && trip="hard page-faults/s ${hpf} then ${hpf2} > ${MAX_HPF}"
  fi

  if [ -n "$trip" ]; then
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      echo "[$(date -u +%FT%TZ)] TRIP ($trip) but restart budget ${MAX_RESTARTS} exhausted -- leaving it alone, MANUAL ATTENTION NEEDED" | tee -a "$WLOG"
      exit 2
    fi
    echo "[$(date -u +%FT%TZ)] TRIP: $trip -- killing pipeline and relaunching with --resume (restart #$((restarts+1)))" | tee -a "$WLOG"
    kill_pipeline
    relaunch
    restarts=$((restarts+1))
  fi

  sleep "$POLL"
done
