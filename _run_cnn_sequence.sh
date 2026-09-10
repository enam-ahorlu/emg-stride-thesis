#!/usr/bin/env bash
# Stage 3: run jobs_window_cnn.txt strictly one at a time (GPU serialises).
set -u
JOBS="jobs_window_cnn.txt"
i=0
while IFS= read -r line; do
  case "$line" in ''|\#*) continue;; esac
  i=$((i+1))
  echo "=========== [CNN job $i] $(date -u +%FT%TZ) ==========="
  echo "$line"
  eval "$line"
  rc=$?
  echo "[CNN job $i] exit $rc at $(date -u +%FT%TZ)"
  if [ $rc -ne 0 ]; then echo "[CNN job $i] FAILED — stopping sequence"; exit $rc; fi
done < "$JOBS"
echo "=========== ALL CNN JOBS DONE $(date -u +%FT%TZ) ==========="
