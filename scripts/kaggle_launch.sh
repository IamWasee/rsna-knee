#!/usr/bin/env bash
# Push a notebook with push_kernel.py, retrying ONLY Kaggle-side start rejections.
#
#   scripts/kaggle_launch.sh <slug> <push_kernel.py args...>
#
# On 2026-10-07 six of eight GPU pushes ended in ERROR within a minute or two
# with an empty log, and identical pushes minutes later ran. That pattern -- an
# ERROR inside the first ~2.5 min and a log with no output from the notebook --
# is retried, at most 3 times. Any other ERROR is the notebook's own failure (a
# refused rehearsal, a crash, a missing input) and stops here, so quota is not
# re-spent on it and the cause is not hidden. Never pushes while the slug's
# previous version is still running.
set -u
slug="$1"; shift
owner="abdullahwasee"
status() { timeout 60 kaggle kernels status "$owner/$slug" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*'; }

case "$(status)" in
  *RUNNING*|*QUEUED*) echo "$slug: previous version still running -- not pushing"; exit 1;;
esac

tmp=$(mktemp -d)
for try in 1 2 3 4; do
  python3 "$(dirname "$0")/push_kernel.py" "$@" >/dev/null 2>&1 || { echo "push failed"; exit 1; }
  # the status can still describe the previous version for a few seconds
  t0=$(date +%s); s=""; sleep 45
  while [ $(( $(date +%s) - t0 )) -lt 150 ]; do
    s=$(status); sleep 20
    case "$s" in *ERROR*|*CANCEL*|*COMPLETE*) break;; esac
  done
  case "$s" in
    *RUNNING*|*QUEUED*) echo "$(date +%H:%M) $slug: $s after try $try"; exit 0;;
    *ERROR*)
      rm -f "$tmp"/*.log
      timeout 120 kaggle kernels output "$owner/$slug" -p "$tmp" --file-pattern '\.log$' >/dev/null 2>&1
      n=$(cat "$tmp"/*.log 2>/dev/null | wc -c | tr -d ' ')
      if [ "${n:-0}" -lt 50 ] && [ "$try" -lt 4 ]; then
        echo "$(date +%H:%M) $slug: rejected at start (empty log), retry $try"; sleep 120; continue
      fi
      echo "$(date +%H:%M) $slug: ERROR with a $n-byte log -- the notebook's own failure; not retrying"; exit 2;;
    *) echo "$(date +%H:%M) $slug: $s"; exit 0;;
  esac
done
echo "$slug: still rejected after 3 retries"; exit 3
