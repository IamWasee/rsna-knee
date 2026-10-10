#!/usr/bin/env bash
# Install (or remove with --remove) the Kaggle watcher as a launchd job: every 30 min,
# survives restarts, no Claude usage. Report: data/audit/watch/report.md
set -eu
P="$HOME/Library/LaunchAgents/com.kaggleknee.watch.plist"
R="$(cd "$(dirname "$0")/.." && pwd)"
launchctl unload "$P" 2>/dev/null || true
if [ "${1:-}" = "--remove" ]; then rm -f "$P"; echo "watcher removed"; exit 0; fi
mkdir -p "$R/data/audit/watch" "$HOME/Library/LaunchAgents"
cat > "$P" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.kaggleknee.watch</string>
  <key>ProgramArguments</key><array><string>/usr/bin/python3</string><string>$R/scripts/kaggle_watch.py</string></array>
  <key>WorkingDirectory</key><string>$R</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
  <key>StartInterval</key><integer>1800</integer>
  <key>StandardOutPath</key><string>$R/data/audit/watch/launchd.log</string>
  <key>StandardErrorPath</key><string>$R/data/audit/watch/launchd.err</string>
</dict></plist>
EOF
launchctl load "$P"
launchctl list | grep kaggleknee && echo "watcher installed: runs every 30 min"
