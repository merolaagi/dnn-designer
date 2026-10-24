#!/usr/bin/env bash
# Run the Deep Network Designer in the background on port 8770, so Saathi and the browser reach it.
#
#   ./serve.sh                 restart it (and wait until it answers)
#   ./serve.sh install         start it by itself whenever you log in to the Mac, and restart it if it stops
#   ./serve.sh uninstall       stop doing that (and stop it)
#   ./serve.sh stop            stop it
#   ./serve.sh status          is it running, which version, is it set to start by itself
#   ./serve.sh log             follow its log (Ctrl-C stops following, not the server)
#   ./serve.sh setup-browser   install the browser the Support agent chats in (once)
#   ./serve.sh setup-video     install what the Video agent watches with: yt-dlp, ffmpeg, Whisper (once)
#
# Once installed, macOS (launchd) keeps it running: after a restart of the Mac, after a crash, after you
# log out and in. ./serve.sh then restarts it through launchd, so there is only ever one copy.

set -uo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
PORT="${PORT:-8770}"
LOG="data/server.log"
LABEL="com.saathi.dnn"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"
mkdir -p data

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
PYABS="$(cd "$(dirname "$(command -v "$PY")")" && pwd)/$(basename "$PY")"

installed() { [ -f "$PLIST" ]; }

wait_up() {
  for _ in $(seq 1 90); do
    sleep 0.5
    hello=$(curl -s "http://127.0.0.1:$PORT/api/app/hello" 2>/dev/null || true)
    if [ -n "$hello" ]; then
      version=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1]).get("version","?"))' "$hello" 2>/dev/null || echo "?")
      echo "==> running: version $version at http://127.0.0.1:$PORT$(installed && echo ' (starts by itself)')"
      return 0
    fi
  done
  echo "==> it did not answer within 45 seconds; the end of its log:"
  tail -n 30 "$LOG"
  return 1
}

stop_browser() {
  # the browser process (browser_service.py) is restarted too, so a new version of it is used
  local b
  b=$(lsof -ti tcp:"${SAATHI_BROWSER_PORT:-8772}" -sTCP:LISTEN 2>/dev/null || true)
  [ -n "$b" ] && kill $b 2>/dev/null || true
}

kill_port() {
  local pids
  pids=$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
  [ -z "$pids" ] && return 0
  echo "==> stopping the server on port $PORT"
  kill $pids 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 0.5
    lsof -ti tcp:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || return 0
  done
  pids=$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
}

write_plist() {
  mkdir -p "$HOME/Library/LaunchAgents"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYABS</string><string>-m</string><string>uvicorn</string><string>main:app</string>
    <string>--host</string><string>127.0.0.1</string><string>--port</string><string>$PORT</string>
  </array>
  <key>WorkingDirectory</key><string>$HERE</string>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>$(dirname "$PYABS"):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$HERE/$LOG</string>
  <key>StandardErrorPath</key><string>$HERE/$LOG</string>
</dict>
</plist>
EOF
}

case "${1:-}" in
  log)
    exec tail -n 40 -f "$LOG"
    ;;
  setup-browser)
    echo "==> installing Playwright for $PY"
    "$PY" -m pip install --quiet playwright || "$PY" -m pip install --quiet --break-system-packages playwright || exit 1
    echo "==> installing its Chromium (used when Google Chrome is not there)"
    "$PY" -m playwright install chromium || exit 1
    echo "==> the browser is ready"
    exit 0
    ;;
  setup-video)
    pipi() { "$PY" -m pip install --quiet --upgrade "$@" || "$PY" -m pip install --quiet --upgrade --break-system-packages "$@"; }
    echo "==> installing yt-dlp (fetches the video) for $PY"
    pipi yt-dlp || exit 1
    if command -v ffmpeg >/dev/null 2>&1; then
      echo "==> ffmpeg is already there"
    elif command -v brew >/dev/null 2>&1; then
      echo "==> installing ffmpeg with Homebrew"
      brew install ffmpeg || pipi imageio-ffmpeg || exit 1
    else
      echo "==> installing ffmpeg (a copy for Python)"
      pipi imageio-ffmpeg || exit 1
    fi
    if [ "$(uname -m)" = "arm64" ] && [ "$(uname -s)" = "Darwin" ]; then
      echo "==> installing mlx-whisper (writes down what is said, on the Mac's GPU)"
      pipi mlx-whisper || exit 1
      echo "==> downloading the Whisper model (about 1.6 GB, once)"
      "$PY" -c 'import json, pathlib; c = pathlib.Path("data/video.json"); m = (json.loads(c.read_text()) if c.exists() else {}).get("whisper", "mlx-community/whisper-large-v3-turbo"); from huggingface_hub import snapshot_download; snapshot_download(m)' || exit 1
    else
      echo "==> installing faster-whisper (writes down what is said)"
      pipi faster-whisper || exit 1
    fi
    echo "==> the Video agent is ready (it uses the vision model under Settings → Reading invoices for the pictures)"
    exit 0
    ;;
  install)
    launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
    stop_browser
    kill_port
    write_plist
    launchctl bootstrap "$DOMAIN" "$PLIST" || { echo "==> macOS did not take it; see the message above"; exit 1; }
    echo "==> it now starts by itself when you log in, and restarts if it stops"
    wait_up
    exit $?
    ;;
  uninstall)
    launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
    rm -f "$PLIST"
    stop_browser
    kill_port
    echo "==> it no longer starts by itself, and it is stopped (./serve.sh starts it by hand)"
    exit 0
    ;;
  status)
    hello=$(curl -s "http://127.0.0.1:$PORT/api/app/hello" 2>/dev/null || true)
    if [ -n "$hello" ]; then echo "running: $hello"; else echo "not running"; fi
    installed && echo "starts by itself: yes" || echo "starts by itself: no (./serve.sh install)"
    exit 0
    ;;
  stop)
    stop_browser
    if installed; then
      launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
      kill_port
      echo "==> stopped. It starts by itself again at your next login, or now with ./serve.sh"
    else
      kill_port
      echo "==> stopped"
    fi
    exit 0
    ;;
esac

# restart
stop_browser
if installed; then
  write_plist                                       # picks up a moved folder or a new Python
  if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    echo "==> restarting (macOS keeps it running)"
    launchctl kickstart -k "$DOMAIN/$LABEL"
  else
    kill_port
    launchctl bootstrap "$DOMAIN" "$PLIST"
  fi
else
  kill_port
  echo "==> starting on port $PORT (log: $LOG)"
  nohup "$PY" -m uvicorn main:app --host 127.0.0.1 --port "$PORT" >> "$LOG" 2>&1 &
fi
wait_up
