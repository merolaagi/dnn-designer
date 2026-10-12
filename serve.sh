#!/usr/bin/env bash
# (Re)start the Deep Network Designer in the background on port 8770, so Saathi and the browser reach it
# and the terminal stays free. Whatever was already listening on the port is stopped first.
#
#   ./serve.sh          restart it
#   ./serve.sh stop     stop it
#   ./serve.sh log      follow its log (Ctrl-C stops following, not the server)
#   ./serve.sh setup-browser   install the browser the Support agent chats in (once)

set -uo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8770}"
LOG="data/server.log"
mkdir -p data

if [ "${1:-}" = "log" ]; then
  exec tail -n 40 -f "$LOG"
fi

PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

if [ "${1:-}" = "setup-browser" ]; then
  echo "==> installing Playwright for $PY"
  "$PY" -m pip install --quiet playwright || "$PY" -m pip install --quiet --break-system-packages playwright || exit 1
  echo "==> installing its Chromium (used when Google Chrome is not there)"
  "$PY" -m playwright install chromium || exit 1
  echo "==> the browser is ready"
  exit 0
fi

# the browser process (browser_service.py) is restarted too, so a new version of it is used
bpids=$(lsof -ti tcp:"${SAATHI_BROWSER_PORT:-8772}" -sTCP:LISTEN 2>/dev/null || true)
[ -n "$bpids" ] && kill $bpids 2>/dev/null || true

pids=$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$pids" ]; then
  echo "==> stopping the server on port $PORT"
  kill $pids 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 0.5
    lsof -ti tcp:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break
  done
  pids=$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
fi
[ "${1:-}" = "stop" ] && { echo "==> stopped"; exit 0; }

echo "==> starting on port $PORT (log: $LOG)"
nohup "$PY" -m uvicorn main:app --host 127.0.0.1 --port "$PORT" >> "$LOG" 2>&1 &

for _ in $(seq 1 60); do
  sleep 0.5
  hello=$(curl -s "http://127.0.0.1:$PORT/api/app/hello" 2>/dev/null || true)
  if [ -n "$hello" ]; then
    version=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1]).get("version","?"))' "$hello" 2>/dev/null || echo "?")
    echo "==> running: version $version at http://127.0.0.1:$PORT"
    exit 0
  fi
done
echo "==> it did not answer within 30 seconds; the end of its log:"
tail -n 30 "$LOG"
exit 1
