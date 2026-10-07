#!/usr/bin/env bash
# Launch the LeafMachine3 Electron GUI against the default LM3_settings.yaml.
#
#   ./launch_gui.sh            launch (no-op if already up) and wait for the server
#   ./launch_gui.sh restart    stop whatever is running, then launch fresh
#   ./launch_gui.sh status     report what is listening, which interpreter, which window
#   ./launch_gui.sh stop       stop the GUI and its server
#
# Uses only the checkout's uv-locked desktop group. Install it first with:
#   uv sync --frozen --extra <gpu|cpu|macos> --group desktop
#   uv run --frozen --no-sync lm3-desktop install
# The shell owns only the GUI it launches; Electron manages its own backend lifetime.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$HERE/app"
UV="$HERE/.venv/bin/uv"

export LM3_HOST="${LM3_HOST:-127.0.0.1}"
export LM3_PORT="${LM3_PORT:-8766}"
export DISPLAY="${DISPLAY:-:1}"

export LM3_ROOT="$HERE"
unset LM3_PYTHON CONDA_PREFIX VIRTUAL_ENV ELECTRON_RUN_AS_NODE ELECTRON_NO_ATTACH_CONSOLE || true

url="http://$LM3_HOST:$LM3_PORT"
log="${LM3_GUI_LOG:-$HERE/logs/gui.log}"
# Match on the absolute app path so this never touches another checkout's Electron.
pattern="$APP/node_modules/electron"

healthz() { curl -s -m 2 -o /dev/null -w '%{http_code}' "$url/healthz" 2>/dev/null || echo 000; }

# The WM reparents the window, so `xwininfo -root -children` will NOT list it and it
# looks like it never mapped; -tree finds it. Match the WM_CLASS in parens, not the
# title -- a plain grep for "leafmachine" also matches any VS Code window whose title
# happens to name a LeafMachine file, which reads as a mapped app window that is not.
window_line() { xwininfo -root -tree 2>/dev/null | grep -F '("leafmachine3"' | head -1 || true; }

do_stop() {
  local killed=0
  if pgrep -f "$pattern" >/dev/null 2>&1; then pkill -f "$pattern" 2>/dev/null || true; killed=1; fi
  # Never signal a backend by name. An attached CLI server belongs to its own caller;
  # Electron's shutdown request/watchdog handles only the backend that Electron owns.
  for _ in $(seq 1 10); do
    pgrep -f "$pattern" >/dev/null 2>&1 || break
    sleep 1
  done
  pkill -9 -f "$pattern" 2>/dev/null || true
  [[ $killed -eq 1 ]] && echo "stopped GUI and server" || echo "nothing running"
}

do_status() {
  local code; code="$(healthz)"
  printf 'port %s : %s\n' "$LM3_PORT" \
    "$([[ $code == 200 ]] && echo 'LM3 server up' || echo "no LM3 server (healthz=$code)")"
  printf 'uv      : %s\n' "$UV"
  local win; win="$(window_line)"
  printf 'window  : %s\n' "${win:-not mapped}"
  printf 'log     : %s\n' "$log"
}

do_launch() {
  [[ -x "$UV" ]] || {
    echo "missing uv desktop environment -- run uv sync --frozen --extra <gpu|cpu|macos> --group desktop, then uv run --frozen --no-sync lm3-desktop install" >&2
    exit 1
  }
  if [[ "$(healthz)" == 200 ]] && pgrep -f "$pattern" >/dev/null 2>&1; then
    echo "already up at $url"; do_status; return 0
  fi
  mkdir -p "$(dirname "$log")"
  # --no-sync preserves the hardware extra selected by the installation command. The desktop
  # command verifies that environment before npm starts, and Electron does its own frozen sync.
  ( cd "$HERE" && setsid --fork "$UV" run --frozen --no-sync lm3-desktop start >"$log" 2>&1 </dev/null )
  for i in $(seq 1 60); do
    [[ "$(healthz)" == 200 ]] && { echo "server up after ${i}s"; break; }
    sleep 1
  done
  [[ "$(healthz)" == 200 ]] || { echo "server did not come up; tail of $log:" >&2; tail -20 "$log" >&2; exit 1; }
  # The window maps a few seconds after the server answers /healthz. Poll for it, so a
  # "not mapped" below means the shell really failed rather than that we looked early.
  for _ in $(seq 1 30); do [[ -n "$(window_line)" ]] && break; sleep 1; done
  do_status
}

case "${1:-launch}" in
  launch)  do_launch ;;
  restart) do_stop; do_launch ;;
  stop)    do_stop ;;
  status)  do_status ;;
  *) sed -n '2,8p' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
