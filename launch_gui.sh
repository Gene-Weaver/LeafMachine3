#!/usr/bin/env bash
# Launch the LeafMachine3 Electron GUI against the default LM3_settings.yaml.
#
#   ./launch_gui.sh            launch (no-op if already up) and wait for the server
#   ./launch_gui.sh restart    stop whatever is running, then launch fresh
#   ./launch_gui.sh status     report what is listening, which interpreter, which window
#   ./launch_gui.sh stop       stop the GUI and its server
#
# Four environment gotchas are handled here. Every one of them fails SILENTLY or
# misleadingly, so none of them should be left to the caller:
#
#   (1) INTERPRETER. The Electron shell resolves the backend in the order
#       LM3_PYTHON -> VIRTUAL_ENV -> CONDA_PREFIX -> <checkout>/bin/python3 -> `lm3`
#       on PATH (app/main.js: resolveBackendLaunch). This checkout's venv lives at
#       .venv_LM3/, NOT bin/, so the checkout candidate never matches. Launched from a
#       conda-active shell it therefore picks miniconda base, which has fastapi and can
#       serve the whole UI -- but has no ultralytics, so the server starts clean and then
#       EVERY stage worker dies with ModuleNotFoundError once a run begins.
#   (2) ELECTRON_RUN_AS_NODE=1 is set by VS Code's terminal. Inherited, the electron
#       binary behaves as plain node: require("electron") returns the binary's path
#       STRING, app/ipcMain are undefined, and it exits without opening a window.
#   (3) PORT 8765 is occupied on this machine by an unrelated app (~/Dropbox/3D_Topo).
#       It 404s on /healthz, so the shell won't mis-attach -- it tries to spawn its own
#       uvicorn, which cannot bind. Default to 8766 instead.
#   (4) DISPLAY is unset in non-graphical/agent shells, and such shells also reap the
#       process group on exit -- hence DISPLAY=:1 and setsid.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$HERE/app"
ELECTRON="$APP/node_modules/electron/dist/electron"

export LM3_HOST="${LM3_HOST:-127.0.0.1}"
export LM3_PORT="${LM3_PORT:-8766}"
export DISPLAY="${DISPLAY:-:1}"

# (1) Pin the interpreter, and clear the two prefixes that would otherwise win the
#     fallback chain. LM3_PYTHON already outranks them; unsetting them means a stale
#     conda prefix cannot be inherited by the server or its spawned stage workers.
export LM3_PYTHON="${LM3_PYTHON:-$HERE/.venv_LM3/bin/python}"
unset CONDA_PREFIX VIRTUAL_ENV || true
# (2)
unset ELECTRON_RUN_AS_NODE ELECTRON_NO_ATTACH_CONSOLE || true

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
  # Quitting the shell normally stops the server too, but a shell killed outright
  # leaves it holding the port -- and the next launch would ADOPT that orphan and run
  # whatever stale code it started with. Reap it explicitly.
  if pgrep -f 'uvicorn leafmachine3.server.app' >/dev/null 2>&1; then
    pkill -f 'uvicorn leafmachine3.server.app' 2>/dev/null || true; killed=1
  fi
  for _ in $(seq 1 10); do
    pgrep -f "$pattern" >/dev/null 2>&1 || pgrep -f 'uvicorn leafmachine3.server.app' >/dev/null 2>&1 || break
    sleep 1
  done
  pkill -9 -f "$pattern" 2>/dev/null || true
  pkill -9 -f 'uvicorn leafmachine3.server.app' 2>/dev/null || true
  [[ $killed -eq 1 ]] && echo "stopped GUI and server" || echo "nothing running"
}

do_status() {
  local code; code="$(healthz)"
  printf 'port %s : %s\n' "$LM3_PORT" \
    "$([[ $code == 200 ]] && echo 'LM3 server up' || echo "no LM3 server (healthz=$code)")"
  printf 'python  : %s\n' "$LM3_PYTHON"
  local server; server="$(pgrep -af 'uvicorn leafmachine3.server.app' | head -1 || true)"
  [[ -n $server ]] && printf 'serving : %s\n' "$server"
  local win; win="$(window_line)"
  printf 'window  : %s\n' "${win:-not mapped}"
  printf 'log     : %s\n' "$log"
}

do_launch() {
  [[ -x "$LM3_PYTHON" ]] || { echo "LM3_PYTHON is not runnable: $LM3_PYTHON" >&2; exit 1; }
  [[ -x "$ELECTRON" ]] || { echo "missing electron -- run: (cd $APP && npm install)" >&2; exit 1; }
  if [[ "$(healthz)" == 200 ]] && pgrep -f "$pattern" >/dev/null 2>&1; then
    echo "already up at $url"; do_status; return 0
  fi
  mkdir -p "$(dirname "$log")"
  # Invoke the dist binary directly rather than `npm start`, so no node/nvm is needed
  # (nvm only loads in interactive shells).
  #
  # `setsid --fork`, not a bare `setsid`: bare setsid only calls setsid(2) and EXECS IN
  # PLACE when it is not already a process-group leader, so electron stayed a direct child
  # of this script and the shell blocked in wait() on it -- `launch_gui.sh restart` never
  # returned even though the app was up and healthy. --fork guarantees a fork, so electron
  # is reparented to init and this script can exit. The new session also stops an agent or
  # CI shell from reaping the app when it exits.
  ( cd "$APP" && setsid --fork "$ELECTRON" . >"$log" 2>&1 </dev/null )
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
