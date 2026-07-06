#!/usr/bin/env bash
# Workshop pod entrypoint for the single-container (proot) miles-codecontests
# image. Starts the Harbor agent server in the background, waits for it to be
# healthy on localhost, then execs JupyterLab in the foreground.
#
# All services run in THIS container and talk over localhost:
#   - Harbor agent server on :11000 (background, proot subprocess backend)
#   - JupyterLab on :8888 (foreground)
#
# Loopback wiring (MILES_HOST_IP / MILES_ROUTER_EXTERNAL_HOST / AGENT_SERVER_URL)
# and the subprocess-env defaults are baked into the image ENV, so there is
# nothing to configure here. The full foreground command is passed as arguments
# (the launcher passes `jupyter lab <per-pod flags>`) and exec'd verbatim.
set -uo pipefail

EX=/workspace/miles_aai/examples/experimental/qwen3-codecontests
WORK="$EX/runtime/work"
mkdir -p "$WORK"

# Backstop: clear any stale aiter JIT-compile lock. With kernels prebaked into
# the image nothing compiles at runtime, so no FileBaton lock is created and this
# normally deletes nothing. It only matters if a future base-image change adds a
# kernel that compiles at runtime and a user interrupts that first compile (an
# orphaned lock makes aiter's wait() loop forever). Cheap insurance, not the
# primary fix.
rm -f /app/aiter/aiter/jit/build/lock_* 2>/dev/null || true

echo "=== Starting Harbor (proot) ==="
(
  cd "$EX" && PYTHONPATH=/workspace/miles_aai exec setsid \
    "${HARBOR_PYTHON:-python3}" harbor/server.py --port 11000 --max-concurrent 8
) </dev/null >"$WORK/harbor.log" 2>&1 &

for i in $(seq 1 30); do
  if curl -sf http://localhost:11000/health >/dev/null 2>&1; then
    echo "Harbor up"
    break
  fi
  echo "  harbor attempt $i/30..."
  sleep 2
done

unset GITHUB_TOKEN EXA_API_KEY
echo "=== Starting JupyterLab ==="
cd "$EX"
exec "$@"
