#!/usr/bin/env bash
# Launch the SINGLE-IMAGE qwen3-codecontests workshop container and drop into a
# bash shell. One container runs everything: Megatron/SGLang trainer, the Harbor
# agent server, and JupyterLab. Harbor grades tasks on the bare host via the
# `subprocess` env's proot backend -- NO docker socket, NO swe-net, NO per-task
# containers.
#
# The image is SELF-CONTAINED: the miles source, the workshop, and the baked
# CodeContests dataset all live at /workspace/miles_aai inside the image. We do
# NOT mount the host repo (that would shadow the baked code/dataset). Only the
# HF cache (model weights) and the runtime dir (logs/trials, optional) are
# mounted for persistence across container restarts.
#
# From the shell, start JupyterLab yourself:
#     jupyter lab "$NB" --ip=0.0.0.0 --port=8888 --allow-root --no-browser
# then open http://<host>:8888 with the token it prints.
set -euo pipefail

# Baked-in paths inside the image (match Dockerfile.workshop ENV).
REPO_DIR="/workspace/miles_aai"
EX="$REPO_DIR/examples/experimental/qwen3-codecontests"

# Host dirs mounted for persistence (override via env). Keeping these on the host
# means weights/logs/trials survive `docker rm`; everything else is baked.
HF_CACHE="${HF_CACHE:-$HOME/cc_workshop/hf_cache}"   # HuggingFace cache (weights)
WORK_DIR="${WORK_DIR:-$HOME/cc_workshop/work}"        # logs + Harbor trials
GPU="${GPU:-7}"                                       # CC_HIP_VISIBLE_DEVICES
WANDB_KEY="${WANDB_KEY:-}"                            # optional; empty => W&B off
IMAGE="${IMAGE:-miles_workshop:v1}"
NAME="${NAME:-cc_workshop}"
PORT="${PORT:-8888}"
HARBOR_PORT="${HARBOR_PORT:-11000}"

mkdir -p "$WORK_DIR" "$HF_CACHE"

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  # GPU + SGLang needs: device access, --ipc host, large shm. NO repo mount
  # (baked code is used), NO docker.sock, NO swe-net. The runtime dir is bind
  # mounted at the image's baked EX/runtime so logs/trials land on the host.
  docker run -d --name "$NAME" --hostname "$NAME" \
    --device /dev/kfd --device /dev/dri \
    --privileged --ipc host --shm-size 32g --security-opt label=disable \
    -p "${PORT}:8888" \
    -p "${HARBOR_PORT}:11000" \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v "$WORK_DIR":"$EX/runtime/work" \
    -e GPU="$GPU" \
    -e CC_HIP_VISIBLE_DEVICES="$GPU" \
    -e WANDB_KEY="$WANDB_KEY" \
    -e WORK_DIR="$EX/runtime/work" \
    -e HARBOR_TRIALS_DIR="$EX/runtime/work/cc_trials" \
    -e MILES_HOST_IP="${MILES_HOST_IP:-127.0.0.1}" \
    -e MILES_ROUTER_EXTERNAL_HOST="${MILES_ROUTER_EXTERNAL_HOST:-127.0.0.1}" \
    "$IMAGE"
fi

NB="examples/experimental/qwen3-codecontests/run_from_scratch_single.ipynb"
echo "Container '$NAME' is up (Jupyter :${PORT}, Harbor :${HARBOR_PORT})."
echo "Open the notebook with:"
echo "    jupyter lab \"$NB\" --ip=0.0.0.0 --port=8888 --allow-root --no-browser"
echo "then browse to:  http://<this-host>:${PORT}/lab/tree/$NB?token=<token>"
exec docker exec -it -w "$REPO_DIR" "$NAME" bash
