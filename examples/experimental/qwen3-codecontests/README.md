# Multi Turn RL using Qwen3-1.7B on CodeContests (Miles + Megatron + Harbor)

Self-contained example to RL-train **Qwen3-1.7B** (dense) on the **CodeContests**
Harbor dataset with Miles + Megatron + Harbor, using the **mini-swe-agent**.

There are two ways to run it:

- **Single image (recommended, Docker-free grading).** One container runs the
  trainer, the Harbor agent server, and JupyterLab. Harbor grades each task on
  the bare host inside a proot sandbox via Harbor's `subprocess`
  environment — no Docker-in-Docker, no per-task containers, no `swe-net`. See
  **[Single-image quickstart](#single-image-quickstart)**.
- **Multi-container (original).** Three containers on a shared Docker network,
  with Harbor spawning a sibling Docker sandbox per task. See
  **[Multi-container quickstart](#multi-container-quickstart)**.

Both drive the same workflow from a notebook; the single-image path uses
`[run_from_scratch_single.ipynb](run_from_scratch_single.ipynb)`, the
multi-container path uses `[run_from_scratch.ipynb](run_from_scratch.ipynb)`.

## Single-image quickstart

### 1. Build the workshop image

Build context is the repo root (the Dockerfile bakes in the miles source, the
workshop, and the CodeContests dataset):

```bash
docker build -f docker/Dockerfile.workshop -t miles_workshop:v1 .   # or: just -f docker/justfile workshop
```

The image installs Harbor (with the `subprocess` environment) from the
`HARBOR_REF` build arg — by default the fork branch that carries it. It also
bakes `proot` + `uv` + `pytest`/`pytest-json-ctrf` so the per-task grader
runs on the bare host, and pre-extracts the CodeContests tasks + difficulty
splits (no §5 download at runtime).

### 2. Launch the single container and drop into its shell

```bash
GPU=0 WANDB_KEY=<optional> bash docker/run_workshop.sh
```

This runs ONE container with GPU access (`--privileged --ipc host`), publishes
Jupyter `:8888` and Harbor `:11000`, identity-mounts the repo, and mounts the HF
cache. **No Docker socket, no `swe-net`.**

### 3. Start JupyterLab and run the notebook

```bash
jupyter lab "examples/experimental/qwen3-codecontests/run_from_scratch_single.ipynb" \
  --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

Run the cells top-to-bottom. They install miles, start Harbor in-process
(`localhost:11000`, `subprocess` env), fetch the Qwen3-1.7B weights, and run
training — all in this one container. §5 (data prep) is a no-op because the
dataset is baked in.

### Notes / caveats (single image)

- **ROCm/MI300-specific & large.** Inherits the same rlsys base as
  `Dockerfile.cc-base` (~75GB). Other accelerators should adapt from the
  per-arch Dockerfiles.
- **Grading sandbox / Kubernetes.** Harbor grades each task on the bare host via
  the `subprocess` environment, which sandboxes with **proot** (userspace,
  ptrace-based path binding). proot needs no namespaces or capabilities, so the
  workshop runs in a fully **non-privileged** pod under the `RuntimeDefault`
  seccomp profile — the pod is the security boundary for untrusted task code;
  there is no Docker-in-Docker and no privileged container.
- **Non-privileged GPU.** Training needs only the ROCm devices (`/dev/kfd`,
  `/dev/dri`) + their group GIDs (`render`, `video`) and `--ipc host`/shm — NOT
  `--privileged` (that was only ever needed for the old Docker-in-Docker). The
  whole pod (training + proot grading) runs non-privileged.
- **Grading network egress.** Each task's `tests/test.sh` (from the dataset, not
  editable) still `curl`s `uv` and `uv add`s pytest at runtime; the image bakes
  these so the original `pytest --ctrf` path resolves, but PyPI/astral egress is
  assumed at grading time.
- **Reset is less hermetic.** With one container there is no `docker restart` to
  clear ray/sglang; §9 kills the processes instead. If endpoint errors persist
  after a reset, relaunch the container.
- **Override knobs:** `HARBOR_ENV_TYPE=docker` restores the legacy DinD path;
  `HARBOR_TASKS_DIR` points at a different task set; `HARBOR_REF` (build arg)
  pins a different Harbor.

## Multi-container quickstart

### 1. Clone the repo

```bash
git clone https://github.com/sreerohi/miles_aai.git
cd miles_aai
```

### 2. Build the three images

Note the different build contexts:

```bash
docker build -f docker/Dockerfile.cc-base    docker/ -t miles_base:v1     # pulls the ~75GB rlsys base
docker build -f docker/Dockerfile.agent-base docker/ -t agent_base:v1
docker build -f docker/Dockerfile.notebook   .       -t cc_notebook:v1    # context = repo root
```

### 3. Launch the notebook container and drop into its shell

```bash
GPU=0 WANDB_KEY=<optional> bash docker/run_notebook.sh
```

This creates the `swe-net` network, mounts the host repo + runtime + Docker
socket, and execs you into a bash shell. Set `GPU` to an idle device (default is `7`).

### 4. Start JupyterLab from inside that shell

```bash
jupyter lab "examples/experimental/qwen3-codecontests/run_from_scratch.ipynb" --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

Copy the printed `http://127.0.0.1:8888/lab?token=...` URL; open it in your browser
as `http://<host-ip>:8888/lab?token=...` (port 8888 is published).

### 5. Run the notebook cells top-to-bottom

- **Config cell** — derives all paths from the repo.
- **§1 Host setup** — creates `swe-net` (if not present).
- **§2** — launches `miles_swe` + `agent_env` from the images you built.
- **§3** — `pip install -e .` miles into `miles_swe`.
- **§4** — starts Harbor.
- **§5 Data prep** — downloads the CodeContests dataset (slow, one-time).
- **§6** — pre-fetches the Qwen3-1.7B weights.
- **§7+** — full training + the live monitors.

## Caveats worth knowing before you start

- The notebook **does not build** `miles_base` / `agent_base` — it only `docker run`s
them, so step 2 is mandatory first.
- Because the sibling containers bind-mount the host repo (`-v $REPO_DIR:$REPO_DIR`),
the `git clone` from step 1 is what makes them work.
- **GPU exposure:** the notebook starts `miles_swe` with `--privileged --ipc host`,
which gives ROCm device access; ensure the host ROCm stack is healthy (`rocm-smi` works).
- **HF:** Qwen3-1.7B and `open-thoughts/CodeContests` are public, so no token is
normally needed; set `HF_TOKEN` only if you hit rate limits.

