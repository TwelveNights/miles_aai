"""Dependency-free builder for the Qwen3 / CodeContests train args.

This is the single source of truth for the command-line flags emitted by
``run-qwen3-codecontests.py``. It has NO ``miles``/``typer`` imports so it can be
unit-tested on a bare host (``test_launcher_args.py``), while the launcher maps
its ``ScriptArgs`` into ``CCArgs`` and calls :func:`build_train_args`.

Ported from ``run.py:execute()`` with the dense-Qwen3 changes:
  * SGLang parsers glm47/glm45 -> qwen25/qwen3
  * --tito-model glm47 -> qwen3
  * expert-model-parallel-size = 1 (dense; no MoE expert parallelism)
  * the SGLANG_ROCM_FUSED_DECODE_MLA=0 workaround is NOT emitted (Qwen3 = GQA,
    not MLA) -- see :func:`rocm_env_vars`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CCArgs:
    # mode / topology
    mode: str = "normal"  # "normal" | "debug_rollout_only"
    async_mode: bool = True
    train_num_gpus: int = 4
    num_gpus_per_node: int = 8
    num_nodes: int = 1

    # checkpoints / data
    hf_checkpoint: str = "Qwen/Qwen3-8B"
    ref_load: str = "/root/Qwen3-8B_torch_dist"
    save_dir: str = "/root/Qwen3-8B_codecontests/"
    save_interval: int = 20
    prompt_data: str = "/root/cc_train.jsonl"
    # Optional resume: when set, miles loads the latest checkpoint here and
    # continues the step count. Empty = fresh training from ref_load/hf_checkpoint.
    load: str = ""

    # rollout / batch
    max_seq_len: int = 16384
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    global_batch_size: int = 8
    num_rollout: int = 3000
    over_sampling_batch_size: int = 0
    # Per-response generation cap (tokens). 0 = use the mode default
    # (8192 sync / 16384 async); >0 overrides it.
    rollout_max_response_len: int = 0

    # qwen3 / codecontests fixed knobs
    tito_model: str = "qwen3"
    tool_call_parser: str = "qwen25"
    reasoning_parser: str = "qwen3"

    # logging
    wandb_key: str = ""
    wandb_project: str = "qwen3-8b-codecontests"
    wandb_run_name: str = "qwen3-8b-codecontests"
    wandb_team: str = ""
    use_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_run_name: str = "qwen3-8b-codecontests"


def build_train_args(a: CCArgs) -> str:
    async_mode = a.async_mode

    roll_temp = 0.7 if async_mode else 0.8
    roll_max_resp = 16384 if async_mode else (a.rollout_max_response_len or 8192)
    seq_len = 65536 if async_mode else a.max_seq_len
    roll_ctx = 65536 if async_mode else a.max_seq_len
    max_tok_per_gpu = 32768 if async_mode else 16384

    ckpt_args = (
        f"--hf-checkpoint {a.hf_checkpoint} "
        f"--ref-load {a.ref_load} "
        f"--save {a.save_dir} "
        f"--save-interval {a.save_interval} "
    )
    if a.load:
        # Resume policy weights + dataset consumption state, but skip the saved
        # optimizer/RNG state: restoring the precision-aware Adam state from a
        # distributed-optimizer checkpoint hits an incompatibility
        # (`unscaled_state.dtype` on a bool). Adam moments restart fresh; at
        # constant lr 1e-6 the impact is minor. Step count + data position are
        # still restored from --load, so no seen data is replayed.
        ckpt_args += f"--load {a.load} --no-load-optim --no-load-rng "

    rollout_args = (
        f"--prompt-data {a.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {a.num_rollout} "
        f"--rollout-batch-size {a.rollout_batch_size} "
        f"--n-samples-per-prompt {a.n_samples_per_prompt} "
        f"--rollout-temperature {roll_temp} "
        f"--rollout-max-response-len {roll_max_resp} "
        f"--max-seq-len {seq_len} "
        f"--rollout-max-context-len {roll_ctx} "
        f"--global-batch-size {a.global_batch_size} "
        "--balance-data "
    )
    if a.over_sampling_batch_size:
        rollout_args += f"--over-sampling-batch-size {a.over_sampling_batch_size} "

    # Dense Qwen3: no expert parallelism. Use TP up to 4, capped by the number
    # of training GPUs so single-GPU colocate runs correctly use TP=1.
    ep_size = 1
    n_gpus = a.num_gpus_per_node
    train_gpus = a.train_num_gpus if async_mode else n_gpus
    rollout_gpus = (n_gpus - a.train_num_gpus) if async_mode else n_gpus
    tp_size = min(4, train_gpus)

    perf_args = (
        f"--tensor-model-parallel-size {tp_size} "
        f"{'--sequence-parallel ' if tp_size > 1 else ''}"
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {ep_size} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {max_tok_per_gpu} "
        "--use-precision-aware-optimizer "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    # Qwen3 dense uses GQA + standard parsers (NOT glm47/glm45).
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        f"--sglang-tool-call-parser {a.tool_call_parser} "
        f"--sglang-reasoning-parser {a.reasoning_parser} "
        "--use-miles-router "
        "--sglang-router-port 31000 "
        f"--sglang-context-length {roll_ctx} "
        "--sglang-allow-auto-truncate "
    )
    if async_mode:
        sglang_args += "--sglang-disable-custom-all-reduce "

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path harbor.swe_agent_function.run "
        "--custom-rm-path harbor.generate.reward_func "
        "--rollout-function-path harbor.generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        f"--tito-model {a.tito_model} "
        "--use-session-server "
        "--session-server-port 30000 "
        # required by the mini-swe-agent harness
        "--tito-allowed-append-roles user tool "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
    )
    if async_mode:
        misc_args += (
            "--attention-backend flash "
            "--accumulate-allreduce-grads-in-fp32 "
            "--update-weight-transfer-mode broadcast "
            f"--update-weight-buffer-size {2 * 1024 ** 3} "
            "--pause-generation-mode in_place "
            "--use-fault-tolerance "
            "--rollout-health-check-first-wait 1800 "
            "--actor-num-nodes 1 "
            f"--actor-num-gpus-per-node {train_gpus} "
            f"--num-gpus-per-node {n_gpus} "
            f"--rollout-num-gpus {rollout_gpus} "
        )
    else:
        misc_args += (
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-backend flash "
            "--colocate "
        )
        # Multi-GPU colocate keeps the rollout engine resident to avoid
        # offload/reload latency. On a single GPU there isn't room for both the
        # SGLang KV-cache pool and the Megatron training step, so let
        # offload-rollout (default level: kv_cache + weight) free the GPU during
        # training to avoid OOM.
        if train_gpus > 1:
            misc_args += "--no-offload-rollout "
        misc_args += (
            f"--actor-num-nodes {a.num_nodes} "
            f"--actor-num-gpus-per-node {a.num_gpus_per_node} "
            f"--rollout-num-gpus {a.num_gpus_per_node} "
        )

    debug_args = "--debug-rollout-only " if a.mode == "debug_rollout_only" else ""

    wandb_args = ""
    if a.wandb_key:
        wandb_args = (
            "--use-wandb "
            f"--wandb-project {a.wandb_project} "
            f"--wandb-group {a.wandb_run_name} "
            f"--wandb-key {a.wandb_key} "
        )
        if a.wandb_team:
            wandb_args += f"--wandb-team {a.wandb_team} "

    prometheus_args = ""
    if a.use_prometheus:
        prometheus_args = (
            "--use-prometheus "
            f"--prometheus-port {a.prometheus_port} "
            f"--prometheus-run-name {a.prometheus_run_name} "
        )

    return (
        f"{ckpt_args}"
        f"{rollout_args}"
        f"{optimizer_args}"
        f"{grpo_args}"
        f"{wandb_args}"
        f"{prometheus_args}"
        f"{perf_args}"
        f"{sglang_args}"
        f"{agent_args}"
        f"{misc_args}"
        f"{debug_args}"
    )


def rocm_env_vars(num_gpus_per_node: int) -> dict:
    """ROCm GPU-visibility env for Ray (per radixark/miles#1118).

    Deliberately does NOT set ``SGLANG_ROCM_FUSED_DECODE_MLA`` -- that workaround is
    only needed for the GLM MLA decode path; Qwen3 uses standard GQA attention.
    """
    all_gpus = ",".join(str(i) for i in range(num_gpus_per_node))
    return {
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "HIP_VISIBLE_DEVICES": all_gpus,
        "SGLANG_SET_CPU_AFFINITY": "0",
    }
