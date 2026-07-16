"""Live training monitor for Section 8 of run_from_scratch.ipynb.

Public entry point:
    watch_training(work_dir=None, poll=8)

Each refresh prints: the pinned W&B run link, the train.log tail (interesting
lines), the CURRENT step's per-sample rollout panel (reused from rollout_monitor),
and live hover-able Plotly charts (step_time vs rollout_time, raw_reward,
truncated) once the first training step completes. Interrupt/stop to end.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime

from IPython.display import clear_output, display

import rollout_monitor  # the notebook-monitoring/ dir is on sys.path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"\[?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")  # keep in sync with rollout_monitor
WB = re.compile(r"(https?://\S*wandb\.ai/\S+)")  # W&B run/project URL printed by training
LOGPAT = re.compile(r"step \d+:|Finish rollout|update_weights|saved checkpoint|watchdog|Traceback|rollout \d+:|perf \d+:")


def _ensure_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import nbformat  # noqa: F401  (plotly mime rendering requires nbformat>=4.2.0)
    except ModuleNotFoundError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "plotly", "nbformat"])
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    return go, make_subplots


def wandb_url(tlog):  # prefer the run link so it never scrolls away
    run = proj = None
    if os.path.exists(tlog):
        for raw in open(tlog, errors="ignore"):
            for m in WB.finditer(ANSI.sub("", raw)):
                u = m.group(1).rstrip(").,]")
                if "/runs/" in u:
                    run = u
                elif proj is None:
                    proj = u
    return run or proj


def parse_steps(tlog):
    s = {}
    if not os.path.exists(tlog):
        return s
    for raw in open(tlog, errors="ignore"):
        ln = ANSI.sub("", raw)
        m = re.search(r"rollout (\d+): \{", ln)
        if m:
            d = s.setdefault(int(m.group(1)), {})
            for k in ("raw_reward", "truncated"):
                mm = re.search(rf"'rollout/{k}': ([0-9.eE+-]+)", ln)
                if mm:
                    d[k] = float(mm.group(1))
        m = re.search(r"perf (\d+): \{", ln)
        if m:
            # step_time and rollout_time live in (different) `perf N: {...}` dicts; grab whichever
            # this line carries. step_t comes straight from perf/step_time (the trainer's own
            # measurement) -- we do NOT time the loop ourselves, so step 0 has a real value too.
            d = s.setdefault(int(m.group(1)), {})
            for key in ("rollout_time", "step_time"):
                mm = re.search(rf"'perf/{key}': ([0-9.eE+-]+)", ln)
                if mm:
                    d[key] = float(mm.group(1))
            # avg turns per rollout (mean over the step's samples), logged as agent/turns_mean
            mm = re.search(r"'agent/turns_mean': ([0-9.eE+-]+)", ln)
            if mm:
                d["turns_mean"] = float(mm.group(1))
        m = re.search(r"step (\d+): \{.*'train/step'", ln)
        if m:
            d = s.setdefault(int(m.group(1)), {})
            t = TS.search(ln)
            if t:
                d["ts"] = datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    return s


def render_charts(s):
    go, make_subplots = _ensure_plotly()
    xs = sorted(s)
    step_t = [s[n].get("step_time") for n in xs]  # perf/step_time reported by the trainer (per step)
    roll_t = [s[n].get("rollout_time") for n in xs]
    reward = [s[n].get("raw_reward") for n in xs]
    trunc = [s[n].get("truncated") for n in xs]
    turns = [s[n].get("turns_mean") for n in xs]  # agent/turns_mean: avg agent turns per rollout sample

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("step_time vs rollout_time (s)", "rollout/raw_reward",
                                        "rollout/truncated", "agent/turns_mean (avg turns per rollout)"))
    fig.add_trace(go.Scatter(x=xs, y=step_t, name="step_time", mode="lines+markers"), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=roll_t, name="rollout_time", mode="lines+markers"), 1, 1)
    fig.add_trace(go.Scatter(x=xs, y=reward, name="raw_reward", mode="lines+markers"), 2, 1)
    fig.add_trace(go.Scatter(x=xs, y=trunc, name="truncated", mode="lines+markers"), 3, 1)
    fig.add_trace(go.Scatter(x=xs, y=turns, name="turns_mean", mode="lines+markers"), 4, 1)
    fig.update_yaxes(title_text="seconds", row=1, col=1)
    fig.update_yaxes(title_text="raw_reward (pass rate)", row=2, col=1)
    fig.update_yaxes(title_text="truncated fraction", row=3, col=1)
    fig.update_yaxes(title_text="avg turns / rollout", row=4, col=1)
    fig.update_xaxes(title_text="rollout / training step", row=4, col=1)
    fig.update_layout(height=1040, hovermode="x unified", margin=dict(l=70, r=20, t=40, b=45))
    display(fig)
    print(f"{'step':>4} {'step_t':>7} {'roll_t':>7} {'reward':>7} {'trunc':>6} {'turns':>6}")
    for i, n in enumerate(xs):
        print(f"{n:>4} {(('%.0f' % step_t[i]) if step_t[i] is not None else '-'):>7} "
              f"{(('%.0f' % roll_t[i]) if roll_t[i] is not None else '-'):>7} "
              f"{(('%.3f' % reward[i]) if reward[i] is not None else '-'):>7} "
              f"{(('%.3f' % trunc[i]) if trunc[i] is not None else '-'):>6} "
              f"{(('%.2f' % turns[i]) if turns[i] is not None else '-'):>6}")
    print("step_t = perf/step_time reported by the trainer for each step (its own measurement, "
          "including step 0); not wall-clock timed by this monitor.")


def watch_training(work_dir=None, poll=8, raw=True, tail_lines=5):
    """W&B link + train.log tail + current-step rollout panel + live charts. Interrupt to stop.

    raw=True (default) shows the full unfiltered train.log tail (a live stream, the last
    ``tail_lines`` lines); raw=False shows only curated milestone lines (step/rollout/perf/
    checkpoint/traceback).
    """
    work = rollout_monitor.default_work_dir(work_dir)
    tlog, trials, harbor_log = f"{work}/train.log", f"{work}/cc_trials", f"{work}/harbor.log"
    try:
        while True:
            s = parse_steps(tlog)
            step_ts = sorted(v["ts"] for v in s.values() if "ts" in v)
            n_done = len(step_ts)  # completed training steps
            cur_step = n_done  # the rollout step currently in progress
            clear_output(wait=True)

            # (1) Pinned W&B link.
            url = wandb_url(tlog)
            print(f"W&B run: {url}" if url else "W&B run: (link not in train.log yet)")

            # (2) train.log tail. Default = full unfiltered stream (last tail_lines, like
            # `tail -f`); raw=False = only step/rollout/perf/checkpoint/traceback lines.
            if raw:
                loglines = [ANSI.sub("", l) for l in rollout_monitor._tail(tlog, tail_lines).splitlines()]
                header = f"=== train.log (raw tail, last {tail_lines}) ==="
            else:
                loglines = [ANSI.sub("", l) for l in rollout_monitor._tail(tlog, 300).splitlines()
                            if LOGPAT.search(l)][-tail_lines:]
                header = "=== train.log (tail) ==="
            print(f"\n{header}")
            print("\n".join(loglines) or "(waiting for log...)")

            # (3) Per-sample rollout panel for the CURRENT step (scope to samples after the last finished step).
            scope_ep = step_ts[-1] if step_ts else None
            plines, _d, _r = rollout_monitor.rollout_panel(tlog, trials, harbor_log, scope_ep=scope_ep,
                                                           maxrows=None, title=f"step {cur_step} rollout",
                                                           include_tail=False)
            if any(l.startswith("[") for l in plines):  # only when this step has produced samples
                print()
                print("\n".join(plines))

            # (4) Charts after the first full rollout + training step.
            if n_done >= 1:
                print()
                render_charts(s)
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\n[stopped]")
