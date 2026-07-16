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

from IPython.display import display

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


def _ensure_widgets():
    """Return the go module if go.FigureWidget (ipywidgets) is usable, else None.

    FigureWidget lets us mutate the chart in place instead of re-sending the whole
    figure every poll. If ipywidgets isn't available we fall back to display(fig).
    """
    go, _ = _ensure_plotly()
    try:
        import ipywidgets  # noqa: F401
        go.FigureWidget()
        return go
    except Exception:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ipywidgets"])
            import ipywidgets  # noqa: F401
            go.FigureWidget()
            return go
        except Exception:
            return None


def wandb_url(tlog, state=None):  # prefer the run link so it never scrolls away
    """Return the W&B run (or project) URL from train.log.

    With ``state`` (a dict), the resolved run link is cached the first time it is
    seen and scanning stops thereafter; only newly appended bytes are read on
    subsequent calls. Without ``state`` the whole file is re-scanned (unused now,
    kept for callers/tests).
    """
    if state is not None:
        if state.get("run"):
            return state["run"]
        run, proj = state.get("run"), state.get("proj")
        for raw in rollout_monitor._read_new(tlog, state):
            for m in WB.finditer(ANSI.sub("", raw)):
                u = m.group(1).rstrip(").,]")
                if "/runs/" in u:
                    run = u
                elif proj is None:
                    proj = u
        state["run"], state["proj"] = run, proj
        return run or proj
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


def _parse_line(ln, s):
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


def parse_steps(tlog, state=None):
    """Parse per-step metrics from train.log into {step: {...}}.

    With ``state`` (a dict), accumulates into ``state['s']`` across polls and reads
    only bytes appended since the last call (offset tracked in ``state``); the log
    shrinking (new run) resets both. Without ``state`` the whole file is parsed.
    """
    if state is not None:
        if os.path.exists(tlog) and os.path.getsize(tlog) < state.get(tlog, 0):
            state["s"] = {}  # log truncated (new run) -> drop stale steps
        s = state.setdefault("s", {})
        for raw in rollout_monitor._read_new(tlog, state):
            _parse_line(ANSI.sub("", raw), s)
        return s
    s = {}
    if not os.path.exists(tlog):
        return s
    for raw in open(tlog, errors="ignore"):
        _parse_line(ANSI.sub("", raw), s)
    return s


def _chart_arrays(s):  # (xs, step_t, roll_t, reward, trunc, turns) aligned by step
    xs = sorted(s)
    return (xs,
            [s[n].get("step_time") for n in xs],  # perf/step_time reported by the trainer (per step)
            [s[n].get("rollout_time") for n in xs],
            [s[n].get("raw_reward") for n in xs],
            [s[n].get("truncated") for n in xs],
            [s[n].get("turns_mean") for n in xs])  # agent/turns_mean: avg agent turns per rollout sample


def _new_figure(go, make_subplots, widget=False):
    """4-subplot figure with 5 empty traces. widget=True -> mutable go.FigureWidget."""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("step_time vs rollout_time (s)", "rollout/raw_reward",
                                        "rollout/truncated", "agent/turns_mean (avg turns per rollout)"))
    fig.add_trace(go.Scatter(x=[], y=[], name="step_time", mode="lines+markers"), 1, 1)
    fig.add_trace(go.Scatter(x=[], y=[], name="rollout_time", mode="lines+markers"), 1, 1)
    fig.add_trace(go.Scatter(x=[], y=[], name="raw_reward", mode="lines+markers"), 2, 1)
    fig.add_trace(go.Scatter(x=[], y=[], name="truncated", mode="lines+markers"), 3, 1)
    fig.add_trace(go.Scatter(x=[], y=[], name="turns_mean", mode="lines+markers"), 4, 1)
    fig.update_yaxes(title_text="seconds", row=1, col=1)
    fig.update_yaxes(title_text="raw_reward (pass rate)", row=2, col=1)
    fig.update_yaxes(title_text="truncated fraction", row=3, col=1)
    fig.update_yaxes(title_text="avg turns / rollout", row=4, col=1)
    fig.update_xaxes(title_text="rollout / training step", row=4, col=1)
    fig.update_layout(height=1040, hovermode="x unified", margin=dict(l=70, r=20, t=40, b=45))
    return go.FigureWidget(fig) if widget else fig


def _fill_traces(fig, s):
    xs, step_t, roll_t, reward, trunc, turns = _chart_arrays(s)
    for tr, y in zip(fig.data, (step_t, roll_t, reward, trunc, turns)):
        tr.x, tr.y = xs, y


def update_figure(fig, s):  # in-place mutation of a persistent FigureWidget (no re-display)
    with fig.batch_update():
        _fill_traces(fig, s)


def step_table(s):  # text table of per-step metrics (printed in the text region)
    xs, step_t, roll_t, reward, trunc, turns = _chart_arrays(s)
    lines = [f"{'step':>4} {'step_t':>7} {'roll_t':>7} {'reward':>7} {'trunc':>6} {'turns':>6}"]
    for i, n in enumerate(xs):
        lines.append(f"{n:>4} {(('%.0f' % step_t[i]) if step_t[i] is not None else '-'):>7} "
                     f"{(('%.0f' % roll_t[i]) if roll_t[i] is not None else '-'):>7} "
                     f"{(('%.3f' % reward[i]) if reward[i] is not None else '-'):>7} "
                     f"{(('%.3f' % trunc[i]) if trunc[i] is not None else '-'):>6} "
                     f"{(('%.2f' % turns[i]) if turns[i] is not None else '-'):>6}")
    lines.append("step_t = perf/step_time reported by the trainer for each step (its own measurement, "
                 "including step 0); not wall-clock timed by this monitor.")
    return "\n".join(lines)


def render_charts(s):  # fallback path: build + display a fresh figure, then the table
    go, make_subplots = _ensure_plotly()
    fig = _new_figure(go, make_subplots, widget=False)
    _fill_traces(fig, s)
    display(fig)
    print(step_table(s))


def _build_text(tlog, trials, harbor_log, s, step_ts, cur_step, url, raw, tail_lines,
                trial_cache, harbor_state):
    """Assemble the text region (W&B link + log tail + rollout panel + step table)."""
    out = [f"W&B run: {url}" if url else "W&B run: (link not in train.log yet)"]

    # train.log tail. Default = full unfiltered stream (last tail_lines, like `tail -f`);
    # raw=False = only step/rollout/perf/checkpoint/traceback lines.
    if raw:
        loglines = [ANSI.sub("", l) for l in rollout_monitor._tail(tlog, tail_lines).splitlines()]
        header = f"=== train.log (raw tail, last {tail_lines}) ==="
    else:
        loglines = [ANSI.sub("", l) for l in rollout_monitor._tail(tlog, 300).splitlines()
                    if LOGPAT.search(l)][-tail_lines:]
        header = "=== train.log (tail) ==="
    out += ["", header, "\n".join(loglines) or "(waiting for log...)"]

    # Per-sample rollout panel for the CURRENT step (scope to samples after the last finished step).
    scope_ep = step_ts[-1] if step_ts else None
    plines, _d, _r = rollout_monitor.rollout_panel(tlog, trials, harbor_log, scope_ep=scope_ep,
                                                   maxrows=None, title=f"step {cur_step} rollout",
                                                   include_tail=False, trial_cache=trial_cache,
                                                   harbor_state=harbor_state)
    if any(l.startswith("[") for l in plines):  # only when this step has produced samples
        out += ["", "\n".join(plines)]

    if len(step_ts) >= 1:  # step table once the first step has completed
        out += ["", step_table(s)]
    return "\n".join(out)


def watch_training(work_dir=None, poll=8, raw=True, tail_lines=5):
    """W&B link + train.log tail + current-step rollout panel + live charts. Interrupt to stop.

    Incremental by design: each poll reads only bytes appended to train.log/harbor.log
    since the last poll and re-parses only trial dirs whose files changed, so refresh
    cost stays roughly flat as the run grows. The charts update in place via a
    go.FigureWidget when ipywidgets is available; otherwise they fall back to an
    update-in-place display handle refreshed only when a new step appears.

    raw=True (default) shows the full unfiltered train.log tail (a live stream, the last
    ``tail_lines`` lines); raw=False shows only curated milestone lines (step/rollout/perf/
    checkpoint/traceback).
    """
    work = rollout_monitor.default_work_dir(work_dir)
    tlog, trials, harbor_log = f"{work}/train.log", f"{work}/cc_trials", f"{work}/harbor.log"

    # Cross-poll state (created once; reset naturally when the cell is re-run). parse_steps
    # and wandb_url each need their OWN dict because both track a byte offset into tlog.
    pstate, wstate, trial_cache, harbor_state = {}, {}, {}, {}

    go, make_subplots = _ensure_plotly()
    use_widget = _ensure_widgets() is not None  # FigureWidget (in-place) available?
    from IPython.display import Pretty

    text_out = fig = text_h = fig_h = None
    if use_widget:  # primary path: persistent FigureWidget + Output text region
        import ipywidgets
        text_out = ipywidgets.Output()
        fig = _new_figure(go, make_subplots, widget=True)
        display(ipywidgets.VBox([text_out, fig]))
    else:  # fallback: update-in-place display handles (no ipywidgets required)
        text_h = display(Pretty("(waiting for log...)"), display_id=True)

    last_charted = -1
    try:
        while True:
            s = parse_steps(tlog, state=pstate)
            step_ts = sorted(v["ts"] for v in s.values() if "ts" in v)
            n_done = len(step_ts)  # completed training steps
            cur_step = n_done  # the rollout step currently in progress
            url = wandb_url(tlog, state=wstate)
            text = _build_text(tlog, trials, harbor_log, s, step_ts, cur_step, url, raw, tail_lines,
                               trial_cache, harbor_state)

            if use_widget:
                text_out.clear_output(wait=True)  # scoped to the text region; leaves the chart alone
                with text_out:
                    print(text)
                if n_done >= 1:
                    update_figure(fig, s)  # in-place; no re-display
            else:
                # display(display_id=True) returns a handle in a real kernel; guard
                # against None (e.g. plain interpreter) so the loop never crashes.
                if text_h is not None:
                    text_h.update(Pretty(text))
                else:
                    print(text)
                if n_done >= 1 and n_done != last_charted:  # throttle: only resend the figure on a new step
                    nf = _new_figure(go, make_subplots, widget=False)
                    _fill_traces(nf, s)
                    if fig_h is None:
                        fig_h = display(nf, display_id=True)
                    elif hasattr(fig_h, "update"):
                        fig_h.update(nf)
                    last_charted = n_done
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\n[stopped]")
