#!/usr/bin/env python3
"""Extract CodeContests (or any Harbor-packaged HF dataset) into task dirs.

The dataset ``open-thoughts/CodeContests`` ships parquet files with two columns:
``path`` (task id, e.g. ``code_contests-0000``) and ``task_binary`` (a gzip tar
archive of that task's Harbor directory: instruction.md, environment/Dockerfile,
tests/, ...). This script downloads the dataset and extracts every task into

    <out>/<task_id>/...

It depends ONLY on ``pyarrow`` + ``tarfile`` (+ ``huggingface_hub`` for download).
No SkyRL, no Harbor adapter. Logic mirrors the dataset's own
``extract_parquet_tasks.py`` with hardened (path-traversal-safe) extraction.

Usage:
    # download from HF then extract
    python extract_codecontests.py --dataset open-thoughts/CodeContests --out ~/harbor_tasks_cc

    # extract from a parquet you already have (offline)
    python extract_codecontests.py --parquet /path/tasks.parquet --out ~/harbor_tasks_cc

    # only the first N tasks (quick validation)
    python extract_codecontests.py --dataset open-thoughts/CodeContests --out ~/harbor_tasks_cc --limit 20
"""

from __future__ import annotations

import argparse
import io
import os
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath


def _is_within(base: Path, target: Path) -> bool:
    try:
        return os.path.commonpath([str(base.resolve()), str(target.resolve())]) == str(
            base.resolve()
        )
    except Exception:
        return False


def _sanitize_member_name(name: str) -> str:
    parts = [p for p in PurePosixPath(name).parts if p not in ("..", ".", "", "/")]
    return str(PurePosixPath(*parts)) if parts else ""


def _safe_extract_tar(archive_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as tf:
        for member in tf.getmembers():
            name = _sanitize_member_name(member.name)
            if not name:
                continue
            if ".snapshot" in PurePosixPath(name).parts:
                continue
            target = (dest_dir / name).resolve()
            if not _is_within(dest_dir, target):
                raise RuntimeError(f"unsafe path in archive: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                with src, open(target, "wb") as dst:
                    dst.write(src.read())


def _rewrite_base_image(target: Path) -> None:
    """Optionally rewrite the task's ``environment/Dockerfile`` FROM line.

    When ``CC_TASK_BASE_IMAGE`` is set (e.g. ``cc_task_base:v1``), point each
    task container at a prebuilt base that already has uv + mini-swe-agent
    (with the ``litellm[proxy]`` extra) installed. This turns Harbor's per-trial
    ``uv tool install`` into a fast no-op and guarantees the fastapi/orjson proxy
    deps are present (no ``ModuleNotFoundError`` -> AgentError). No-op when the
    env var is unset, so default behavior (``FROM ubuntu:24.04``) is unchanged.
    Idempotent, and applied to already-extracted tasks on re-run.
    """
    base = os.environ.get("CC_TASK_BASE_IMAGE")
    if not base:
        return
    dockerfile = target / "environment" / "Dockerfile"
    if not dockerfile.exists():
        return
    try:
        text = dockerfile.read_text()
        new = re.sub(r"^FROM\s+\S+", f"FROM {base}", text, count=1, flags=re.MULTILINE)
        if new != text:
            dockerfile.write_text(new)
    except Exception as e:  # noqa: BLE001
        print(f"  WARN failed to rewrite base image for {target.name}: {e}")


def _extract_one(args: tuple) -> bool:
    rel_path, data, out_dir_str = args
    if not isinstance(rel_path, str) or not isinstance(data, (bytes, bytearray, memoryview)):
        return False
    out_dir = Path(out_dir_str)
    parts = [p for p in PurePosixPath(rel_path).parts if p not in ("..", "")]
    target = (out_dir / Path(*parts)).resolve() if parts else (out_dir / "task_unknown")
    if not _is_within(out_dir, target):
        return False
    if target.exists() and (target / "instruction.md").exists():
        _rewrite_base_image(target)  # patch base image even if already extracted
        return True  # idempotent: already extracted
    try:
        _safe_extract_tar(bytes(data), target)
        _rewrite_base_image(target)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  WARN failed to extract {rel_path}: {e}")
        return False


def _find_task_parquets(root: Path) -> list[Path]:
    import pyarrow.parquet as pq

    found = []
    for f in sorted(root.glob("**/*.parquet")):
        try:
            names = pq.read_schema(f).names
        except Exception:  # noqa: BLE001
            continue
        if "path" in names and "task_binary" in names:
            found.append(f)
    return found


def extract_parquet(parquet_path: Path, out_dir: Path, limit: int | None = None, workers: int = 8) -> int:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["path", "task_binary"])
    paths = table.column("path").to_pylist()
    data = table.column("task_binary").to_pylist()
    if limit is not None:
        paths, data = paths[:limit], data[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, d, str(out_dir)) for p, d in zip(paths, data)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_extract_one, jobs, chunksize=32))
    return sum(results)


def prepare(
    *,
    dataset: str | None,
    parquet: str | None,
    out: str,
    limit: int | None = None,
    workers: int = 8,
) -> str:
    out_dir = Path(os.path.expanduser(out)).resolve()

    if parquet:
        pqs = [Path(os.path.expanduser(parquet))]
    else:
        if not dataset:
            raise ValueError("either --dataset or --parquet is required")
        from huggingface_hub import snapshot_download

        print(f"downloading {dataset} ...")
        snap = Path(snapshot_download(repo_id=dataset, repo_type="dataset"))
        print(f"  -> {snap}")
        pqs = _find_task_parquets(snap)
        if not pqs:
            raise RuntimeError(f"no parquet with (path, task_binary) columns found under {snap}")

    total = 0
    for p in pqs:
        print(f"extracting {p.name} ...")
        total += extract_parquet(p, out_dir, limit=limit, workers=workers)
    print(f"done: {total} task dirs under {out_dir}")
    return str(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract Harbor CodeContests tasks (no SkyRL).")
    ap.add_argument("--dataset", default="open-thoughts/CodeContests", help="HF dataset id")
    ap.add_argument("--parquet", default=None, help="extract from a local parquet instead of downloading")
    ap.add_argument("--out", default=os.path.expanduser("~/harbor_tasks_cc"))
    ap.add_argument("--limit", type=int, default=None, help="only first N tasks")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    prepare(
        dataset=None if args.parquet else args.dataset,
        parquet=args.parquet,
        out=args.out,
        limit=args.limit,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
