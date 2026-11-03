#!/usr/bin/env python3
"""生产口径的擦除并发基准：K 路 prepare 并行，比较 ERASE_PIPE 取值。

和 batch_pipeline 一样用 stage_worker 起进程、同样的线程环境变量，
只跳过翻译/压制（那两段不占擦除显存，且会单独测）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEV = Path(os.environ.get("VE_ROOT") or BASE)
PY = sys.executable
WORKER = DEV / "scripts" / "stage_worker.py"
CONFIG = DEV / "config.yaml"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def env_for(pipe: int):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    env.setdefault("ERASE_TORCH_THREADS", "8")
    env.setdefault("ERASE_CV_THREADS", "4")
    env.setdefault("ERASE_X264_THREADS", "12")
    env["ERASE_PIPE"] = str(pipe)
    return env


def gpu_rows(stop):
    rows = []
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"], capture_output=True,
                text=True, timeout=10).stdout.strip().splitlines()
            if out:
                u, m = [float(x) for x in out[0].split(",")[:2]]
                rows.append((u, m))
        except Exception:  # noqa: BLE001
            pass
        stop.wait(0.5)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcs", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--pipe", type=int, default=1)
    ap.add_argument("--tag", default="k4p1")
    args = ap.parse_args()

    root = BASE / "probe" / "bjobs" / args.tag
    shutil.rmtree(root, ignore_errors=True)
    jobs = []
    for s in args.srcs:
        src = Path(s)
        job = root / src.parent.name
        job.mkdir(parents=True)
        shutil.copyfile(src, job / "input.mp4")
        jobs.append(job)

    import threading
    stop = threading.Event()
    acc: list = []
    th = threading.Thread(target=lambda: acc.extend(gpu_rows(stop)), daemon=True)
    th.start()

    env = env_for(args.pipe)
    t0 = time.time()
    running, times, fails = [], {}, []
    pending = list(jobs)
    while pending or running:
        while pending and len(running) < args.jobs:
            job = pending.pop(0)
            f = open(job / "prepare.log", "ab")
            p = subprocess.Popen(
                [PY, str(WORKER), "prepare", "--job", str(job), "--target", "ja",
                 "--config", str(CONFIG)],
                stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            running.append({"job": job, "proc": p, "f": f, "t": time.time()})
            log(f"[{job.name}] prepare START")
        time.sleep(1)
        for r in list(running):
            if r["proc"].poll() is None:
                continue
            r["f"].close()
            dt = time.time() - r["t"]
            ok = (r["job"] / "state.json").exists()
            times[r["job"].name] = round(dt, 1)
            if not ok:
                fails.append(r["job"].name)
            log(f"[{r['job'].name}] {'OK' if ok else 'FAIL'} {dt:.0f}s")
            running.remove(r)
    total = time.time() - t0
    stop.set()
    th.join(timeout=5)

    utils = [u for u, _ in acc] or [0]
    out = {"tag": args.tag, "pipe": args.pipe, "jobs": args.jobs,
           "clips": len(jobs), "wall_s": round(total, 1), "per_clip": times,
           "fails": fails,
           "gpu": {"util_mean": round(statistics.mean(utils), 1),
                   "util_p50": sorted(utils)[len(utils) // 2],
                   "mem_max_mb": max(m for _, m in acc) if acc else 0}}
    p = BASE / "probe" / f"batchprep_{args.tag}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log("RESULT " + json.dumps({k: v for k, v in out.items() if k != "per_clip"},
                               ensure_ascii=False))


if __name__ == "__main__":
    main()
