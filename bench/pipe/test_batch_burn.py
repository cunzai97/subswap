#!/usr/bin/env python3
"""batch_pipeline.finish_job + 压制线程池的自检（不碰 GPU/ffmpeg/翻译）。

验：
  1. finish_job 会把成片/qc/日志/report 落到 out/，并把 prep 记账写全
  2. finish_job 在压制失败时返回 False 且不落成片
  3. 三支片用线程池压制，墙钟明显低于串行（压制之间互不排队）
"""
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import os
import tempfile

BASE = Path(__file__).resolve().parents[2]
DEV = Path(os.environ.get("VE_ROOT") or BASE)
SCRATCH = Path(tempfile.mkdtemp(prefix="ve_test_burn_"))
sys.path.insert(0, str(DEV / "scripts"))

import batch_pipeline as B  # noqa: E402

TMP = SCRATCH / "batch_job"
OUT = SCRATCH / "batch_out"


class Args:
    fresh = False
    target = "ja"


def make_job(name, dur=90.0, fail=False):
    job = TMP / name
    shutil.rmtree(job, ignore_errors=True)
    (job / "work" / "qc").mkdir(parents=True)
    (job / "work" / "qc" / "qc_00.jpg").write_bytes(b"jpg")
    (job / "work" / "sub.ass").write_text("[Events]\n", encoding="utf-8")
    (job / "input.mp4").write_bytes(b"fake")
    (job / "report.json").write_text('{"duration": %s}' % dur, encoding="utf-8")
    job.joinpath("marker").write_text("x")
    return job


def fake_run_stage(stage, job, args, logpath, env, timeout):
    """压制在这里只睡一会，模拟 NVENC 出片。"""
    time.sleep(0.6)
    if job.name.startswith("bad"):
        return 1, 0.6, False
    (job / "output.mp4").write_bytes(b"out")
    (job / "report.json").write_text(
        '{"duration": 90.0, "render_check": {"ok": true}, "seconds_burn": 0.6}',
        encoding="utf-8")
    return 0, 0.6, False


def main():
    fail = 0
    B.run_stage = fake_run_stage
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    prep = {}

    # 1) 单支成功路径
    job = make_job("good1")
    name, ok = B.finish_job(job, Args(), {}, OUT, prep)
    out_mp4 = OUT / "good1_ja.mp4"
    if not (ok and out_mp4.exists() and prep["good1"]["burn_ok"] is True):
        print(f"FAIL 成功路径 ok={ok} 成片={out_mp4.exists()} prep={prep.get('good1')}")
        fail += 1
    if not (OUT / "good1_qc" / "qc_00.jpg").exists() \
            or not (OUT / "good1_ja.ass").exists() \
            or not (OUT / "good1_ja.report.json").exists():
        print("FAIL 交付物不齐（qc/ass/report）")
        fail += 1

    # 2) 失败路径必须有账、无成片
    bad = make_job("bad1")
    name, ok = B.finish_job(bad, Args(), {}, OUT, prep)
    if ok or (OUT / "bad1_ja.mp4").exists() or prep["bad1"]["burn_ok"] is not False:
        print(f"FAIL 失败路径 ok={ok} prep={prep.get('bad1')}")
        fail += 1

    # 3) 并发压制：3 支 × 0.6s 串行 ≈ 1.8s，池子里应接近 0.6s
    jobs = [make_job(f"par{i}", fail=False) for i in range(3)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(B.finish_job, j, Args(), {}, OUT, prep) for j in jobs]
        for f in futs:
            f.result()
    par = time.time() - t0

    prep2 = {}
    ser_jobs = [make_job(f"ser{i}", fail=False) for i in range(3)]
    t0 = time.time()
    for j in ser_jobs:
        B.finish_job(j, Args(), {}, OUT, prep2)
    ser = time.time() - t0
    print(f"并发压制 {par:.2f}s / 串行 {ser:.2f}s")
    if par > ser * 0.7:
        print("FAIL 并发压制没提速")
        fail += 1

    print("RESULT:", "PASS" if not fail else f"{fail} 项失败")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
