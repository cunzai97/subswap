#!/usr/bin/env python3
"""两份成片差异落在哪里：判断“改动”是擦得更干净，还是擦到了不该擦的地方。

逐帧解码（抽 1/25 帧），算 |A-B| > 阈值 的像素，看它们是否落在字幕带/cue 框里。
"""
from __future__ import annotations

import os
import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEV = Path(os.environ.get("VE_ROOT") or BASE)
sys.path.insert(0, str(DEV / "scripts"))
sys.path.insert(0, str(DEV / "webapp"))

import numpy as np  # noqa: E402


def frames(path, w, h, step=25):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    n = 0
    size = w * h * 3
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        if n % step == 0:
            yield n, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        n += 1
    p.stdout.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--band", default="0.55,0.95")
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--thr", type=int, default=12)
    args = ap.parse_args()

    import pickle
    cache = sorted((BASE / "probe" / "cache").glob("*.pkl"))[-1]
    with open(cache, "rb") as f:
        d = pickle.load(f)
    meta = d["meta"]
    w, h = meta["w"], meta["h"]
    b0, b1 = [float(x) for x in args.band.split(",")]
    fb0 = int(h * b0) & ~1
    fb1 = int(h * b1) & ~1

    boxes = [c["box"] for c in d["cues"] if c.get("box")]
    stat = []
    for (n, fa), (_, fb) in zip(frames(args.a, w, h, args.step),
                                frames(args.b, w, h, args.step)):
        df = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(axis=2)
        hot = df > args.thr
        tot = int(hot.sum())
        if not tot:
            continue
        inband = int(hot[fb0:fb1].sum())
        # 落在某条 cue 框里（外扩 20px）的算“字幕区”
        inbox = 0
        ys, xs = np.nonzero(hot)
        for bx0, by0, bx1, by1 in boxes:
            m = (xs >= bx0 - 20) & (xs <= bx1 + 20) & (ys >= by0 - 20) & (ys <= by1 + 30)
            inbox += int(m.sum())
        stat.append({"frame": n, "hot": tot, "inband_pct": round(100 * inband / tot, 1),
                     "inbox_pct": round(100 * min(inbox, tot) / tot, 1)})

    if not stat:
        print("RESULT " + json.dumps({"diff_frames": 0}))
        return
    hot = [s["hot"] for s in stat]
    out = {"diff_frames": len(stat),
           "hot_mean_px": round(float(np.mean(hot)), 1),
           "hot_max_px": int(max(hot)),
           "inband_pct_mean": round(float(np.mean([s["inband_pct"] for s in stat])), 1),
           "inbox_pct_mean": round(float(np.mean([s["inbox_pct"] for s in stat])), 1),
           "worst": sorted(stat, key=lambda s: -s["hot"])[:5]}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("RESULT " + json.dumps({k: v for k, v in out.items() if k != "worst"}))


if __name__ == "__main__":
    main()
