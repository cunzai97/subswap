#!/usr/bin/env python3
"""给擦除段做函数级计时：找出 230s 里到底是谁在花时间。

做法：把 pipeline / erase_ref 里几个热函数用装饰器包一层累计墙钟，
只跑前 N 个镜头，结束打印柱状表。不落盘成片（writer 换成空写）。
"""
from __future__ import annotations

import os
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEV = Path(os.environ.get("VE_ROOT") or BASE)
sys.path.insert(0, str(DEV / "webapp"))
sys.path.insert(0, str(DEV / "scripts"))

import yaml  # noqa: E402

ACC: dict[str, list] = {}


def wrap(mod, name, label=None, skip_first=0):
    """给 mod.name 加计时。"""
    label = label or name
    fn = getattr(mod, name)
    state = {"n": 0}

    def inner(*a, **k):
        state["n"] += 1
        if state["n"] <= skip_first:
            return fn(*a, **k)
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            rec = ACC.setdefault(label, [0.0, 0])
            rec[0] += dt
            rec[1] += 1

    setattr(mod, name, inner)


def wrap_gen(mod, name, label=None):
    """生成器函数要按 next() 的时间记，不能只记创建开销。"""
    label = label or name
    fn = getattr(mod, name)

    def inner(*a, **k):
        gen = fn(*a, **k)
        while True:
            t0 = time.perf_counter()
            try:
                item = next(gen)
            except StopIteration:
                return
            dt = time.perf_counter() - t0
            rec = ACC.setdefault(label, [0.0, 0])
            rec[0] += dt
            rec[1] += 1
            yield item

    setattr(mod, name, inner)


class NullWriter:
    def __init__(self):
        self.n = 0
        self.stdin = self

    def write(self, b):
        t0 = time.perf_counter()
        self.n += 1
        dt = time.perf_counter() - t0
        rec = ACC.setdefault("encode.write", [0.0, 0])
        rec[0] += dt
        rec[1] += 1

    def close(self):
        pass

    def wait(self):
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(BASE / "probe" / "work" / "p1" / "input.mp4"))
    ap.add_argument("--shots", type=int, default=12)
    ap.add_argument("--pipe", type=int, default=1)
    args = ap.parse_args()

    import convert as C
    import pipeline as P
    import erase_ref as R

    src = Path(args.src)
    cache = BASE / "probe" / "cache"
    pkl = None
    for p in sorted(cache.glob("*.pkl")):
        with open(p, "rb") as f:
            d = pickle.load(f)
        if Path(d["src"]).name == "input.mp4" or Path(d["src"]).exists():
            pkl = d
    if pkl is None:
        raise SystemExit("先跑一次 erase_bench 生成提取缓存")

    cfg = yaml.safe_load((DEV / "config.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("inpaint", {})["fixed_w"] = 768

    wrap_gen(P, "iter_frames", "decode(iter_frames)")
    wrap(R, "_glyph_mask", "glyph_mask")
    wrap(R, "_flow_to", "flow(DIS)")
    wrap(R, "_warp_with", "warp")
    wrap(R, "_photometric", "photometric")
    wrap(R, "_lama_fill", "lama_fill(GPU)")
    wrap(P, "temporal_stabilize", "temporal_stab")
    wrap(P, "track_mask", "track_mask(legacy)")

    shots = pkl["shots"][:args.shots]
    per_shot = pkl["per_shot"][:args.shots]
    cues = [c for c in pkl["cues"] if c["start"] < shots[-1]["end"]]
    meta = dict(pkl["meta"])
    meta["dur"] = shots[-1]["end"]

    t0 = time.perf_counter()
    frames = 0
    n = C.erase(src, meta, pkl["layout"], shots, per_shot, cfg,
                on_prog=lambda f, m: None, log=lambda m: None,
                mask_mode="stroke", cues=cues, pipe=args.pipe)
    total = time.perf_counter() - t0
    print(json.dumps({"total_s": round(total, 1), "shots": len(shots),
                      "pipe": args.pipe}, ensure_ascii=False))
    rows = sorted(ACC.items(), key=lambda kv: -kv[1][0])
    for label, (s, c) in rows:
        print(f"{label:22s} {s:8.2f}s  n={c:5d}  {100*s/total:5.1f}%")
    print(f"{'TOTAL':22s} {total:8.2f}s")


if __name__ == "__main__":
    main()
