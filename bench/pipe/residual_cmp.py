#!/usr/bin/env python3
"""同一支片：两份 erased.mp4 的残留复查对比（用同一份 cues/per_shot）。

用来判断质量改动到底有没有把“擦不干净”变少——只看 OCR 复查命中数，
不看肉眼感觉。两份成片必须是同一支片、同一帧数。
"""
from __future__ import annotations

import os
import argparse
import json
import pickle
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEV = Path(os.environ.get("VE_ROOT") or BASE)
sys.path.insert(0, str(DEV / "webapp"))
sys.path.insert(0, str(DEV / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="A 成片（老代码）")
    ap.add_argument("--b", required=True, help="B 成片（新代码）")
    ap.add_argument("--cache", default="")
    args = ap.parse_args()

    import convert as C
    import pipeline as P

    cache = args.cache
    if not cache:
        cands = sorted((BASE / "probe" / "cache").glob("*.pkl"))
        cache = str(cands[-1])
    with open(cache, "rb") as f:
        d = pickle.load(f)
    meta = d["meta"]
    shots, per_shot = d["shots"], d["per_shot"]
    layout = d["layout"]
    source = d["source"]

    out = {}
    for tag, path in (("A", args.a), ("B", args.b)):
        pm = P.probe(path)
        if abs(pm["dur"] - meta["dur"]) > 0.5:
            print(f"警告：{tag} 时长 {pm['dur']} 与源 {meta['dur']} 不符")
        log = lambda m: None  # noqa: E731
        scan = C.residual_scan(Path(path), pm, layout, shots, per_shot, source, log)
        rep = C._report_scan(scan)
        out[tag] = rep
        print(f"{tag}: {json.dumps(rep, ensure_ascii=False)}")
    a = out["A"].get("suspects", 0)
    b = out["B"].get("suspects", 0)
    print(f"\n残留命中 A={a} B={b}")
    print("RESULT " + json.dumps({"a": a, "b": b}))


if __name__ == "__main__":
    main()
