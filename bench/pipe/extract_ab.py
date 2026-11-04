#!/usr/bin/env python3
"""extract（OCR 建轨）提速 A/B：只换 ocr 配置，比时间 + cue 集合是否一致。

判据：cue 的 (文本, 起, 止) 与基线逐条比对。任何丢 cue / 文本变化 / 时间漂移
> 0.12s 都算回归。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("VE_ROOT") or BASE)
sys.path.insert(0, str(ROOT / "webapp"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402


def run_extract(C, P, norm, meta, layout, ocr):
    t0 = time.perf_counter()
    cues, source, shots, per_shot = C.extract(norm, meta, layout, "auto",
                                              lambda m: None, ocr=ocr)
    dt = time.perf_counter() - t0
    return dt, cues, shots


def key(c, nd=2):
    return (c["text"].strip(), round(c["start"], nd), round(c["end"], nd))


def cmp_cues(base, new, tol=0.12):
    """按起点对齐比较：文本必须相同，起止时间差 ≤ tol。"""
    out = {"missing": [], "extra": [], "text_changed": [], "time_shift": []}
    bs = sorted(base, key=lambda c: c["start"])
    ns = sorted(new, key=lambda c: c["start"])
    used = set()
    for b in bs:
        hit = None
        for i, n in enumerate(ns):
            if i in used:
                continue
            if abs(n["start"] - b["start"]) <= tol:
                hit = i
                break
        if hit is None:
            out["missing"].append(key(b))
            continue
        used.add(hit)
        n = ns[hit]
        if n["text"].strip() != b["text"].strip():
            out["text_changed"].append({"base": key(b), "new": key(n)})
        elif abs(n["end"] - b["end"]) > tol:
            out["time_shift"].append({"base": key(b), "new": key(n),
                                      "d_end": round(n["end"] - b["end"], 3)})
    for i, n in enumerate(ns):
        if i not in used:
            out["extra"].append(key(n))
    return out


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else
               "/home/pan/work/review-scratch/video-erase/probe/chain_src/ep04.mp4")
    import convert as C
    import pipeline as P

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    work = Path(os.environ.get("VE_TMP") or "/home/pan/work/review-scratch/video-erase/probe/ext_ab")
    work.mkdir(parents=True, exist_ok=True)
    norm = C.normalize(src, work, lambda m: None)
    meta = P.probe(str(norm))
    layout = C.detect_layout(norm, meta, lambda m: None)
    print(f"素材 {meta['w']}x{meta['h']} {meta['dur']:.1f}s")

    variants = [
        ("base", {"det_limit_side_len": 736, "det_limit_type": "min"}, {}),
        ("max960", {"det_limit_side_len": 960, "det_limit_type": "max"}, {}),
        ("max1280", {"det_limit_side_len": 1280, "det_limit_type": "max"}, {}),
        ("max1600", {"det_limit_side_len": 1600, "det_limit_type": "max"}, {}),
        ("min384", {"det_limit_side_len": 384, "det_limit_type": "min"}, {}),
        ("min480", {"det_limit_side_len": 480, "det_limit_type": "min"}, {}),
        ("min560", {"det_limit_side_len": 560, "det_limit_type": "min"}, {}),
    ]
    base_cues = None
    rows = []
    for name, ocr, _extra in variants:
        dt, cues, shots = run_extract(C, P, norm, meta, layout, ocr)
        row = {"name": name, "extract_s": round(dt, 1), "cues": len(cues),
               "shots": len(shots), "ocr": ocr}
        if base_cues is None:
            base_cues = cues
        else:
            d = cmp_cues(base_cues, cues)
            row["regress"] = {k: len(v) for k, v in d.items()}
            row["detail"] = {k: v[:5] for k, v in d.items() if v}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "detail"},
                         ensure_ascii=False))
    out = work / f"extract_ab_{int(time.time())}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = all(all(n == 0 for n in r.get("regress", {}).values()) for r in rows[1:])
    print("RESULT " + json.dumps({"ok": ok, "best": min(
        rows[1:], key=lambda r: r["extract_s"])["name"] if len(rows) > 1 else None}))
    print(f"明细 {out}")


if __name__ == "__main__":
    main()
