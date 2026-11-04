#!/usr/bin/env python3
"""时域稳定化调参台：只渲染指定镜头区间，跑得快，用来对比抖动指标。

用法：
  python3 bench/pipe/tune_stab.py --state <pkl> --shots 0-8 --out probe/stab_a.mp4
  # 环境变量照样生效：ERASE_TIERS / ERASE_PIPE / VE_STAB_ALPHA 等
然后：
  python3 bench/pipe/jitter_scan.py --erased probe/stab_a.mp4 --state <pkl> \
      --shots 0-8 [--crop-range]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEV = Path(os.environ.get("VE_ROOT") or BASE)
sys.path.insert(0, str(DEV / "webapp"))
sys.path.insert(0, str(DEV / "scripts"))

import yaml  # noqa: E402


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--shots", default="0-8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="stab")
    args = ap.parse_args()

    import convert as C
    import pipeline as P

    st = pickle.load(open(args.state, "rb"))
    cfg = yaml.safe_load((DEV / "config.yaml").read_text(encoding="utf-8"))
    a, _, b = args.shots.partition("-")
    lo, hi = int(a), int(b or a)
    meta = st["meta"]
    out = Path(args.out)
    work = out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    segs = []
    t_all = time.time()
    model = C.get_model(cfg)
    for i in range(lo, min(hi + 1, len(st["shots"]))):
        shot = st["shots"][i]
        tracks = C.filter_tracks_by_layout(st["per_shot"][i], st["layout"], meta)
        seg = work / f"shot_{i:04d}.h264"
        w = P.open_writer(str(seg), meta["w"], meta["h"], meta["fps"], force_x264=True)
        t0 = time.time()
        try:
            r = P.erase_shot(model, st["src"], shot, tracks, meta, w, cfg, "stroke",
                             amp=C._MODEL_AMP, cues=st["cues"])
        finally:
            w.stdin.close()
            w.wait()
        segs.append(seg)
        log(f"镜头 {i} {shot['start']:.2f}-{shot['end']:.2f} {r[0]} 帧 "
            f"{time.time()-t0:.1f}s mode={r[4]}")
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in segs))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", str(out)], check=True)
    # 落一份「只含这段」的 state，jitter_scan 直接用
    t_lo = st["shots"][lo]["start"]
    t_hi = st["shots"][min(hi, len(st["shots"]) - 1)]["end"]
    trim = {
        "meta": meta,
        "src": st["src"],
        "shots": [{"start": s_["start"] - t_lo, "end": s_["end"] - t_lo}
                  for s_ in st["shots"][lo:hi + 1]],
        "cues": [dict(c, start=c["start"] - t_lo, end=c["end"] - t_lo)
                 for c in st.get("cues", [])
                 if c["start"] < t_hi and c["end"] > t_lo],
        "layout": st.get("layout"),
        "per_shot": [[dict(t, start=t["start"] - t_lo, end=t["end"] - t_lo)
                      for t in ps] for ps in st["per_shot"][lo:hi + 1]],
    }
    with open(out.with_suffix(".pkl"), "wb") as f:
        pickle.dump(trim, f)
    log(f"输出 {out} 总耗时 {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
