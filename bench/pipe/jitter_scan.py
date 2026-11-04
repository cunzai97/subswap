#!/usr/bin/env python3
"""擦除段抖动体检：找出「有字幕 + 背景在动」的帧里，擦除区是不是在闪。

判据（沿用 bench v6 的 d2 思路，但只对有运动的帧报告）：
  - 先用 TELEA 把字幕框内像素补掉，再算 DIS 光流 → 光流只反映背景运动，
    不会被静止的白字边缘带偏；
  - 把上一帧 warp 到当前帧，算掩码内残差 d2_mask、掩码外残差 d2_bg；
  - ratio = d2_mask / d2_bg：接近 1 说明「改过的区域跟周围背景一起自然运动」，
    >>1 就是「周围在动、这块在原地抖」——用户看到的模糊块抖动就是这个；
  - 同时记录运动量 motion（掩码外帧间差），只对 motion ≥ --min-motion 的帧报告。

用法：
  python3 bench/pipe/jitter_scan.py --erased probe/erased_p1.mp4 \
      --state <prepare 缓存 pkl> --out probe/jitter_p1.json [--topn 12]
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent


def log(m):
    print(m, flush=True)


def frames_iter(path, w, h, crop=None):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if crop:
        x, y, cw, ch = crop
        cmd += ["-vf", f"crop={cw}:{ch}:{x}:{y}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 9)
    fw = crop[2] if crop else w
    fh = crop[3] if crop else h
    n = fw * fh * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(fh, fw, 3)
    p.stdout.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--erased", required=True)
    ap.add_argument("--state", required=True, help="prepare 缓存 pkl（含 cues/shots/meta）")
    ap.add_argument("--out", default="")
    ap.add_argument("--topn", type=int, default=12)
    ap.add_argument("--min-motion", type=float, default=1.2)
    ap.add_argument("--half", type=int, default=2, help="光流半分辨率倍数")
    args = ap.parse_args()

    st = pickle.load(open(args.state, "rb"))
    meta = st["meta"]
    cues = st.get("cues") or []
    shots = st["shots"]
    w, h, fps = meta["w"], meta["h"], meta["fps"]

    # 字幕带（整幅宽，够看背景运动）
    ys = [c["box"][1] for c in cues if c.get("box")] + [c["box"][3] for c in cues if c.get("box")]
    if not ys:
        log("没有 cue 框，退出")
        return
    y0 = max(0, int(min(ys)) - 60) & ~1
    y1 = min(h, int(max(ys)) + 60) & ~1
    bw, bh = w, y1 - y0
    log(f"字幕带 {bw}x{bh} @y{y0}")

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    s = 1.0 / max(1, args.half)
    sw, sh = (int(bw * s) // 2 * 2, int(bh * s) // 2 * 2)

    prev = None
    prev_filled = None
    rows = []
    k = 0
    for fr in frames_iter(args.erased, w, h, crop=(0, y0, bw, bh)):
        t = k / fps
        mask = np.zeros((bh, bw), np.uint8)
        for c in cues:
            if c.get("box") and c["start"] - 0.04 <= t <= c["end"] + 0.04:
                bx0, by0, bx1, by1 = c["box"]
                mask[max(0, int(by0) - y0):min(bh, int(by1) - y0),
                     max(0, int(bx0)):min(bw, int(bx1))] = 255
        if prev is not None and mask.any():
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            pgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            # 字幕框内先 TELEA 补齐：让光流只看背景
            m3 = cv2.dilate(mask, np.ones((15, 15), np.uint8))
            gf = cv2.inpaint(gray, m3, 5, cv2.INPAINT_TELEA)
            pgf = cv2.inpaint(pgray, m3, 5, cv2.INPAINT_TELEA)
            a_s = cv2.resize(pgf, (sw, sh), interpolation=cv2.INTER_AREA)
            b_s = cv2.resize(gf, (sw, sh), interpolation=cv2.INTER_AREA)
            flow = dis.calc(b_s, a_s, None)              # 当前 → 上一帧
            flow = cv2.resize(flow, (bw, bh), interpolation=cv2.INTER_LINEAR) / s
            gx, gy = np.meshgrid(np.arange(bw, dtype=np.float32),
                                 np.arange(bh, dtype=np.float32))
            warped = cv2.remap(prev, gx + flow[..., 0], gy + flow[..., 1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            d = np.abs(warped.astype(np.int16) - fr.astype(np.int16)).mean(axis=2)
            inside = mask > 0
            outside = ~inside
            # 掩码外只看「真的在动」的区域，静止背景的 d2 天然接近 0，比值会爆
            mot_map = np.abs(gf.astype(np.int16) - pgf.astype(np.int16))
            moving = (mot_map > 6) & outside
            d2m = float(d[inside].mean()) if inside.any() else 0.0
            d2b = float(d[moving].mean()) if moving.sum() > 40 else 0.0
            motion = float(mot_map[outside].mean()) if outside.any() else 0.0
            if moving.sum() > 40:
                rows.append({"f": k, "t": round(t, 2), "d2_mask": round(d2m, 2),
                             "d2_bg": round(d2b, 2),
                             "ratio": round(d2m / max(d2b, 1e-6), 2),
                             "motion": round(motion, 2)})
            prev_filled = fr.copy()
        prev = fr.copy()
        k += 1
    log(f"扫完 {k} 帧，参与统计 {len(rows)} 帧")

    # 按镜头汇总（ratio 取中位数，motion 取中位数）
    per_shot = []
    for i, sh in enumerate(shots):
        sub = [r for r in rows if sh["start"] <= r["t"] < sh["end"]]
        if len(sub) < 4:
            continue
        sub.sort(key=lambda r: -r["ratio"])
        ratios = sorted(r["ratio"] for r in sub)
        mots = sorted(r["motion"] for r in sub)
        per_shot.append({
            "shot": i, "start": round(sh["start"], 2), "end": round(sh["end"], 2),
            "frames": len(sub),
            "ratio_p50": round(ratios[len(ratios) // 2], 2),
            "ratio_p90": round(ratios[int(0.9 * (len(ratios) - 1))], 2),
            "motion_p50": round(mots[len(mots) // 2], 2),
            "worst": sub[:3],
        })
    per_shot.sort(key=lambda r: -r["ratio_p90"])
    log("\n抖动最重的镜头（ratio_p90 高 = 改过的区域比周围背景抖得多）")
    log(f"{'镜头':>4} {'区间':>15} {'帧':>4} {'motion中位':>9} {'ratio中位':>9} {'ratio_p90':>9}")
    for r in per_shot[:args.topn]:
        log(f"{r['shot']:>4} {r['start']:>6.2f}-{r['end']:<6.2f} {r['frames']:>4} "
            f"{r['motion_p50']:>9.2f} {r['ratio_p50']:>9.2f} {r['ratio_p90']:>9.2f}")
    all_ratio = sorted(r["ratio"] for r in rows)
    log(f"\n全片 ratio 中位 {all_ratio[len(all_ratio)//2]:.2f}，"
        f"p90 {all_ratio[int(0.9*(len(all_ratio)-1))]:.2f}，"
        f"ratio>1.5 的帧 {sum(1 for r in rows if r['ratio'] > 1.5)} / {len(rows)}")
    mot_rows = [r for r in rows if r["motion"] >= args.min_motion]
    mr = sorted(r["ratio"] for r in mot_rows)
    if mr:
        log(f"运动帧（motion≥{args.min_motion}，{len(mot_rows)} 帧）ratio 中位 "
            f"{mr[len(mr)//2]:.2f}，p90 {mr[int(0.9*(len(mr)-1))]:.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "erased": args.erased, "rows": rows, "shots": per_shot,
            "ratio_p50": all_ratio[len(all_ratio) // 2],
        }, indent=1))
        log(f"明细写到 {args.out}")


if __name__ == "__main__":
    main()
