#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LaMa 字幕带逐帧擦除基准（M0）

用法:
  python bench_lama.py input.mp4 output.mp4 --start 0 --dur 20 \
      --rect 0.06,0.72,0.94,0.90 --batch 4 --fp16

输出:
  - output.mp4   擦除后的样片
  - stdout       fps / 单帧耗时 / 显存峰值 / 日产能估算

说明:
  本脚本用固定矩形模拟"镜头级掩码"，只用于测吞吐和看观感。
  真实流水线的掩码来自 OCR 检测 + 镜头级并集，见 PLAN.md §3.3。
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", path,
    ])
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den)


def iter_frames(path, start, dur, w, h):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start}", "-t", f"{dur}",
           "-i", path, "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    finally:
        p.stdout.close()
        p.wait()


def open_writer(path, w, h, fps):
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-an",
           "-c:v", "libx264", "-crf", "14", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def match_color(src, dst, mask):
    """把 dst 掩码区的均值/方差对齐到 src 的环带，防止亮度跳变"""
    ring = (cv2.dilate(mask, np.ones((11, 11), np.uint8)) > 0) & (mask == 0)
    hole = mask > 0
    if ring.sum() < 50 or hole.sum() == 0:
        return dst
    out = dst.astype(np.float32)
    for c in range(3):
        s = src[..., c].astype(np.float32)
        s_m, s_s = s[ring].mean(), s[ring].std() + 1e-6
        d = out[..., c]
        d_m, d_s = d[hole].mean(), d[hole].std() + 1e-6
        out[..., c][hole] = (d[hole] - d_m) * (s_s / d_s) + s_m
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--model", default=str(Path.home() / "models/lama/big-lama.pt"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=20.0)
    ap.add_argument("--rect", default="0.06,0.72,0.94,0.90",
                    help="字幕矩形，相对坐标 x0,y0,x1,y1")
    ap.add_argument("--pad", type=int, default=8, help="掩码外扩像素")
    ap.add_argument("--max-w", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=False)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--color-match", dest="color_match", action="store_true", default=True)
    ap.add_argument("--no-color-match", dest="color_match", action="store_false")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA/ROCm 不可用")
    dev = "cuda"
    print(f"[env] {torch.__version__} hip={torch.version.hip} "
          f"dev={torch.cuda.get_device_name(0)}")

    model = torch.jit.load(args.model, map_location=dev).eval()
    if args.fp16:
        try:
            model = model.half()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] fp16 失败，退回 fp32: {e}")
            args.fp16 = False

    w, h, fps = probe(args.src)
    rx0, ry0, rx1, ry1 = [float(x) for x in args.rect.split(",")]
    rect = (int(rx0 * w), int(ry0 * h), int(rx1 * w), int(ry1 * h))
    margin = 24
    y0, y1 = max(0, rect[1] - margin), min(h, rect[3] + margin)
    ch, cw = y1 - y0, w

    mask = np.zeros((ch, cw), np.uint8)
    mask[max(0, rect[1] - y0 - args.pad): rect[3] - y0 + args.pad,
         max(0, rect[0] - args.pad): rect[2] + args.pad] = 255
    alpha = cv2.GaussianBlur(mask, (0, 0), 2.5).astype(np.float32) / 255.0
    alpha3 = alpha[..., None]

    scale = min(1.0, args.max_w / cw)
    sw = max(8, int(cw * scale) // 8 * 8)
    sh = max(8, int(ch * scale) // 8 * 8)
    mask_s = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    msk_t = torch.from_numpy((mask_s > 0).astype(np.float32))[None, None].to(dev)
    if args.fp16:
        msk_t = msk_t.half()

    print(f"[input] {w}x{h}@{fps:g}  band={cw}x{ch} -> {sw}x{sh}  "
          f"batch={args.batch} fp16={args.fp16}")

    writer = open_writer(args.dst, w, h, fps)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total = 0

    def flush(batch):
        nonlocal total
        if not batch:
            return
        crops = np.stack([f[y0:y1] for f in batch])
        if (sw, sh) != (cw, ch):
            interp = cv2.INTER_AREA if sw < cw else cv2.INTER_LINEAR
            crops = np.stack([
                cv2.resize(c, (sw, sh), interpolation=interp) for c in crops
            ])
        img = torch.from_numpy(crops[:, :, :, ::-1].copy())
        img = img.permute(0, 3, 1, 2).float().div_(255).to(dev)
        if args.fp16:
            img = img.half()
        with torch.inference_mode():
            out = model(img, msk_t.expand(img.shape[0], -1, -1, -1))
        out = out.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        out = (out * 255).round().astype(np.uint8)
        for i, f in enumerate(batch):
            o = out[i][:, :, ::-1]
            if (sw, sh) != (cw, ch):
                o = cv2.resize(o, (cw, ch), interpolation=cv2.INTER_LINEAR)
            crop = f[y0:y1]
            if args.color_match:
                o = match_color(crop, o, mask)
            f[y0:y1] = (o * alpha3 + crop * (1 - alpha3)).astype(np.uint8)
            writer.stdin.write(f.tobytes())
        total += len(batch)
        if total % 100 < len(batch):
            dt = time.time() - t0
            print(f"  {total} frames  {total / dt:.1f} fps", flush=True)

    batch = []
    for f in iter_frames(args.src, args.start, args.dur, w, h):
        batch.append(f)
        if len(batch) >= args.batch:
            flush(batch)
            batch = []
    flush(batch)

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    writer.stdin.close()
    writer.wait()

    fps_total = total / elapsed if elapsed > 0 else 0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    ep_frames = 75 * 60 * 25
    ep_min = ep_frames / fps_total / 60 if fps_total else float("inf")
    cards = 40 * ep_frames / (fps_total * 86400) if fps_total else float("inf")

    print("\n===== 结果 =====")
    print(f"帧数          : {total}")
    print(f"总耗时        : {elapsed:.1f} s")
    print(f"吞吐(含编解码): {fps_total:.1f} fps   ({1000 / fps_total:.1f} ms/帧)")
    print(f"显存峰值      : {peak:.0f} MiB")
    print(f"单集(75min)   : {ep_min:.1f} 分钟 GPU 时间")
    print(f"40 部/天需要  : {cards:.2f} 张同规格卡")


if __name__ == "__main__":
    main()
