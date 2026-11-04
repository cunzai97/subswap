#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""短剧字幕擦除 + 翻译 + 压制 全链路（v1）

流程：
  拆条 -> 镜头级 OCR 掩码 -> LaMa 逐帧擦除 -> 拼接
       -> 翻译（云端 API 或 mock）-> 重断句 -> ASS -> NVENC 压制 -> 残留质检

用法：
  python pipeline.py assets/sources/namenyuan_1080.mp4 --out outputs/cli/namenyuan

翻译：
  设置 TRANSLATE_BASE_URL / TRANSLATE_API_KEY / TRANSLATE_MODEL 走 OpenAI 兼容接口；
  未设置时用内置 mock（仅用于验证链路，不用于交付）。
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time

PROF = {}


def _prof(name, dt):
    if os.environ.get("ERASE_PROFILE") == "1":
        PROF[name] = PROF.get(name, 0.0) + dt
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

# onnxruntime-gpu 需要找到 torch 自带的 CUDA/cuDNN 动态库
_NVIDIA = Path.home() / ".local/lib/python3.12/site-packages/nvidia"
if _NVIDIA.exists():
    _libs = [str(x) for x in _NVIDIA.glob("*/lib")]
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        _libs + [os.environ.get("LD_LIBRARY_PATH", "")])
sys.path.insert(0, str(ROOT / "scripts"))
from cues import plan_cues, wrap_lines, merge_lines, check_cues  # noqa: E402

OCR = None
OCR_MIN_SCORE = 0.3   # 印地语等非中文行识别置信度低，框仍要用来擦


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


_OCR_OPTS: dict = {}


def set_ocr_opts(**kw):
    """覆盖 det 输入尺寸等参数（convert.extract 从 config.yaml:ocr 传进来）。

    rapidocr 默认 `det_limit_type=min, det_limit_side_len=736`：它会把**短边**放大到
    736——字幕带只有 192px 高时等于放大 3.8 倍再检测，实测 211ms/帧；改成
    `max/960`（长边封顶）后同一帧 28ms，快了 7.5 倍，字高 63px 还剩 56px。
    """
    global OCR
    add = {k: v for k, v in kw.items() if v is not None}
    if add and any(_OCR_OPTS.get(k) != v for k, v in add.items()):
        _OCR_OPTS.update(add)
        OCR = None  # 参数变了必须重建会话，否则老进程里生效不了


def get_ocr():
    global OCR
    if OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        kw = dict(_OCR_OPTS)
        if os.environ.get("OCR_USE_CUDA", "1") == "1":
            try:
                OCR = RapidOCR(det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True,
                               **kw)
            except Exception as e:  # noqa: BLE001
                log(f"OCR GPU 初始化失败，退回 CPU：{e}")
                OCR = RapidOCR(**kw)
        else:
            OCR = RapidOCR(**kw)
    return OCR


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:format=duration",
        "-of", "json", str(path),
    ])
    j = json.loads(out)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "w": int(s["width"]), "h": int(s["height"]),
        "fps": float(num) / float(den),
        "dur": float(j["format"]["duration"]),
    }


def has_audio(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
    ]).decode().strip()
    return bool(out)


def iter_frames(path, w, h, start=0.0, dur=None, fps=None, crop=None):
    """crop=(x,y,cw,ch)：只解码该区域，此时 w/h 为区域宽高"""
    cmd = ["ffmpeg", "-v", "error"]
    # 裸 H.264 分段没有时间轴，ffmpeg -ss 0 会 seek 失败导致 0 帧
    if start > 0.001:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    vf = []
    if crop:
        vf.append(f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}")
    if fps:
        vf.append(f"fps={fps:g}")
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    # 调用方读够帧就停，ffmpeg 的 broken pipe 是预期行为，别刷日志
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         bufsize=10 ** 8)
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


def detect_shots(path, dur, thr=0.16, min_shot=0.5):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-vf", "scale=160:90,fps=10,format=gray", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = 160 * 90
    prev, cuts, i = None, [], 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            f = np.frombuffer(buf, np.uint8).reshape(90, 160).astype(np.int16)
            if prev is not None and np.abs(f - prev).mean() / 255.0 > thr:
                t = i / 10.0
                if not cuts or t - cuts[-1] >= min_shot:
                    cuts.append(t)
            prev = f
            i += 1
    finally:
        p.stdout.close()
        p.wait()
    bounds = [0.0] + [c for c in cuts if c < dur - 0.2] + [dur]
    shots = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < min_shot and shots:
            shots[-1]["end"] = b
        else:
            shots.append({"start": a, "end": b})
    return shots


def ocr_boxes(frame, band, y_off=0, min_w_frac=0.05, max_w=0):
    """只在字幕带里跑 OCR：面积小一半，速度约 1.8 倍
    frame 可以是整帧，也可以是已裁好的字幕带（此时 band=(0,1), y_off=裁切起点）
    min_w_frac=0 时不按宽度过滤，用于复查残留的单字
    max_w>0 时先把字幕带等比缩到该宽度再识别：det 成本随面积走，1080→720 约快
    2.5 倍，字幕字高 60px 缩完还有 40px，远高于识别下限；框按比例还原。
    """
    global OCR
    h, w = frame.shape[:2]
    fb0, fb1 = int(h * band[0]), int(h * band[1])
    yb0, yb1 = fb0 + y_off, fb1 + y_off
    strip = frame[fb0:fb1]
    scale = 1.0
    if max_w and strip.shape[1] > max_w:
        scale = max_w / float(strip.shape[1])
        strip = cv2.resize(strip, (int(round(strip.shape[1] * scale)),
                                   int(round(strip.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
    try:
        res, _ = get_ocr()(strip, use_cls=False)
    except Exception as e:  # noqa: BLE001  cuDNN FE 偶发失败，整进程退回 CPU
        log(f"OCR 推理失败，退回 CPU：{type(e).__name__} {str(e)[:120]}")
        OCR = None
        os.environ["OCR_USE_CUDA"] = "0"
        res, _ = get_ocr()(strip, use_cls=False)
    inv = 1.0 / scale
    out = []
    for box, text, score in (res or []):
        if score < OCR_MIN_SCORE:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, x1 = max(0, int(min(xs) * inv)), min(w, int(max(xs) * inv))
        y0 = max(0, int(min(ys) * inv)) + yb0
        y1 = min(h + y_off, int(max(ys) * inv) + yb0)
        if x1 - x0 < min_w_frac * w:
            continue
        out.append(((x0, y0, x1, y1), text.strip(), float(score)))
    return out


def stroke_mask(frame, box, pad=8, dilate=7):
    """框内做笔画级二值化：背景用中值滤波估计，浅色/深色字都能抓到"""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 31)
    diff = cv2.absdiff(gray, bg)
    th, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = max(float(th), 18.0)
    m = (diff >= th).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if dilate > 1:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m, x0, y0


def norm_text(x):
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", x or "")


CJK_RE = re.compile(r"[\u3400-\u9fff，。！？、；：…—]")


def cjk_text(texts):
    """一轨多帧的识别结果 → 中文字幕（去掉印地语乱码）"""
    from collections import Counter
    cnt = Counter()
    for s in texts:
        c = "".join(CJK_RE.findall(s or ""))
        if c:
            cnt[c] += 1
    return cnt.most_common(1)[0][0] if cnt else ""


def _sim(a, b):
    import difflib
    return difflib.SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


def merge_cues(cues, gap=0.6, sim=0.72):
    """按“时间相接 + 文本相似”合并：OCR 抖动会把一句话拆成好几条，
    繁简混用/错一个字都按相似处理；合并时保留较长的那句。"""
    out = []
    for c in sorted(cues, key=lambda x: (x["start"], -(len(x.get("text") or "")))):
        hit = None
        for m in out:
            if c["start"] <= m["end"] + gap and c["end"] >= m["start"] - gap \
                    and _sim(c["text"], m["text"]) >= sim:
                hit = m
                break
        if hit:
            hit["start"] = min(hit["start"], c["start"])
            hit["end"] = max(hit["end"], c["end"])
            if len(c["text"]) > len(hit["text"]):
                hit["text"] = c["text"]
        else:
            out.append(dict(c))
    return out


def _side_by_side(a, b):
    """两截同一行字幕：横向基本不重叠；同一位置的淡入淡出不算"""
    if not a or not b:
        return False
    ha, hb = a[3] - a[1], b[3] - b[1]
    if abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) > 0.75 * max(ha, hb, 1):
        return False
    wa, wb = a[2] - a[0], b[2] - b[0]
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return inter < 0.35 * max(min(wa, wb), 1)


def drop_junk(cues):
    """同时间段里被更长的字幕盖住的短碎片（多半是印地语误识成汉字）"""
    out = []
    for c in cues:
        dur = c["end"] - c["start"]
        if dur <= 0:
            continue
        covered = sum(max(0.0, min(c["end"], o["end"]) - max(c["start"], o["start"]))
                      for o in cues
                      if o is not c and len(o["text"]) >= len(c["text"]) + 2
                      and not _side_by_side(c.get("box"), o.get("box")))
        if covered > 0.8 * dur:
            continue
        out.append(c)
    return out


def merge_overlaps(cues, tol=0.12):
    """同一时刻并排的两条字幕（多轨/切点）并成一句，翻译才不会分裂成三行"""
    out = []
    for c in sorted(cues, key=lambda x: (x["start"], x["end"])):
        if c["end"] - c["start"] < 0.05:      # 脏数据：负时长/零时长
            continue
        if out and c["start"] < out[-1]["end"] - tol:
            m = out[-1]
            if c["text"] and c["text"] not in m["text"]:
                # OCR 把一句话拆成左右两截时，按 x 位置拼，不能按开始时间
                if (m.get("box") and c.get("box")
                        and _side_by_side(m["box"], c["box"])):
                    parts = ([c["text"], m["text"]] if c["box"][0] < m["box"][0]
                             else [m["text"], c["text"]])
                    m["text"] = " ".join(p for p in parts if p)
                else:
                    m["text"] = (m["text"] + " " + c["text"]).strip()
            m["end"] = max(m["end"], c["end"])
            if m.get("box") and c.get("box"):
                m["box"] = (min(m["box"][0], c["box"][0]), min(m["box"][1], c["box"][1]),
                            max(m["box"][2], c["box"][2]), max(m["box"][3], c["box"][3]))
        else:
            out.append(dict(c))
    return out


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def _box_at(tr, t):
    best = tr["boxes"][0]
    for bt, box in tr["boxes"]:
        if abs(bt - t) < abs(best[0] - t):
            best = (bt, box)
    return best[1]


def analyze_shot(path, shot, meta, band, sample_fps=2.0, mask_mode="stroke",
                 keep_any_text=False, ocr_max_w=0, reuse_static=True,
                 static_thr=2.5):
    """抽样 OCR → 按 IoU 串成文字轨（tracks），取代整镜头并集掩码

    注意：不用 ffmpeg 的 fps 滤镜抽样——它会把首帧推到区间边界（实测偏 0.2~0.5s），
    这里按原始帧率解码、每 step 帧取一帧，时间戳才准。

    ocr_max_w：识别前把字幕带等比缩到的宽度（0=不缩）。det 成本随面积走。
    reuse_static：字幕带和上一帧几乎一样时直接复用上一帧的识别结果——一条字幕
        通常持续 2~4s，4fps 下就是 8~16 次重复识别，实测能省掉一半以上的 OCR。
        判据是灰度平均绝对差 ≤ static_thr，淡入淡出/换行都会超阈值重新识别。
    """
    w, h, fps = meta["w"], meta["h"], meta["fps"]
    fb0 = int(h * band[0]) & ~1
    fb1 = int(h * band[1]) & ~1
    ch = fb1 - fb0
    step = max(1, int(round(fps / sample_fps)))
    dt = step / fps
    tracks = []
    prev_gray = None
    prev_boxes = None
    for k, fr in enumerate(iter_frames(path, w, ch, shot["start"],
                                       shot["end"] - shot["start"], None,
                                       crop=(0, fb0, w, ch))):
        if k % step:
            continue
        t = shot["start"] + k / fps
        boxes = None
        if reuse_static and prev_gray is not None:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if float(cv2.absdiff(g, prev_gray).mean()) <= static_thr:
                boxes = prev_boxes
            prev_gray = g
        if boxes is None:
            boxes = ocr_boxes(fr, (0.0, 1.0), y_off=fb0, max_w=ocr_max_w)
            if reuse_static:
                prev_gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            prev_boxes = boxes
        for box, text, score in boxes:
            best, bi = None, 0.35
            cur_cjk = "".join(CJK_RE.findall(text or ""))
            cur_key = cur_cjk if not keep_any_text else norm_text(text)[:10]
            for tr in tracks:
                if t - tr["end"] > 0.75:
                    continue
                # 同一位置换了台词：另起一轨，别把两条字幕连成一条
                last_key = tr.get("last_cjk") if not keep_any_text else tr.get("last_norm")
                v = _iou(box, tr["last_box"])
                if cur_key and last_key and cur_key != last_key and v > 0.5:
                    continue
                if v > bi:
                    best, bi = tr, v
            if best is not None:
                best["boxes"].append((t, box))
                best["last_box"] = box
                best["end"] = t + dt
                best["texts"].append(text)
                if cur_cjk:
                    best["last_cjk"] = cur_cjk
                if keep_any_text:
                    best["last_norm"] = cur_key
            else:
                tracks.append({"start": t, "end": t + dt,
                               "boxes": [(t, box)], "last_box": box,
                               "texts": [text], "last_cjk": cur_cjk,
                               "last_norm": cur_key if keep_any_text else None})
    for tr in tracks:
        # 中文字幕场景只取汉字（滤掉印地语误识）；外文字幕场景取出现最多的原文
        if keep_any_text:
            from collections import Counter
            cnt = Counter(x.strip() for x in tr["texts"] if (x or "").strip())
            tr["text"] = cnt.most_common(1)[0][0] if cnt else ""
        else:
            tr["text"] = cjk_text(tr["texts"])
        # 抽样 OCR 偶尔漏掉末尾一个字（实测“難道不是嗎”→“難道不是”），
        # 整条轨的位置基本不动，只要并集框没明显变大就用并集，避免漏字残留
        bs = [b for _, b in tr["boxes"]]
        ub = (min(b[0] for b in bs), min(b[1] for b in bs),
              max(b[2] for b in bs), max(b[3] for b in bs))
        mw = float(np.median([b[2] - b[0] for b in bs])) or 1.0
        mh = float(np.median([b[3] - b[1] for b in bs])) or 1.0
        if (ub[2] - ub[0]) <= 1.25 * mw and (ub[3] - ub[1]) <= 1.35 * mh:
            tr["ubox"] = ub

    cues = [{"start": tr["start"], "end": tr["end"], "text": tr["text"]}
            for tr in tracks if tr["text"]]
    merged = drop_junk(merge_cues(cues))
    merged = [c for c in merged if c["end"] - c["start"] >= 0.2]

    # 掩码时间比台词宽一点，但别宽太多：0.8s 会在台词出现前擦到干净画面
    # （实测 60s 处提前整块擦牛仔布）。0.4s 足够覆盖 4fps 抽样的抖动
    for tr in tracks:
        tr["text_start"], tr["text_end"] = tr["start"], tr["end"]
        tr["start"] = max(shot["start"], tr["start"] - 0.4)
        tr["end"] = min(shot["end"], tr["end"] + 0.15)
    return tracks, merged


_NVENC_OK = None


def nvenc_ok():
    """探测本机 NVENC；失败就回 x264，不影响单机无 N 卡环境。"""
    global _NVENC_OK
    if _NVENC_OK is None:
        try:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                 "color=c=black:s=256x256:r=25", "-t", "0.2", "-c:v", "h264_nvenc",
                 "-f", "null", "-"],
                check=True, timeout=20, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            _NVENC_OK = True
        except Exception:  # noqa: BLE001
            _NVENC_OK = False
    return _NVENC_OK


def open_writer(path, w, h, fps, force_x264=False):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-an"]
    # 中间层默认 NVENC：与 x264 crf14 实测 PSNR 50.9dB/SSIM 0.995，快 3.6 倍，
    # 把 CPU 留给 OCR/光流；ERASE_WRITER=x264 可强制回软编。
    if not force_x264 and os.environ.get("ERASE_WRITER", "auto").lower() != "x264" \
            and nvenc_ok():
        cq = os.environ.get("ERASE_NVENC_CQ", "14")
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(cq), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-crf", "14", "-preset", "veryfast"]
        x264_threads = os.environ.get("ERASE_X264_THREADS")
        if x264_threads:
            cmd += ["-threads", str(max(1, int(x264_threads)))]
    if str(path).endswith(".h264"):
        # 逐镜头裸流要按字节拼接；B 帧的显示顺序跨段会让 mux 丢帧（实测 3/375）
        cmd += ["-bf", "0"]
    cmd += ["-pix_fmt", "yuv420p", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def erase_band_rect(tracks, w, h):
    """擦除引擎实际处理的字幕带（原字幕带 + 参考填充要的上下文）。

    erase_ref 和 overlay 写入器必须用同一个矩形，否则叠回去会错位。
    """
    x0, y0, x1, y1 = _band_rect(tracks, w, h)
    cx = max(96, int(round(0.35 * (x1 - x0))))
    cy = max(32, int(round(0.25 * (y1 - y0))))
    return (max(0, x0 - cx) & ~1, max(0, y0 - cy) & ~1,
            min(w, x1 + cx) & ~1, min(h, y1 + cy) & ~1)


def open_overlay_writer(path, src, meta, start, dur, band_rect, force_x264=False):
    """只喂「字幕带」的中间层写入器。

    原路径：Python 每帧搬整帧（1080x1920x3 = 6.2MB）给 ffmpeg 编码。
    这条路：ffmpeg 自己把源片解出来，把我们这条字幕带 patch 叠回去再编码，
    Python 侧每帧只搬 1MB 左右，整帧解码/编码全留在 ffmpeg 里。

    返回的 Popen 上挂了 band_rect 属性，擦除引擎据此只写带内像素。
    """
    bx0, by0, bx1, by1 = band_rect
    bw, bh = bx1 - bx0, by1 - by0
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-ss", f"{start:.6f}", "-t", f"{dur:.6f}", "-i", str(src),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{bw}x{bh}",
           "-r", f"{meta['fps']:g}", "-i", "-", "-an",
           "-filter_complex",
           f"[0:v]crop={bw}:{bh}:{bx0}:{by0}[base];[base][1:v]overlay=0:0:format=yuv420[v]",
           "-map", "[v]"]
    if not force_x264 and os.environ.get("ERASE_WRITER", "auto").lower() != "x264" \
            and nvenc_ok():
        cq = os.environ.get("ERASE_NVENC_CQ", "14")
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(cq), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-crf", "14", "-preset", "veryfast"]
        x264_threads = os.environ.get("ERASE_X264_THREADS")
        if x264_threads:
            cmd += ["-threads", str(max(1, int(x264_threads)))]
    if str(path).endswith(".h264"):
        cmd += ["-bf", "0"]
    cmd += ["-pix_fmt", "yuv420p", str(path)]
    w = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    w.band_rect = (bx0, by0, bx1, by1)
    return w


def match_color(src, dst, mask):
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


def textlike_mask(band, k=31, min_th=30):
    """不依赖 OCR 的“像文字”检测：亮笔画 + 暗描边，用来清洗背景样本"""
    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    op = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    cl = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    hi = cv2.subtract(g, op)
    lo = cv2.subtract(cl, g)
    th1, _ = cv2.threshold(hi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2, _ = cv2.threshold(lo, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (hi >= max(float(th1), min_th)).astype(np.uint8) \
        | (lo >= max(float(th2), min_th)).astype(np.uint8)
    return cv2.dilate(m, np.ones((5, 5), np.uint8))


def _band_rect(tracks, w, h, mx=40, my=28):
    bs = [box for tr in tracks for _, box in tr["boxes"]]
    # ffmpeg 的 crop 对 yuv420 会取偶数尺寸，这里先对齐，否则 rawvideo 读取错位
    x0 = max(0, min(b[0] for b in bs) - mx) & ~1
    y0 = max(0, min(b[1] for b in bs) - my) & ~1
    x1 = min(w, max(b[2] for b in bs) + mx) & ~1
    y1 = min(h, max(b[3] for b in bs) + my) & ~1
    return x0, y0, x1, y1


def track_mask(band, tracks, t, xo, yo, mask_mode="stroke", pad=8, dilate=7,
               fast=False, fallback=0.12, expand=0.0, bright=0.0):
    """当前帧、当前时刻活跃字幕轨的掩码（band 局部坐标）"""
    h, w = band.shape[:2]
    m = np.zeros((h, w), np.uint8)
    tl = None
    gray = None
    for tr in tracks:
        if not (tr["start"] - 0.08 <= t <= tr["end"] + 0.08):
            continue
        bx0, by0, bx1, by1 = tr.get("ubox") or _box_at(tr, t)
        box = (bx0 - xo, by0 - yo, bx1 - xo, by1 - yo)
        # OCR 框常把首尾字裁掉一点；expand>0 时左右各放宽一点
        xpad = int((box[2] - box[0]) * expand) + 10 if expand > 0 else 0
        mb = (max(0, box[0] - xpad), max(0, box[1]),
              min(w, box[2] + xpad), min(h, box[3]))
        if fast or mask_mode == "rect":
            m[max(0, mb[1] - pad):min(h, mb[3] + pad),
              max(0, mb[0] - pad):min(w, mb[2] + pad)] = 255
        else:
            sm, mx, my = stroke_mask(band, box, pad=pad, dilate=dilate)
            if tl is None:
                tl = textlike_mask(band)
            cbx0, cby0 = max(0, int(mb[0]) - pad - dilate), max(0, int(mb[1]) - pad - dilate)
            cbx1, cby1 = min(w, int(mb[2]) + pad + dilate), min(h, int(mb[3]) + pad + dilate)
            if cbx1 <= cbx0 or cby1 <= cby0:
                continue
            if sm.max() == 0:
                # 笔画没抓到：只有框里确实有低对比度文字时才整框用 textlike
                sub = tl[cby0:cby1, cbx0:cbx1]
                if sub.mean() < 0.01:
                    continue
                canvas = np.zeros((h, w), np.uint8)
                canvas[cby0:cby1, cbx0:cbx1] = sub * 255
            else:
                canvas = np.zeros((h, w), np.uint8)
                mh, mw = sm.shape
                canvas[my:my + mh, mx:mx + mw] = sm
                # textlike 只补“笔画附近”的描边/抗锯齿，避免把头发纹理整片擦掉
                near = cv2.dilate(canvas, np.ones((pad + 7, pad + 7), np.uint8))
                canvas[(near > 0) & (tl > 0)] = 255
                if expand > 0:
                    # 放宽后的整行区域里，textlike 的笔画也一起盖（补首尾字）
                    ry0, ry1 = max(0, int(box[1]) - 10), min(h, int(box[3]) + 10)
                    sub = tl[ry0:ry1, mb[0]:mb[2]]
                    region = canvas[ry0:ry1, mb[0]:mb[2]]
                    region[sub > 0] = 255
                    canvas[ry0:ry1, mb[0]:mb[2]] = region

            rx0, ry0 = max(0, int(mb[0]) - pad), max(0, int(mb[1]) - pad)
            rx1, ry1 = min(w, int(mb[2]) + pad), min(h, int(mb[3]) + pad)
            if rx1 <= rx0 or ry1 <= ry0:
                continue
            # 白衣服上的白字：textlike 也漏检，首尾看背景亮度直接整行矩形兜底
            if bright > 0 and expand > 0:
                if gray is None:
                    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
                segs = []
                for k in range(4):
                    a = mb[0] + (mb[2] - mb[0]) * k // 4
                    b = mb[0] + (mb[2] - mb[0]) * (k + 1) // 4
                    vals = gray[mb[1]:mb[3], a:b]
                    vals = vals[tl[mb[1]:mb[3], a:b] == 0]
                    segs.append(float(np.median(vals)) if vals.size else 0.0)
                if min(segs[0], segs[3]) >= bright:
                    m[ry0:ry1, rx0:rx1] = 255
                    continue
            # 白底白字：局部 Otsu 只抓到中间几个字，首尾字会留白。
            # textlike 里仍有大量笔画没被笔画掩码盖住时，整行矩形兜底。
            if fallback > 0 and tl is not None:
                tlb = tl[ry0:ry1, rx0:rx1]
                cb = canvas[ry0:ry1, rx0:rx1]
                if int(tlb.sum()) >= 400:
                    need = tlb & ~(cv2.dilate(
                        cb, np.ones((pad + 7, pad + 7), np.uint8)) > 0)
                    if float(need.sum()) / float(tlb.sum()) > fallback:
                        m[ry0:ry1, rx0:rx1] = 255
                        continue
            m = np.maximum(m, canvas)
    return m


def temporal_median(frames, raws, masks, state, radius=1, thr=18.0,
                    use_flow=True):
    """运动补偿的时域中值：只在掩码内做，压掉逐帧重绘的随机纹理。

    和「混合上一帧结果」的区别：这里每个候选都是**当帧重新生成的**填充，
    中值只挑掉离谱的，不会把上一次模糊过的结果再 warp 一遍——混合法反复
    构图会越抹越糊（实测把红印花抹平成一条亮带），中值不会。

    state 跨批保存上一批的最后一帧，批边界也能取到左邻。
    """
    if not frames or radius <= 0:
        return frames
    dis = state.get("dis")
    if use_flow and dis is None:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        state["dis"] = dis
    h, w = frames[0].shape[:2]

    def gray_flat(band, mask):
        g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        if mask is not None and mask.any():
            g = cv2.inpaint(g, cv2.dilate(mask, np.ones((15, 15), np.uint8)), 5,
                            cv2.INPAINT_TELEA)
        return g

    def flow_map(cur_g, ref_g):
        """返回把 ref 对齐到 cur 的 remap 坐标增量。"""
        s = min(1.0, 1080.0 / max(h, w))
        sw, sh = max(16, int(round(w * s))), max(16, int(round(h * s)))
        a = cv2.resize(ref_g, (sw, sh), interpolation=cv2.INTER_AREA)
        b = cv2.resize(cur_g, (sw, sh), interpolation=cv2.INTER_AREA)
        fl = dis.calc(b, a, None)
        if (sw, sh) != (w, h):
            fl = cv2.resize(fl, (w, h), interpolation=cv2.INTER_LINEAR) / s
        return fl

    items = list(zip(frames, raws, masks))
    prev = state.get("prev_item")
    gcache = {}
    for i, (_, raw, _m) in enumerate(items):
        gcache[i] = gray_flat(raw, None)
    gcache[-1] = state.get("prev_gray")
    out = []
    for i, (band, raw, mk) in enumerate(items):
        if mk is None or not mk.any():
            out.append(band)
            continue
        cands = [band]
        for d in (-radius, radius):
            j = i + d
            nj = items[j][0] if 0 <= j < len(items) else (
                prev[0] if (j < 0 and prev is not None) else None)
            if nj is None:
                continue
            gj = gcache.get(j)
            if gj is None and not (0 <= j < len(items)):
                gj = gcache.get(-1)
            if gj is None:
                continue
            fl = flow_map(gcache[i], gj)
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            wr = cv2.remap(nj, gx + fl[..., 0], gy + fl[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            # 只信和当前帧对得上的候选（差值超阈值说明对齐失败）
            diff = np.abs(wr.astype(np.int16) - band.astype(np.int16)).mean(axis=2)
            wr = np.where((diff < thr)[..., None], wr, band)
            cands.append(wr)
        if len(cands) > 1:
            stack = np.stack(cands)
            med = np.median(stack, axis=0).astype(np.uint8)
            r = band.copy()
            r[mk > 0] = med[mk > 0]
            out.append(r)
        else:
            out.append(band)
    state["prev_item"] = (frames[-1].copy(), raws[-1].copy(), masks[-1])
    state["prev_gray"] = gray_flat(raws[-1], None)
    return out


def temporal_stabilize(frames, masks, state, alpha=0.4, use_flow=True, thr=0.0):
    """运动补偿的时域平滑：只混合掩码内像素，压掉逐帧重绘的纹理跳动。
    state 跨帧保存上一帧结果；看不清/切镜时自动放弃混合（避免拖影）。
    低运动镜头直接混合（光流误差常常比待平滑的抖动还大）"""
    if alpha <= 0 or not frames:
        return frames
    dis = state.get("dis")
    if use_flow and dis is None:
        # DIS 对象构造不便宜，按镜头缓存；原来每批都重建一次
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        state["dis"] = dis
    for fr, mk in zip(frames, masks):
        h, w = fr.shape[:2]
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        prev, pg = state.get("prev"), state.get("gray")
        if prev is None or prev.shape != fr.shape:
            state["prev"], state["gray"] = fr.copy(), gray
            continue
        if use_flow:
            s = min(1.0, 540.0 / max(h, w))
            sw, sh = max(16, int(round(w * s))), max(16, int(round(h * s)))
            a_s = cv2.resize(pg, (sw, sh), interpolation=cv2.INTER_AREA)
            b_s = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
            flow = dis.calc(b_s, a_s, None)          # 当前帧 → 上一帧
            if (sw, sh) != (w, h):
                flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / s
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            warped = cv2.remap(prev, gx + flow[..., 0], gy + flow[..., 1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            warped = prev
        region = cv2.dilate(mk, np.ones((21, 21), np.uint8))
        a0 = alpha
        outside = region == 0
        if outside.any():
            d = np.abs(warped.astype(np.int16) - fr.astype(np.int16)).mean(axis=2)
            ok = float((d[outside] < 28).mean())
            # 整帧一致率只用来挡「切镜/大遮挡」；运动大的镜头整帧一致率天然低
            # （实测运动帧 ok 常掉到 0.5 以下），一旦在这里否决，抖动最大的镜头
            # 恰恰一点平滑都拿不到——所以开了逐像素可信度就只留一道粗闸。
            if thr > 0:
                if ok < 0.3:
                    a0 = 0.0
            elif ok < 0.7:
                a0 = 0.0
            elif ok < 0.9:
                a0 *= 0.5
        if a0 > 0:
            ys, xs = np.where(region > 0)
            if ys.size:
                # 只处理掩码外接框：框外 alpha 恒为 0，原来整带做 float 混合纯属浪费。
                # 模糊留 16px 余量（sigma=4，核半径约 12），结果与整图一致。
                mg = 16
                ry0, ry1 = max(0, int(ys.min()) - mg), min(h, int(ys.max()) + mg + 1)
                rx0, rx1 = max(0, int(xs.min()) - mg), min(w, int(xs.max()) + mg + 1)
                amap = cv2.GaussianBlur(region[ry0:ry1, rx0:rx1], (0, 0), 4.0)
                amap = amap.astype(np.float32) * (a0 / 255.0)
                if thr > 0:
                    # 逐像素可信度看「背景对齐得好不好」，不看掩码内部：
                    # 掩码内部是这一帧新填的内容，和上一帧填的本来就不一样（那就是
                    # 要平滑掉的抖动），拿它自己判可信度等于把自己否掉。
                    dd = np.abs(warped[ry0:ry1, rx0:rx1].astype(np.int16)
                                - fr[ry0:ry1, rx0:rx1].astype(np.int16)).mean(axis=2)
                    reg = region[ry0:ry1, rx0:rx1]
                    good = ((dd < thr) & (reg == 0)).astype(np.float32)
                    # 背景一致率的大范围均值当可信度：掩码内部本来就没有背景可比，
                    # 用周围一圈的平均代替。原来用 TELEA 外推，单帧 162ms，太贵。
                    conf = cv2.blur(good, (81, 81))
                    amap = amap * np.clip(conf * 1.6, 0.0, 1.0)
                a = amap[..., None]
                sub = fr[ry0:ry1, rx0:rx1].astype(np.float32)
                wsub = warped[ry0:ry1, rx0:rx1].astype(np.float32)
                fr[ry0:ry1, rx0:rx1] = (sub * (1.0 - a) + wsub * a).astype(np.uint8)
        state["prev"] = fr.copy()
        state["gray"] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    return frames


def erase_shot(model, path, shot, tracks, meta, writer, cfg, mask_mode="stroke",
               dbg_name="shot", amp=False, cues=None):
    """生产入口：默认参考帧优先（v6），可配置回退到旧的逐帧 LaMa。"""
    engine = str((cfg.get("erase") or {}).get("engine") or "ref").lower()
    if engine == "legacy":
        return erase_shot_legacy(model, path, shot, tracks, meta, writer, cfg,
                                 mask_mode, dbg_name=dbg_name, amp=amp)
    from erase_ref import erase_shot_ref
    return erase_shot_ref(sys.modules[__name__], model, path, shot, tracks, meta,
                          writer, cfg, mask_mode, dbg_name=dbg_name, amp=amp,
                          cues=cues)


def erase_shot_legacy(model, path, shot, tracks, meta, writer, cfg, mask_mode="stroke",
                      dbg_name="shot", amp=False):
    """两种模式：
    temporal —— 机位基本不动：用同镜头里“字幕不在该像素上”的帧估计清空背景，
                逐帧把字幕像素换成真实背景；无清空样本的残留用 Telea 补
    lama     —— 镜头在动：逐帧笔画掩码 + LaMa
    """
    w, h, fps = meta["w"], meta["h"], meta["fps"]
    dur = shot["end"] - shot["start"]
    # 每个镜头必须写“取整到同一帧网格”的帧数，否则逐镜头多一两帧，
    # 几分钟后声画就会整体偏掉（实测 2 分钟漂 0.7s）
    n_exp = int(round(shot["end"] * fps)) - int(round(shot["start"] * fps))
    x0, y0, x1, y1 = _band_rect(tracks, w, h)
    bw, bh = x1 - x0, y1 - y0
    batch_size = int(cfg["inpaint"]["batch"])
    cast_half = bool(cfg["inpaint"]["fp16"]) and not amp
    sw = int(cfg["inpaint"]["fixed_w"])
    sh = int(cfg["inpaint"]["fixed_h"])
    fit_canvas = bool(cfg["inpaint"].get("fit_canvas", True))
    fit = min(sw / bw, sh / bh, 1.0)
    fit_w = max(2, int(round(bw * fit)))
    fit_h = max(2, int(round(bh * fit)))
    cw = min(sw, (fit_w + 7) // 8 * 8) if fit_canvas else sw
    ch = min(sh, (fit_h + 7) // 8 * 8) if fit_canvas else sh
    cw, ch = max(8, cw), max(8, ch)
    cpx, cpy = (cw - fit_w) // 2, (ch - fit_h) // 2
    feather = float(cfg["inpaint"]["feather_sigma"])
    pad = int(cfg["mask"].get("mask_pad", 8))
    dilate = int(cfg["mask"].get("dilate", 7))
    fallback = float(cfg["mask"].get("stroke_fallback", 0.12))
    expand = float(cfg["mask"].get("expand_ratio", 0.0))
    bright = float(cfg["mask"].get("bright_fallback", 0.0))
    bh_list = [b[3] - b[1] for tr in tracks for _, b in tr["boxes"]]
    shift = max(16, min(int(np.median(bh_list)) + 24 if bh_list else 90, bh // 2))

    # ---- 第一遍：只解码字幕带，统计每像素“无字幕帧”的下界颜色 + 机位运动量 ----
    # 字幕是亮字：取干净样本的最小值即可，漏检的白字也会被自动剔掉（比中位数还稳）
    t0 = time.time()
    M = np.full((bh, bw, 3), 255, np.uint8)
    S = np.zeros((bh, bw, 3), np.float32)
    S2 = np.zeros((bh, bw, 3), np.float32)
    C = np.zeros((bh, bw), np.uint16)
    union = np.zeros((bh, bw), np.uint8)
    diffs = []
    prev_g = prev_clean = None
    n = 0
    for k, fr in enumerate(iter_frames(path, bw, bh, shot["start"], dur, None,
                                       crop=(x0, y0, bw, bh))):
        t = shot["start"] + k / fps
        # 背景采样一律用矩形：把字的描边/光晕整个剔掉，笔画掩码会漏掉抗锯齿
        m = track_mask(fr, tracks, t, x0, y0, "rect", pad=pad, dilate=dilate)
        # OCR 漏检的字幕也要从背景样本里剔掉，否则背景会被白字污染
        clean = (m == 0) & (textlike_mask(fr) == 0)
        cm = clean[..., None]
        np.minimum(M, fr, out=M, where=cm)
        f = fr.astype(np.float32)
        S += f * cm
        S2 += (f * f) * cm
        C += clean
        np.maximum(union, m, out=union)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if prev_g is not None:
            sel = clean & prev_clean
            if sel.any():
                gd = np.abs(g[sel].astype(np.int16) - prev_g[sel].astype(np.int16))
                diffs.append(float(gd.mean()))
        prev_g, prev_clean = g, clean
        n += 1

    motion = float(np.median(diffs)) if diffs else 99.0
    Cn = np.maximum(C, 1)[..., None]
    mean = S / Cn
    var = np.maximum(S2 / Cn - mean * mean, 0).mean(axis=2)
    std = np.sqrt(var)
    use = union > 0
    # 背景板用多帧均值，不能用逐像素最小值：min 会系统性偏暗，
    # 在牛仔布/衣服纹理上会拼出黑块（实测 60s 处整块发黑）
    bg = np.clip(mean, 0, 255)
    ok = (C >= 3) & (std < 8.0)
    # 背景要和周围一致：样本太少/被别的姿态污染时按局部中值剔掉
    bg_u8 = np.clip(bg, 0, 255).astype(np.uint8)
    bgm = cv2.medianBlur(bg_u8, 15)
    dev = np.abs(bg_u8.astype(np.int16) - bgm.astype(np.int16)).max(axis=2)
    ok &= dev < 24
    frac = float((ok & use).sum()) / max(int(use.sum()), 1)

    # 只有真正锁死的机位才用背景板：稍有漂移时背景板和实拍帧对不上，
    # 字幕位置会出现方块接缝（实测 60s 处整块发黑/拼接）
    static = n >= 8 and dur >= 3.0 and motion <= 0.1 and frac >= 0.6
    if os.environ.get("ERASE_FORCE_LAMA"):
        static = False
    mode = "temporal" if static else "lama"
    log(f"    带 {bw}x{bh} 运动 {motion:.2f} 可用清空 {frac*100:.0f}% → {mode}")
    # 逐帧 LaMa 的纹理每帧都会变，落在静止背景上像“呼吸”；只在掩码内做时域平滑
    stab = float(cfg["inpaint"].get("temporal_ema", 0.0) or 0.0)
    if static:
        stab = 0.0
    stab_state = {"prev": None, "gray": None}
    if os.environ.get("ERASE_DEBUG"):
        dbg = ROOT / "outputs" / "debug" / dbg_name
        dbg.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg) + "_bg.png", np.clip(bg, 0, 255).astype(np.uint8))
        cv2.imwrite(str(dbg) + "_ok.png", ok.astype(np.uint8) * 255)

    total, masked, t_w, n_res = 0, 0, time.time(), 0
    last_bytes = None

    def flush(frames, times):
        nonlocal total, masked, n_res, last_bytes
        n_real = len(frames)
        real_bands = [f[y0:y1, x0:x1].copy() for f in frames]
        # 最终掩码用笔画级：整行矩形会让 LaMa 在牛仔布/衣服上补出黑块；
        # 笔画掩码 + 足够膨胀能把白字和黑描边一起盖住（dilate=12）
        mk_mode = "rect" if mask_mode == "rect" else "stroke"
        t_p = time.perf_counter()
        mk = [track_mask(b, tracks, t, x0, y0, mk_mode, pad=pad, dilate=dilate,
                         fallback=fallback, expand=expand, bright=bright)
              for b, t in zip(real_bands, times)]
        _prof("mask", time.perf_counter() - t_p)
        masked += int(sum(int((m > 0).sum()) for m in mk))

        if static:
            # 有“同一像素在别的帧没有字幕”的样本：直接换成真实背景
            for b, m in zip(real_bands, mk):
                fill = (m > 0) & ok
                b[fill] = np.clip(bg[fill], 0, 255).round().astype(np.uint8)
            need = []
            # 位移借纹理只在残留很小的时候用；整行常驻字幕这种大块残留
            # 上下平移会把背景涂成一条横带，直接交给 LaMa
            small = max(400, int(0.02 * bw * bh))
            for b, m in zip(real_bands, mk):
                r = (m > 0) & ~ok
                if 0 < int(r.sum()) < small:
                    cand = np.zeros_like(b)
                    valid = np.zeros((bh, bw), bool)
                    for dy in (shift, -shift):
                        ys = slice(max(0, dy), bh + min(0, dy))
                        yd = slice(max(0, -dy), bh + min(0, -dy))
                        use = r[yd] & (m[ys] == 0) & ~valid[yd]
                        cand[yd][use] = b[ys][use]
                        valid[yd] |= use
                    got = r & valid
                    b[got] = cand[got]
                    if got.any():
                        # 位移拷贝的接缝柔一下，否则手臂/衣服上会出现方块边界
                        soft = cv2.GaussianBlur(b, (0, 0), 2.0)
                        gd = cv2.dilate(got.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                        b[gd] = soft[gd]
                    r = r & ~valid
                need.append(r)
            # 残留交给 LaMa 时只盖真正缺样本的笔画，别扩成整行：
            # 扩成矩形后模型几乎没有上下文，实测会补出黑块/假字
            inp = [cv2.dilate(r.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                   for r in need]
            for b, r in zip(real_bands, need):
                if 0 < int(r.sum()) < 40:
                    b[:] = cv2.inpaint(b, r.astype(np.uint8) * 255, 3,
                                       cv2.INPAINT_TELEA)
        else:
            need = [m > 0 for m in mk]
            inp = need
        n_res += int(sum(int(r.sum()) for r in need))

        todo = [i for i, r in enumerate(need) if int(r.sum()) >= 40]
        if todo:
            t_p = time.perf_counter()
            bands = real_bands + [real_bands[-1]] * (batch_size - n_real)
            msks = inp + [np.zeros_like(inp[-1])] * (batch_size - n_real)
            # 每镜头按实际字幕带裁画布：固定 1024x256 会给窄带补大量空边，
            # 对模型结果没有贡献却要付全部算力。等比缩放+补边，避免拉伸。
            canv, mcanv, metas = [], [], []
            interp = cv2.INTER_AREA if fit < 1.0 else cv2.INTER_LINEAR
            for b, r in zip(bands, msks):
                im = cv2.resize(b, (fit_w, fit_h), interpolation=interp)
                mm = cv2.resize((r * 255).astype(np.uint8), (fit_w, fit_h),
                                interpolation=cv2.INTER_NEAREST)
                canv.append(cv2.copyMakeBorder(im, cpy, ch - fit_h - cpy,
                                               cpx, cw - fit_w - cpx,
                                               cv2.BORDER_REPLICATE))
                mcanv.append(cv2.copyMakeBorder(mm, cpy, ch - fit_h - cpy,
                                                cpx, cw - fit_w - cpx,
                                                cv2.BORDER_CONSTANT, value=0))
                metas.append((cpx, cpy, fit_w, fit_h, bw, bh))
            img = torch.from_numpy(np.stack(canv)[:, :, :, ::-1].copy()).permute(0, 3, 1, 2)
            img = img.float().div_(255).cuda()
            if cast_half:
                img = img.half()
            msk_t = torch.from_numpy((np.stack(mcanv) > 0).astype(np.float32)).cuda()[:, None]
            if cast_half:
                msk_t = msk_t.half()
            _prof("prep", time.perf_counter() - t_p)
            t_p = time.perf_counter()
            with torch.inference_mode():
                if amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        out = model(img, msk_t)
                else:
                    out = model(img, msk_t)
            torch.cuda.synchronize()
            _prof("model", time.perf_counter() - t_p)
            # fp16/autocast 下 FFC 会吐 NaN；整批回退 fp32 重算，坏像素不填会留原字幕
            t_p = time.perf_counter()
            out = out.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            finite = np.isfinite(out).all(axis=3)
            if not bool(finite.all()) and (cast_half or amp):
                P.log("    fp16 输出含 NaN，整批回退 fp32")
                with torch.inference_mode():
                    out = model(img.float(), msk_t.float())
                out = out.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                finite = np.isfinite(out).all(axis=3)
            out = (np.nan_to_num(out, nan=0.0) * 255).round().clip(0, 255).astype(np.uint8)
            for i in todo:
                r = need[i]
                ys2, xs2 = np.where(r)
                bx0, bx1 = max(0, int(xs2.min()) - 10), min(bw, int(xs2.max()) + 11)
                by0, by1 = max(0, int(ys2.min()) - 10), min(bh, int(ys2.max()) + 11)
                a_mask = (r * 255).astype(np.uint8)
                alpha = cv2.GaussianBlur(a_mask, (0, 0), feather)
                px, py, nw, nh, ww, hh = metas[i]
                o = cv2.resize(out[i][py:py + nh, px:px + nw][..., ::-1], (ww, hh),
                               interpolation=cv2.INTER_LINEAR)
                v = cv2.resize(finite[i][py:py + nh, px:px + nw].astype(np.uint8) * 255,
                               (ww, hh), interpolation=cv2.INTER_NEAREST) > 127
                b = real_bands[i]
                sub_o = o[by0:by1, bx0:bx1]
                sub_c = b[by0:by1, bx0:bx1]
                if cfg["inpaint"]["color_match"]:
                    sub_o = match_color(sub_c, sub_o, a_mask[by0:by1, bx0:bx1])
                a = alpha[by0:by1, bx0:bx1].astype(np.float32) / 255.0
                a *= v[by0:by1, bx0:bx1]
                a = a[..., None]
                b[by0:by1, bx0:bx1] = (sub_o * a + sub_c * (1 - a)).astype(np.uint8)
            _prof("post", time.perf_counter() - t_p)

        t_s = time.perf_counter()
        if stab > 0:
            temporal_stabilize(real_bands, mk, stab_state, stab,
                               use_flow=motion > 1.2)
        _prof("stab", time.perf_counter() - t_s)

        for i in range(n_real):
            frames[i][y0:y1, x0:x1] = real_bands[i]
            if total >= n_exp:
                continue
            last_bytes = frames[i].tobytes()
            writer.stdin.write(last_bytes)
            total += 1

    buf, ts = [], []
    t_first = time.time()
    for k, fr in enumerate(iter_frames(path, w, h, shot["start"], dur, None)):
        if total >= n_exp:
            break
        buf.append(fr)
        ts.append(shot["start"] + k / fps)
        if len(buf) >= batch_size:
            t_f = time.perf_counter()
            flush(buf, ts)
            _prof("flush", time.perf_counter() - t_f)
            buf, ts = [], []
    if buf:
        t_f = time.perf_counter()
        flush(buf, ts)
        _prof("flush", time.perf_counter() - t_f)
    while total < n_exp and last_bytes is not None:
        writer.stdin.write(last_bytes)
        total += 1

    dt = time.time() - t0
    log(f"    第二遍 {time.time()-t_w:.1f}s 共 {dt:.1f}s 残留 {n_res/max(masked,1)*100:.0f}%")
    if os.environ.get("ERASE_PROFILE") == "1":
        _prof("firstpass", t_first - t0)
        _prof("io", (time.time() - t_w) - PROF.get("flush", 0.0))
        log("    PROF " + " ".join(f"{k}={v:.2f}s" for k, v in PROF.items()))
        PROF.clear()
    coverage = masked / max(total * bh * bw, 1)
    return total, dt, (total / dt if dt else 0), coverage, mode


# ---------------- 翻译 ----------------

MOCK = {
    "三年了，你终于回来了": "Three years. You finally came back.",
    "这句话我在心里练了一千遍": "I rehearsed this line a thousand times.",
    "可你连看都不看我一眼": "But you won't even look at me.",
    "我从来没想过要离开你": "I never once thought of leaving you.",
    "那这三年，你去了哪里": "Then where have you been for three years?",
    "我去做了一个了断": "I went to settle it. Once and for all.",
    "从今天起，没有人能再伤害你": "From today, no one will ever hurt you again.",
    "包括我自己": "Including me.",
    "别走。": "Don't go.",
}


def _clean(s):
    return "".join(ch for ch in s if ch not in "，。！？、,.!? ")


def mock_translate(cues):
    out = []
    for c in cues:
        key = _clean(c["text"])
        hit = next((v for k, v in MOCK.items() if _clean(k) == key), None)
        out.append(hit or f"[EN] {c['text']}")
    return out


def api_translate(cues, base, key, model):
    prompt = (
        "把下面 JSON 里的中文台词译成地道的美式英语口语。"
        "只输出 JSON 数组 [{\"id\":1,\"en\":\"...\"}]，id 一一对应。"
        "要求：简短口语，不解释，不用中文标点。\n"
        + json.dumps([{"id": i, "zh": c["text"]} for i, c in enumerate(cues)],
                     ensure_ascii=False)
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    content = data["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        content = content.split("\n", 1)[1] if "\n" in content else content
    arr = json.loads(content)
    by_id = {int(x["id"]): x["en"] for x in arr}
    return [by_id.get(i, c["text"]) for i, c in enumerate(cues)]


TRANS_SYS = (
    "You are a professional subtitle translator for vertical short dramas. "
    "Translate each numbered Traditional Chinese line into natural, spoken American English.\n"
    "Rules:\n"
    "- Translate every line; keep the meaning and the emotional tone. Do not explain, merge, or skip.\n"
    "- Keep each line short: at most 48 characters total, so it fits two lines of 24 chars.\n"
    "- Keep character names in pinyin; use natural English forms of address.\n"
    "- OCR may have typos or missing characters; infer from context.\n"
    "- Output ONLY a JSON array like [{\"id\":0,\"en\":\"...\"}] with one entry per input id."
)


TRANS_SYS_ZH = (
    "You are a professional subtitle translator. Translate each numbered English line "
    "into natural, spoken Simplified Chinese.\n"
    "Rules:\n"
    "- Translate every line; keep the meaning and the emotional tone. Do not explain, merge, or skip.\n"
    "- Keep each line short: at most 26 characters, so it fits one or two subtitle lines.\n"
    "- Keep character names natural.\n"
    "- OCR may have typos or missing characters; infer from context.\n"
    "- Output ONLY a JSON array like [{\"id\":0,\"en\":\"...\"}] with one entry per input id."
)


# 目标语言 -> 提示词里的语言名；key 用 ISO 短码
PLAIN_PROMPT = {
    "en": "将以下文本翻译成自然、口语化的英语，保留人物称呼和语气，只输出译文：\n\n{text}",
    "zh": "将以下文本翻译成自然、口语化的简体中文，保留人物称呼和语气，只输出译文：\n\n{text}",
    "ja": "将以下文本翻译成自然、口语化的日语，保留人物称呼和语气，"
          "同一部剧的人称和语气保持一致。每条译文尽量控制在 24 个字符以内，"
          "能一行显示就一行。只输出译文：\n\n{text}",
}

LANG_NAME = {
    "en": "natural, spoken American English",
    "zh": "natural, spoken Simplified Chinese",
    "ja": "natural, spoken Japanese",
    "ko": "natural, spoken Korean",
    "es": "natural, spoken Latin American Spanish",
    "pt": "natural, spoken Brazilian Portuguese",
    "ar": "natural, spoken Modern Standard Arabic",
    "ru": "natural, spoken Russian",
    "th": "natural, spoken Thai",
    "vi": "natural, spoken Vietnamese",
    "id": "natural, spoken Indonesian",
    "fr": "natural, spoken French",
    "de": "natural, spoken German",
}

TRANS_SYS_GENERIC = (
    "You are a professional subtitle translator for short dramas. "
    "Translate each numbered line into {lang}.\n"
    "Rules:\n"
    "- Translate every line; keep the meaning and the emotional tone. Do not explain, merge, or skip.\n"
    "- Keep each line short: at most {n} characters, so it fits one or two subtitle lines.\n"
    "- Keep character names natural; use natural forms of address.\n"
    "- OCR may have typos or missing characters; infer from context.\n"
    "- Output ONLY a JSON array like [{{\"id\":0,\"en\":\"...\"}}] with one entry per input id."
)


def _sys_prompt(tcfg):
    tgt = str(tcfg.get("target") or "en").lower()
    if tgt.startswith("zh") or tgt.startswith("cn"):
        return TRANS_SYS_ZH
    if tgt.startswith("en"):
        return TRANS_SYS
    return TRANS_SYS_GENERIC.format(lang=LANG_NAME.get(tgt, tgt),
                                    n=int(tcfg.get("max_chars", 26)))


def _extract_json_array(s):
    """容忍 ```json 围栏 / 前后废话，取第一个能解析的 JSON 数组"""
    dec = json.JSONDecoder()
    i = s.find("[")
    while i != -1:
        try:
            arr, _ = dec.raw_decode(s[i:])
            if isinstance(arr, list):
                return arr
        except Exception:  # noqa: BLE001
            pass
        i = s.find("[", i + 1)
    return None


def load_translate_key(cfg):
    t = cfg.get("translate") or {}
    key = os.environ.get("TRANSLATE_API_KEY") or t.get("api_key")
    if key:
        return key
    f = t.get("api_key_file")
    if f:
        try:
            d = json.loads(Path(os.path.expanduser(f)).read_text(encoding="utf-8"))
            name = t.get("api_key_name")
            for e in d.get("api_keys") or []:
                if e.get("enabled") and (not name or e.get("name") == name):
                    return str(e.get("key") or "")
            if d.get("api_key"):
                return str(d["api_key"])
        except Exception as e:  # noqa: BLE001
            log(f"读取翻译 Key 失败：{e}")
    return None


_REQ_LOCK = threading.Lock()
_LAST_REQ = [0.0]


def _chat_once(base, key, model, messages, timeout, temperature, effort, max_tokens,
               min_gap=0.0):
    if min_gap > 0:
        with _REQ_LOCK:
            wait = _LAST_REQ[0] + min_gap - time.time()
            if wait > 0:
                time.sleep(wait)
            _LAST_REQ[0] = time.time()
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens}
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


def _effort_for(model, tcfg):
    if any(k in model.lower() for k in ("qwen", "deepseek", "kimi")):
        return tcfg.get("reasoning_effort") or ""
    return ""


def _translate_batch(cues, ids, models, base, key, tcfg):
    payload = [{"id": i, "zh": cues[i]["text"]} for i in ids]
    msgs = [{"role": "system", "content": _sys_prompt(tcfg)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    content = ""
    attempts = int(tcfg.get("retries", 2)) + 1
    for attempt in range(attempts):
        model = models[attempt % len(models)]
        try:
            content = _chat_once(
                base, key, model, msgs,
                float(tcfg.get("timeout", 150)),
                float(tcfg.get("temperature", 0.2)),
                _effort_for(model, tcfg),
                int(tcfg.get("max_tokens", 4096)),
                float(tcfg.get("min_gap", 0)))
            arr = _extract_json_array(content)
            if arr:
                out = {}
                for x in arr:
                    try:
                        en = str(x.get("en") or "").replace("\n", " ").strip()
                        if en:
                            out[int(x["id"])] = en
                    except Exception:  # noqa: BLE001
                        continue
                if out:
                    return out
        except urllib.error.HTTPError as e:
            # 429/502 是上游排队或限流：换模型 + 退避，别把批次打穿
            wait = float(tcfg.get("rate_sleep", 6)) * (attempt + 1) \
                if e.code in (429, 502, 503) else 0.6 * (attempt + 1)
            content = f"<error: HTTP {e.code}>"
            log(f"  批 {ids[0]}..{ids[-1]} 用 {model} 失败 HTTP {e.code}，换模型")
            time.sleep(wait + random.uniform(0, 1.5))
            continue
        except Exception as e:  # noqa: BLE001
            content = f"<error: {e}>"
        time.sleep(0.6 * (attempt + 1))
    log(f"翻译批失败 ids={ids[0]}..{ids[-1]}：{str(content)[:120]}")
    return {}


def _translate_one(text, base, key, model, tcfg):
    """整句直译：给翻译专用模型（Hy-MT2）用，比 JSON 批量更稳"""
    tpl = tcfg.get("plain_prompt") or (
        "将以下文本翻译成英语，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}")
    msgs = [{"role": "user", "content": tpl.replace("{text}", text)}]
    attempts = int(tcfg.get("retries", 2)) + 1
    for attempt in range(attempts):
        try:
            content = _chat_once(
                base, key, model, msgs,
                float(tcfg.get("timeout", 150)),
                float(tcfg.get("temperature", 0.2)),
                "",
                int(tcfg.get("max_tokens", 512)),
                float(tcfg.get("min_gap", 0)))
            out = content.strip().strip("`").strip().strip('"').strip()
            out = (out.replace("’", "'").replace("‘", "'")
                      .replace("“", '"').replace("”", '"'))
            if out:
                return out
        except urllib.error.HTTPError as e:
            time.sleep(float(tcfg.get("rate_sleep", 3)) * (attempt + 1)
                       + random.uniform(0, 0.8))
            continue
        except Exception:  # noqa: BLE001
            time.sleep(0.3 * (attempt + 1))
    return ""


KANA_RE = re.compile(r"[\u3040-\u30ff]")


def _translation_ok(text, target):
    """译文语言粗检：只用来发现“原文没翻/翻了别的语言”，不评判质量"""
    v = (text or "").strip()
    if not v:
        return False
    tgt = (target or "en").lower()
    if tgt.startswith("zh") or tgt.startswith("cn"):
        return bool(CJK_RE.search(v))
    if tgt.startswith("ja"):
        return bool(KANA_RE.search(v))
    if tgt.startswith("ko"):
        return bool(re.search(r"[\uac00-\ud7af]", v))
    cjk = len(CJK_RE.findall(v))
    return cjk <= max(2, int(0.2 * len(v)))


def llm_translate(cues, cfg, base, key, model):
    tcfg = dict(cfg.get("translate") or {})
    tgt = str(tcfg.get("target") or "en").lower()
    if (tcfg.get("style") or "json") == "plain" and tgt not in ("en",):
        key = "ja" if tgt.startswith("ja") else (
            "zh" if tgt.startswith(("zh", "cn")) else tgt)
        tcfg["plain_prompt"] = PLAIN_PROMPT.get(key) or (
            "Translate the following subtitle into {lang}. Keep names and tone. "
            "Output only the translation:\n\n{{text}}"
        ).format(lang=LANG_NAME.get(tgt, tgt))
    bs = max(1, int(tcfg.get("batch", 20)))
    conc = max(1, int(tcfg.get("concurrency", 1)))
    models = [m for m in (tcfg.get("models") or [model]) if m] or [model]
    # 同一句台词在短剧里会重复出现，去重后再译，省请求次数也省额度
    texts = [c["text"] for c in cues]
    uniq = list(dict.fromkeys(texts))
    idx = {t: i for i, t in enumerate(uniq)}
    ucues = [{"text": t} for t in uniq]

    if (tcfg.get("style") or "json") == "plain":
        log(f"翻译：{len(cues)} 条（去重 {len(uniq)} 条）逐句直译 → {tgt}，"
            f"并发 {conc}，model={model}")
        t0 = time.time()
        results = {}
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = {ex.submit(_translate_one, t, base, key, model, tcfg): i
                    for i, t in enumerate(uniq)}
            done = 0
            for fu in as_completed(futs):
                i = futs[fu]
                v = fu.result()
                if v:
                    results[i] = v
                done += 1
                if done % 25 == 0 or done == len(uniq):
                    log(f"  翻译进度 {done}/{len(uniq)} 条")
        en, bad = [], 0
        for t in texts:
            v = results.get(idx[t]) or ""
            if not _translation_ok(v, tgt):
                bad += 1
                v = v or "[?]"
            en.append(v)
        log(f"翻译完成 {len(texts)-bad}/{len(texts)} 条，{time.time()-t0:.1f}s")
        return en

    batches = [list(range(i, min(i + bs, len(ucues))))
               for i in range(0, len(ucues), bs)]
    log(f"翻译：{len(cues)} 条（去重 {len(uniq)} 条）/ {len(batches)} 批 × {bs}，"
        f"并发 {conc}，模型轮换 {models}")
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {}
        for bi, ids in enumerate(batches):
            order = models[bi % len(models):] + models[:bi % len(models)]
            futs[ex.submit(_translate_batch, ucues, ids, order, base, key, tcfg)] = ids
        done = 0
        for fu in as_completed(futs):
            results.update(fu.result())
            done += 1
            log(f"  翻译进度 {done}/{len(batches)} 批，{len(results)}/{len(uniq)} 条")
    missing = [i for i in range(len(uniq)) if i not in results]
    if missing:
        log(f"翻译缺 {len(missing)} 条，逐条重试")
        for i in missing:
            results.update(_translate_batch(ucues, [i], models, base, key, tcfg))
    en, bad = [], 0
    for t in texts:
        v = results.get(idx[t]) or ""
        if not _translation_ok(v, tgt):
            bad += 1
            v = v or "[?]"
        en.append(v)
    log(f"翻译完成 {len(texts)-bad}/{len(texts)} 条，{time.time()-t0:.1f}s")
    return en


def translate(cues, cfg):
    t = cfg.get("translate") or {}
    tgt = str(t.get("target") or "en").lower()
    base = os.environ.get("TRANSLATE_BASE_URL") or t.get("base_url")
    key = load_translate_key(cfg)
    model = os.environ.get("TRANSLATE_MODEL") or t.get("model") or ""
    enabled = bool(t.get("enable", True))
    if enabled and base and key:
        log(f"翻译：API {base} model={model} → {tgt}")
        try:
            return llm_translate(cues, cfg, base, key, model)
        except Exception as e:  # noqa: BLE001
            log(f"翻译失败：{e}")
            if not tgt.startswith("en"):
                raise RuntimeError(f"翻译失败，拒绝用英语 mock 冒充 {tgt}") from e
    else:
        if not tgt.startswith("en"):
            raise RuntimeError(f"未配置翻译后端，无法生成 {tgt} 字幕")
        log("翻译：未配置 API，使用 mock（仅验证链路）")
    return mock_translate(cues)


# ---------------- 输出 ----------------

ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: EN,{font},{size},&H00FFFFFF,&H000000FF,&H99000000,&H99000000,0,0,0,0,100,100,0,0,1,3,1,2,60,60,{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ts(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def write_ass(cues, path, meta, max_chars, max_lines, font_size=None,
              margin_v=None, font=None):
    """font_size/margin_v 不给时按画面尺寸推算（竖屏 1080 的默认口径）；
    给定时用于对齐原字幕的实际字号和位置"""
    size = int(font_size) if font_size else int(meta["w"] * 0.048)
    mv = int(margin_v) if margin_v is not None else int(meta["h"] * 0.11)
    head = ASS_HEAD.format(w=meta["w"], h=meta["h"],
                           font=font or "Noto Sans CJK SC",
                           size=size, mv=mv)
    lines = [head]
    for c in cues:
        text = "\\N".join(wrap_lines(c["en"], max_chars)[:max_lines])
        lines.append(f"Dialogue: 0,{ts(c['start'])},{ts(c['end'])},EN,,0,0,0,,{text}\n")
    Path(path).write_text("".join(lines), encoding="utf-8")


def qa_residual(path, meta, rect, band):
    w, h = meta["w"], meta["h"]
    hit, total = 0, 0
    for k, fr in enumerate(iter_frames(path, w, h, 0, None, fps=2)):
        total += 1
        boxes = ocr_boxes(fr, band)
        if any(b[0][0] < rect[2] and b[0][2] > rect[0] and
               b[0][1] < rect[3] and b[0][3] > rect[1] for b in boxes):
            hit += 1
    return hit, total


def speech_segments(path, workdir):
    """用 ffmpeg silencedetect 取语音段（零依赖；Silero 可后续替换）"""
    cmd = ["ffmpeg", "-v", "info", "-i", str(path),
           "-af", "silencedetect=noise=-32dB:d=0.25", "-f", "null", "-"]
    err = subprocess.run(cmd, capture_output=True).stderr.decode("utf-8", "ignore")
    sil, cur = [], None
    for line in err.splitlines():
        if "silence_start:" in line:
            cur = float(line.split("silence_start:")[1].split()[0])
        elif "silence_end:" in line and cur is not None:
            sil.append((cur, float(line.split("silence_end:")[1].split()[0])))
            cur = None
    dur = probe(path)["dur"]
    segs, t = [], 0.0
    for a, b in sil:
        if a > t:
            segs.append((t, a))
        t = max(t, b)
    if t < dur:
        segs.append((t, dur))
    return segs


def snap_cues(cues, segs, tol=0.6):
    """把字幕边界吸附到最近的人声起止，修正 OCR 抽样的半秒级误差"""
    if not segs:
        return cues
    starts = [a for a, _ in segs]
    ends = [b for _, b in segs]
    for c in cues:
        a = min(starts, key=lambda x: abs(x - c["start"]))
        if abs(a - c["start"]) <= tol and a < c["end"] - 0.2:
            c["start"] = round(a, 3)
        b = min(ends, key=lambda x: abs(x - c["end"]))
        if abs(b - c["end"]) <= tol and b > c["start"] + 0.2:
            c["end"] = round(b, 3)
    return cues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "cli"))
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--model", default=str(ROOT / "models" / "big-lama.pt"))
    ap.add_argument("--band", default=None, help="字幕带 0.55,0.95 覆盖配置")
    ap.add_argument("--mask-mode", default=None, choices=["stroke", "rect"])
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--no-burn", action="store_true", help="只擦除，不翻译不压制")
    ap.add_argument("--all-text", action="store_true", help="连非中文字幕行一起擦")
    ap.add_argument("--erase-provider", default=None, choices=["local", "cloud"],
                    help="擦除引擎；默认读 config.erase.provider")
    ap.add_argument("--erase-quality", default=None, choices=["pro", "standard"],
                    help="云端档位：pro=1元/分钟，standard=0.4元/分钟")
    ap.add_argument("--target", default=None,
                    help="目标语言（en/zh/ja/...）；默认读 config.translate.target")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    target = str(args.target or (cfg.get("translate") or {}).get("target") or "en").lower()
    cfg.setdefault("translate", {})["target"] = target
    globals()["OCR_MIN_SCORE"] = float(cfg["mask"].get("ocr_conf", 0.3))
    erase_provider = str(args.erase_provider
                         or (cfg.get("erase") or {}).get("provider") or "local").lower()
    erase_quality = str(args.erase_quality
                        or ((cfg.get("erase") or {}).get("cloud") or {}).get("quality")
                        or "pro").lower()
    cloud_mode = erase_provider == "cloud"
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    work = out / "segments"
    work.mkdir(parents=True, exist_ok=True)

    meta = probe(src)
    log(f"输入 {meta['w']}x{meta['h']} @{meta['fps']:.2f}  {meta['dur']:.1f}s")

    t_start = time.time()
    shots = detect_shots(src, meta["dur"])
    log(f"拆条：{len(shots)} 个镜头 {[(round(s['start'],2), round(s['end'],2)) for s in shots]}")

    if not cloud_mode:
        if not torch.cuda.is_available():
            sys.exit("GPU 不可用")
        backend = str(cfg["inpaint"].get("backend", "native")).lower()
        inpaint_fp16 = bool(cfg["inpaint"]["fp16"])
        if backend == "native":
            from lama_native import load_native
            model = load_native(args.model, device="cuda")
            # native 后端：权重保持 fp32，卷积在 autocast 下走 fp16；
            # FFT 分支内部强制 fp32（CUDA ComplexHalf 不可用）
            use_amp = inpaint_fp16
            log(f"LaMa 已加载（native，卷积 {'fp16' if use_amp else 'fp32'}）："
                f"{torch.cuda.get_device_name(0)}")
        else:
            model = torch.jit.load(args.model, map_location="cuda").eval()
            use_amp = False
            if inpaint_fp16:
                model = model.half()
                log("警告：TorchScript 后端 fp16 会因 ComplexHalf 产生 NaN，建议改 backend: native")
            log(f"LaMa 已加载（script）：{torch.cuda.get_device_name(0)}")

        # 预热：固定形状跑几次，把 cuDNN 的算法选择一次性付掉
        with torch.inference_mode():
            bs = int(cfg["inpaint"]["batch"])
            dummy = torch.zeros(bs, 3, int(cfg["inpaint"]["fixed_h"]),
                                int(cfg["inpaint"]["fixed_w"]), device="cuda")
            if inpaint_fp16 and backend != "native":
                dummy = dummy.half()
            dmask = torch.zeros_like(dummy[:, :1])
            ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp \
                else torch.no_grad()
            with ctx:
                for _ in range(3):
                    model(dummy, dmask)
        torch.cuda.synchronize()
        log("预热完成")
    else:
        model = None
        log(f"擦除引擎：云端（{'精细化版' if erase_quality == 'pro' else '标准版'}），"
            f"跳过 LaMa/GPU 预热")

    band = tuple(float(x) for x in args.band.split(",")) if args.band \
        else tuple(cfg["mask"]["band"])
    mask_mode = args.mask_mode or cfg["mask"]["mode"]
    log(f"字幕带 {band}  掩码模式 {mask_mode if not cloud_mode else 'cloud'}")
    cloud_layout = {"band": band,
                    "align": 8 if sum(band) / 2 < 0.5 else 2,
                    "detected": True}

    vad_segs = []
    if cfg["vad"]["enable"] and not args.no_vad and has_audio(src):
        try:
            vad_segs = speech_segments(src, work)
            log(f"VAD：{len(vad_segs)} 段语音")
        except Exception as e:  # noqa: BLE001
            log(f"VAD 失败，跳过：{e}")

    all_cues, fps_list, cov_list = [], [], []
    mode_count = {"cloud": 1} if cloud_mode else {"temporal": 0, "lama": 0, "ref": 0}
    box_u = None
    cjk_only = bool(cfg["mask"].get("cjk_only", True)) and not args.all_text

    # 整片一条输出流：按镜头分段另存再 concat 会累积容器时长误差（2 分钟能漂 0.7s）
    # 云端模式不写中间片，OCR 分析完直接传整片给云端
    erased_v = out / "erased_v.mp4"
    writer = None if cloud_mode else open_writer(erased_v, meta["w"], meta["h"],
                                                 meta["fps"])

    for i, shot in enumerate(shots):
        tracks, cues = analyze_shot(src, shot, meta, band,
                                    sample_fps=float(cfg["ocr"]["sample_fps"]),
                                    mask_mode=mask_mode,
                                    keep_any_text=bool(args.all_text))
        if cjk_only:
            # 只擦中文字幕：印地语等非中文行 OCR 读不出来，硬擦会闪。
            # 单个汉字的真字幕要保留（“蕭”这种），但整行印地语被误读成一个
            # 汉字时框很宽，用字宽/字高比排掉
            keep = []
            for tr in tracks:
                n = len(CJK_RE.findall(tr["text"]))
                if n >= 2:
                    keep.append(tr)
                elif n == 1:
                    bs = [b for _, b in tr["boxes"]]
                    tw = float(np.median([b[2] - b[0] for b in bs]))
                    th = float(np.median([b[3] - b[1] for b in bs])) or 1.0
                    if tw <= 2.2 * th:
                        keep.append(tr)
            tracks = keep
        else:
            # 非中文行只能整行常驻：检测时有时无会让字幕闪
            for tr in tracks:
                if not tr["text"]:
                    tr["start"], tr["end"] = shot["start"], shot["end"]
        if tracks:
            xs0 = min(b[0] for tr in tracks for _, b in tr["boxes"])
            ys0 = min(b[1] for tr in tracks for _, b in tr["boxes"])
            xs1 = max(b[2] for tr in tracks for _, b in tr["boxes"])
            ys1 = max(b[3] for tr in tracks for _, b in tr["boxes"])
            box_u = (min(box_u[0], xs0), min(box_u[1], ys0),
                     max(box_u[2], xs1), max(box_u[3], ys1)) if box_u else (xs0, ys0, xs1, ys1)
        all_cues.extend(cues)
        if not tracks:
            log(f"镜头 {i}: 无字幕" + ("" if cloud_mode else "，原样写入"))
            if cloud_mode:
                continue
            n_exp = int(round(shot["end"] * meta["fps"])) \
                - int(round(shot["start"] * meta["fps"]))
            last, nw = None, 0
            for fr in iter_frames(src, meta["w"], meta["h"], shot["start"],
                                  shot["end"] - shot["start"], None):
                if nw >= n_exp:
                    break
                last = fr
                writer.stdin.write(fr.tobytes())
                nw += 1
            while nw < n_exp and last is not None:
                writer.stdin.write(last.tobytes())
                nw += 1
        elif cloud_mode:
            log(f"镜头 {i}: {len(tracks)} 条字幕轨（整片交云端擦除）")
        else:
            frames, dt, efps, cover, mode = erase_shot(
                model, src, shot, tracks, meta, writer, cfg, mask_mode,
                dbg_name=f"seg_{i:04d}", amp=use_amp)
            mode_count[mode] += 1
            fps_list.append(efps)
            cov_list.append(cover)
            log(f"镜头 {i}: {mode} 掩码 {cover*100:.1f}%  "
                f"{frames} 帧 {dt:.1f}s ({efps:.1f} fps)  "
                f"显存 {torch.cuda.max_memory_allocated()/2**20:.0f}MiB")

    erased = out / "erased.mp4"
    if cloud_mode:
        import erase_cloud as _ec
        erase_info = _ec.erase(src, erased, cfg, log=log, meta=meta,
                               layout=cloud_layout, quality=erase_quality)
    else:
        writer.stdin.close()
        writer.wait()
        if has_audio(src):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(erased_v),
                            "-i", str(src), "-map", "0:v:0", "-map", "1:a?",
                            "-c", "copy", "-shortest", str(erased)], check=True)
        else:
            erased_v.replace(erased)
        erase_info = {"provider": "local", "mask_mode": mask_mode}
        log("拼接完成")

    # 跨镜头合并同文：镜头切点会把一条字幕劈成两段
    all_cues = merge_overlaps(drop_junk(merge_cues(all_cues)))
    all_cues = snap_cues(all_cues, vad_segs, float(cfg["vad"]["tol"]))

    # 翻译 + 断句；--no-burn 只做擦除，不依赖翻译服务
    log(f"字幕条数：{len(all_cues)}")
    for c in all_cues:
        c["zh"] = c["text"]
    final = out / "final.mp4"
    problems = []
    if args.no_burn:
        import shutil
        shutil.copyfile(erased, final)
        final_cues = all_cues
        log("跳过翻译与压制（--no-burn）")
    else:
        en = translate(all_cues, cfg)

        is_ja = target.startswith("ja")
        max_chars = 18 if is_ja else (24 if meta["h"] > meta["w"] else 42)
        max_lines = 2
        cps = 9.0 if is_ja else 17.0
        final_cues = []
        for c, t in zip(all_cues, en):
            cues, issues = plan_cues(t, c["start"], c["end"], max_chars=max_chars,
                                     max_lines=max_lines, cps=cps)
            for x in cues:
                x["en"] = x["text"]
                final_cues.append(x)
            problems.extend(issues)
            problems.extend(check_cues(cues, max_chars=max_chars,
                                       max_lines=max_lines, cps=cps, lang=target))
        log(f"断句后 {len(final_cues)} 条，问题 {len(problems)} 个：{problems[:5]}")

        ass = out / "final.ass"
        write_ass(final_cues, ass, meta, max_chars, max_lines,
                  font="Noto Sans CJK JP" if is_ja else None)

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        base = ["ffmpeg", "-v", "error", "-y", "-i", str(erased),
                "-vf", f"ass={ass}", "-c:a", "copy",
                "-movflags", "+faststart", str(final)]
        try:
            subprocess.run(base + ["-c:v", "h264_nvenc", "-preset", "p5",
                                   "-cq", "21", "-b:v", "0"], check=True)
        except subprocess.CalledProcessError as e:
            log(f"NVENC 压制失败（exit {e.returncode}），改用 libx264")
            final.unlink(missing_ok=True)
            subprocess.run(base + ["-c:v", "libx264", "-preset", "veryfast",
                                   "-crf", "18"], check=True)
        log("压制完成")

    if box_u:   # QA 只统计真正处理过的区域（否则留下的印地语行会被误算作残留）
        pad_q = 6
        rect0 = (max(0, box_u[0] - pad_q), max(0, box_u[1] - pad_q),
                 min(meta["w"], box_u[2] + pad_q), min(meta["h"], box_u[3] + pad_q))
    else:
        rect0 = (int(meta["w"] * 0.05), int(meta["h"] * band[0]),
                 int(meta["w"] * 0.95), int(meta["h"] * band[1]))
    qa_hit, qa_total = qa_residual(erased, meta, rect0, band)

    report = {
        "src": str(src),
        "out": str(final),
        "duration": meta["dur"],
        "shots": len(shots),
        "cues": len(final_cues),
        "erase": erase_info,
        "erase_fps": round(float(np.mean(fps_list)), 2) if fps_list else None,
        "erase_seconds": round(time.time() - t_start, 1),
        "residual_frames": f"{qa_hit}/{qa_total}",
        "mask_coverage": round(float(np.mean(cov_list)), 4) if cov_list else 0,
        "erase_modes": mode_count,
        "vad_segments": len(vad_segs),
        "ocr_cuda": os.environ.get("OCR_USE_CUDA", "1") == "1",
        "cue_issues": problems,
        "cue_debug": [[round(c["start"], 2), round(c["end"], 2), c["zh"]]
                      for c in all_cues],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    log("报告：" + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
