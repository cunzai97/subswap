#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频字幕转换核心：擦除原字幕 -> OCR -> 翻译 -> 按原字幕位置字号重新压制

设计口径（兼容性优先）：
  - 分辨率、纵横比、fps 都不写死；字幕带、字号、边距全部从素材实测
  - 带旋转元数据的素材先转正（手机/微信素材常见）
  - VFR 源先归一化到 CFR，否则逐镜头写帧会声画漂移
  - 无音轨素材不报错
  - 源语言 zh 走中文字幕过滤（滤掉印地语等误识），其他语言走原文取众数
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "webapp" / "vendor"))
import pipeline as P  # noqa: E402
import erase_cloud as E  # noqa: E402

try:
    import wordninja as _wordninja
except Exception:  # noqa: BLE001
    _wordninja = None

LANG_CJK = {"zh", "ja", "ko"}
# 实测：中文 42 号字在 720p 上 OCR 框高 31px（0.74 em）
FONT_COEF = {"cjk": 1.15, "latin": 1.15}
CHAR_W = {"cjk": 1.00, "latin": 0.52}

_MODEL = None
_MODEL_AMP = False


def resolve_erase(cfg, opts=None):
    """opts.erase / config.erase.provider -> (provider, quality)

    取值：local | cloud_pro | cloud_standard（cloud 默认精细化版）
    """
    opts = opts or {}
    raw = str(opts.get("erase") or "").strip().lower()
    if not raw:
        raw = str(((cfg or {}).get("erase") or {}).get("provider") or "local").lower()
    if raw in ("cloud", "cloud_pro", "pro"):
        return "cloud", "pro"
    if raw in ("cloud_standard", "cloud_std", "standard"):
        return "cloud", "standard"
    return "local", None


def cloud_key_ready(cfg):
    return E.key_ready(cfg)


def get_model(cfg=None):
    """加载擦除模型。

    backend: native（默认）= 原生 PyTorch 重建 big-lama，fp32 权重 + 卷积
    autocast fp16（FFT 分支内部强制 fp32）；比 TorchScript fp32 快约 1.7×。
    backend: script = 旧 TorchScript（fp32）。
    """
    global _MODEL, _MODEL_AMP
    if _MODEL is None:
        import torch
        icfg = (cfg or {}).get("inpaint") or {}
        backend = str(icfg.get("backend", "native")).lower()
        if backend == "native":
            from lama_native import load_native
            _MODEL = load_native(str(ROOT / "models/big-lama.pt"), device="cuda")
            _MODEL_AMP = bool(icfg.get("fp16", True))
        else:
            _MODEL = torch.jit.load(str(ROOT / "models/big-lama.pt"),
                                    map_location="cuda").eval()
            _MODEL_AMP = False
        with torch.inference_mode():
            bs = max(1, int(icfg.get("batch", 8)))
            d = torch.zeros(bs, 3, int(icfg.get("fixed_h", 256)),
                            int(icfg.get("fixed_w", 1024)), device="cuda")
            dm = torch.zeros_like(d[:, :1])
            if _MODEL_AMP:
                with torch.autocast("cuda", dtype=torch.float16):
                    for _ in range(2):
                        _MODEL(d, dm)
            else:
                for _ in range(2):
                    _MODEL(d, dm)
        torch.cuda.synchronize()
    return _MODEL


# ---------------- 基础工具 ----------------

def ffprobe_streams(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
         str(path)], capture_output=True, check=True).stdout
    return json.loads(out)


def has_audio(path):
    try:
        d = ffprobe_streams(path)
    except Exception:  # noqa: BLE001
        return False
    return any(s.get("codec_type") == "audio" for s in d.get("streams", []))


def rotation_of(info):
    for s in info.get("streams", []):
        if s.get("codec_type") != "video":
            continue
        for sd in s.get("side_data_list") or []:
            if "rotation" in sd:
                try:
                    return int(float(sd["rotation"]))
                except Exception:  # noqa: BLE001
                    pass
        r = (s.get("tags") or {}).get("rotate")
        if r:
            try:
                return int(float(r))
            except Exception:  # noqa: BLE001
                pass
    return 0


def is_vfr(info):
    for s in info.get("streams", []):
        if s.get("codec_type") != "video":
            continue
        r, a = s.get("r_frame_rate"), s.get("avg_frame_rate")
        if not r or not a:
            continue
        try:
            rf = eval(r.replace("/", "/"))  # noqa: S307  形如 30000/1001
            af = eval(a.replace("/", "/"))  # noqa: S307
        except Exception:  # noqa: BLE001
            continue
        if af > 0 and abs(rf - af) / max(af, 1e-6) > 0.02:
            return True
    return False


def normalize(src: Path, work: Path, log):
    """转正 + 归一化帧率；不需要时原样返回"""
    info = ffprobe_streams(src)
    rot = rotation_of(info)
    vfr = is_vfr(info)
    if not rot and not vfr:
        return src
    vf = []
    # 旋转交给 ffmpeg 的 autorotate：它会把 display matrix 应用到像素并清掉元数据。
    # 自己 transpose + 默认 autorotate 会转两次（实测又变回竖屏）。
    if rot:
        log(f"素材带 {rot}° 旋转，先转正")
    if vfr:
        log("素材是可变帧率，先归一化到恒定帧率")
    if rot:
        vf.append("null")     # 有滤镜链才会触发 autorotate
    out = work / "normalized.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if vfr:
        cmd += ["-vsync", "cfr"]
    cmd += ["-c:v", "libx264", "-crf", "12", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "copy"] if has_audio(src) else ["-an"]
    cmd += [str(out)]
    subprocess.run(cmd, check=True)
    return out


# ---------------- 版面探测 ----------------

def detect_layout(src: Path, meta, log):
    """抽样 OCR 全画面，找出真正的字幕行，返回带范围/字高/边距/对齐方式"""
    w, h, fps, dur = meta["w"], meta["h"], meta["fps"], meta["dur"]
    n_sample = 16
    step = max(1.0, dur / n_sample)
    boxes = []
    for i in range(n_sample):
        t = min(dur - 0.05, i * step + 0.3)
        # 窗口给足 1s：低帧率/VFR 源 0.12s 可能一帧都取不到
        fr = next(iter(P.iter_frames(str(src), w, h, t, 1.0, None)), None)
        if fr is None:
            continue
        for box, text, score in P.ocr_boxes(fr, (0.0, 1.0)):
            if score < 0.55:
                continue
            if (box[2] - box[0]) < 0.04 * w:      # 太窄的多半是噪点
                continue
            boxes.append((box, text, score))
    if not boxes:
        log("版面探测：没抓到文字，退回画面下方 40% 作为字幕带")
        return {"band": (0.60, 0.98), "font_h": max(18, int(h * 0.045)),
                "align": 2, "margin_v": int(h * 0.06), "line_w": int(w * 0.86),
                "rows": [], "detected": False}

    # 按 y 中心聚类成"行"
    boxes.sort(key=lambda x: (x[0][1] + x[0][3]) / 2)
    rows = []
    for box, text, score in boxes:
        yc = (box[1] + box[3]) / 2
        bh = box[3] - box[1]
        for r in rows:
            if abs(yc - r["yc"]) < 0.9 * max(r["h"], bh):
                r["boxes"].append(box)
                r["texts"].append(text)
                r["yc"] = float(sum((b[1] + b[3]) / 2 for b in r["boxes"]) / len(r["boxes"]))
                r["h"] = float(sorted(b[3] - b[1] for b in r["boxes"])[len(r["boxes"]) // 2])
                break
        else:
            rows.append({"yc": yc, "h": bh, "boxes": [box], "texts": [text]})
    need = max(2, int(0.15 * n_sample))
    rows = [r for r in rows if len(r["boxes"]) >= need]
    if not rows:
        rows = [{"yc": max((b[1] + b[3]) / 2 for b, _t, _s in boxes), "h": 24,
                 "boxes": [b for b, _t, _s in boxes], "texts": [""]}]
    # 字幕行的典型特征：横向最长、文本最完整。误检（'O'/'7' 这种）框小但可能很稳定，
    # 按样本数排序会把它们选成主行，所以按“框宽 × 文本长度”综合排
    def row_score(r):
        w_med = float(sorted(b[2] - b[0] for b in r["boxes"])[len(r["boxes"]) // 2])
        tmax = max(len(t or "") for t in r["texts"])
        return w_med * (1 + min(tmax, 40) / 10.0)

    rows.sort(key=row_score, reverse=True)
    main = rows[0]
    yc, mh = float(main["yc"]), float(main["h"])
    if yc < h * 0.5:                   # 顶部字幕
        align = 8
        margin = int(max(8, yc - mh / 2))
    else:                              # 底部字幕：用主行中心/字高，别用最大框底边
        align = 2
        margin = int(max(8, h - (yc + mh / 2)))
    pad = 1.5 * mh
    band = (max(0.0, (yc - pad) / h), min(1.0, (yc + pad) / h))
    line_w = float(sorted(b[2] - b[0] for b in main["boxes"])[len(main["boxes"]) // 2])
    # 只把主行和紧邻的第二行留给轨道过滤；否则衣服/招牌上的大字也会被当字幕擦掉
    near = [r for r in rows if abs(float(r["yc"]) - yc) <= 1.6 * mh]
    layout = {
        "band": (round(band[0], 3), round(band[1], 3)),
        "font_h": int(round(mh)),
        "align": align,
        "margin_v": margin,
        "line_w": int(min(line_w * 1.25, w * 0.92)),
        "rows": [round(float(r["yc"]) / h, 3) for r in near],
        "main_yc": round(yc / h, 3),
        "detected": True,
    }
    log(f"版面探测：字幕带 y{band[0]:.2f}-{band[1]:.2f} 字高 {layout['font_h']}px "
        f"{'顶部' if align == 8 else '底部'} 边距 {margin}px 行宽 {layout['line_w']}px")
    return layout


# ---------------- 字幕抽取 ----------------

CJK_RE = P.CJK_RE


def filter_tracks_by_layout(tracks, layout, meta):
    """只保留落在主字幕行上的轨；误检框（'O'/'7' 这类）会把掩码撑到人物身上"""
    if not layout.get("detected"):
        return tracks
    h = meta["h"]
    fh = max(8, int(layout["font_h"]))
    centers = []
    for yc in (layout.get("rows") or []):
        centers.append(yc * h)
    if not centers:
        return tracks
    keep = []
    for tr in tracks:
        bs = [b for _, b in tr["boxes"]]
        yc = float(sorted((b[1] + b[3]) / 2 for b in bs)[len(bs) // 2])
        hh = float(sorted(b[3] - b[1] for b in bs)[len(bs) // 2])
        if hh > 3.0 * fh:            # 明显不是字幕行的框
            continue
        if min(abs(yc - c) for c in centers) <= 1.3 * fh:
            keep.append(tr)
    return keep


def _med(vals):
    s = sorted(vals)
    return float(s[len(s) // 2])


def filter_scene_clusters(tracks, layout, meta):
    """同一镜头里 3 条以上、横向散开的窄轨 = 账本/招牌等画面文字；
    真字幕是整行，偶尔一个短句也只会是单条，不会成簇"""
    lw = max(1, int(layout.get("line_w") or meta["w"] * 0.5))
    fh = max(8, int(layout.get("font_h") or 24))
    st = {}
    for tr in tracks:
        bs = [b for _, b in tr["boxes"]]
        st[id(tr)] = (_med([b[2] - b[0] for b in bs]),
                      _med([(b[0] + b[2]) / 2 for b in bs]),
                      _med([(b[1] + b[3]) / 2 for b in bs]),
                      _med([b[3] - b[1] for b in bs]))
    # 只把“单字大小”的窄轨当画面文字；被 OCR 拆成两半的字幕仍是宽条，
    # 不能误伤（宽高比 >1.8 的短半句仍按字幕保留）
    narrow = [tr for tr in tracks
              if st[id(tr)][0] < 0.30 * lw and st[id(tr)][0] <= 1.8 * st[id(tr)][3]]
    drop = set()
    for tr in narrow:
        if id(tr) in drop:
            continue
        cl = [tr]
        for o in narrow:
            if o is tr or abs(st[id(o)][2] - st[id(tr)][2]) > 1.5 * fh:
                continue
            if min(tr["end"], o["end"]) >= max(tr["start"], o["start"]) - 1.0:
                cl.append(o)
        xs = [st[id(x)][1] for x in cl]
        starts = [x["start"] for x in cl]
        if (len(cl) >= 3 and max(xs) - min(xs) >= 0.22 * meta["w"]
                and max(starts) - min(starts) <= 2.5):
            drop.update(id(x) for x in cl)
    return [tr for tr in tracks if id(tr) not in drop]


def filter_tracks_cjk(tracks):
    """中文字幕场景：只留真中文轨，排掉外文字幕被误读成单个汉字的情况"""
    keep = []
    for tr in tracks:
        n = len(CJK_RE.findall(tr["text"]))
        if n >= 2:
            keep.append(tr)
        elif n == 1:
            bs = [b for _, b in tr["boxes"]]
            tw = float(sorted(b[2] - b[0] for b in bs)[len(bs) // 2])
            th = float(sorted(b[3] - b[1] for b in bs)[len(bs) // 2]) or 1.0
            if tw <= 2.2 * th:
                keep.append(tr)
    return keep


KANA_HANGUL = re.compile(r"[\u3040-\u30ff\uac00-\ud7af]")
LATIN_WORD = re.compile(r"[A-Za-z\u00c0-\u024f][A-Za-z\u00c0-\u024f'’-]*")


def clean_latin_text(text):
    """OCR 后处理：去首尾噪声、修连字符/引号、补首字母大写"""
    t = (text or "").replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip(" -–—•·|")
    t = re.sub(r"^[-–—]\s*", "", t)
    # RapidOCR 的英文行经常不吐空格：整行无空格时用词频模型恢复
    if _wordninja is not None and " " not in t:
        letters = re.sub(r"[^A-Za-z]", "", t)
        if len(letters) >= 8:
            tail = t.rstrip()[-1] if t.rstrip()[-1:] in ".?!" else ""
            parts = _wordninja.split(t)
            if parts and tail:
                parts[-1] += tail
            t = " ".join(parts)
    toks = t.split()

    def junk(tok):
        return len(tok) == 1 and (tok.isdigit() or (tok.isalpha() and tok not in "AI"))

    while toks and junk(toks[-1]):
        toks.pop()
    while toks and junk(toks[0]):
        toks.pop(0)
    t = " ".join(toks)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"\.{3,}", "…", t)
    if t and t[0].islower() and t[0].isascii():
        t = t[0].upper() + t[1:]
    return t


def clean_cues(cues, source):
    """去掉 OCR 噪声和碎片；保留“好/行/哟”这类单字台词"""
    out = []
    for c in cues:
        start, end = float(c["start"]), float(c["end"])
        dur = end - start
        t = (c.get("text") or "").strip()
        if source in ("zh", "ja", "ko"):
            # 单字台词真实存在（好/行/哟），只靠时长滤掉 OCR 碎片
            if dur < 0.35:
                continue
            if len(CJK_RE.findall(t)) < 1 and not KANA_HANGUL.search(t):
                continue
            t = t.strip(" -–—•·|")
            # OCR 把一句话拆成左右两截时中间会留空格，中文字幕不需要
            t = re.sub(r"(?<=[\u3400-\u9fff，。！？、；：])\s+(?=[\u3400-\u9fff])", "", t)
        else:
            if len(t) < 2 or dur < 0.45:
                continue
            t = clean_latin_text(t)
            words = LATIN_WORD.findall(t)
            scripts = len(CJK_RE.findall(t)) + len(KANA_HANGUL.findall(t))
            if not words and scripts < 2:
                continue
            if len(words) == 1 and len(words[0]) < 4:
                continue
        if not t:
            continue
        out.append({"start": start, "end": end, "text": t,
                    **({"box": c["box"]} if c.get("box") else {})})
    return out


def guess_source(cues):
    txt = "".join(c["text"] for c in cues)
    if not txt:
        return "zh"
    cjk = len(CJK_RE.findall(txt))
    return "zh" if cjk / max(len(txt), 1) > 0.2 else "other"


_HAN_RE = re.compile(r"[\u3400-\u9fff]")
_LATIN_EDGE = re.compile(r"^[A-Za-z][A-Za-z0-9'’\-\.]{1,}\s*")
_LATIN_TAIL = re.compile(r"\s*[A-Za-z][A-Za-z0-9'’\-\.]{1,}$")
_LATIN_CAPS = re.compile(r"\s*[A-Z]{5,}\s*")


def strip_scene_latin(cues):
    """中文字幕带里的球衣/招牌拉丁字会被 OCR 并进台词，交付前清掉。"""
    out = []
    for c in cues:
        t = c["text"]
        if not _HAN_RE.search(t):
            continue
        t = _LATIN_TAIL.sub("", _LATIN_EDGE.sub("", t))
        t = _LATIN_CAPS.sub(" ", t).strip()
        if not _HAN_RE.search(t):
            continue
        c = dict(c)
        c["text"] = t
        out.append(c)
    return out


def extract(src: Path, meta, layout, source, log, on_prog=None, ocr=None):
    band = tuple(layout["band"])
    ocfg = ocr or {}
    ocr_max_w = int(ocfg.get("max_w", 0) or 0)
    reuse_static = bool(ocfg.get("reuse_static", True))
    static_thr = float(ocfg.get("static_thr", 2.5))
    P.set_ocr_opts(det_limit_side_len=ocfg.get("det_limit_side_len"),
                   det_limit_type=ocfg.get("det_limit_type"))
    shots = P.detect_shots(str(src), meta["dur"])
    log(f"拆条：{len(shots)} 个镜头")
    all_cues = []
    per_shot = []
    for i, shot in enumerate(shots):
        keep_any = source != "zh"
        tracks, cues = P.analyze_shot(str(src), shot, meta, band, sample_fps=4.0,
                                      keep_any_text=keep_any,
                                      ocr_max_w=ocr_max_w, reuse_static=reuse_static,
                                      static_thr=static_thr)
        tracks = filter_tracks_by_layout(tracks, layout, meta)
        if source == "zh":
            tracks = filter_tracks_cjk(tracks)
        tracks = filter_scene_clusters(tracks, layout, meta)
        # 字幕文本只能来自“留下并擦除”的轨：账本/招牌等被版面过滤的文字
        # 不能混进翻译和重压字幕（实测 23.5s 账本字被并进台词）
        all_cues += [{"start": tr.get("text_start", tr["start"]),
                      "end": tr.get("text_end", tr["end"]),
                      "text": tr["text"],
                      "box": tr.get("ubox") or tr["last_box"]}
                     for tr in tracks if tr["text"]]
        per_shot.append(tracks)
        if on_prog:
            on_prog((i + 1) / len(shots))
    cues = P.merge_overlaps(P.drop_junk(P.merge_cues(all_cues)))
    cues = clean_cues(cues, source)
    cues = [c for c in cues if c["end"] - c["start"] >= 0.15]
    cues.sort(key=lambda c: c["start"])
    if source == "auto":
        source = guess_source(cues)
    if source == "zh":
        cues = strip_scene_latin(cues)
    log(f"字幕：{len(cues)} 条（源语言判定 {source}）")
    return cues, source, shots, per_shot


# ---------------- 擦除 ----------------

def qc_frames(erased, meta, layout, cues, source, work, log):
    """擦除后抽帧：cue 中点取字幕带，再 OCR 一遍看还有没有源语言字"""
    import cv2
    qdir = work / "qc"
    shutil.rmtree(qdir, ignore_errors=True)
    qdir.mkdir(parents=True)
    band = layout["band"]
    y0 = int(meta["h"] * band[0]) & ~1
    y1 = int(meta["h"] * band[1]) & ~1
    bh = y1 - y0
    n = min(12, len(cues))
    picks = [cues[round(i * (len(cues) - 1) / max(1, n - 1))] for i in range(n)] if n else []
    thumbs, suspects = [], []
    for i, c in enumerate(picks):
        t = round((float(c["start"]) + float(c["end"])) / 2, 2)
        fr = next(P.iter_frames(str(erased), meta["w"], bh, t, 1.0, None,
                                crop=(0, y0, meta["w"], bh)), None)
        if fr is None:
            continue
        hit = ""
        for _box, txt, score in P.ocr_boxes(fr, (0.0, 1.0)):
            if score < 0.5:
                continue
            if source in ("zh", "ja", "ko"):
                if CJK_RE.search(txt or "") or KANA_HANGUL.search(txt or ""):
                    hit = (txt or "").strip()
                    break
            else:
                words = LATIN_WORD.findall(txt or "")
                if len(words) >= 2 and sum(len(w) for w in words) >= 6:
                    hit = (txt or "").strip()
                    break
        name = f"qc_{i:02d}.jpg"
        cv2.imwrite(str(qdir / name), fr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        thumbs.append({"t": t, "img": name})
        if hit:
            suspects.append({"t": t, "text": hit})
    log(f"自动抽帧 {len(thumbs)} 张，疑似残留 {len(suspects)} 处")
    return {"thumbs": thumbs, "suspects": suspects}


def _stream_frames(path) -> int | None:
    """裸流/MP4 都按 packet 数读帧；读不到返回 None，不猜。"""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
            text=True)
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def _concat_segments(segs, out, meta, log, expected=None):
    """把逐镜头 H.264 裸流按顺序拼成 MP4（流复制，不重编码）。

    B 帧跨段会让 mux 丢帧（实测 3/375），所以写裸流时强制 -bf 0；
    这里再用 packet 数做硬断言：分段和成片任一不等就直接失败，
    不再只比 ±0.3s 时长（25fps 下那是 7 帧的盲区）。
    """
    parts = [(s, _stream_frames(s)) for s in segs]
    seg_sum = sum(n for _s, n in parts if n is not None)
    if all(n is not None for _s, n in parts):
        if expected is not None and seg_sum != expected:
            raise RuntimeError(f"分段帧数 {seg_sum} != 期望 {expected}，拒绝拼接")
    else:
        log("警告：部分分段读不到帧数，跳过分段求和断言")
    tmp = out.with_name(out.stem + ".h264")
    with open(tmp, "wb") as fw:
        for s in segs:
            with open(s, "rb") as fr:
                shutil.copyfileobj(fr, fw, 1 << 20)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-r", f"{meta['fps']:g}",
                    "-f", "h264", "-i", str(tmp), "-an", "-c:v", "copy",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    tmp.unlink(missing_ok=True)
    got = _stream_frames(out)
    if got is None:
        log("警告：拼接后读不到帧数，跳过成片帧数断言")
    elif all(n is not None for _s, n in parts) and got != seg_sum:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"拼接丢帧：成片 {got} 帧，分段合计 {seg_sum} 帧")
    elif expected is not None and got != expected:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"成片帧数 {got} != 期望 {expected}，拒绝出片")
    dur = P.probe(str(out))["dur"]
    if abs(dur - meta["dur"]) > 0.3:
        log(f"警告：分段拼接 {dur:.2f}s 与素材 {meta['dur']:.2f}s 偏差较大")
    return out


class _SerialModel:
    """把 LaMa 调用串行化。

    多镜头并行时 GPU 一次只能跑一个 batch（模型不是线程安全的，16GB 卡也放不下
    两份前向激活）；能并行的是镜头里那些纯 CPU 的活——解码、字形掩码、光流、
    光度校验、合成。所以锁只包住 model(...) 这一层，不包住整个镜头。
    """

    def __init__(self, model):
        self._model = model
        self._lock = threading.Lock()

    def __call__(self, *a, **k):
        with self._lock:
            return self._model(*a, **k)

    def __getattr__(self, name):
        return getattr(self._model, name)


def erase(src: Path, meta, layout, shots, per_shot, cfg, on_prog, log,
          mask_mode="stroke", cues=None, pipe=None):
    """逐镜头擦除并拼接。

    pipe=1 串行（老行为）；pipe>1 同时推进 N 个镜头，靠“一个镜头在等 GPU、
    另一个在跑 CPU”把擦除段里空转的算力吃回来。ERASE_PIPE 环境变量可覆盖。
    分段文件按镜头号落盘，谁先完成都不影响拼接顺序。
    """
    if pipe is None:
        pipe = int(os.environ.get("ERASE_PIPE", "0") or 0)
    if int(pipe) <= 0:
        pipe = int(((cfg or {}).get("worker") or {}).get("erase_pipe", 0) or 0)
    if int(pipe) <= 0:
        # 单跑（网页/CLI）默认 4 路：实测 1080x1920 单片擦除 281s -> 94s，
        # 峰值显存约 6GB。批量跑由 batch_pipeline 按并发数下调。
        pipe = 4 if (os.cpu_count() or 4) >= 12 else 2
    pipe = max(1, int(pipe))
    model = get_model(cfg)
    if pipe > 1:
        model = _SerialModel(model)
    work = src.parent
    erased = work / "erased.mp4"
    seg_dir = work / "segments"
    shutil.rmtree(seg_dir, ignore_errors=True)
    seg_dir.mkdir(parents=True)
    segs = [seg_dir / f"shot_{i:04d}.h264" for i in range(len(shots))]
    t0 = time.time()
    total_frames = 0
    n_done = 0
    n_shots = 0
    p_lock = threading.Lock()

    # 只喂字幕带的中间层：ffmpeg 自己解码源片、把带贴回去再编码。
    # 省掉 Python 侧每帧 6.2MB 的整帧搬运（实测见 docs/ERASE_WRITE_OVERLAY.md）。
    write_mode = (os.environ.get("ERASE_WRITE")
                  or ((cfg or {}).get("worker") or {}).get("erase_write") or "overlay")
    overlay_ok = (str(write_mode).lower() == "overlay"
                  and str(((cfg or {}).get("erase") or {}).get("engine", "ref")) != "legacy")

    def run_shot(i, force_x264=False):
        """返回 (镜头号, 帧数, 是否走了擦除, 模式)；异常原样抛出。"""
        shot = shots[i]
        tracks = filter_tracks_by_layout(per_shot[i], layout, meta)
        band = P.erase_band_rect(tracks, meta["w"], meta["h"]) \
            if (overlay_ok and tracks) else None
        if band is not None:
            writer = P.open_overlay_writer(str(segs[i]), src, meta, shot["start"],
                                           shot["end"] - shot["start"], band,
                                           force_x264=force_x264)
        else:
            writer = P.open_writer(str(segs[i]), meta["w"], meta["h"], meta["fps"],
                                   force_x264=force_x264)
        try:
            if not tracks:
                n = int(round(shot["end"] * meta["fps"])) - int(round(shot["start"] * meta["fps"]))
                last = None
                got = 0
                for k, fr in enumerate(P.iter_frames(str(src), meta["w"], meta["h"],
                                                     shot["start"], shot["end"] - shot["start"], None)):
                    if got >= n:
                        break
                    last = fr
                    writer.stdin.write(fr.tobytes())
                    got += 1
                # 分段必须帧数精确，否则拼接后整片会漂
                while last is not None and got < n:
                    writer.stdin.write(last.tobytes())
                    got += 1
                return i, got, False, "copy"
            n, dt, fps, cov, mode = P.erase_shot(model, str(src), shot, tracks, meta,
                                                 writer, cfg, mask_mode,
                                                 amp=_MODEL_AMP, cues=cues)
            return i, n, True, mode
        finally:
            try:
                writer.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            if writer.wait():
                raise RuntimeError(f"镜头 {i} 中间层编码失败")

    def collect(i, n, erased_any, mode):
        nonlocal total_frames, n_done, n_shots
        with p_lock:
            total_frames += n
            n_shots += 1
            if erased_any:
                n_done += 1
            if on_prog:
                el = max(1e-6, time.time() - t0)
                on_prog(n_shots / max(1, len(shots)),
                        f"镜头 {n_shots}/{len(shots)}（完成 #{i + 1}）· {mode} · "
                        f"{total_frames / el:.1f} fps")

    def safe_shot(i):
        """单个镜头失败时换软编重试一次：并发高时 NVENC 抢不到会话/显存，
        直接抛会让整支片白跑（实测 4 路并发时 ep07 镜头 74 就这样挂过）。"""
        try:
            return run_shot(i)
        except Exception as e:  # noqa: BLE001
            if os.environ.get("ERASE_WRITER", "auto").lower() == "x264":
                raise
            log(f"镜头 {i} 失败（{type(e).__name__}），改用 x264 中间层重试")
            return run_shot(i, force_x264=True)

    if pipe > 1 and len(shots) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=pipe, thread_name_prefix="erase") as ex:
            futs = [ex.submit(safe_shot, i) for i in range(len(shots))]
            for fu in futs:
                collect(*fu.result())
    else:
        for i in range(len(shots)):
            collect(*safe_shot(i))
    _concat_segments(segs, erased, meta, log, expected=total_frames)
    return erased


# ---------------- 残留复查 / 自动返修 ----------------

def _residual_hit(text, want, source):
    """复查到的字必须和原台词有共同字/词，避免把招牌等画面文字当残留抹掉"""
    if not text or not want:
        return False
    if source == "zh":
        return bool(set(CJK_RE.findall(text)) & set(CJK_RE.findall(want)))
    if source == "ja":
        pat = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
    elif source == "ko":
        pat = re.compile(r"[\uac00-\ud7af\u3400-\u9fff]")
    else:
        got = {w.lower() for w in LATIN_WORD.findall(text) if len(w) >= 3}
        return bool(got & {w.lower() for w in LATIN_WORD.findall(want) if len(w) >= 3})
    return bool(set(pat.findall(text)) & set(pat.findall(want)))


def _sample_times(tr, shot):
    # 只在字幕真实出现的时间窗内取样；5 个点覆盖短闪残留
    # （实测 0.12s 的首字残留会落在 3 点取样的空档里）
    a = max(shot["start"], float(tr.get("text_start", tr["start"])))
    b = min(shot["end"], float(tr.get("text_end", tr["end"])))
    if b <= a:
        a = max(shot["start"], float(tr["start"]))
        b = min(shot["end"], float(tr["end"]))
    if b <= a:
        return [round(a, 3)]
    ts = [a + (b - a) * f for f in (0.12, 0.31, 0.5, 0.69, 0.88)]
    return sorted({round(t, 3) for t in ts
                   if shot["start"] <= t <= shot["end"]})


def residual_scan(erased, meta, layout, shots, per_shot, source, log,
                  only_shots=None):
    """逐条字幕复查：只 OCR 该字幕自己的框（外扩一点），命中原台词才算残留。

    only_shots：返修后只复查被改过的镜头，其余镜头上一轮已干净，不再重复扫。
    """
    w, h, fps = meta["w"], meta["h"], meta["fps"]
    band = layout["band"]
    fb0 = int(h * band[0]) & ~1
    fb1 = int(h * band[1]) & ~1
    ch = fb1 - fb0
    scan_fps = 8.0
    suspects, n_tracks, n_ocr = [], 0, 0
    for si, shot in enumerate(shots):
        if only_shots is not None and si not in only_shots:
            continue
        tracks = filter_tracks_by_layout(per_shot[si], layout, meta)
        if not tracks:
            continue
        plans = [(tr, _sample_times(tr, shot)) for tr in tracks]
        need = sorted({t for _tr, ts in plans for t in ts})
        if not need:
            continue
        frames, idx = {}, 0
        for k, fr in enumerate(P.iter_frames(str(erased), w, ch, shot["start"],
                                             shot["end"] - shot["start"], scan_fps,
                                             crop=(0, fb0, w, ch))):
            t = shot["start"] + k / scan_fps
            while idx < len(need) and t >= need[idx] - 0.5 / scan_fps:
                frames[need[idx]] = fr
                idx += 1
            if idx >= len(need):
                break
        for tr, ts in plans:
            n_tracks += 1
            for t in ts:
                fr = frames.get(t)
                if fr is None:
                    continue
                bx0, by0, bx1, by1 = tr.get("ubox") or P._box_at(tr, t)
                mx = max(24, int(0.035 * w))
                my = max(12, int(0.25 * (by1 - by0)))
                cx0, cx1 = max(0, int(bx0) - mx), min(w, int(bx1) + mx)
                cy0 = max(fb0, int(by0) - my) - fb0
                cy1 = min(fb1, int(by1) + my) - fb0
                crop = fr[cy0:cy1, cx0:cx1]
                if crop.size == 0:
                    continue
                hit = ""
                for _b, txt, score in P.ocr_boxes(crop, (0.0, 1.0), min_w_frac=0.0):
                    n_ocr += 1
                    txt = (txt or "").strip()
                    if score >= 0.45 and _residual_hit(txt, tr["text"], source):
                        hit = txt
                        break
                if hit:
                    suspects.append({"shot": si, "t": round(float(t), 2),
                                     "text": hit, "cue": tr["text"], "tr": tr,
                                     "box": [int(bx0), int(by0), int(bx1), int(by1)]})
                    break
    log(f"残留复查：{n_tracks} 条字幕 / {n_ocr} 次识别，疑似残留 {len(suspects)} 处")
    return {"tracks": n_tracks, "ocr": n_ocr, "suspects": suspects}


def _report_scan(scan):
    return {"tracks": scan["tracks"], "ocr": scan["ocr"],
            "suspects": len(scan["suspects"]),
            "hits": [{"t": s["t"], "text": s["text"], "cue": s["cue"]}
                     for s in scan["suspects"][:20]]}


def _lama_patch_batch(model, frames, masks, cfg, log=None):
    """整行矩形掩码 + LaMa；只对有残留的镜头跑，正常镜头零开销"""
    import cv2
    import numpy as np
    import torch
    sw = int(cfg["inpaint"].get("fixed_w", 1024))
    sh = int(cfg["inpaint"].get("fixed_h", 256))
    feather = float(cfg["inpaint"].get("feather_sigma", 4.0)) or 4.0
    items, metas, canv, mcanv = [], [], [], []
    for fr, mk in zip(frames, masks):
        ys, xs = np.where(mk > 0)
        if not ys.size:
            continue
        bx0, by0 = int(xs.min()), int(ys.min())
        bx1, by1 = int(xs.max()) + 1, int(ys.max()) + 1
        # 留足上下文的边：整行遮掉后 LaMa 要靠周围纹理补出衣服/背景
        mg = max(48, int(0.75 * (by1 - by0)))
        cx0, cy0 = max(0, bx0 - mg), max(0, by0 - mg)
        cx1, cy1 = min(fr.shape[1], bx1 + mg), min(fr.shape[0], by1 + mg)
        reg, mreg = fr[cy0:cy1, cx0:cx1], mk[cy0:cy1, cx0:cx1]
        hh, ww = reg.shape[:2]
        s = min(sw / ww, sh / hh, 1.0)
        nw, nh = max(2, int(round(ww * s))), max(2, int(round(hh * s)))
        nw, nh = min(nw, sw), min(nh, sh)
        im = cv2.resize(reg, (nw, nh),
                        interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
        mm = cv2.resize((mreg > 0).astype(np.uint8) * 255, (nw, nh),
                        interpolation=cv2.INTER_NEAREST)
        px, py = (sw - nw) // 2, (sh - nh) // 2
        canv.append(cv2.copyMakeBorder(im, py, sh - nh - py, px, sw - nw - px,
                                       cv2.BORDER_REPLICATE))
        mcanv.append(cv2.copyMakeBorder(mm, py, sh - nh - py, px, sw - nw - px,
                                        cv2.BORDER_CONSTANT, value=0))
        items.append((fr, mk, (cx0, cy0, cx1, cy1)))
        metas.append((px, py, nw, nh))
    if not items:
        return frames
    img = torch.from_numpy(np.stack(canv)[:, :, :, ::-1].copy()).permute(0, 3, 1, 2)
    img = img.float().div_(255).cuda()
    msk_t = torch.from_numpy((np.stack(mcanv) > 0).astype(np.float32)).cuda()[:, None]
    with torch.inference_mode():
        if _MODEL_AMP:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(img, msk_t)
        else:
            out = model(img, msk_t)
    torch.cuda.synchronize()
    out = out.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    finite = np.isfinite(out).all(axis=3)
    if not bool(finite.all()) and _MODEL_AMP:
        # 大 rect 掩码偶尔在 autocast fp16 下整批 NaN；静默跳过 = 返修无效，
        # 会留下灰字残影。用 fp32 权重重跑这一批。
        if log:
            log("模型 fp16 输出含 NaN，整批回退 fp32")
        with torch.inference_mode():
            out = model(img, msk_t)
        torch.cuda.synchronize()
        out = out.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        finite = np.isfinite(out).all(axis=3)
    out = (np.nan_to_num(out, nan=0.0) * 255).round().clip(0, 255).astype(np.uint8)
    for (fr, mk, (cx0, cy0, cx1, cy1)), (px, py, nw, nh), o, fin in zip(
            items, metas, out, finite):
        ww, hh = cx1 - cx0, cy1 - cy0
        o2 = cv2.resize(o[py:py + nh, px:px + nw][..., ::-1], (ww, hh),
                        interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(fin[py:py + nh, px:px + nw].astype(np.uint8) * 255,
                           (ww, hh), interpolation=cv2.INTER_NEAREST) > 127
        mreg = mk[cy0:cy1, cx0:cx1]
        alpha = cv2.GaussianBlur((mreg > 0).astype(np.uint8) * 255, (0, 0), feather)
        a = (alpha.astype(np.float32) / 255.0) * valid
        a = a[..., None]
        fr[cy0:cy1, cx0:cx1] = (
            o2.astype(np.float32) * a
            + fr[cy0:cy1, cx0:cx1].astype(np.float32) * (1 - a)).astype(np.uint8)
    return frames


def patch_erased(erased, out_path, meta, shots, per_shot, suspects, layout, cfg,
                 log, on_prog=None):
    """只重擦有残留的镜头分段，其余分段原样流复制拼接；帧序/时长不变。

    首擦已按镜头写成 segments/shot_XXXX.h264，因此返修不再整片重编码：
    一个 suspect 只重写它所在的镜头（通常 0.5–5s），拼接用裸流 concat。
    """
    import cv2
    erased = Path(erased)
    out_path = Path(out_path)
    w, h, fps = meta["w"], meta["h"], meta["fps"]
    model = get_model(cfg)
    seg_dir = Path(erased).parent / "segments"
    if not seg_dir.is_dir():
        raise RuntimeError(f"缺少镜头分段目录 {seg_dir}（旧任务无法分段返修）")
    by_shot = {}
    for s in suspects:
        by_shot.setdefault(s["shot"], []).append(s["tr"])
    pad = max(int(cfg["mask"].get("mask_pad", 8)), 16)
    dilate = max(int(cfg["mask"].get("dilate", 7)), 20)
    expand = max(float(cfg["mask"].get("expand_ratio", 0.0)), 0.18)
    batch_size = max(1, int(cfg["inpaint"].get("batch", 4)))
    stab = float(cfg["inpaint"].get("temporal_ema", 0.0) or 0.0)
    backups, todo = [], sorted(by_shot)
    frames_total, patched_total = 0, 0
    try:
        for j, si in enumerate(todo):
            seg = seg_dir / f"shot_{si:04d}.h264"
            if not seg.exists():
                log(f"返修跳过镜头 {si}：分段不存在")
                continue
            trs = list({id(tr): tr for tr in by_shot[si]}.values())
            shot = shots[si]
            bak = seg.with_suffix(".orig.h264")
            shutil.copyfile(seg, bak)
            backups.append((seg, bak))
            tmp = seg.with_suffix(".tmp.h264")
            writer = P.open_writer(str(tmp), w, h, fps)
            stab_state = {"prev": None, "gray": None}
            buf_f, buf_m = [], []
            k, patched, rc = 0, 0, None

            def flush():
                nonlocal buf_f, buf_m
                if not buf_f:
                    return
                _lama_patch_batch(model, buf_f, buf_m, cfg, log)
                if stab > 0:
                    P.temporal_stabilize(buf_f, buf_m, stab_state, stab)
                for f in buf_f:
                    writer.stdin.write(f.tobytes())
                buf_f, buf_m = [], []

            try:
                for fr in P.iter_frames(str(seg), w, h):
                    t = shot["start"] + k / fps
                    mk = P.track_mask(fr, trs, t, 0, 0, "rect",
                                      pad=pad, dilate=dilate, expand=expand)
                    if mk is not None and int(mk.max()) > 0:
                        buf_f.append(fr)
                        buf_m.append(mk)
                        patched += 1
                        if len(buf_f) >= batch_size:
                            flush()
                    else:
                        flush()
                        writer.stdin.write(fr.tobytes())
                        if stab > 0:
                            stab_state["prev"] = fr
                            stab_state["gray"] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                    k += 1
                flush()
            finally:
                try:
                    writer.stdin.close()
                except Exception:  # noqa: BLE001
                    pass
                rc = writer.wait()
            if rc:
                raise RuntimeError(f"返修镜头 {si} 编码失败 rc={rc}")
            if k == 0:
                raise RuntimeError(f"返修镜头 {si} 解码 0 帧（分段损坏？）")
            tmp.replace(seg)
            frames_total += k
            patched_total += patched
            if on_prog:
                on_prog((j + 1) / max(len(todo), 1),
                        f"重擦镜头 {si} ({patched} 帧)")
        order = [seg_dir / f"shot_{i:04d}.h264" for i in range(len(shots))]
        _concat_segments(order, out_path, meta, log)
    except Exception:
        for seg, bak in backups:
            if bak.exists():
                shutil.copyfile(bak, seg)
                bak.unlink(missing_ok=True)
        raise
    else:
        for _seg, bak in backups:
            bak.unlink(missing_ok=True)
    return frames_total, patched_total


def auto_repair(erased, meta, layout, shots, per_shot, source, cfg, work, log,
                on_prog=None):
    """擦完自动复查 + 只返修有残留的镜头，最多 repair_rounds 轮；
    任何一轮失败都保留上一轮结果，不影响出片"""
    ercfg = cfg.get("erase") or {}
    max_repair = max(0, int(ercfg.get("repair_rounds", 2)))
    rounds, repaired, patched = [], [], 0
    only = None
    for r in range(max_repair + 1):
        scan = residual_scan(erased, meta, layout, shots, per_shot, source, log,
                             only_shots=only)
        rounds.append(_report_scan(scan))
        if not scan["suspects"] or r >= max_repair:
            break
        only = {s["shot"] for s in scan["suspects"]}
        if on_prog:
            on_prog((r + 0.2) / max_repair,
                    f"第 {r + 1} 轮返修 {len(scan['suspects'])} 处")
        out = work / f"erased_patch{r + 1}.mp4"
        rcfg = copy.deepcopy(cfg)
        rcfg.setdefault("mask", {})
        # 残留常是 OCR 漏掉的边缘字（如行尾“点”），返修框要比首擦更宽，
        # 否则同样的 ubox 会再次漏掉同一块像素。
        rcfg["mask"]["mask_pad"] = max(int(rcfg["mask"].get("mask_pad", 8)), 24)
        rcfg["mask"]["dilate"] = max(int(rcfg["mask"].get("dilate", 7)), 20)
        rcfg["mask"]["expand_ratio"] = max(
            float(rcfg["mask"].get("expand_ratio", 0.0)), 0.35)
        rcfg.setdefault("inpaint", {})
        rcfg["inpaint"]["fixed_w"] = max(int(rcfg["inpaint"].get("fixed_w", 1024)), 1024)
        rcfg["inpaint"]["fixed_h"] = 256
        # 返修这一轮不用邻帧补洞：补洞依赖邻居的真实像素，遇到描边/阴影会把残影再贴回来，
        # 返修就该用最稳的纯模型重擦（掩码已经放大了）。
        rcfg.setdefault("erase", {})["tf_radius"] = 0
        try:
            _n, pf = patch_erased(
                erased, out, meta, shots, per_shot, scan["suspects"], layout, rcfg,
                log, on_prog=(lambda f, _msg="": on_prog(
                    (r + 0.2 + 0.8 * f) / max_repair, "重擦残留镜头")) if on_prog else None)
        except Exception as e:  # noqa: BLE001
            log(f"返修失败，保留上一轮结果：{type(e).__name__} {e}")
            out.unlink(missing_ok=True)
            break
        os.replace(out, erased)
        patched += pf
        repaired += [s["shot"] for s in scan["suspects"]]
    return {"enabled": True, "rounds": rounds,
            "suspects_before": rounds[0]["suspects"] if rounds else 0,
            "suspects_after": rounds[-1]["suspects"] if rounds else 0,
            "repaired_shots": sorted(set(repaired)),
            "patched_frames": patched}


# ---------------- 翻译 ----------------

LANG_ALIAS = {"zh": "zh", "cn": "zh", "en": "en", "ja": "ja", "ko": "ko", "es": "es",
              "pt": "pt", "ar": "ar", "ru": "ru", "th": "th", "vi": "vi",
              "id": "id", "fr": "fr", "de": "de"}


def pick_translator(preferred=None):
    """按顺序探测可用的 OpenAI 兼容翻译后端"""
    cands = []
    env_base = __import__("os").environ.get("TRANSLATE_BASE_URL")
    if preferred:
        cands.append(preferred)
    if env_base:
        cands.append({"base_url": env_base,
                      "model": __import__("os").environ.get("TRANSLATE_MODEL", ""),
                      "api_key": __import__("os").environ.get("TRANSLATE_API_KEY", ""),})
    # 本地翻译服务（Hy-MT2 翻译专用模型，llama-server :1509）
    cands += [
        {"base_url": "http://127.0.0.1:1509/v1", "model": "Hy-MT2-7B-Q4_K_M.gguf",
         "api_key": "local", "kind": "plain"},
    ]
    for c in cands:
        try:
            key = c.get("api_key") or ""
            if not key and c.get("api_key_file"):
                key = P.load_translate_key({"translate": c})
            if not key:
                continue
            if probe_backend(c["base_url"], key, c["model"]):
                return dict(c, api_key=key)
        except Exception:  # noqa: BLE001
            continue
    return None


def probe_backend(base, key, model):
    import urllib.request
    body = json.dumps({"model": model, "max_tokens": 8,
                       "messages": [{"role": "user", "content": "ping"}]}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                headers={"Content-Type": "application/json",
                                         "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


LANG_EN = {"zh": "Simplified Chinese", "en": "English", "ja": "Japanese",
           "ko": "Korean", "es": "Spanish", "pt": "Portuguese", "ar": "Arabic",
           "ru": "Russian", "th": "Thai", "vi": "Vietnamese", "id": "Indonesian",
           "fr": "French", "de": "German"}

PY_PROMPT = {
    "zh": "把下面的影视字幕翻译成自然、口语化的简体中文，保留人物称呼和语气，只输出译文：\n\n{text}",
    "ja": "把下面的影视字幕翻译成自然、口语化的日语，保留人物称呼和语气，"
          "同一部剧的人称和语气保持一致。每条译文尽量控制在 24 个字符以内，"
          "能一行显示就一行。只输出译文：\n\n{text}",
    "en": "Translate the following subtitle into natural, conversational English. "
          "Keep names and tone. Output only the translation:\n\n{text}",
}


def translate(cues, target, log, backend=None):
    lang = LANG_ALIAS.get(target, target)
    be = backend or pick_translator()
    if not be:
        raise RuntimeError("没有可用的翻译后端（1509 / 8790 都不通）")
    log(f"翻译后端：{be['base_url']} · {be['model']} → {lang}")
    kind = be.get("kind", "json")
    tpl = PY_PROMPT.get(lang) or (
        "Translate the following text into {name}. Output only the translation, "
        "no explanation:\n\n{text}").format(name=LANG_EN.get(lang, lang), text="{text}")
    tcfg = {
        "target": lang,
        "style": "plain" if kind == "plain" else "json",
        "plain_prompt": tpl,
        "base_url": be["base_url"],
        "model": be["model"],
        "models": [be["model"]],
        "api_key": be["api_key"],
        "batch": 8,
        "concurrency": 2,
        "min_gap": 0,
        "timeout": 180,
        "retries": 3,
        "temperature": 0.2,
        "max_tokens": 4096,
        "max_chars": 26,
    }
    cfg = {"translate": tcfg}
    out = P.llm_translate(cues, cfg, be["base_url"], be["api_key"], be["model"])
    return out, lang


# ---------------- 输出 ----------------

def font_scale(lang, source):
    """源文字高 -> 目标字号：中文字形接近满格，拉丁字形较矮/较宽，不能都乘 1.15"""
    tgt_cjk = lang in LANG_CJK
    src_cjk = source in LANG_CJK
    if tgt_cjk:
        return 1.0 if src_cjk else 1.15
    return 0.85 if src_cjk else 1.05


def subtitle_metrics(layout, lang, source, screen_w=None):
    script = "cjk" if lang in LANG_CJK else "latin"
    font = max(12, int(round(layout["font_h"] * font_scale(lang, source))))
    line_w = int(layout["line_w"])
    if lang == "ja":
        # 字号保持与原字幕一致；只放宽行宽到 12 字，避免把日语词硬拆开。
        # 行宽仍限制在左右 60px 安全边距内。
        if screen_w:
            line_w = max(line_w, int(font * 12))
            line_w = min(line_w, int(screen_w) - 120)
    cw = CHAR_W[script] * font
    max_chars = max(8, int(line_w / max(cw, 1)))
    if lang == "ja":
        max_chars = min(max_chars, 18)
    return font, max_chars


def _sentence_units(text, cap):
    """按句读切段；单段超长再按词/字切。避免把半句话甩到下一条 cue"""
    parts = [p.strip() for p in re.split(r"(?<=[.!?;:。！？；：])\s*", text) if p.strip()]
    if CJK_RE.search(text) or KANA_HANGUL.search(text):
        parts = [q for p in parts for q in re.split(r"(?<=[、，,])", p) if q.strip()]
    units = []
    for p in parts:
        if len(p) <= cap:
            units.append(p)
            continue
        if (CJK_RE.search(p) or KANA_HANGUL.search(p)) and " " not in p:
            units.extend(p[i:i + cap] for i in range(0, len(p), cap))
            continue
        cur = ""
        for w in p.split():
            cand = (cur + " " + w).strip()
            if cur and len(cand) > cap:
                units.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            units.append(cur)
    return units


def fit_rows(c, text, font, max_chars, layout, lang):
    """整句优先；放不下先缩字号；再按句读拆成连续 cue，绝不截断"""
    start, end = float(c["start"]), float(c["end"])
    dur = max(0.1, end - start)
    lines = P.wrap_lines(text, max_chars)
    if len(lines) <= 2:
        return [(start, end, "\\N".join(lines), None)]
    cw = CHAR_W["cjk" if lang in LANG_CJK else "latin"]
    for scale in (0.92, 0.85, 0.78, 0.70):
        f = max(12, int(font * scale))
        mc = max(8, int(layout["line_w"] / max(cw * f, 1)))
        ls = P.wrap_lines(text, mc)
        if len(ls) <= 2:
            return [(start, end, "\\N".join(ls), f)]
    cap = max_chars * 2
    cjk_join = lang in LANG_CJK
    chunks, cur = [], ""
    for u in _sentence_units(text, cap):
        if cjk_join and cur and u and not (cur[-1].isascii() and u[0].isascii()):
            cand = cur + u
        else:
            cand = (cur + " " + u).strip()
        if cur and len(cand) > cap:
            chunks.append(cur)
            cur = u
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    if not chunks:
        chunks = ["\\N".join(lines[i:i + 2]) for i in range(0, len(lines), 2)]
    chunks = ["\\N".join(P.wrap_lines(ch, max_chars)) for ch in chunks]
    weights = [max(1, len(re.sub(r"\s", "", ch))) for ch in chunks]
    total = sum(weights)
    rows, t = [], start
    for ch, wt in zip(chunks, weights):
        dt = dur * wt / total
        rows.append((t, min(end, t + dt), ch, None))
        t += dt
    if rows:
        rows[-1] = (rows[-1][0], end, rows[-1][2], rows[-1][3])
    return rows


LANG_FONT = {"zh": "Noto Sans CJK SC", "ja": "Noto Sans CJK JP",
             "ko": "Noto Sans CJK KR", "ar": "Noto Sans Arabic",
             "th": "Noto Sans Thai"}


def os_font(lang):
    name = LANG_FONT.get(lang, "Noto Sans")
    try:
        out = subprocess.run(["fc-match", name], capture_output=True, timeout=5).stdout.decode()
        if "DejaVu" in out and name != "Noto Sans":
            return "Noto Sans"
    except Exception:  # noqa: BLE001
        pass
    return name


def build_ass(cues, out_ass, meta, layout, lang, targets, source="auto"):
    font, max_chars = subtitle_metrics(layout, lang, source, meta["w"])
    fit_layout = dict(layout)
    if lang == "ja":
        fit_layout["line_w"] = min(meta["w"] - 120,
                                   max(int(layout["line_w"]), int(font * 12)))
    rows = []
    for c, t in zip(cues, targets):
        txt = (t or "").strip()
        if not txt:
            continue
        for a, b, body, fs in fit_rows(c, txt, font, max_chars, fit_layout, lang):
            rows.append({"start": a, "end": b, "body": body, "fs": fs,
                         "box": c.get("box")})
    # libass 会把时间重叠的两条事件上下错开；先裁掉重叠，再用 \pos 固定到原字幕位置
    rows.sort(key=lambda r: (r["start"], r["end"]))
    for i in range(len(rows) - 1):
        if rows[i]["end"] > rows[i + 1]["start"]:
            rows[i]["end"] = max(rows[i]["start"] + 0.05,
                                 rows[i + 1]["start"] - 0.02)
    default_bottom = int(meta["h"]) - int(layout.get("margin_v", 0))
    for r in rows:
        box = r.get("box")
        if box and len(box) == 4:
            cx = int(round((float(box[0]) + float(box[2])) / 2))
            yb = int(round(float(box[3])))
        else:
            cx = int(meta["w"]) // 2
            yb = default_bottom
        cx = max(font, min(int(meta["w"]) - font, cx))
        yb = max(font + 2, min(int(meta["h"]) - 2, yb))
        tag = f"\\an2\\pos({cx},{yb})"
        if r["fs"]:
            tag += f"\\fs{r['fs']}"
        r["en"] = "{" + tag + "}" + r["body"]
    text = P.ASS_HEAD.format(w=meta["w"], h=meta["h"], font=os_font(lang),
                             size=font, mv=layout["margin_v"])
    align = layout.get("align", 2)
    head_lines = text.splitlines()
    for i, ln in enumerate(head_lines):
        if ln.startswith("Style: EN,"):
            f = ln.split(",")
            f[18] = str(align)          # Alignment
            head_lines[i] = ",".join(f)
    text = "\n".join(head_lines) + "\n"
    body = []
    for r in rows:
        body.append(f"Dialogue: 0,{P.ts(r['start'])},{P.ts(r['end'])},EN,,0,0,0,,{r['en']}\n")
    Path(out_ass).write_text(text + "".join(body), encoding="utf-8")
    return font, max_chars


def burn(video: Path, ass: Path, out: Path, log, audio_src: Path | None = None):
    """压制字幕。erased.mp4 是纯视频，音轨从原始/归一化素材单独取。"""
    src_audio = audio_src if audio_src and has_audio(audio_src) else None
    if src_audio is None and has_audio(video):
        src_audio = video
    base = ["ffmpeg", "-v", "error", "-y", "-i", str(video)]
    if src_audio is not None and src_audio != video:
        base += ["-i", str(src_audio)]
    if src_audio is not None:
        audio = ["-map", "0:v:0",
                 "-map", "1:a:0" if src_audio != video else "0:a:0",
                 "-c:a", "copy"]
        # 不用 -shortest：音轨比画面短时会把尾部画面一起切掉（实测 5ed 少 4 帧）。
        # 音轨更长时用 -t 卡到画面时长，保证不掉帧也不留长尾。
        try:
            vm = P.probe(str(video))
            if P.probe(str(src_audio))["dur"] > vm["dur"] + 0.02:
                audio += ["-t", f"{vm['dur'] + 0.5 / vm['fps']:.3f}"]
        except Exception:  # noqa: BLE001
            pass
    else:
        audio = ["-an"]

    def cmd_for(venc):
        return base + ["-vf", f"ass={ass}"] + venc + ["-pix_fmt", "yuv420p"] \
            + audio + ["-movflags", "+faststart", str(out)]

    # 长片逐镜头跑完后 torch 缓存可能吃到几 GB，NVENC 会因显存不足开不了；
    # 先归还缓存，再失败时退 CPU x264，避免整支视频在最后一秒失败。
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    try:
        subprocess.run(cmd_for(["-c:v", "h264_nvenc", "-preset", "p5",
                                "-cq", "20", "-b:v", "0"]), check=True)
    except subprocess.CalledProcessError as e:
        log(f"NVENC 压制失败（exit {e.returncode}），改用 libx264")
        Path(out).unlink(missing_ok=True)
        x264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        th = os.environ.get("ERASE_X264_THREADS")
        if th:
            x264 += ["-threads", str(max(1, int(th)))]
        subprocess.run(cmd_for(x264), check=True)
    log(f"压制完成 {time.time()-t0:.1f}s")


def font_ok(lang):
    """字体族是否真的装了；缺字体时 libass 会静默换字体或不出字。"""
    name = LANG_FONT.get(lang, "Noto Sans")
    try:
        fam = subprocess.check_output(
            ["fc-match", "-f", "%{family}", name], text=True, timeout=5).strip()
    except Exception:  # noqa: BLE001
        return True, name, "fc-match 不可用"
    return (name.lower() in fam.lower()), name, fam


def verify_burn(out, erased, state, log):
    """压制后验收：帧数/时长/音轨/ASS 事件/字体/字幕带是否真的变了。

    返回 report 里的 render_check；任何一项硬失败直接抛错，不让坏片静默交付。
    """
    import cv2
    out, erased = Path(out), Path(erased)
    meta = state["meta"]
    cues = state.get("cues") or []
    targets = state.get("targets") or []
    pairs = [(c, t) for c, t in zip(cues, targets) if (t or "").strip()]
    ass = Path(state["ass"])
    events = sum(1 for ln in ass.read_text(encoding="utf-8").splitlines()
                 if ln.startswith("Dialogue:")) if ass.exists() else 0
    if not events:
        raise RuntimeError("ASS 里没有可渲染的字幕事件，拒绝出片")

    exp = _stream_frames(erased)
    got = _stream_frames(out)
    if exp is not None and got is not None and abs(got - exp) > 1:
        raise RuntimeError(f"压制帧数 {got} != 擦除源 {exp}（±1），拒绝出片")
    dur_in = P.probe(str(erased))["dur"]
    dur_out = P.probe(str(out))["dur"]
    if abs(dur_out - dur_in) > 0.2:
        raise RuntimeError(f"压制时长 {dur_out:.2f}s != 擦除源 {dur_in:.2f}s，拒绝出片")
    src_audio = has_audio(state.get("src") or "")
    out_audio = has_audio(out)
    if src_audio and not out_audio:
        raise RuntimeError("源有音轨但成片没有，拒绝出片")

    ok, font_name, font_fam = font_ok(state.get("target_lang") or state.get("target") or "en")
    if not ok:
        log(f"警告：字体族 {font_name} 未精确命中（实际 {font_fam}），"
            f"继续用渲染检查判定")

    # 抽最长的 3 条字幕中点，比较字幕带；白字差异像素应远超压缩噪声
    w, h = int(meta["w"]), int(meta["h"])
    band = state["layout"]["band"]
    fb0 = int(h * band[0]) & ~1
    fb1 = int(h * band[1]) & ~1
    ch = max(2, fb1 - fb0)
    picks = sorted(pairs, key=lambda p: p[0]["end"] - p[0]["start"], reverse=True)[:3]
    if not picks:
        raise RuntimeError("没有已翻译的字幕，拒绝出片")
    hits = []
    for c, _t in picks:
        t = (float(c["start"]) + float(c["end"])) / 2
        a = next(P.iter_frames(str(erased), w, ch, t, 0.2, None,
                               crop=(0, fb0, w, ch)))
        b = next(P.iter_frames(str(out), w, ch, t, 0.2, None,
                               crop=(0, fb0, w, ch)))
        d = cv2.absdiff(a, b).max(axis=2)
        hits.append(int((d > 40).sum()))
    if max(hits) < 200:
        raise RuntimeError(f"压制后字幕带没有明显变化（差异像素 {hits}），"
                           f"字幕可能没渲染")
    log(f"压制验收：{got} 帧 / {dur_out:.2f}s / 字幕带差异像素 {hits}")
    return {"ok": True, "frames": got, "duration": round(dur_out, 2),
            "audio": bool(out_audio), "ass_events": events,
            "font": font_name, "font_match": bool(ok), "band_hits": hits}


# ---------------- 主流程 ----------------

def _json_default(o):
    try:
        return float(o)
    except Exception:  # noqa: BLE001
        return str(o)


def _save_state(job_dir, state):
    (Path(job_dir) / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=1, default=_json_default),
        encoding="utf-8")


def _load_state(job_dir):
    return json.loads((Path(job_dir) / "state.json").read_text(encoding="utf-8"))


def prepare(job_dir: Path, opts: dict, log, on_stage=None):
    """OCR/擦除/返修/QC，产出 erased.mp4 + state.json；不翻译、不压制。"""
    job_dir = Path(job_dir)
    src_in = job_dir / "input.mp4"
    work = job_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    cfg = dict(opts.get("cfg") or {})
    t_start = time.time()

    def stage(name, frac=0.0):
        if on_stage:
            on_stage(name, frac)

    stage("准备素材", 0.02)
    src = normalize(src_in, work, log)
    meta = P.probe(str(src))
    log(f"素材 {meta['w']}x{meta['h']} @{meta['fps']:.3g}fps {meta['dur']:.1f}s")

    # OCR 参数要先落地：版面探测自己就会建会话，晚设置等于不生效
    _ocr_cfg = cfg.get("ocr") or {}
    P.set_ocr_opts(
        det_limit_side_len=int(os.environ.get("VE_OCR_DET_LEN")
                               or _ocr_cfg.get("det_limit_side_len") or 0) or None,
        det_limit_type=os.environ.get("VE_OCR_DET_TYPE")
        or _ocr_cfg.get("det_limit_type"))

    stage("探测字幕位置", 0.06)
    layout = detect_layout(src, meta, log)

    source = (opts.get("source") or "auto").lower()
    stage("读取字幕", 0.1)
    cues, source, shots, per_shot = extract(
        src, meta, layout, source, log,
        on_prog=lambda f: stage("读取字幕", 0.1 + 0.15 * f),
        ocr=cfg.get("ocr"))
    if not cues:
        raise RuntimeError("素材里没找到字幕，无法确定要擦除/翻译的内容")

    stage("擦除原字幕", 0.26)
    # 竖屏 1080 宽字幕带用 768 画布：实测擦除快 ~13%，肉眼无差
    if meta["h"] > meta["w"] and meta["w"] <= 1200:
        inp_cfg = dict(cfg.get("inpaint") or {})
        inp_cfg["fixed_w"] = min(int(inp_cfg.get("fixed_w", 1024)), 768)
        cfg["inpaint"] = inp_cfg
    # TorchScript 后端不支持 half 输入（ComplexHalf NaN）：显式关掉，避免手滑改
    if str((cfg.get("inpaint") or {}).get("backend", "native")).lower() != "native" \
            and bool(cfg["inpaint"].get("fp16")):
        cfg["inpaint"]["fp16"] = False
        log("TorchScript 后端不使用 fp16（native 才支持），已自动关闭")
    mask_mode = opts.get("mask_mode") or cfg["mask"].get("mode", "stroke")
    provider, quality = resolve_erase(cfg, opts)
    if provider == "cloud":
        erased = work / "erased.mp4"
        erase_info = E.erase(
            src, erased, cfg, log=log, meta=meta, layout=layout, quality=quality,
            on_prog=lambda f, msg="": stage(
                "云端擦除" + (f" · {msg}" if msg else ""), 0.26 + 0.48 * f))
    else:
        erased = erase(src, meta, layout, shots, per_shot, cfg,
                       on_prog=lambda f, msg="": stage("擦除原字幕", 0.26 + 0.5 * f),
                       log=log, mask_mode=mask_mode, cues=cues)
        erase_info = {"provider": "local", "mask_mode": mask_mode}
        if bool((cfg.get("erase") or {}).get("auto_repair", True)):
            try:
                erase_info["repair"] = auto_repair(
                    erased, meta, layout, shots, per_shot, source, cfg, work, log,
                    on_prog=lambda f, msg="": stage(
                        "返修残留" + (f" · {msg}" if msg else ""), 0.76 + 0.02 * f))
            except Exception as e:  # noqa: BLE001
                log(f"残留返修异常，保留首次擦除结果：{type(e).__name__} {e}")
                erase_info["repair"] = {"enabled": True,
                                        "error": f"{type(e).__name__}: {e}"}
    qc = qc_frames(erased, meta, layout, cues, source, work, log)

    state = {
        "job": str(job_dir),
        "src": str(src),
        "erased": str(erased),
        "source": source,
        "target": (opts.get("target") or "en").lower(),
        "backend": opts.get("backend"),
        "mask_mode": mask_mode if provider == "local" else "cloud",
        "meta": meta,
        "layout": layout,
        "cues": cues,
        "erase": erase_info,
        "qc": qc,
        "seconds_prepare": round(time.time() - t_start, 1),
        "resolution": f"{meta['w']}x{meta['h']}",
        "fps": meta["fps"],
        "duration": round(meta["dur"], 2),
    }
    _save_state(job_dir, state)
    stage("擦除完成", 0.78)
    return state


def translate_state(job_dir: Path, opts: dict, log, on_stage=None):
    """读 state.json，翻译并排版，写回 targets/font/ass。"""
    job_dir = Path(job_dir)
    state = _load_state(job_dir)
    t_start = time.time()
    if on_stage:
        on_stage("翻译", 0.8)

    target = (opts.get("target") or state.get("target") or "en").lower()
    targets, lang = translate(state["cues"], target, log,
                              backend=opts.get("backend") or state.get("backend"))

    if on_stage:
        on_stage("排版压制", 0.9)
    ass = job_dir / "work" / "sub.ass"
    font, max_chars = build_ass(state["cues"], ass, state["meta"], state["layout"],
                                lang, targets, source=state["source"])
    log(f"字幕样式：字号 {font}px "
        f"{'顶部' if state['layout'].get('align') == 8 else '底部'} "
        f"边距 {state['layout']['margin_v']}px 每行 {max_chars} 字")

    state.update({"target_lang": lang, "targets": targets, "ass": str(ass),
                  "font_size": font, "max_chars": max_chars,
                  "seconds_translate": round(time.time() - t_start, 1)})
    _save_state(job_dir, state)
    return state


def burn_state(job_dir: Path, log, on_stage=None):
    """读 state.json，压字幕并写 report.json。"""
    job_dir = Path(job_dir)
    state = _load_state(job_dir)
    if on_stage:
        on_stage("压制", 0.95)
    t0 = time.time()
    out = job_dir / "output.mp4"
    (job_dir / "report.json").unlink(missing_ok=True)
    burn(Path(state["erased"]), Path(state["ass"]), out, log,
         audio_src=Path(state["src"]))
    burn_s = round(time.time() - t0, 1)
    render = verify_burn(out, state["erased"], state, log)
    if on_stage:
        on_stage("完成", 1.0)

    total = round(state.get("seconds_prepare", 0)
                  + state.get("seconds_translate", 0) + burn_s, 1)
    report = {
        "source_lang": state["source"],
        "target_lang": state["target_lang"],
        "cues": len(state["cues"]),
        "translated": sum(1 for t in state["targets"] if (t or "").strip()),
        "layout": state["layout"],
        "font_size": state["font_size"],
        "max_chars": state["max_chars"],
        "mask_mode": state["mask_mode"],
        "erase": state["erase"],
        "qc": state["qc"],
        "seconds": total,
        "seconds_prepare": state.get("seconds_prepare"),
        "seconds_translate": state.get("seconds_translate"),
        "seconds_burn": burn_s,
        "resolution": state["resolution"],
        "fps": state["fps"],
        "duration": state["duration"],
        "audio": has_audio(out),
        "render_check": render,
        "pairs": [[round(c["start"], 2), round(c["end"], 2), c["text"], t]
                  for c, t in zip(state["cues"], state["targets"])][:400],
    }
    (job_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return out, report


def convert(job_dir: Path, opts: dict, log, on_stage=None):
    """opts: {source: auto|zh|..., target: zh|en|..., config: path}"""
    prepare(job_dir, opts, log, on_stage)
    translate_state(job_dir, opts, log, on_stage)
    return burn_state(job_dir, log, on_stage)
