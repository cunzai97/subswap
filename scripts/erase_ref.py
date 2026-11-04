#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参考帧优先的字幕擦除引擎（bench v6 的生产移植）。

路由：
  1. 逐帧检测真实白色字幕（按画面像素，不按 cue 时间）——无字帧原样保留；
  2. 有字帧按连续段分组，每段取最近的“无字帧”当参考；
  3. DIS 光流把参考帧 warp 到当前帧；字幕区内的光流先用 TELEA 补齐，避免白字污染流场；
  4. 参考帧与当前帧做光度校验，通过则用真实像素填充；失败或无参考时交给模型兜底；
  5. 只对模型兜底的帧做时域稳定，参考填充的帧不动（真实像素不需要平滑）。

模块通过参数拿到 pipeline 模块对象，避免 CLI 以 __main__ 运行时产生第二份模块状态。
"""
from __future__ import annotations

import tempfile
import time

import cv2
import numpy as np
import torch

REF_PHOTO = 12.0
REF_FEATHER = 1.0
MODEL_FEATHER = 2.5
MODEL_STAB = 0.45

# 第三档（ProPainter）默认参数：阈值沿用 bench/video_inpaint/HYBRID_ROUTING.md
PP_TEX = 16.0          # 背景 Laplacian 方差中位数 >= 16 → 难
PP_MOTION = 2.5        # 相邻帧灰度差中位数 >= 2.5 → 难
PP_GUARD_MEAN = 20.0   # PP 与 LaMa 掩码内平均差 > 20 → 判 PP 幻觉，回退 LaMa
PP_GUARD_P90 = 85.0    # 或 p90 > 85 → 回退
PP_MAX_FRAMES = 900    # 单镜头最多送给 PP 的帧数（防止一镜到底的片子把耗时打爆）


def tier_cfg(cfg):
    """读 erase.tiers 配置；ERASE_TIERS=0/off 可全局关掉第三档。"""
    import os
    ecfg = cfg.get("erase") or {}
    on = bool(ecfg.get("tiers", False))
    env = os.environ.get("ERASE_TIERS", "").strip().lower()
    if env in ("0", "off", "false", "no"):
        on = False
    elif env in ("1", "on", "true", "yes"):
        on = True
    def _env(name, dflt):
        v = os.environ.get(name)
        return float(v) if v not in (None, "") else dflt

    return {
        "on": on,
        "tex": _env("VE_PP_TEX", float(ecfg.get("pp_tex", PP_TEX))),
        "motion": _env("VE_PP_MOTION", float(ecfg.get("pp_motion", PP_MOTION))),
        "guard_mean": float(ecfg.get("pp_guard_mean", PP_GUARD_MEAN)),
        "guard_p90": float(ecfg.get("pp_guard_p90", PP_GUARD_P90)),
        "max_frames": int(ecfg.get("pp_max_frames", PP_MAX_FRAMES)),
        "frame_motion": float(ecfg.get("pp_frame_motion", PP_MOTION)),
        "chunk": int(ecfg.get("pp_chunk", 80)),
        "subvideo": int(ecfg.get("pp_subvideo", 80)),
        "neighbor": int(ecfg.get("pp_neighbor", 10)),
        "scale": float(ecfg.get("pp_scale", 1.0)),
        "vpad": float(ecfg.get("pp_vpad", 0.8)),
    }


def _pp_rect(tracks, w, h, fh, vpad, clip=None):
    """PP 只吃字幕带本身（不需要参考填充那么大的上下文），显存和耗时都省一半。"""
    ub = [tr["ubox"] for tr in tracks if tr.get("ubox")]
    if not ub:
        return None
    tx0 = min(b[0] for b in ub)
    ty0 = min(b[1] for b in ub)
    tx1 = max(b[2] for b in ub)
    ty1 = max(b[3] for b in ub)
    pad_y = max(24, int(round(vpad * fh)))
    pad_x = max(24, int(round(0.25 * (tx1 - tx0) * 0.1)))
    px0 = max(0, int(tx0) - pad_x) & ~1
    py0 = max(0, int(ty0) - pad_y) & ~1
    px1 = min(w, int(tx1) + pad_x + 1) & ~1
    py1 = min(h, int(ty1) + pad_y + 1) & ~1
    if clip is not None:
        # 必须落在引擎已解码的 band crop 里，否则后面按坐标切片会错位
        cx0, cy0, cx1, cy1 = clip
        px0, py0 = max(px0, cx0), max(py0, cy0)
        px1, py1 = min(px1, cx1), min(py1, cy1)
    # PP 要求 8 的倍数（内部再裁一次，这里直接对齐，避免二次重采样）
    px1 = px0 + ((px1 - px0) // 8) * 8
    py1 = py0 + ((py1 - py0) // 8) * 8
    if px1 - px0 < 64 or py1 - py0 < 32:
        return None
    return int(px0), int(py0), int(px1), int(py1)


def _med(vals, default=0.0):
    return float(np.median(vals)) if vals else default


def _runs_of(ids):
    out = []
    for i in ids:
        if out and i == out[-1][-1] + 1:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _make_dis():
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def _flow_to(ft, fr, dis, s=0.5):
    h, w = ft.shape[:2]
    sw, sh = max(16, int(w * s)), max(16, int(h * s))
    gt = cv2.resize(cv2.cvtColor(ft, cv2.COLOR_BGR2GRAY), (sw, sh),
                    interpolation=cv2.INTER_AREA)
    gr = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (sw, sh),
                    interpolation=cv2.INTER_AREA)
    fl = dis.calc(gt, gr, None)          # current -> ref, half-res pixels
    return fl, sw, sh


def _warp_with(a, fl, s=0.5):
    h, w = a.shape[:2]
    F = cv2.resize(fl, (w, h), interpolation=cv2.INTER_LINEAR) / s
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(a, gx + F[..., 0], gy + F[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _blend(out, alt, mask, sigma):
    if not mask.any():
        return out
    alpha = cv2.GaussianBlur((mask * 255).astype(np.uint8), (0, 0), sigma)
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    return np.clip(alt.astype(np.float32) * a + out.astype(np.float32) * (1 - a),
                   0, 255).round().astype(np.uint8)


def _ring_of(mask, k=21):
    ring = ~cv2.dilate(mask.astype(np.uint8),
                       np.ones((k, k), np.uint8)).astype(bool)
    if ring.sum() < 500:
        ring = ~mask
    return ring


def _photometric(wr, ft, ring):
    if ring.sum() < 50:
        return 1e9
    return float(np.abs(wr[ring].astype(np.int16)
                        - ft[ring].astype(np.int16)).mean())


def _glyph_mask(frame, zone, params):
    """在字幕搜索区里找白色字形组件；返回 (mask, component_px, merged, bbox)。

    merged=1 表示有白字和画面里的大块白色物体（白衣服/墙）粘成一个组件、
    被尺寸门限丢掉；bbox 是保留下来的字形的并集框（搜索区坐标，None=无）。
    是否整行兜底由调用方看“框相对台词行框是否缺了一截”决定——白背景片
    里大亮块很常见，但字都找到了就不该无脑扩大掩码。
    """
    b, g, r = cv2.split(frame)
    cand = ((b > params["white_min"]) & (g > params["white_min"])
            & (r > params["white_min"])
            & ((r.astype(np.int16) - b.astype(np.int16)) < params["white_sat"])
            ).astype(np.uint8)
    z = np.zeros_like(cand)
    z[zone[1]:zone[3], zone[0]:zone[2]] = 1
    cand &= z
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    keep = np.zeros_like(cand)
    total = 0
    merged = 0
    xs0 = ys0 = 10 ** 9
    xs1 = ys1 = -1
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < params["min_area"] or h < params["min_h"]:
            continue
        if h <= params["max_h"] and w <= params["max_w"]:
            keep[lab == i] = 1
            total += int(a)
            xs0, ys0 = min(xs0, x), min(ys0, y)
            xs1, ys1 = max(xs1, x + w), max(ys1, y + h)
        else:
            # 白字贴着白衣服时两者连成一个超尺寸亮块：整块被门限拒绝，
            # 首字会残留。只置标记，是否兜底由调用方按行框缺口决定。
            merged = 1
    bbox = (xs0, ys0, xs1, ys1) if total > 0 else None
    if total > 0:
        # 保留字形并集框内的白色像素，确保首尾字/标点不漏；
        # 调用方会用 OCR 行框把这里限制在原字幕范围内。
        px, py = params["pad_x"], params["pad_y"]
        bx0, by0 = max(0, xs0 - px), max(0, ys0 - py)
        bx1, by1 = min(cand.shape[1], xs1 + px), min(cand.shape[0], ys1 + py)
        keep[by0:by1, bx0:bx1] = cand[by0:by1, bx0:bx1]
    return keep, total, merged, bbox


def _active_limit(P, h, w, tracks, t, x0, y0, pad, cues=None):
    """当前时刻活跃字幕的 OCR 行框（局部坐标，外扩 pad）。

    优先用 merge 后的 cue 行框：原始 tracks 的口型时间更宽，会把下一条
    cue 的框（甚至白衣服/徽章）并进来，导致整条横向被抹平。
    """
    lim = np.zeros((h, w), np.uint8)

    def add_box(bb):
        # OCR 行框常把首尾字裁掉一点；左右各放宽 12%，再叠 pad。
        ew = int(round((int(bb[2]) - int(bb[0])) * 0.12))
        bx0 = max(0, int(bb[0]) - x0 - pad - ew)
        by0 = max(0, int(bb[1]) - y0 - pad)
        bx1 = min(w, int(bb[2]) - x0 + pad + ew)
        by1 = min(h, int(bb[3]) - y0 + pad)
        if bx1 > bx0 and by1 > by0:
            lim[by0:by1, bx0:bx1] = 255

    if cues:
        for c in cues:
            if c.get("box") and c["start"] - 0.08 <= t <= c["end"] + 0.08:
                add_box(c["box"])
        if lim.any():
            return lim
    for tr in tracks:
        if not (tr["start"] - 0.08 <= t <= tr["end"] + 0.08):
            continue
        add_box(tr.get("ubox") or P._box_at(tr, t))
    return lim


def _lama_fill(P, model, bands, masks, cfg, amp, feather):
    """固定 1024x256（或 fit_canvas）画布上跑模型，只回填掩码内像素。"""
    if not bands:
        return bands
    sw = int(cfg["inpaint"].get("fixed_w", 1024))
    sh = int(cfg["inpaint"].get("fixed_h", 256))
    bh, bw = bands[0].shape[:2]
    fit_canvas = bool(cfg["inpaint"].get("fit_canvas", False))
    fit = min(sw / bw, sh / bh, 1.0)
    fit_w = max(2, int(round(bw * fit)))
    fit_h = max(2, int(round(bh * fit)))
    cw = min(sw, (fit_w + 7) // 8 * 8) if fit_canvas else sw
    ch = min(sh, (fit_h + 7) // 8 * 8) if fit_canvas else sh
    cw, ch = max(8, cw), max(8, ch)
    cpx, cpy = (cw - fit_w) // 2, (ch - fit_h) // 2
    bs = max(1, int(cfg["inpaint"].get("batch", 8)))
    cast_half = bool(cfg["inpaint"].get("fp16")) and not amp

    n_real = len(bands)
    pad = max(0, bs - n_real)
    items = list(zip(bands, masks))
    items += [(bands[-1], np.zeros_like(masks[-1]))] * pad
    canv, mcanv = [], []
    interp = cv2.INTER_AREA if fit < 1.0 else cv2.INTER_LINEAR
    for b, m in items:
        im = cv2.resize(b, (fit_w, fit_h), interpolation=interp)
        mm = cv2.resize((m > 0).astype(np.uint8) * 255, (fit_w, fit_h),
                        interpolation=cv2.INTER_NEAREST)
        canv.append(cv2.copyMakeBorder(im, cpy, ch - fit_h - cpy, cpx,
                                       cw - fit_w - cpx, cv2.BORDER_REPLICATE))
        mcanv.append(cv2.copyMakeBorder(mm, cpy, ch - fit_h - cpy, cpx,
                                        cw - fit_w - cpx, cv2.BORDER_CONSTANT,
                                        value=0))
    img = torch.from_numpy(np.stack(canv)[:, :, :, ::-1].copy()).permute(0, 3, 1, 2)
    img = img.float().div_(255).cuda()
    if cast_half:
        img = img.half()
    msk_t = torch.from_numpy((np.stack(mcanv) > 0).astype(np.float32)).cuda()[:, None]
    if cast_half:
        msk_t = msk_t.half()
    with torch.inference_mode():
        if amp:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(img, msk_t)
        else:
            out = model(img, msk_t)
    torch.cuda.synchronize()
    out = out.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    finite = np.isfinite(out).all(axis=3)
    if not bool(finite.all()) and amp:
        # 实测大矩形掩码在 native autocast fp16 下偶尔整批 NaN；坏像素静默不填
        # 会留下灰字残影（返修两轮也修不掉）。整批用 fp32 权重重跑一次。
        P.log("    fp16 输出含 NaN，整批回退 fp32")
        with torch.inference_mode():
            out = model(img, msk_t)
        torch.cuda.synchronize()
        out = out.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        finite = np.isfinite(out).all(axis=3)
        if not bool(finite.all()):
            P.log("    fp32 仍有 NaN，坏像素保持原样")
    out = (np.nan_to_num(out, nan=0.0) * 255).round().clip(0, 255).astype(np.uint8)

    for i in range(n_real):
        r = masks[i] > 0
        if not r.any():
            continue
        ys, xs = np.where(r)
        bx0, bx1 = max(0, int(xs.min()) - 10), min(bw, int(xs.max()) + 11)
        by0, by1 = max(0, int(ys.min()) - 10), min(bh, int(ys.max()) + 11)
        a_mask = (r * 255).astype(np.uint8)
        alpha = cv2.GaussianBlur(a_mask, (0, 0), feather)
        o = cv2.resize(out[i][cpy:cpy + fit_h, cpx:cpx + fit_w][..., ::-1],
                       (bw, bh), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(finite[i][cpy:cpy + fit_h, cpx:cpx + fit_w].astype(np.uint8) * 255,
                       (bw, bh), interpolation=cv2.INTER_NEAREST) > 127
        b = bands[i]
        sub_o = o[by0:by1, bx0:bx1]
        sub_c = b[by0:by1, bx0:bx1]
        if cfg["inpaint"].get("color_match"):
            sub_o = P.match_color(sub_c, sub_o, a_mask[by0:by1, bx0:bx1])
        a = alpha[by0:by1, bx0:bx1].astype(np.float32) / 255.0
        a *= v[by0:by1, bx0:bx1]
        a = a[..., None]
        b[by0:by1, bx0:bx1] = (sub_o * a + sub_c * (1 - a)).astype(np.uint8)
    return bands


def _active_frames(tracks, n, fps, t0):
    if n <= 0:
        return 0
    active = np.zeros(n, bool)
    for tr in tracks:
        a = int(np.floor((tr["start"] - 0.08 - t0) * fps))
        b = int(np.ceil((tr["end"] + 0.08 - t0) * fps))
        a, b = max(0, a), min(n - 1, b)
        if b >= a:
            active[a:b + 1] = True
    return int(active.sum())


def _pp_shot(P, fmm, mmm, todo, geo, pprect, tracks, shot, fps, tier, mcfg,
             mask_mode, cues, log):
    """对给定帧号跑 ProPainter。返回 {帧号: 裁片内擦除结果（BGR）}。

    掩码 = max(白字笔画膨胀, 生产笔画掩码)，与 bench v6 兜底口径一致；
    掩码外像素 PP 也不许改（pp_engine 只贴回掩码内）。
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import pp_engine
    except Exception as e:  # noqa: BLE001
        log(f"    ProPainter 不可用（{type(e).__name__}: {e}），本镜头继续走 LaMa")
        return {}
    if not pp_engine.available():
        log("    ProPainter 权重/仓库缺失，本镜头继续走 LaMa")
        return {}
    x0, y0, bw, bh = geo
    px0, py0, px1, py1 = pprect
    cw, ch, ox, oy = px1 - px0, py1 - py0, px0 - x0, py0 - y0
    pad = int(mcfg.get("mask_pad", 8))
    dilate = max(3, int(mcfg.get("dilate", 20)))
    fallback = float(mcfg.get("stroke_fallback", 0.12))
    expand = float(mcfg.get("expand_ratio", 0.0))
    bright = float(mcfg.get("bright_fallback", 0.0))
    kernel = np.ones((dilate, dilate), np.uint8)
    frames, masks, keys = [], [], []
    for k in todo:
        t = shot["start"] + k / fps
        crop = np.ascontiguousarray(fmm[k][oy:oy + ch, ox:ox + cw])
        gm = cv2.dilate((mmm[k] > 0).astype(np.uint8) * 255, kernel)[oy:oy + ch, ox:ox + cw]
        try:
            mt = P.track_mask(crop, tracks, t, px0, py0, mask_mode, pad=pad,
                              dilate=dilate, fallback=fallback, expand=expand,
                              bright=bright)
        except Exception:  # noqa: BLE001
            mt = np.zeros((ch, cw), np.uint8)
        m = (((gm > 0) | (mt > 0)).astype(np.uint8) * 255)
        m = np.ascontiguousarray(
            m & (_active_limit(P, ch, cw, tracks, t, px0, py0, pad, cues=cues) > 0))
        if not m.any():
            continue
        frames.append(crop)
        masks.append(m)
        keys.append(k)
    if not frames:
        return {}
    out = pp_engine.inpaint(frames, masks, log=log, chunk=tier["chunk"],
                            subvideo=tier["subvideo"], neighbor=tier["neighbor"],
                            scale=tier["scale"])
    return dict(zip(keys, out))


def erase_shot_ref(P, model, path, shot, tracks, meta, writer, cfg,
                   mask_mode="stroke", dbg_name="shot", amp=False, cues=None):
    t_start = time.time()
    w, h, fps = meta["w"], meta["h"], meta["fps"]
    dur = shot["end"] - shot["start"]
    n_exp = int(round(shot["end"] * fps)) - int(round(shot["start"] * fps))
    if n_exp <= 0:
        return 0, 0.0, 0.0, 0.0, "ref"
    bx0, by0, bx1, by1 = P._band_rect(tracks, w, h)
    x0, y0, x1, y1 = bx0, by0, bx1, by1
    # 参考填充需要比字幕框更大的背景做光流和光度校验；v6 直接用了整幅宽度。
    cx = max(96, int(round(0.35 * (x1 - x0))))
    cy = max(32, int(round(0.25 * (y1 - y0))))
    x0, x1 = max(0, x0 - cx) & ~1, min(w, x1 + cx) & ~1
    y0, y1 = max(0, y0 - cy) & ~1, min(h, y1 + cy) & ~1
    bw, bh = x1 - x0, y1 - y0

    mcfg = cfg["mask"]
    pad = int(mcfg.get("mask_pad", 8))
    dilate = max(3, int(mcfg.get("dilate", 20)))
    fallback = float(mcfg.get("stroke_fallback", 0.12))
    expand = float(mcfg.get("expand_ratio", 0.0))
    bright = float(mcfg.get("bright_fallback", 0.0))
    ecfg = cfg.get("erase") or {}
    ref_photo = float(ecfg.get("ref_photo", REF_PHOTO))
    ref_feather = float(ecfg.get("ref_feather", REF_FEATHER))
    model_feather = float(ecfg.get("model_feather", MODEL_FEATHER))
    model_stab = float(ecfg.get("model_stab", MODEL_STAB))
    tier = tier_cfg(cfg)

    boxes = [b for tr in tracks for _, b in tr["boxes"]]
    if not boxes:
        return legacy("没有字幕框")
    uboxes = [tr["ubox"] for tr in tracks if tr.get("ubox")]
    fh = float(np.median([b[3] - b[1] for b in uboxes])) if uboxes \
        else float(np.median([b[3] - b[1] for b in boxes])) * 1.15
    if fh <= 0:
        fh = max(16.0, bh * 0.45)
    params = {
        "white_min": int(mcfg.get("white_level", 185)),
        "white_sat": int(mcfg.get("white_sat", 22)),
        "min_area": max(12, int(round(0.003 * fh * fh))),
        "min_h": max(4, int(round(0.10 * fh))),
        "max_h": max(24, int(round(1.20 * fh))),
        "max_w": max(40, min(bw, int(round(3.50 * fh)))),
        "pad_x": max(6, int(round(0.12 * fh))),
        "pad_y": max(4, int(round(0.08 * fh))),
        "min_px": max(60, int(round(0.016 * fh * fh))),
    }
    zpad = max(6, int(round(0.10 * fh)))
    zone = (max(0, min(b[0] for b in boxes) - x0 - zpad),
            max(0, min(b[1] for b in boxes) - y0 - zpad),
            min(bw, max(b[2] for b in boxes) - x0 + zpad),
            min(bh, max(b[3] for b in boxes) - y0 + zpad))

    def legacy(why):
        P.log(f"    白字门控不可用（{why}），整镜头回退 legacy LaMa")
        return P.erase_shot_legacy(model, path, shot, tracks, meta, writer, cfg,
                                   mask_mode, dbg_name=dbg_name, amp=amp)

    with tempfile.TemporaryDirectory(prefix="erase_ref_") as td:
        fpath = f"{td}/frames.raw"
        mpath = f"{td}/masks.raw"
        present, n = [], 0
        merged_n = 0
        tex_v, motion_v, prev_g = [], [], None
        mot_k = np.zeros(n_exp, np.float32)
        zone_mask = None
        if tier["on"]:
            zone_mask = np.zeros((bh, bw), bool)
            za, zb = max(0, by0 - y0), min(bh, by1 - y0)
            zc, zd = max(0, bx0 - x0), min(bw, bx1 - x0)
            if zb > za and zd > zc:
                zone_mask[za:zb, zc:zd] = True
            else:
                zone_mask = None
        kernel = np.ones((dilate, dilate), np.uint8)
        padt = pad + max(8, dilate // 2)
        with open(fpath, "wb") as fw, open(mpath, "wb") as mw:
            for k, fr in enumerate(P.iter_frames(path, bw, bh, shot["start"], dur,
                                                 None, crop=(x0, y0, bw, bh))):
                if k >= n_exp:
                    break
                t = shot["start"] + k / fps
                m = np.zeros((bh, bw), np.uint8)
                total = 0
                if zone[2] > zone[0] and zone[3] > zone[1]:
                    gl, total, merged, kbox = _glyph_mask(fr, zone, params)
                    if total >= params["min_px"]:
                        m = cv2.dilate(gl * 255, kernel)
                        if merged and kbox is not None:
                            # 白字和白衣服/白墙粘成大组件，连通域法会漏字。
                            # 只在缺字的一侧补“一个字宽”的窄条，不再整行矩形，
                            # 否则白背景镜头会被整片抹平。
                            gx0, gx1 = kbox[0] + x0, kbox[2] + x0
                            hit_cap = False
                            for tr in tracks:
                                if tr["start"] - 0.08 <= t <= tr["end"] + 0.08:
                                    bb = tr.get("ubox") or P._box_at(tr, t)
                                    gap = 0.35 * fh
                                    if gx0 - bb[0] <= gap and bb[2] - gx1 <= gap:
                                        continue
                                    cap_w = max(1, int(round(1.5 * fh)))
                                    ry0 = max(0, int(bb[1]) - y0 - padt)
                                    ry1 = min(bh, int(bb[3]) - y0 + padt)
                                    if gx0 - bb[0] > gap:
                                        rx1 = min(bw, gx0 - x0)
                                        rx0 = max(0, int(bb[0]) - x0 - padt)
                                        rx0 = max(rx0, rx1 - cap_w)
                                        if rx1 > rx0 and ry1 > ry0:
                                            m[ry0:ry1, rx0:rx1] = 255
                                            hit_cap = True
                                    if bb[2] - gx1 > gap:
                                        rx0 = max(0, gx1 - x0)
                                        rx1 = min(bw, int(bb[2]) - x0 + padt)
                                        rx1 = min(rx1, rx0 + cap_w)
                                        if rx1 > rx0 and ry1 > ry0:
                                            m[ry0:ry1, rx0:rx1] = 255
                                            hit_cap = True
                            if hit_cap:
                                merged_n += 1
                        if m.any():
                            m = cv2.bitwise_and(
                                m, _active_limit(P, bh, bw, tracks, t, x0, y0, pad,
                                             cues=cues))
                        present.append(k)
                if tier["on"]:
                    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                    bgm = (m == 0)
                    if zone_mask is not None:
                        bgm = bgm & zone_mask
                    if bgm.any():
                        tv = float(cv2.Laplacian(g, cv2.CV_32F)[bgm].var())
                        tex_v.append(tv)
                        if prev_g is not None:
                            mv = float(cv2.absdiff(g, prev_g)[bgm].mean())
                            motion_v.append(mv)
                            if k < len(mot_k):
                                mot_k[k] = mv
                    prev_g = g
                fw.write(fr.tobytes())
                mw.write(m.tobytes())
                n += 1
        if n == 0:
            return 0, time.time() - t_start, 0.0, 0.0, "ref"

        active = _active_frames(tracks, n, fps, shot["start"])
        if not present:
            return legacy("0 个有字帧")
        # cue 时间窗常常比烧录字幕宽很多；只要确实看到白字就按像素门控走。
        # 只有几乎完全检测不到时才怀疑是彩色/低对比字幕，回退旧链路。
        if len(present) < 3:
            return legacy(f"有字帧仅 {len(present)}/{active}")

        present_set = set(present)
        cleans = [i for i in range(n) if i not in present_set]
        runs = _runs_of(present)
        run_ref = {}
        for run in runs:
            center = (run[0] + run[-1]) / 2.0
            run_ref[(run[0], run[-1])] = (
                min(cleans, key=lambda c: abs(c - center)) if cleans else None)
        run_of = np.full(n, -1, np.int32)
        for ri, run in enumerate(runs):
            run_of[run[0]:run[-1] + 1] = ri

        P.log(f"    白字门控 {len(present)}/{n} 帧，参考段 {len(runs)}，"
              f"无字帧 {len(cleans)}，粘连兜底 {merged_n} 帧，字形阈值 px≥{params['min_px']} h{params['min_h']}-"
              f"{params['max_h']} w≤{params['max_w']}")

        fmm = np.memmap(fpath, dtype=np.uint8, mode="r", shape=(n, bh, bw, 3))
        mmm = np.memmap(mpath, dtype=np.uint8, mode="r", shape=(n, bh, bw))
        dis = _make_dis()

        # ---- 三级分流：一档参考帧填充（逐帧、真实像素）/ 二档 LaMa / 三档 ProPainter ----
        # 分类按「镜头」做，不按帧：模型逐帧切换会让纹理在帧间跳变，比字幕更显眼。
        pp_bands, pp_rect, pp_why = {}, None, ""
        tex_med, mot_med = _med(tex_v), _med(motion_v)
        if tier["on"]:
            why = []
            # 阈值 <= 0 表示该判据不启用：生产带上纹理值天然很高（脸/衣服/铁网），
            # bench 那套 16/2.5 的绝对阈值搬过来会把九成镜头判成「难」。
            if tier["tex"] > 0 and tex_med >= tier["tex"]:
                why.append(f"纹理 {tex_med:.1f}≥{tier['tex']:g}")
            if tier["motion"] > 0 and mot_med >= tier["motion"]:
                why.append(f"运动 {mot_med:.1f}≥{tier['motion']:g}")
            if not cleans:
                why.append("本段无干净参考帧")
            todo = []
            if why:
                pp_rect = _pp_rect(tracks, w, h, fh, tier["vpad"],
                                   clip=(x0, y0, x1, y1))
                # 只把这些必进模型的帧送 PP：本段没有干净参考帧，或本帧背景运动大
                # （参考帧 warp 会错位）。无字帧不需要任何模型。
                need = [k for k in present
                        if k < n and (run_ref.get(
                            (runs[run_of[k]][0], runs[run_of[k]][-1])) is None
                            or (k < len(mot_k) and tier["frame_motion"] > 0
                                and mot_k[k] >= tier["frame_motion"]))]
                budget = tier["max_frames"]
                if len(need) >= 3 and pp_rect is not None:
                    for run in _runs_of(need):
                        a, b = max(0, run[0] - 2), min(n - 1, run[-1] + 2)
                        if b - a + 1 > budget:
                            b = a + budget - 1
                        todo.extend(range(a, b + 1))
                        budget -= (b - a + 1)
                        if budget <= 0:
                            break
            if why and todo and pp_rect is not None:
                pp_why = "+".join(why)
                P.log(f"    三级分流：{pp_why} → ProPainter（裁片 "
                      f"{pp_rect[2]-pp_rect[0]}x{pp_rect[3]-pp_rect[1]}，"
                      f"{len(todo)}/{sum(len(r) for r in runs)} 帧）")
                pp_bands = _pp_shot(P, fmm, mmm, todo, (x0, y0, bw, bh), pp_rect,
                                    tracks, shot, fps, tier, mcfg, mask_mode,
                                    cues, P.log)
                if not pp_bands:
                    pp_why, pp_rect = "", None
            else:
                extra = "（参考帧能覆盖，不需要 PP）" if why else ""
                P.log(f"    三级分流：纹理 {tex_med:.1f} 运动 {mot_med:.1f}"
                      f"{extra} → LaMa")
        pp_st = {"ok": 0, "rej": 0}
        stab_state = {"prev": None, "gray": None}
        batch_size = max(1, int(cfg["inpaint"].get("batch", 8)))
        ref_n = model_n = raw_n = no_ref = photo_fail = masked_px = 0
        total = 0
        last_bytes = None
        buf, bands, model_masks, bidx = [], [], [], []

        def flush():
            nonlocal total, last_bytes
            if not buf:
                return
            idx = [i for i, m in enumerate(model_masks) if m is not None]
            if idx:
                filled = _lama_fill(P, model, [bands[i] for i in idx],
                                    [model_masks[i] for i in idx], cfg, amp,
                                    model_feather)
                for i, b in zip(idx, filled):
                    bands[i] = b
                # 三档仲裁：PP 结果只在掩码内生效；和 LaMa 差太多说明 PP 在幻觉，
                # 退回 LaMa（bench 的安全闸口径）
                if pp_bands:
                    ox, oy = pp_rect[0] - x0, pp_rect[1] - y0
                    cw, ch = pp_rect[2] - pp_rect[0], pp_rect[3] - pp_rect[1]
                    for i in idx:
                        pr = pp_bands.get(bidx[i])
                        if pr is None:
                            continue
                        ppb = bands[i].copy()
                        ppb[oy:oy + ch, ox:ox + cw] = pr
                        mm = model_masks[i]
                        if not mm.any():
                            continue
                        d = np.abs(ppb.astype(np.int16)
                                   - bands[i].astype(np.int16)).mean(axis=2)[mm]
                        if (d.mean() <= tier["guard_mean"]
                                and np.percentile(d, 90) <= tier["guard_p90"]):
                            bands[i] = ppb
                            pp_st["ok"] += 1
                        else:
                            pp_st["rej"] += 1
            has_model = any(m is not None for m in model_masks)
            if has_model:
                P.temporal_stabilize(
                    bands,
                    [(m.astype(np.uint8) * 255 if m is not None
                      else np.zeros((bh, bw), np.uint8)) for m in model_masks],
                    stab_state, model_stab, use_flow=True)
            elif model_stab > 0:
                # 没有模型帧的批次也要把状态推到最后一帧，
                # 否则下一段模型帧会从几批之前的旧帧做光流，稳定性失真。
                last = bands[-1]
                stab_state["prev"] = last.copy()
                stab_state["gray"] = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
            for fr, b in zip(buf, bands):
                fr[y0:y1, x0:x1] = b
                if total < n_exp:
                    last_bytes = fr.tobytes()
                    writer.stdin.write(last_bytes)
                    total += 1
            buf.clear()
            bands.clear()
            model_masks.clear()
            bidx.clear()

        for k, fr in enumerate(P.iter_frames(path, w, h, shot["start"], dur, None)):
            if k >= n_exp:
                break
            band = fr[y0:y1, x0:x1].copy()
            mref = mmm[k] > 0 if k < n else np.zeros((bh, bw), bool)
            mt = None
            if k < n and run_of[k] >= 0 and mref.any():
                masked_px += int(mref.sum())
                run = runs[run_of[k]]
                r = run_ref.get((run[0], run[-1]))
                filled = False
                if r is not None:
                    near = (cv2.dilate(mref.astype(np.uint8),
                                       np.ones((11, 11), np.uint8)) > 0) & ~mref
                    b, g, rc = cv2.split(band)
                    whiteish = ((b > 190) & (g > 190) & (rc > 190)
                                & ((rc.astype(np.int16) - b.astype(np.int16)) < 30))
                    mfill = mref | (near & whiteish)
                    mfill = mfill & (_active_limit(
                        P, bh, bw, tracks, shot["start"] + k / fps,
                        x0, y0, pad, cues=cues) > 0)
                    fl, sw, sh = _flow_to(band, fmm[r], dis)
                    md = cv2.dilate(mfill.astype(np.uint8),
                                    np.ones((31, 31), np.uint8))
                    mdh = cv2.resize(md, (sw, sh), interpolation=cv2.INTER_NEAREST)
                    fl = np.stack([
                        cv2.inpaint(fl[..., 0], mdh, 3, cv2.INPAINT_TELEA),
                        cv2.inpaint(fl[..., 1], mdh, 3, cv2.INPAINT_TELEA)],
                        axis=-1)
                    fl = cv2.GaussianBlur(fl, (0, 0), 0.8)
                    wr = _warp_with(fmm[r], fl)
                    ring = _ring_of(mfill)
                    d = _photometric(wr, band, ring)
                    if d <= ref_photo:
                        off = (band[ring].astype(np.float32)
                               - wr[ring].astype(np.float32)).mean(axis=0)
                        wr = np.clip(wr.astype(np.float32) + off, 0, 255)
                        band = _blend(band, wr.round().astype(np.uint8),
                                      mfill, ref_feather)
                        filled = True
                        ref_n += 1
                    else:
                        photo_fail += 1
                if not filled:
                    mt = P.track_mask(band, tracks, shot["start"] + k / fps,
                                      x0, y0, mask_mode, pad=pad, dilate=dilate,
                                      fallback=fallback, expand=expand,
                                      bright=bright)
                    mt = (mt > 0) | mref
                    mt = mt & (_active_limit(
                        P, bh, bw, tracks, shot["start"] + k / fps,
                        x0, y0, pad, cues=cues) > 0)
                    if r is None:
                        no_ref += 1
                    model_n += 1
            else:
                raw_n += 1
            buf.append(fr)
            bands.append(band)
            model_masks.append(mt)
            bidx.append(k)
            if len(buf) >= batch_size:
                flush()
        flush()
        while total < n_exp and last_bytes is not None:
            writer.stdin.write(last_bytes)
            total += 1
        del fmm, mmm

    dt = time.time() - t_start
    coverage = masked_px / max(total * bh * bw, 1)
    pp_txt = ""
    if pp_st["ok"] or pp_st["rej"]:
        pp_txt = f"  PP {pp_st['ok']}（回退 {pp_st['rej']}）"
    P.log(f"    ref {ref_n} / model {model_n}（无参考 {no_ref}，光度 {photo_fail}）"
          f" / 原样 {raw_n}{pp_txt}  掩码 {coverage*100:.1f}%  "
          f"{total} 帧 {dt:.1f}s ({total/dt if dt else 0:.1f} fps)")
    return total, dt, (total / dt if dt else 0.0), coverage, "ref"
