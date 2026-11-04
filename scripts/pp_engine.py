#!/usr/bin/env python3
"""ProPainter 封装：进程内调用、模型常驻，供擦除引擎的第三档（最难镜头）使用。

设计要点
--------
1. **进程内**：生产解释器（/usr/bin/python3）本来就能 import ProPainter 的模块，
   没必要像 bench 那样每个 chunk fork 一个子进程重新加载 190MB 权重；模型在这里
   只加载一次并常驻，chunk 之间复用。
2. **按 chunk 跑**：PP 会把 frames + 双向光流全放在显存里，整镜头一次跑必炸 16G。
   默认 chunk=80 帧（与官方 `--subvideo_length` 一致），超过自动切。
3. **只在掩码内替换**：PP 输出整幅图，外面一圈是我们自己的原像素，只取掩码区
   （膨胀 5px 的 binary 掩码）贴回，掩码外的帧保持逐位不变。
4. **可降分辨率**：`scale < 1` 时按比例缩到 PP 处理尺寸再放大回原尺寸，用显存换质量。

对外只有 `inpaint(frames, masks, **opts)`：BGR uint8 进、同尺寸 BGR uint8 出。
"""
from __future__ import annotations

import contextlib
import os
import sys
import time

import cv2
import numpy as np

PP_ROOT = os.environ.get(
    "VE_PP_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "bench", "video_inpaint", "repos", "ProPainter"))

_MODELS = None
_TORCH = None


def available():
    return os.path.isdir(PP_ROOT) and os.path.isfile(
        os.path.join(PP_ROOT, "weights", "ProPainter.pth"))


def _imports():
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    if PP_ROOT not in sys.path:
        sys.path.insert(0, PP_ROOT)
    import torch
    from model.modules.flow_comp_raft import RAFT_bi
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    from model.propainter import InpaintGenerator
    from core.utils import to_tensors
    from utils.download_util import load_file_from_url

    _TORCH = dict(torch=torch, RAFT_bi=RAFT_bi,
                  RFC=RecurrentFlowCompleteNet,
                  Inpaint=InpaintGenerator, to_tensors=to_tensors,
                  load_url=load_file_from_url)
    return _TORCH


def _url(*a, **kw):  # pragma: no cover - 只是网络兜底
    return None


def _models(device, half):
    """一次加载、常驻；half 会就地转 fp16（同一进程里只能选一种精度）。"""
    global _MODELS
    m = _imports()
    if _MODELS is not None:
        return _MODELS
    torch = m["torch"]
    url = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"

    def w(name):
        p = os.path.join(PP_ROOT, "weights", name)
        if os.path.isfile(p):
            return p
        return m["load_url"](url=url + name, model_dir="weights", progress=True,
                             file_name=None)

    raft = m["RAFT_bi"](w("raft-things.pth"), device)
    rfc = m["RFC"](w("recurrent_flow_completion.pth"))
    for p in rfc.parameters():
        p.requires_grad = False
    rfc.to(device).eval()
    net = m["Inpaint"](model_path=w("ProPainter.pth")).to(device).eval()
    if half:
        rfc, net = rfc.half(), net.half()
    _MODELS = dict(raft=raft, rfc=rfc, net=net, half=half, device=device)
    return _MODELS


def _ref_index(mid, neighbor_ids, length, ref_stride, ref_num):
    out = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                out.append(i)
        return out
    start = max(0, mid - ref_stride * (ref_num // 2))
    end = min(length, mid + ref_stride * (ref_num // 2))
    for i in range(start, end, ref_stride):
        if i not in neighbor_ids:
            if len(out) > ref_num:
                break
            out.append(i)
    return out


def _dilate(mask, iters):
    if iters <= 0:
        return mask
    m = (np.asarray(mask) > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    for _ in range(iters):
        m = cv2.dilate(m, k)
    return m


def _pil(rgb):
    from PIL import Image
    return [Image.fromarray(f) for f in rgb]


def _pil_l(masks):
    from PIL import Image
    return [Image.fromarray((np.asarray(m) > 0).astype(np.uint8) * 255) for m in masks]


def _run_chunk(frames, masks, opts, device, half, log):
    """frames 是 RGB（PP 的 to_tensors 走 PIL/RGB 通道序）。"""
    torch = _imports()["torch"]
    to_tensors = _imports()["to_tensors"]
    M = _models(device, half)
    h0, w0 = frames[0].shape[:2]
    pw, ph = (w0 - w0 % 8, h0 - h0 % 8)
    scale = float(opts.get("scale", 1.0) or 1.0)
    if scale < 0.999:
        pw, ph = max(16, int(pw * scale) // 8 * 8), max(16, int(ph * scale) // 8 * 8)
    pw, ph = max(8, pw), max(8, ph)
    rs = [cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA
                     if pw < w0 else cv2.INTER_CUBIC) for f in frames]
    d_f = int(opts.get("flow_dilation", 4))
    d_m = int(opts.get("mask_dilation", 4))
    rs_m = [cv2.resize((np.asarray(m) > 0).astype(np.uint8) * 255, (pw, ph),
                       interpolation=cv2.INTER_NEAREST) for m in masks]
    flow_masks = [_dilate(m, d_f) for m in rs_m]
    masks_d = [_dilate(m, d_m) for m in rs_m]

    with torch.no_grad():
        t = to_tensors()(_pil(rs)).unsqueeze(0) * 2 - 1
        fm = to_tensors()(_pil_l(flow_masks)).unsqueeze(0)
        md = to_tensors()(_pil_l(masks_d)).unsqueeze(0)
        t, fm, md = t.to(device), fm.to(device), md.to(device)
        n = t.size(1)
        short_clip = 12 if t.size(-1) <= 640 else 8 if t.size(-1) <= 720 else \
            4 if t.size(-1) <= 1280 else 2
        iters = int(opts.get("raft_iter", 20))
        if n > short_clip:
            fs, bs = [], []
            for f in range(0, n, short_clip):
                e = min(n, f + short_clip)
                a, b = M["raft"](t[:, max(0, f - 1):e], iters=iters)
                fs.append(a)
                bs.append(b)
                torch.cuda.empty_cache()
            gt = (torch.cat(fs, dim=1), torch.cat(bs, dim=1))
        else:
            gt = M["raft"](t, iters=iters)
            torch.cuda.empty_cache()

        # RAFT 必须 fp32（PP 官方也是先算光流再转半精度）
        if half:
            t, fm, md = t.half(), fm.half(), md.half()
            gt = (gt[0].half(), gt[1].half())

        sub = int(opts.get("subvideo", 80))
        fl = gt[0].size(1)
        if fl > sub:
            pf, pb = [], []
            pad = 5
            for f in range(0, fl, sub):
                s = max(0, f - pad)
                e = min(fl, f + sub + pad)
                ps = max(0, f) - s
                pe = e - min(fl, f + sub)
                sub_bi, _ = M["rfc"].forward_bidirect_flow(
                    (gt[0][:, s:e], gt[1][:, s:e]), fm[:, s:e + 1])
                sub_bi = M["rfc"].combine_flow(
                    (gt[0][:, s:e], gt[1][:, s:e]), sub_bi, fm[:, s:e + 1])
                pf.append(sub_bi[0][:, ps:e - s - pe])
                pb.append(sub_bi[1][:, ps:e - s - pe])
                torch.cuda.empty_cache()
            gt = (torch.cat(pf, dim=1), torch.cat(pb, dim=1))
        else:
            pred, _ = M["rfc"].forward_bidirect_flow(gt, fm)
            gt = M["rfc"].combine_flow(gt, pred, fm)
            torch.cuda.empty_cache()

        masked_in = t * (1 - md)
        prop_len = min(100, sub)
        if n > prop_len:
            uf, um = [], []
            pad = 10
            for f in range(0, n, prop_len):
                s0 = max(0, f - pad)
                e0 = min(n, f + prop_len + pad)
                ps = max(0, f) - s0
                pe = e0 - min(n, f + prop_len)
                b, tt, _, _, _ = md[:, s0:e0].size()
                bi_sub = (gt[0][:, s0:e0 - 1], gt[1][:, s0:e0 - 1])
                pi, lm = M["net"].img_propagation(masked_in[:, s0:e0], bi_sub,
                                                 md[:, s0:e0], "nearest")
                uf.append(t[:, s0:e0] * (1 - md[:, s0:e0])
                          + pi.view(b, tt, 3, ph, pw) * md[:, s0:e0])
                um.append(lm.view(b, tt, 1, ph, pw))
                torch.cuda.empty_cache()
            out_f, out_m = [], []
            for fi, f in enumerate(range(0, n, prop_len)):
                s0 = max(0, f - pad)
                e0 = min(n, f + prop_len + pad)
                ps = max(0, f) - s0
                pe = e0 - min(n, f + prop_len)
                out_f.append(uf[fi][:, ps:e0 - s0 - pe])
                out_m.append(um[fi][:, ps:e0 - s0 - pe])
            uframes, umasks = torch.cat(out_f, dim=1), torch.cat(out_m, dim=1)
        else:
            b, tt, _, _, _ = md.size()
            pi, lm = M["net"].img_propagation(masked_in, gt, md, "nearest")
            uframes = t * (1 - md) + pi.view(b, tt, 3, ph, pw) * md
            umasks = lm.view(b, tt, 1, ph, pw)
        torch.cuda.empty_cache()

        neigh = int(opts.get("neighbor", 10))
        ref_stride = int(opts.get("ref_stride", 10))
        stride = neigh // 2
        ref_num = sub // ref_stride if n > sub else -1
        comp = [None] * n
        for f in range(0, n, stride):
            nb = [i for i in range(max(0, f - stride), min(n, f + stride + 1))]
            refs = _ref_index(f, nb, n, ref_stride, ref_num)
            sel = uframes[:, nb + refs]
            sel_m = md[:, nb + refs]
            sel_um = umasks[:, nb + refs]
            sel_f = (gt[0][:, nb[:-1]], gt[1][:, nb[:-1]])
            pred = M["net"](sel, sel_f, sel_m, sel_um, len(nb))
            pred = ((pred.view(-1, 3, ph, pw) + 1) / 2).cpu().permute(0, 2, 3, 1).numpy() * 255
            bm = md[0, nb].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            for i, idx in enumerate(nb):
                img = (pred[i].astype(np.uint8) * bm[i]
                       + cv2.resize(rs[idx], (pw, ph)) * (1 - bm[i]))
                img = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_CUBIC)
                if comp[idx] is None:
                    comp[idx] = img.astype(np.uint8)
                else:
                    comp[idx] = (comp[idx].astype(np.float32) * 0.5
                                 + img.astype(np.float32) * 0.5).round().astype(np.uint8)
            torch.cuda.empty_cache()

    # 只保留掩码内像素，外面用原图（PP 会重采样整幅图，掩码外不能有差）
    out = []
    for f, c, m in zip(frames, comp, masks):
        mm = (np.asarray(m) > 0)
        r = f.copy()
        r[mm] = c[mm]
        out.append(r)
    return out


def _bgr2rgb(f):
    return f[:, :, ::-1].copy()


def inpaint(frames, masks, log=None, **opts):
    """frames/masks 逐帧对应（BGR uint8 / 0-255 掩码），返回同尺寸擦除结果。"""
    log = log or (lambda *_: None)
    n = len(frames)
    if n == 0:
        return []
    m = _imports()
    torch = m["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = bool(opts.get("fp16", True)) and device.type == "cuda"
    chunk = max(1, int(opts.get("chunk", 80)))
    out = []
    for f in range(0, n, chunk):
        seg = [_bgr2rgb(x) for x in frames[f:f + chunk]]
        segm = masks[f:f + chunk]
        t0 = time.time()
        out.extend(_run_chunk(seg, segm, opts, device, half, log))
        dt = time.time() - t0
        log(f"      PP chunk {f}-{f + len(seg)} {len(seg)} 帧 {dt:.1f}s "
            f"({len(seg) / dt if dt else 0:.2f} fps)")
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
    return [_bgr2rgb(x) for x in out]


def release():
    """把常驻模型从显存里放掉（批量跑完或切回只有 LaMa 的模式时用）。"""
    global _MODELS
    if _MODELS is None:
        return
    _MODELS = None
    with contextlib.suppress(Exception):
        torch = _imports()["torch"]
        torch.cuda.empty_cache()
