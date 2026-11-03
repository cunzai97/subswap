#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""云端字幕擦除（火山引擎 AI MediaKit）· 与本地 LaMa 同一个调用面

流程：申请上传地址 -> PUT 本地文件 -> 提交异步任务 -> 轮询 -> 下载 -> 对齐源分辨率/音轨
只依赖标准库 + ffmpeg，不碰 torch，云端模式下不需要 GPU。

价格（2026-11，官方文档）：
  标准版 erase-video-subtitle      0.4 元/分钟
  精细化版 erase-video-subtitle-pro 1 元/分钟
按输出时长计费，失败不计费。
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from urllib import error, request

DEFAULT_BASE = "https://mediakit.cn-beijing.volces.com"
TOOLS = {"standard": "erase-video-subtitle", "pro": "erase-video-subtitle-pro"}
PRICES = {"standard": 0.4, "pro": 1.0}
DONE = {"completed", "success", "succeeded", "finished"}
FAIL = {"failed", "error", "canceled", "cancelled"}


class CloudEraseError(RuntimeError):
    pass


# ---------------- 配置 ----------------

def _cloud_cfg(cfg):
    e = (cfg or {}).get("erase") or {}
    c = dict(e.get("cloud") or {})
    return e, c


def resolve_quality(cfg, quality=None):
    q = str(quality or _cloud_cfg(cfg)[1].get("quality") or "pro").lower()
    if q not in TOOLS:
        raise CloudEraseError(f"未知云端擦除档位：{q}（可选 pro / standard）")
    return q


def api_key(cfg):
    _e, c = _cloud_cfg(cfg)
    env_name = c.get("api_key_env") or "ERASE_CLOUD_API_KEY"
    key = os.environ.get(env_name) or c.get("api_key") or ""
    if not key and c.get("api_key_file"):
        p = Path(os.path.expandvars(os.path.expanduser(str(c["api_key_file"]))))
        if p.exists():
            key = p.read_text(encoding="utf-8").strip()
    return str(key or "").strip()


def key_ready(cfg):
    return bool(api_key(cfg))


# ---------------- HTTP ----------------

def _json_call(method, url, key, body=None, timeout=120):
    data = None
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except error.HTTPError as ex:
        detail = ex.read().decode("utf-8", "ignore")[:800]
        raise CloudEraseError(f"云端接口 HTTP {ex.code}：{detail}") from None
    except Exception as ex:  # noqa: BLE001
        raise CloudEraseError(f"云端接口不可达（{url}）：{ex}") from None
    try:
        return json.loads(raw or b"{}")
    except json.JSONDecodeError:
        raise CloudEraseError(f"云端返回不是 JSON：{raw[:300]!r}") from None


def _api_error(data):
    if data.get("success") is False:
        err = data.get("error") or {}
        if isinstance(err, dict):
            code = err.get("code") or err.get("type") or "error"
            msg = err.get("message") or err.get("msg") or json.dumps(err, ensure_ascii=False)
            raise CloudEraseError(f"云端任务失败 [{code}]：{msg}")
        raise CloudEraseError(f"云端任务失败：{err}")
    return data


def _transient(ex):
    """网络抖动/上游 5xx 可重试；4xx（Key/参数错误）不重试"""
    s = str(ex).lower()
    return ("不可达" in s or "timed out" in s or "timeout" in s
            or "connection" in s or "http 5" in s or "http 429" in s)


def _dig(obj, *names):
    """在 result / data / 顶层里找一个字段，兼容接口层的嵌套差异"""
    if not isinstance(obj, dict):
        return None
    for n in names:
        if n in obj and obj[n] not in (None, ""):
            return obj[n]
    for k in ("result", "data", "output"):
        if isinstance(obj.get(k), dict):
            v = _dig(obj[k], *names)
            if v not in (None, ""):
                return v
    return None


def _task_status(data):
    st = _dig(data, "status", "task_status", "state")
    if isinstance(st, dict):
        st = st.get("status") or st.get("state")
    return str(st or "").lower()


def _task_video_url(data):
    return _dig(data, "video_url", "url", "output_url")


def _norm_headers(headers):
    """upload_headers 可能是 {}、[{name,value}] 或 [[k,v]]"""
    if not headers:
        return {}
    if isinstance(headers, dict):
        return {str(k): str(v) for k, v in headers.items()}
    out = {}
    for it in headers:
        if isinstance(it, dict):
            k = it.get("name") or it.get("key") or it.get("Name")
            v = it.get("value") or it.get("Value")
            if k is not None and v is not None:
                out[str(k)] = str(v)
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            out[str(it[0])] = str(it[1])
    return out


def _upload_file(url, path, headers, on_prog=None):
    path = Path(path)
    total = max(1, path.stat().st_size)
    mime = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
            ".mkv": "video/x-matroska", ".webm": "video/webm",
            ".avi": "video/x-msvideo", ".flv": "video/x-flv",
            ".ts": "video/mp2t"}.get(path.suffix.lower(), "application/octet-stream")

    class Reader:
        def __init__(self, fh):
            self.fh = fh
            self.done = 0

        def read(self, n=-1):
            chunk = self.fh.read(n)
            if chunk:
                self.done += len(chunk)
                if on_prog:
                    on_prog(min(1.0, self.done / total))
            return chunk

    h = _norm_headers(headers)
    low = {k.lower() for k in h}
    if "content-type" not in low:
        h["Content-Type"] = mime
    h["Content-Length"] = str(total)
    with open(path, "rb") as fh:
        req = request.Request(url, data=Reader(fh), headers=h, method="PUT")
        try:
            with request.urlopen(req, timeout=600) as r:
                return r.status
        except error.HTTPError as ex:
            detail = ex.read().decode("utf-8", "ignore")[:500]
            raise CloudEraseError(f"上传素材失败 HTTP {ex.code}：{detail}") from None
        except Exception as ex:  # noqa: BLE001
            raise CloudEraseError(f"上传素材失败：{ex}") from None


def _download(url, dst, on_prog=None, timeout=600):
    try:
        with request.urlopen(url, timeout=timeout) as r, open(dst, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if on_prog and total:
                    on_prog(min(1.0, done / total))
    except Exception as ex:  # noqa: BLE001
        raise CloudEraseError(f"下载云端结果失败：{ex}") from None
    if not Path(dst).exists() or Path(dst).stat().st_size < 1024:
        raise CloudEraseError("云端结果为空")


def _probe(path):
    raw = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         "-show_format", str(path)], capture_output=True, check=True).stdout
    info = json.loads(raw)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        raise CloudEraseError(f"文件没有视频流：{path}")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "25/1").split("/")
    fps = float(num) / max(float(den), 1e-6)
    return {"w": int(v["width"]), "h": int(v["height"]), "fps": fps,
            "dur": float(info.get("format", {}).get("duration") or 0),
            "audio": any(s.get("codec_type") == "audio" for s in info.get("streams", []))}


# ---------------- 字幕带 / 参数 ----------------

def _region_boxes(c, layout, quality, log):
    """返回 erase_ratio_location；None=交给云端自动检测。

    云端 Subtitle 模式默认只处理画面下半部（Y>=0.5）。顶部字幕只能改用
    Text 模式 + 手动框选，否则会漏擦；标准版不支持框选，直接报错比白花钱好。
    """
    raw = c.get("region", "auto")
    if isinstance(raw, list):
        if len(raw) == 4 and all(isinstance(v, (int, float)) for v in raw):
            return [{"top_left_x": raw[0], "top_left_y": raw[1],
                     "bottom_right_x": raw[2], "bottom_right_y": raw[3]}]
        return raw
    mode = str(raw or "auto").lower()
    if mode == "never":
        return None
    top = bool(layout and layout.get("align") == 8)
    if mode == "auto" and not top:
        return None
    if quality != "pro":
        if top:
            raise CloudEraseError(
                "字幕在画面上半部，云端标准版只能擦下半部；"
                "请改用云端精细化版，或切回本地擦除")
        log("云端标准版不支持框选区域，已忽略字幕带")
        return None
    band = (layout or {}).get("band") or ((0.05, 0.25) if top else (0.55, 0.95))
    y0 = max(0.0, float(band[0]) - 0.01)
    y1 = min(1.0, float(band[1]) + 0.01)
    if y1 - y0 < 0.02:
        y1 = min(1.0, y0 + 0.02)
    box = {"top_left_x": 0.02, "top_left_y": round(y0, 3),
           "bottom_right_x": 0.98, "bottom_right_y": round(y1, 3)}
    log(f"云端框选：{box}")
    return [box]


def adapt_output(raw, dst, src, meta, log):
    """云端结果 -> 本流程口径：源分辨率/帧率 + 源音轨 + yuv420p CFR

    云端一般同尺寸返回，此时只换音轨不重编码；尺寸/帧率变了才缩放重编。
    """
    raw, dst, src = Path(raw), Path(dst), Path(src)
    info = _probe(raw)
    same_size = (info["w"], info["h"]) == (meta["w"], meta["h"])
    same_fps = abs(info["fps"] - meta["fps"]) < 0.01
    src_audio = _probe(src)["audio"] if src is not None and Path(src).exists() else info["audio"]
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(raw)]
    if src_audio and src is not None and Path(src).exists() and Path(src) != raw:
        cmd += ["-i", str(src)]
    if same_size and same_fps:
        cmd += ["-map", "0:v:0", "-c:v", "copy"]
    else:
        vf = [f"scale={meta['w']}:{meta['h']}:flags=bicubic", "setsar=1"]
        if not same_fps:
            vf.append(f"fps={meta['fps']:g}")
        cmd += ["-map", "0:v:0", "-vf", ",".join(vf),
                "-c:v", "libx264", "-crf", "12", "-preset", "veryfast",
                "-pix_fmt", "yuv420p"]
    if src_audio and src is not None and Path(src).exists() and Path(src) != raw:
        cmd += ["-map", "1:a:0", "-c:a", "copy", "-shortest"]
    elif src_audio:
        cmd += ["-map", "0:a:0", "-c:a", "copy"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(dst)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # MP4 容器装不下源音轨编码（PCM 等）时退回 AAC
        cmd2 = list(cmd)
        if "-c:a" in cmd2:
            i = cmd2.index("-c:a")
            cmd2[i:i + 2] = ["-c:a", "aac", "-b:a", "192k"]
        subprocess.run(cmd2, check=True)
    out = _probe(dst)
    log(f"云端结果对齐：{out['w']}x{out['h']} @{out['fps']:.3g}fps · "
        f"{'缩放重编' if not (same_size and same_fps) else '仅换音轨'}")
    return out


# ---------------- 主入口 ----------------

def erase(src, dst, cfg, log=print, meta=None, layout=None, quality=None,
          on_prog=None):
    """上传整个素材 -> 云端擦除 -> 下载并对齐；返回报告用信息"""
    src, dst = Path(src), Path(dst)
    _e, c = _cloud_cfg(cfg)
    key = api_key(cfg)
    if not key:
        raise CloudEraseError(
            "未配置云端擦除 API Key：设置环境变量 ERASE_CLOUD_API_KEY，"
            "或填写 config.yaml 的 erase.cloud.api_key / api_key_file")
    quality = resolve_quality(cfg, quality)
    base = str(os.environ.get("ERASE_CLOUD_BASE_URL") or c.get("base_url")
               or DEFAULT_BASE).rstrip("/")
    poll = max(1.0, float(c.get("poll_interval", 5)))
    timeout = max(30.0, float(c.get("timeout", 1800)))
    meta = meta or _probe(src)
    boxes = _region_boxes(c, layout, quality, log)
    mode_cfg = str(c.get("mode") or "auto").lower()
    mode = "Text" if (mode_cfg == "auto" and boxes and layout
                      and layout.get("align") == 8) else \
        ("Text" if mode_cfg == "text" else "Subtitle")

    t0 = time.time()
    label = "精细化版" if quality == "pro" else "标准版"
    price = PRICES[quality]
    est = round(meta["dur"] / 60 * price, 2)
    log(f"云端擦除：火山引擎 AI MediaKit · {label} · ¥{price}/分钟 "
        f"· 本片约 ¥{est}")

    # 1) 申请上传地址
    log("云端：申请上传地址…")
    up = _api_error(_json_call("POST", base + "/api/v1/tools-sync/request-media-upload-url",
                               key, {}))
    res = up.get("result") or up
    file_id = res.get("file_id")
    upload_url = res.get("upload_url")
    if not file_id or not upload_url:
        raise CloudEraseError(f"申请上传地址失败：{json.dumps(up, ensure_ascii=False)[:500]}")
    if not str(file_id).startswith("mediakit://"):
        file_id = "mediakit://" + str(file_id)

    # 2) 直传本地文件（纯二进制 PUT；PUT 幂等，网络抖动可重试一次）
    log(f"云端：上传 {src.stat().st_size/1e6:.1f} MB…")
    for attempt in range(2):
        try:
            _upload_file(upload_url, src, res.get("upload_headers") or {},
                         on_prog=(lambda f: on_prog(0.03 + 0.22 * f, "上传素材")
                                  if on_prog else None))
            break
        except CloudEraseError as ex:
            if attempt or not _transient(ex):
                raise
            log(f"上传暂时失败，重试一次：{ex}")
            time.sleep(2)
    if on_prog:
        on_prog(0.26, "已上传，提交任务")

    # 3) 提交任务
    body = {"video_url": file_id, "mode": mode}
    if quality == "pro":
        body["model_version"] = str(c.get("model_version") or "v5.6")
    if boxes:
        body["erase_ratio_location"] = boxes
    extra = c.get("submit_extra") or {}
    if isinstance(extra, dict):
        body.update(extra)
    log(f"云端：提交擦除任务（{mode}"
        + (f" · {body.get('model_version')}" if quality == "pro" else "") + "）…")
    sub = _api_error(_json_call("POST", base + f"/api/v1/tools/{TOOLS[quality]}",
                                key, body))
    task_id = sub.get("task_id")
    if not task_id:
        raise CloudEraseError(f"提交任务失败：{json.dumps(sub, ensure_ascii=False)[:500]}")
    log(f"云端：task_id={task_id}，等待处理…")
    if on_prog:
        on_prog(0.3, "云端处理中")

    # 4) 轮询
    deadline = time.time() + timeout
    last = None
    while True:
        try:
            data = _api_error(_json_call("GET", base + f"/api/v1/tasks/{task_id}",
                                         key, None, timeout=60))
        except CloudEraseError as ex:
            # 任务已经在云端跑，查不到就重试，别因为一次抖动丢掉结果
            if time.time() > deadline or not _transient(ex):
                raise
            log(f"云端查询暂时失败，重试：{ex}")
            time.sleep(poll)
            continue
        st = _task_status(data) or "unknown"
        if st != last:
            log(f"云端状态：{st}")
            last = st
        if st in DONE:
            video_url = _task_video_url(data)
            if not video_url:
                raise CloudEraseError(
                    f"任务完成但没拿到下载地址：{json.dumps(data, ensure_ascii=False)[:500]}")
            break
        if st in FAIL:
            err = _dig(data, "message", "error_msg", "error") or st
            raise CloudEraseError(f"云端任务失败：{err}")
        if time.time() > deadline:
            raise CloudEraseError(
                f"云端任务超时（{timeout:.0f}s，task_id={task_id}）；"
                "可在控制台按 task_id 查询，避免重复提交")
        if on_prog:
            # 进度只做缓慢推进，别假装精确；状态文字里有耗时
            frac = min(0.82, 0.3 + 0.5 * (time.time() - t0) / max(60.0, meta["dur"] * 2.5))
            on_prog(frac, f"云端处理中 {time.time()-t0:.0f}s")
        time.sleep(poll)

    # 5) 下载 + 对齐
    raw = dst.with_name(dst.stem + "_raw.mp4")
    log("云端：下载结果…")
    for attempt in range(2):
        try:
            _download(video_url, raw,
                      on_prog=(lambda f: on_prog(0.84 + 0.1 * f, "下载结果")
                               if on_prog else None))
            break
        except CloudEraseError as ex:
            if attempt or not _transient(ex):
                raise
            log(f"下载暂时失败，重试一次：{ex}")
            time.sleep(2)
    if on_prog:
        on_prog(0.95, "对齐音轨/分辨率")
    out = adapt_output(raw, dst, src, meta, log)
    try:
        raw.unlink()
    except OSError:
        pass
    info = {
        "provider": "cloud",
        "vendor": "volcengine",
        "quality": quality,
        "tool": TOOLS[quality],
        "mode": mode,
        "model_version": body.get("model_version", ""),
        "task_id": task_id,
        "region": boxes or [],
        "cost_est": est,
        "price_per_min": price,
        "seconds": round(time.time() - t0, 1),
        "output": out,
    }
    log(f"云端擦除完成 {info['seconds']:.0f}s（估算 ¥{est}）")
    return info
