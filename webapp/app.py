#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""短剧字幕转换 · 网页版

启动：
  cd /media/pan/学习/work/video-erase/webapp
  /usr/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port 8899

浏览器打开 http://127.0.0.1:8899
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from queue import Queue

import yaml
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import convert as C
import erase_cloud as E

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "webapp"
JOBS = ROOT / "runtime" / "web" / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="字幕转换")
Q: Queue[str] = Queue()
LOCK = threading.Lock()
STATE: dict[str, dict] = {}
LEDGER = JOBS / "index.json"


def load_ledger():
    if LEDGER.exists():
        try:
            return json.loads(LEDGER.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


def save_ledger():
    with LOCK:
        LEDGER.write_text(json.dumps(STATE, ensure_ascii=False, indent=1), encoding="utf-8")


def job_dir(jid: str) -> Path:
    return JOBS / jid


def update(jid: str, **kw):
    with LOCK:
        st = STATE.setdefault(jid, {"id": jid, "state": "pending", "progress": 0.0,
                                    "stage": "排队中", "logs": [], "created": time.time()})
        st.update(kw)
        st["updated"] = time.time()
    save_ledger()


def add_log(jid: str, line: str):
    with LOCK:
        st = STATE.setdefault(jid, {"id": jid, "logs": []})
        logs = st.setdefault("logs", [])
        logs.append(f"{time.strftime('%H:%M:%S')} {line}")
        if len(logs) > 400:
            del logs[: len(logs) - 400]
    save_ledger()


def run_job(jid: str):
    d = job_dir(jid)
    opts = json.loads((d / "opts.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    # fp16 由 config.yaml 决定；NaN 已在 _lama_fill / _lama_patch_batch 里
    # 整批回退 fp32，网页路径不再单独硬关。
    opts["cfg"] = cfg
    update(jid, state="running", stage="开始", progress=0.01)
    try:
        def on_stage(name, frac):
            update(jid, stage=name, progress=round(min(0.99, frac), 3))

        out, report = C.convert(d, opts, log=lambda m: add_log(jid, m), on_stage=on_stage)
        size = round(out.stat().st_size / 1e6, 1)
        update(jid, state="done", stage="完成", progress=1.0, report=report,
               size_mb=size, elapsed=round(time.time() - STATE[jid]["created"], 1))
        add_log(jid, f"完成：{size} MB")
    except Exception as e:  # noqa: BLE001
        add_log(jid, "出错：" + str(e))
        add_log(jid, traceback.format_exc()[-1500:])
        update(jid, state="error", stage="出错", error=str(e))


def worker_loop():
    while True:
        jid = Q.get()
        update(jid, state="running", stage="启动", progress=0.0)
        run_job(jid)
        Q.task_done()


STATE.update(load_ledger())
for _jid, _st in STATE.items():
    if _st.get("state") in ("running", "pending"):
        _st["state"] = "interrupted"
        _st["stage"] = "服务重启，任务中断"
save_ledger()

threading.Thread(target=worker_loop, daemon=True).start()


# ---------------- 路由 ----------------

@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    gpu = "unknown"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    try:
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        prov, qual = C.resolve_erase(cfg, {})
        erase = {"provider": prov, "quality": qual or "",
                 "cloud_ready": C.cloud_key_ready(cfg),
                 "prices": E.PRICES}
    except Exception:  # noqa: BLE001
        erase = {"provider": "local", "quality": "", "cloud_ready": False,
                 "prices": E.PRICES}
    return {"ok": True, "gpu": gpu, "queue": Q.qsize(),
            "erase": erase,
            "running": [k for k, v in STATE.items() if v.get("state") == "running"]}


@app.get("/api/translate/check")
def translate_check():
    be = C.pick_translator()
    if not be:
        return {"ok": False, "detail": "1509 和 8790 都不通"}
    return {"ok": True, "base_url": be["base_url"], "model": be["model"]}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...),
                     source: str = Form("auto"),
                     target: str = Form("en"),
                     erase: str = Form("")):
    erase = (erase or "").strip().lower()
    if erase not in ("", "local", "cloud", "cloud_pro", "cloud_standard"):
        return JSONResponse({"error": f"未知擦除方式：{erase}"}, status_code=400)
    try:
        cfg0 = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        prov, qual = C.resolve_erase(cfg0, {"erase": erase})
        if prov == "cloud" and not C.cloud_key_ready(cfg0):
            return JSONResponse(
                {"error": "云端擦除未配置 API Key：设置 ERASE_CLOUD_API_KEY "
                          "或 config.yaml 的 erase.cloud.api_key / api_key_file"},
                status_code=400)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"读取配置失败：{e}"}, status_code=500)
    erase_opt = erase or prov

    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = job_dir(jid)
    d.mkdir(parents=True, exist_ok=True)
    dst = d / "input.mp4"
    # 分块写盘：大文件不占内存；超过 4GB 直接拒绝
    total = 0
    with open(dst, "wb") as f:
        while True:
            chunk = await file.read(8 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > 4 << 30:
                f.close()
                shutil.rmtree(d, ignore_errors=True)
                return JSONResponse({"error": "文件超过 4GB 上限"}, status_code=413)
            f.write(chunk)
    try:
        info = C.ffprobe_streams(dst)
        if not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
            raise ValueError("没有视频流")
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(d, ignore_errors=True)
        return JSONResponse({"error": f"无法读取视频：{e}"}, status_code=400)
    (d / "opts.json").write_text(json.dumps({"source": source, "target": target,
                                             "erase": erase_opt},
                                            ensure_ascii=False), encoding="utf-8")
    update(jid, state="pending", stage="排队中", progress=0.0,
           name=(file.filename or "input.mp4")[:120], source=source, target=target,
           erase=erase_opt,
           size_mb=round(dst.stat().st_size / 1e6, 1))
    add_log(jid, f"接收 {file.filename} ({round(dst.stat().st_size/1e6,1)} MB) "
                 f"{source}→{target} · 擦除 "
                 f"{'云端' + ('精细化' if qual == 'pro' else '标准版') if prov == 'cloud' else '本地 GPU'}")
    Q.put(jid)
    return {"id": jid}


@app.get("/api/jobs")
def list_jobs():
    items = sorted(STATE.values(), key=lambda x: -x.get("created", 0))[:30]
    return {"jobs": [{k: v for k, v in it.items() if k != "logs"} for it in items]}


@app.get("/api/jobs/{jid}")
def get_job(jid: str, since: int = 0):
    st = STATE.get(jid)
    if not st:
        return JSONResponse({"error": "not found"}, status_code=404)
    out = dict(st)
    logs = out.get("logs") or []
    out["logs"] = logs[since:] if since else logs
    out["log_total"] = len(logs)
    out["has_video"] = (job_dir(jid) / "output.mp4").exists()
    out["has_ass"] = (job_dir(jid) / "work" / "sub.ass").exists()
    return out


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    st = STATE.get(jid)
    if st and st.get("state") in ("running", "pending"):
        return JSONResponse({"error": "任务进行中，暂不支持删除"}, status_code=409)
    with LOCK:
        STATE.pop(jid, None)
    shutil.rmtree(job_dir(jid), ignore_errors=True)
    save_ledger()
    return {"ok": True}


@app.get("/api/jobs/{jid}/video")
def get_video(jid: str):
    p = job_dir(jid) / "output.mp4"
    if not p.exists():
        return JSONResponse({"error": "还没生成"}, status_code=404)
    return FileResponse(str(p), media_type="video/mp4",
                        filename=f"{jid}_subtitled.mp4",
                        headers={"Content-Range-Supported": "1"})


@app.get("/api/jobs/{jid}/source")
def get_source(jid: str):
    p = job_dir(jid) / "input.mp4"
    if not p.exists():
        return JSONResponse({"error": "没有源文件"}, status_code=404)
    return FileResponse(str(p), media_type="video/mp4")


@app.get("/api/jobs/{jid}/ass")
def get_ass(jid: str):
    p = job_dir(jid) / "work" / "sub.ass"
    if not p.exists():
        return JSONResponse({"error": "还没生成"}, status_code=404)
    return FileResponse(str(p), media_type="text/plain",
                        filename=f"{jid}.ass")


@app.get("/api/jobs/{jid}/qc/{name}")
def get_qc(jid: str, name: str):
    if not (name.startswith("qc_") and name.endswith(".jpg") and "/" not in name):
        return JSONResponse({"error": "bad name"}, status_code=400)
    p = job_dir(jid) / "work" / "qc" / name
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(p), media_type="image/jpeg")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8899, log_level="warning")


if __name__ == "__main__":
    main()
