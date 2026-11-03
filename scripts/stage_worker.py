#!/usr/bin/env python3
"""单阶段 worker：prepare / translate / burn，供 batch_pipeline.py 调用。"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "webapp"))

import yaml  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "translate", "burn"])
    ap.add_argument("--job", required=True)
    ap.add_argument("--target", default="ja")
    ap.add_argument("--source", default="auto")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--fp16", choices=["on", "off"], default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.fp16:
        cfg.setdefault("inpaint", {})["fp16"] = args.fp16 == "on"
    # 显存/并发调参用：ERASE_BATCH 覆盖 inpaint.batch（不改 config.yaml）
    _bs = os.environ.get("ERASE_BATCH", "").strip()
    if _bs:
        cfg.setdefault("inpaint", {})["batch"] = max(1, int(_bs))

    # 并发时限制每进程线程数，把 CPU 留给 OCR/光流/x264
    import torch
    th = int(os.environ.get("ERASE_TORCH_THREADS", "0") or 0)
    if th > 0:
        torch.set_num_threads(th)
    try:
        import cv2
        ct = int(os.environ.get("ERASE_CV_THREADS", "0") or 0)
        if ct > 0:
            cv2.setNumThreads(ct)
    except Exception:  # noqa: BLE001
        pass

    import convert as C

    def log(m):
        print(time.strftime("%H:%M:%S"), m, flush=True)

    opts = {"source": args.source, "target": args.target, "cfg": cfg}
    job = Path(args.job)
    t0 = time.time()
    if args.stage == "prepare":
        st = C.prepare(job, opts, log)
        out = {"stage": "prepare", "seconds": st["seconds_prepare"],
               "cues": len(st["cues"]), "source": st["source"]}
    elif args.stage == "translate":
        st = C.translate_state(job, opts, log)
        out = {"stage": "translate", "seconds": st.get("seconds_translate"),
               "font": st.get("font_size"), "target": st.get("target_lang")}
    else:
        path, rep = C.burn_state(job, log)
        out = {"stage": "burn", "seconds": rep.get("seconds_burn"),
               "out": str(path)}
    out["wall"] = round(time.time() - t0, 1)
    print("RESULT " + json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
