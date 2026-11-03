#!/usr/bin/env python3
"""擦除段基准：同一支片、同一份提取结果，跑不同的 erase 配置，量墙钟 + GPU 利用率。

只做三件事：
  1. 第一次跑时执行 normalize/probe/layout/extract，把结果 pickle 到 probe/cache
  2. 按 --pipe 反复跑 convert.erase()，每轮另存 erased 成片
  3. 全线程采 nvidia-smi，落 csv，算均值/p50/p90；顺带算两轮成片的 PSNR/SSIM

用法：
  python3 probe/erase_bench.py --tag base --pipe 1 --tag2 pipe2 --pipe2 2
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEV = Path(os.environ.get("VE_ROOT") or BASE)
sys.path.insert(0, str(DEV / "webapp"))
sys.path.insert(0, str(DEV / "scripts"))

import yaml  # noqa: E402

CACHE = BASE / "probe" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class GpuSampler(threading.Thread):
    def __init__(self, csv_path: Path, interval=0.5):
        super().__init__(daemon=True)
        self.csv = csv_path
        self.interval = interval
        self.stop_flag = threading.Event()
        self.rows = []

    def run(self):
        q = "timestamp,utilization.gpu,memory.used,power.draw"
        with open(self.csv, "w", buffering=1) as f:
            f.write(q + "\n")
            while not self.stop_flag.is_set():
                try:
                    out = subprocess.run(
                        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10).stdout.strip()
                except Exception:  # noqa: BLE001
                    out = ""
                if out:
                    line = out.splitlines()[0]
                    f.write(line + "\n")
                    parts = [p.strip() for p in line.split(",")]
                    try:
                        self.rows.append((float(parts[1]), float(parts[2])))
                    except Exception:  # noqa: BLE001
                        pass
                self.stop_flag.wait(self.interval)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=5)


def summarize(rows):
    if not rows:
        return {}
    util = [r[0] for r in rows]
    mem = [r[1] for r in rows]
    s = sorted(util)

    def pct(p):
        if not s:
            return 0.0
        i = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
        return s[i]

    busy = [u for u in util if u >= 5]
    return {
        "samples": len(util),
        "util_mean": round(statistics.mean(util), 1),
        "util_p50": round(pct(50), 1),
        "util_p90": round(pct(90), 1),
        "busy_share": round(len(busy) / len(util), 3),
        "util_mean_busy": round(statistics.mean(busy), 1) if busy else 0.0,
        "mem_max_mb": round(max(mem), 0),
    }


def prepare_extract(src: Path, cfg, tag: str):
    """normalize/probe/layout/extract，结果按 src 哈希缓存。"""
    import hashlib
    key = hashlib.sha1(str(src).encode()).hexdigest()[:12] + "_" + str(src.stat().st_mtime_ns)
    cache = CACHE / f"{key}.pkl"
    if cache.exists():
        log(f"复用提取缓存 {cache.name}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    import convert as C
    import pipeline as P

    work = BASE / "probe" / "work" / tag
    work.mkdir(parents=True, exist_ok=True)
    # 拷一份自己的素材：normalize 可能原样返回，那样 erase 会把分段/成片写进
    # 别人正在用的 runtime 目录（实测踩过），必须隔离。
    own = work / "input.mp4"
    if not own.exists() or own.stat().st_size != src.stat().st_size:
        import shutil as _sh
        _sh.copyfile(src, own)
    t0 = time.time()
    norm = C.normalize(own, work, log)
    meta = P.probe(str(norm))
    log(f"素材 {meta['w']}x{meta['h']} @{meta['fps']:.3g}fps {meta['dur']:.1f}s")
    layout = C.detect_layout(norm, meta, log)
    cues, source, shots, per_shot = C.extract(norm, meta, layout, "auto", log,
                                             on_prog=lambda f: None)
    log(f"提取完成 {time.time()-t0:.1f}s cues={len(cues)} shots={len(shots)}")
    data = {"src": str(norm), "meta": meta, "layout": layout, "cues": cues,
            "source": source, "shots": shots, "per_shot": per_shot,
            "extract_s": round(time.time() - t0, 1)}
    with open(cache, "wb") as f:
        pickle.dump(data, f)
    return data


def run_once(data, cfg, pipe: int, tag: str):
    import convert as C

    src = Path(data["src"])
    c = yaml.safe_load(yaml.safe_dump(cfg))
    meta = data["meta"]
    if meta["h"] > meta["w"] and meta["w"] <= 1200:
        c.setdefault("inpaint", {})["fixed_w"] = min(
            int(c["inpaint"].get("fixed_w", 1024)), 768)
    sampler = GpuSampler(BASE / "probe" / f"gpu_{tag}.csv")
    sampler.start()
    t0 = time.time()
    stage = {"last": t0}
    shot_t = []

    def on_prog(frac, msg):
        now = time.time()
        shot_t.append((round(now - stage["last"], 1), msg))
        stage["last"] = now

    erased = C.erase(src, meta, data["layout"], data["shots"], data["per_shot"],
                     c, on_prog=on_prog, log=log, mask_mode="stroke",
                     cues=data["cues"], pipe=pipe)
    dt = time.time() - t0
    sampler.stop()
    stats = summarize(sampler.rows)
    dst = BASE / "probe" / f"erased_{tag}.mp4"
    subprocess.run(["cp", "-f", str(erased), str(dst)], check=True)
    frames = probe_frames(dst)
    rep = {"tag": tag, "pipe": pipe, "seconds": round(dt, 1),
           "frames": frames, "gpu": stats,
           "shots": len(data["shots"]), "out": str(dst),
           "shot_log": shot_t}
    return rep


def probe_frames(path: Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,duration", "-of", "csv=p=0",
         str(path)], capture_output=True, text=True).stdout.strip()
    try:
        return int(out.split(",")[0])
    except Exception:  # noqa: BLE001
        return None


def psnr(a: Path, b: Path):
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", str(a), "-i", str(b),
         "-lavfi", "psnr", "-f", "null", "-"], capture_output=True, text=True)
    txt = (r.stderr or "") + (r.stdout or "")
    for tok in txt.replace("\n", " ").split():
        if tok.startswith("average:"):
            return tok
    return txt[-200:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--pipe", type=int, default=1)
    ap.add_argument("--tag2", default="pipe2")
    ap.add_argument("--pipe2", type=int, default=2)
    ap.add_argument("--only", choices=["1", "2", "both"], default="both")
    ap.add_argument("--tag3", default="pipe4")
    ap.add_argument("--pipe3", type=int, default=0)
    ap.add_argument("--save", action="store_true", default=True)
    args = ap.parse_args()

    cfg = yaml.safe_load((DEV / "config.yaml").read_text(encoding="utf-8"))
    src = Path(args.src).resolve()
    data = prepare_extract(src, cfg, args.tag)
    log(f"提取耗时 {data.get('extract_s')}s，镜头 {len(data['shots'])}")

    reps = []
    if args.only in ("1", "both"):
        reps.append(run_once(data, cfg, args.pipe, args.tag))
        log(json.dumps(reps[-1]["gpu"], ensure_ascii=False))
    if args.only in ("2", "both"):
        reps.append(run_once(data, cfg, args.pipe2, args.tag2))
        log(json.dumps(reps[-1]["gpu"], ensure_ascii=False))
    if args.pipe3:
        reps.append(run_once(data, cfg, args.pipe3, args.tag3))
        log(json.dumps(reps[-1]["gpu"], ensure_ascii=False))

    out = {"src": str(src), "extract_s": data.get("extract_s"), "runs": reps}
    if len(reps) >= 2:
        out["speedup"] = round(reps[0]["seconds"] / max(1e-6, reps[1]["seconds"]), 3)
        out["psnr"] = psnr(Path(reps[0]["out"]), Path(reps[1]["out"]))
        out["frames_equal"] = reps[0]["frames"] == reps[1]["frames"]
    for r in reps:
        r.pop("shot_log", None)
    p = BASE / "probe" / f"bench_{args.tag}_{int(time.time())}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log("RESULT " + json.dumps({k: v for k, v in out.items() if k != "runs"},
                               ensure_ascii=False))
    log(f"明细 {p}")


if __name__ == "__main__":
    main()
