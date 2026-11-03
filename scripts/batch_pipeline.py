#!/usr/bin/env python3
"""多视频并发流水线：K 路 prepare（OCR/擦除/返修）→ 统一翻译 → 统一压制。

用法：
  python3 scripts/batch_pipeline.py --src-dir /home/pan/视频 --out outputs/batch_auto --jobs 2

设计：prepare 阶段不加载 Hy-MT2，K 个进程共享 V100；翻译阶段临时拉起 Hy-MT2，
翻译完即卸载，把 9GB 显存还给下一批。压制阶段用 NVENC，CPU 只做封装。
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
WORKER = ROOT / "scripts/stage_worker.py"
HOST = "127.0.0.1"
PORT = 1509
HY_BIN = os.environ.get(
    "HYMT2_BIN", "/home/pan/work/turboquant/build-v100-cuda/bin/llama-server")
HY_MODEL = os.environ.get(
    "HYMT2_MODEL", "/media/pan/未来/LLM/hy-mt2/Hy-MT2-7B-Q4_K_M.gguf")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def healthy(timeout=2.0):
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def start_translator(logpath):
    """已运行则复用；否则拉起并等待健康。返回 (proc, started_by_us)。"""
    if healthy():
        log("Hy-MT2 已在运行，复用")
        return None, False
    log("启动 Hy-MT2（翻译完会卸载）")
    cmd = [HY_BIN, "-m", HY_MODEL, "--device", "CUDA0", "--host", HOST,
           "--port", str(PORT), "-ngl", "99", "-np", "4", "-c", "32768", "--jinja"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    f = open(logpath, "ab")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    for _ in range(180):
        if healthy():
            log("Hy-MT2 ready")
            return p, True
        if p.poll() is not None:
            f.close()
            raise RuntimeError(f"Hy-MT2 启动失败，日志 {logpath}")
        time.sleep(1)
    p.terminate()
    f.close()
    raise TimeoutError("Hy-MT2 启动超时")


def stop_translator(p):
    if p is None:
        return
    p.terminate()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
    log("Hy-MT2 已卸载")


def base_env(jobs=4, erase_pipe=0, writer="auto"):
    env = os.environ.copy()
    # 镜头级流水：一路 prepare 里同时推进几支镜头。实测 K=4 时 2 路最稳
    # （pipe=4 峰值贴到 16.05GB/16GB）；K<=2 时 4 路能吃到 3x 擦除提速。
    pipe = int(erase_pipe) if erase_pipe else max(2, min(4, 8 // max(1, jobs)))
    env["ERASE_PIPE"] = str(pipe)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    env.setdefault("ERASE_TORCH_THREADS", "8")
    env.setdefault("ERASE_CV_THREADS", "4")
    # 中间层编码：NVENC 每个实例占 372MB 显存，并发越高越挤；x264 不吃显存，
    # 代价是吃 CPU（实测 1080x1920 约 80fps/路）。>=5 路时改 x264 才能保住并发。
    if writer == "x264" or (writer == "auto" and jobs >= 5):
        env["ERASE_WRITER"] = "x264"
    # x264 中间层时按流水路数分线程，否则 12×2 路会把 CPU 抢空
    if env.get("ERASE_WRITER", "").lower() == "x264":
        env["ERASE_X264_THREADS"] = str(max(2, 12 // max(1, pipe)))
    else:
        env.setdefault("ERASE_X264_THREADS", "12")
    return env


def worker_cmd(stage, job, args):
    cmd = [PY, str(WORKER), stage, "--job", str(job),
           "--target", args.target, "--source", args.source]
    if args.fp16:
        cmd += ["--fp16", args.fp16]
    return cmd


def media_duration(path: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)], text=True, timeout=20)
        return float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return 0.0


def source_fingerprint(path: Path) -> dict:
    """大小 + 首尾 1MB 哈希：同名换片也能识别，不用整片读一遍。"""
    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        head = f.read(1 << 20)
        h.update(head)
        if size > (1 << 20):
            f.seek(size - (1 << 20))
            h.update(f.read(1 << 20))
    return {"size": size, "sha256_head_tail": h.hexdigest()}


def load_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def prepare_done(job: Path) -> bool:
    st = load_json(job / "state.json")
    return bool(st and st.get("cues") and st.get("layout") and st.get("meta")
                and st.get("erased") and Path(st["erased"]).exists()
                and st.get("seconds_prepare") is not None)


def translate_done(job: Path, target: str) -> bool:
    st = load_json(job / "state.json")
    if not st or not st.get("targets") or not st.get("ass"):
        return False
    if not Path(st["ass"]).exists():
        return False
    tgt = str(st.get("target_lang") or "").lower()
    return tgt == target.lower() or tgt.startswith(target.lower())


def burn_done(job: Path) -> bool:
    rep = load_json(job / "report.json")
    return bool(rep and (rep.get("render_check") or {}).get("ok")
                and (job / "output.mp4").exists())


def kill_proc(p: subprocess.Popen):
    """连子进程一起收掉，避免 ffmpeg/worker 残留继续占 GPU。"""
    if p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def run_stage(stage, job, args, logpath, env, timeout):
    t0 = time.time()
    timed_out = False
    with open(logpath, "ab") as f:
        p = subprocess.Popen(worker_cmd(stage, job, args), stdout=f,
                             stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            f.write(f"\n[{time.strftime('%H:%M:%S')}] {stage} 超时 "
                    f"{timeout:.0f}s，终止进程组\n".encode())
            f.flush()
            kill_proc(p)
            rc = -9
    return rc, round(time.time() - t0, 1), timed_out


def finish_job(job: Path, args, env, out: Path, prep: dict) -> tuple[str, bool]:
    """压制一支片并落盘交付物。

    从 main 里搬出来的：压制只吃 NVENC/CPU，和翻译（占 GPU 显存）、以及别的
    支的压制都能同时跑，所以它进线程池，翻译完一支就立刻开始，不再等全批翻译完。
    只碰自己这支的 job 目录和 outputs 里自己那几个文件名。
    """
    rec = prep.setdefault(job.name, {})
    dur = media_duration(job / "input.mp4")
    burn_timeout = max(600.0, 2.0 * dur) if dur > 0 else 600.0
    reused = not args.fresh and burn_done(job)
    if reused:
        rep = load_json(job / "report.json") or {}
        dt = round(float(rep.get("seconds_burn") or 0), 1)
        log(f"[{job.name}] burn 已完成，复用（{dt:.0f}s）")
        ok = True
        rec["burn_reused"] = True
    else:
        log(f"[{job.name}] burn START（超时 {burn_timeout:.0f}s）")
        rc, dt, timed_out = run_stage("burn", job, args,
                                      job / "burn.log", env, burn_timeout)
        ok = rc == 0 and (job / "output.mp4").exists()
        log(f"[{job.name}] burn {'OK' if ok else 'FAIL'} {dt:.0f}s rc={rc}"
            + ("（超时）" if timed_out else ""))
        rec["burn_timeout"] = timed_out
    rec["burn_ok"] = ok
    rec["burn_s"] = dt
    if not ok:
        return job.name, False
    stem = job.name
    shutil.copyfile(job / "output.mp4", out / f"{stem}_{args.target}.mp4")
    qc_src = job / "work" / "qc"
    if qc_src.is_dir():
        shutil.copytree(qc_src, out / f"{stem}_qc", dirs_exist_ok=True)
    for name, suffix in (("work/sub.ass", "ass"),
                         ("prepare.log", "prepare.log"),
                         ("translate.log", "translate.log"),
                         ("burn.log", "burn.log")):
        p = job / name
        if p.exists():
            shutil.copyfile(p, out / f"{stem}_{args.target}.{suffix}")
    rp = job / "report.json"
    if rp.exists():
        rd = load_json(rp) or {}
        if qc_src.is_dir():
            rd["qc_dir"] = f"{stem}_qc"
        (out / f"{stem}_{args.target}.report.json").write_text(
            json.dumps(rd, ensure_ascii=False, indent=1), encoding="utf-8")
    return stem, True


def run_prepare(pending, args, env):
    pending = list(pending)
    running, done = [], {}
    # 已完成的 prepare 直接复用，不再吃 GPU/CPU
    remain = []
    for job in pending:
        if not args.fresh and prepare_done(job):
            st = load_json(job / "state.json") or {}
            dt = round(float(st.get("seconds_prepare") or 0), 1)
            done[job.name] = {"prepare_ok": True, "prepare_s": dt,
                              "rc": 0, "reused": True}
            log(f"[{job.name}] prepare 已完成，复用（{dt:.0f}s）")
        else:
            remain.append(job)
    pending = remain

    while pending or running:
        while pending and len(running) < args.jobs:
            job = pending.pop(0)
            dur = media_duration(job / "input.mp4")
            timeout = max(1800.0, 8.0 * dur) if dur > 0 else 1800.0
            logpath = job / "prepare.log"
            log(f"[{job.name}] prepare START（超时 {timeout:.0f}s）")
            f = open(logpath, "ab")
            p = subprocess.Popen(worker_cmd("prepare", job, args), stdout=f,
                                 stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True)
            running.append({"job": job, "proc": p, "log": f, "t0": time.time(),
                            "deadline": time.time() + timeout, "timeout": timeout})
        time.sleep(1)
        for r in list(running):
            rc = r["proc"].poll()
            if rc is None:
                if time.time() > r["deadline"]:
                    r["log"].write(f"\n[{time.strftime('%H:%M:%S')}] prepare 超时 "
                                   f"{r['timeout']:.0f}s，终止进程组\n".encode())
                    r["log"].flush()
                    kill_proc(r["proc"])
                    rc = -9
                    r["timed_out"] = True
                else:
                    continue
            r["log"].close()
            dt = round(time.time() - r["t0"], 1)
            ok = rc == 0 and (r["job"] / "state.json").exists()
            log(f"[{r['job'].name}] prepare {'OK' if ok else 'FAIL'} {dt:.0f}s rc={rc}"
                + ("（超时）" if r.get("timed_out") else ""))
            done[r["job"].name] = {"prepare_ok": ok, "prepare_s": dt, "rc": rc,
                                   "timeout": bool(r.get("timed_out"))}
            running.remove(r)
    return done


def sampler_start(path):
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    f = open(path, "wb")
    p = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits", "-lms", "1000"],
        stdout=f, stderr=subprocess.DEVNULL, env=env)
    return p, f


def sampler_stop(p, f):
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
    f.close()


def gpu_stats(path):
    utils, mems = [], []
    try:
        for line in Path(path).read_text(errors="replace").splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                utils.append(float(parts[1]))
                mems.append(float(parts[2]))
            except ValueError:
                continue
    except Exception:  # noqa: BLE001
        return {}
    if not utils:
        return {}
    return {"samples": len(utils),
            "util_mean": round(sum(utils) / len(utils), 1),
            "util_p50": round(sorted(utils)[len(utils) // 2], 1),
            "util_max": max(utils),
            "util_ge50_pct": round(100 * sum(1 for u in utils if u >= 50) / len(utils), 1),
            "mem_mean_mb": round(sum(mems) / len(mems)),
            "mem_max_mb": max(mems)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="*", default=[])
    ap.add_argument("--src-dir", default="/home/pan/视频")
    ap.add_argument("--out", default=str(ROOT / "outputs/batch_auto"))
    ap.add_argument("--work", default=str(ROOT / "runtime/batch_auto"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--target", default="ja")
    ap.add_argument("--source", default="auto")
    ap.add_argument("--fp16", choices=["on", "off"], default=None)
    ap.add_argument("--translator", choices=["auto", "external"], default="auto")
    ap.add_argument("--keep-translator", action="store_true")
    ap.add_argument("--keep-work", action="store_true",
                    help="保留 runtime 下的中间文件（默认成功后清理）")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略已有工作目录，强制从头重跑")
    ap.add_argument("--writer", choices=["auto", "nvenc", "x264"], default="auto",
                    help="擦除中间层编码：auto=并发>=5 用 x264，否则 NVENC")
    ap.add_argument("--erase-pipe", type=int, default=0,
                    help="每路 prepare 内并行的镜头数（0=按并发自动：K<=2 用 4，K>=3 用 2）")
    ap.add_argument("--burn-jobs", type=int, default=3,
                    help="压制并发（1=老行为，逐支串行）")
    ap.add_argument("--force-jobs", action="store_true",
                    help="跳过显存自动降并发（默认按 3.1GB/路估算）")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    srcs = [Path(p) for p in args.src] if args.src else sorted(Path(args.src_dir).glob("*.mp4"))
    if args.limit:
        srcs = srcs[:args.limit]
    if not srcs:
        raise SystemExit("没有输入视频")
    # 按空闲显存自动降并发。实测：单路常驻约 2.6GB（LaMa + OCR 上下文），
    # 每开一条镜头流水通道再加约 0.38GB（NVENC 中间层实例，x264 不加）。
    # K=5+pipe2+NVENV 会 OOM、K=6+x264 会 cuFFT 崩，所以这里必须按显存算准。
    try:
        if not args.force_jobs:
            free_mb = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True).strip().splitlines()[0])
            while args.jobs > 1:
                pipe = args.erase_pipe if args.erase_pipe > 0 else max(2, min(4, 8 // args.jobs))
                x264 = args.writer == "x264" or (args.writer == "auto" and args.jobs >= 5)
                per_job = 2600 + (0 if x264 else 380 * pipe)
                if free_mb - 1800 >= per_job * args.jobs:
                    break
                log(f"空闲显存 {free_mb}MB 装不下 {args.jobs} 路（每路约 "
                    f"{per_job}MB，pipe={pipe}），降到 {args.jobs - 1} 路")
                args.jobs -= 1
    except Exception:  # noqa: BLE001
        pass
    out = Path(args.out)
    work = Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    batch_t0 = time.time()
    jobs = []
    for src in srcs:
        job = work / src.stem
        fp = source_fingerprint(src)
        saved = load_json(job / ".source.json")
        reuse = (not args.fresh and job.is_dir()
                 and (job / "input.mp4").exists()
                 and saved and saved.get("fingerprint") == fp)
        if reuse:
            log(f"[{job.name}] 复用已有工作目录（源文件未变）")
        else:
            shutil.rmtree(job, ignore_errors=True)
            job.mkdir(parents=True)
            shutil.copyfile(src, job / "input.mp4")
            (job / ".source.json").write_text(
                json.dumps({"source": str(src), "fingerprint": fp},
                           ensure_ascii=False, indent=1), encoding="utf-8")
        jobs.append(job)

    sampler, sf = sampler_start(work / "gpu.csv")
    env = base_env(args.jobs, args.erase_pipe, args.writer)
    if args.erase_pipe < 0:
        env["ERASE_PIPE"] = "1"  # 负数=关掉流水（老行为），用于 A/B 对照
    translator = None
    started_by_us = False
    try:
        log(f"批处理 {len(jobs)} 支，prepare 并发 {args.jobs}，目标 {args.target}")
        prep = run_prepare(jobs, args, env)
        ok_jobs = [j for j in jobs if prep.get(j.name, {}).get("prepare_ok")]
        if not ok_jobs:
            raise RuntimeError("所有视频 prepare 都失败")

        need_translate = [j for j in ok_jobs
                          if args.fresh or not translate_done(j, args.target)]
        if need_translate:
            if args.translator == "auto":
                translator, started_by_us = start_translator(work / "translator.log")
            elif not healthy():
                raise RuntimeError("翻译后端未运行（--translator external）")
        else:
            log("所有视频翻译已完成，跳过翻译阶段")

        translated = []
        # 压制（NVENC + 封装）不占擦除用的显存，翻译完一支就丢进池子开跑：
        # 尾巴从「翻译 1.5min 串行 + 压制 5.0min 串行」压成 max(两者/并发)。
        burn_jobs = max(1, int(getattr(args, "burn_jobs", 3)))
        pool = ThreadPoolExecutor(max_workers=burn_jobs,
                                  thread_name_prefix="burn")
        burn_futs = []

        def enqueue(job):
            translated.append(job)
            burn_futs.append(pool.submit(finish_job, job, args, env, out, prep))

        for job in ok_jobs:
            if job not in need_translate:
                st = load_json(job / "state.json") or {}
                dt = round(float(st.get("seconds_translate") or 0), 1)
                log(f"[{job.name}] translate 已完成，复用（{dt:.0f}s）")
                prep[job.name]["translate_ok"] = True
                prep[job.name]["translate_s"] = dt
                prep[job.name]["translate_reused"] = True
                enqueue(job)
                continue
            log(f"[{job.name}] translate START")
            rc, dt, timed_out = run_stage("translate", job, args,
                                          job / "translate.log", env, 900.0)
            ok = rc == 0 and (job / "state.json").exists()
            if ok:
                ok = translate_done(job, args.target)
            log(f"[{job.name}] translate {'OK' if ok else 'FAIL'} {dt:.0f}s rc={rc}"
                + ("（超时）" if timed_out else ""))
            prep[job.name]["translate_ok"] = ok
            prep[job.name]["translate_s"] = dt
            prep[job.name]["translate_timeout"] = timed_out
            if ok:
                enqueue(job)

        if started_by_us and not args.keep_translator:
            stop_translator(translator)
            translator = None
        if not translated:
            pool.shutdown(wait=True)
            raise RuntimeError("没有视频完成翻译，无法压制")
        # 等所有压制收尾（此时翻译可能已经结束，压制还在跑）
        for fu in burn_futs:
            try:
                fu.result()
            except Exception as e:  # noqa: BLE001
                log(f"burn 线程异常：{type(e).__name__} {e}")
        pool.shutdown(wait=True)

    finally:
        sampler_stop(sampler, sf)
        if translator is not None:
            stop_translator(translator)

    batch_s = round(time.time() - batch_t0, 1)
    rows, src_total = [], 0.0
    for job in jobs:
        row = {"stem": job.name, **prep.get(job.name, {})}
        rp = job / "report.json"
        if rp.exists():
            d = json.loads(rp.read_text(encoding="utf-8"))
            row.update({"duration": d.get("duration"), "cues": d.get("cues"),
                        "font": d.get("font_size"), "seconds": d.get("seconds"),
                        "seconds_prepare": d.get("seconds_prepare"),
                        "seconds_translate": d.get("seconds_translate"),
                        "seconds_burn": d.get("seconds_burn"),
                        "qc_suspects": len((d.get("qc") or {}).get("suspects") or [])})
            src_total += float(d.get("duration") or 0)
        rows.append(row)
    ok_n = sum(1 for r in rows if r.get("burn_ok"))
    summary = {
        "videos": rows,
        "videos_ok": ok_n,
        "batch_seconds": batch_s,
        "batch_minutes": round(batch_s / 60, 1),
        "source_seconds": round(src_total, 1),
        "realtime_ratio": round(batch_s / src_total, 2) if src_total else None,
        "per_video_seconds": round(batch_s / max(ok_n, 1), 1),
        "videos_per_day_100": round(86400 / (batch_s / max(ok_n, 1)), 1) if ok_n else None,
        "source_hours_per_day_100": round(src_total / batch_s * 24, 2) if batch_s else None,
        "gpu": gpu_stats(work / "gpu.csv"),
    }
    (out / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.keep_work:
        for job in jobs:
            if prep.get(job.name, {}).get("burn_ok"):
                shutil.rmtree(job, ignore_errors=True)
        log("中间文件已清理（--keep-work 可保留）")
    log("批处理完成 " + json.dumps(
        {k: summary[k] for k in ("videos_ok", "batch_minutes", "realtime_ratio",
                                 "videos_per_day_100", "source_hours_per_day_100")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
