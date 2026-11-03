#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""webapp 兼容性回归：容器/编码/旋转/VFR/无音轨/目标语言。

用法（V100 空闲时跑，约 3-4 分钟）：
  cd webapp && /usr/bin/python3 selftest.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))
import convert as C  # noqa: E402

FIX = ROOT / "assets" / "fixtures"          # 固定回归片源，只读
BASE = ROOT / "runtime" / "selftest"         # 每次跑的 job 目录
EN_SRC = ROOT / "assets" / "sources" / "en_1.mp4"
ZH_SRC = ROOT / "assets" / "sources" / "namenyuan_1080.mp4"
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
CFG["inpaint"]["fp16"] = False


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cut(src, out, start=0, dur=12, extra=None):
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(dur),
           "-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k"]
    cmd += extra or []
    cmd += [str(out)]
    run(cmd)


def make_inputs():
    FIX.mkdir(parents=True, exist_ok=True)
    a = FIX / "en_cut.mp4"
    b = FIX / "zh_cut.mp4"
    if not a.exists():
        cut(EN_SRC, a, 0, 12)
    if not b.exists():
        cut(ZH_SRC, b, 55, 12)

    def variant(name, maker):
        p = FIX / name
        if not p.exists():
            maker(p)
        return p

    noaudio = variant("en_noaudio.mp4", lambda p: run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(a), "-an", "-c:v", "copy", str(p)]))
    hevc = variant("en_hevc.mp4", lambda p: run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(a), "-c:v", "libx265", "-crf", "26",
         "-preset", "ultrafast", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1",
         "-c:a", "aac", str(p)]))
    rot = variant("zh_rot90.mp4", lambda p: run(
        ["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", str(b),
         "-c", "copy", str(p)]))
    vfr = variant("zh_vfr.mp4", lambda p: run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(b),
         "-vf", "select='not(mod(n,3))'", "-vsync", "vfr", "-c:v", "libx264",
         "-crf", "18", "-c:a", "aac", str(p)]))
    mov = variant("zh_mov.mov", lambda p: run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(b), "-c", "copy", str(p)]))
    return a, noaudio, hevc, rot, vfr, mov


def probe(path):
    raw = subprocess.check_output(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
         "-show_format", str(path)])
    return json.loads(raw)


def case(name, src, source, target, want_audio, want_size=None, want_kana=False):
    d = BASE / "jobs" / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copyfile(src, d / "input.mp4")
    opts = {"source": source, "target": target, "cfg": CFG}
    t0 = time.time()
    try:
        out, rep = C.convert(d, opts, log=lambda m: None)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        return False

    info = probe(out)
    streams = info.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    problems = []
    if not video:
        problems.append("no video")
    if want_audio and not audio:
        problems.append("audio lost")
    if not want_audio and audio:
        problems.append("unexpected audio")
    if want_size and video:
        got = (video[0].get("width"), video[0].get("height"))
        if got != want_size:
            problems.append(f"size {got} != {want_size}")
    if rep.get("cues", 0) < 1 or rep.get("translated", 0) < rep.get("cues", 0):
        problems.append(f"cues {rep.get('cues')} translated {rep.get('translated')}")
    if bool(rep.get("audio")) != want_audio:
        problems.append(f"report audio={rep.get('audio')}")
    if want_kana:
        ass = (d / "work" / "sub.ass").read_text(encoding="utf-8")
        if not any("\u3040" <= ch <= "\u30ff" for ch in ass):
            problems.append("no kana in ASS")

    dt = time.time() - t0
    if problems:
        print(f"FAIL {name}: {', '.join(problems)} ({dt:.0f}s)")
        return False
    print(f"PASS {name}: {rep['cues']} cues, {video[0]['width']}x{video[0]['height']}, "
          f"audio={bool(audio)}, {dt:.0f}s")
    return True


def main():
    if not EN_SRC.exists() or not ZH_SRC.exists():
        sys.exit("缺少测试片源")
    a, noaudio, hevc, rot, vfr, mov = make_inputs()
    ok = True
    ok &= case("en_plain", a, "other", "zh", True)
    ok &= case("en_noaudio", noaudio, "other", "zh", False)
    ok &= case("en_hevc", hevc, "other", "zh", True)
    ok &= case("zh_rot90", rot, "zh", "en", True, want_size=(1920, 1080))
    ok &= case("zh_vfr", vfr, "zh", "en", True)
    ok &= case("zh_mov", mov, "zh", "en", True)
    ok &= case("zh_to_ja", FIX / "zh_cut.mp4", "zh", "ja", True, want_kana=True)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
