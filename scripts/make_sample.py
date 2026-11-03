#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成一个 24 秒竖屏测试片：4 个不同背景 + 中文硬字幕

用途：没有真实短剧素材时，验证 OCR -> 擦除 -> 翻译 -> 压制 的完整链路。
注意：合成画面只能验证"流程跑通"和"有无明显残影"，真实质量要用真素材评估。
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "samples"
OUT.mkdir(parents=True, exist_ok=True)

CUES = [
    (0.0, 2.5, "三年了，你终于回来了"),
    (2.5, 5.0, "这句话我在心里练了一千遍"),
    (5.0, 7.5, "可你连看都不看我一眼"),
    (7.5, 10.0, "我从来没想过要离开你"),
    (10.0, 12.5, "那这三年，\\N你去了哪里"),
    (12.5, 15.0, "我去做了一个了断"),
    (15.0, 18.0, "从今天起，没有人能再伤害你"),
    (18.0, 21.0, "包括我自己"),
    (21.0, 24.0, "别走。"),
]

ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: CN,Noto Sans CJK SC,64,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,4,1,2,80,80,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ts(t):
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def write_ass(path):
    lines = [ASS_HEAD]
    for a, b, text in CUES:
        lines.append(f"Dialogue: 0,{ts(a)},{ts(b)},CN,,0,0,0,,{text}\n")
    path.write_text("".join(lines), encoding="utf-8")


def ff(cmd):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ass = OUT / "sample.ass"
    write_ass(ass)

    bgs = [
        ("testsrc2=s=1080x1920:r=25:d=6", "bg1.mp4"),
        ("mandelbrot=s=1080x1920:r=25", "bg2.mp4"),
        ("gradients=s=1080x1920:r=25:speed=0.04", "bg3.mp4"),
        ("smptebars=s=1080x1920:r=25", "bg4.mp4"),
    ]
    for src, name in bgs:
        ff(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src, "-t", "6",
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", str(OUT / name)])

    (OUT / "bglist.txt").write_text(
        "".join(f"file '{OUT / n}'\n" for _, n in bgs), encoding="utf-8")
    ff(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
        "-i", str(OUT / "bglist.txt"), "-c", "copy", str(OUT / "bg.mp4")])

    ff(["ffmpeg", "-v", "error", "-y", "-i", str(OUT / "bg.mp4"),
        "-vf", f"ass={ass}", "-c:v", "libx264", "-crf", "18",
        "-preset", "veryfast", "-pix_fmt", "yuv420p", str(OUT / "sample.mp4")])
    print("done:", OUT / "sample.mp4")


if __name__ == "__main__":
    main()
