#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对已擦除成片做 翻译 -> 重断句 -> ASS -> 压制（不重跑擦除）

用法：
  python post_translate.py outputs/cli/xxx --out outputs/cli/xxx_final
  # 读 <dir>/erased.mp4 与 <dir>/report.json 里的 cue_debug
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

import pipeline as P

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir", help="已擦除目录（含 erased.mp4 和 report.json）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--no-burn", action="store_true")
    ap.add_argument("--font-size", type=int, default=None, help="ASS 字号（px），默认按画面推算")
    ap.add_argument("--margin-v", type=int, default=None, help="距底边像素（Alignment=2）")
    ap.add_argument("--font", default=None, help="ASS 字体名")
    ap.add_argument("--max-chars", type=int, default=None, help="每行字符上限")
    ap.add_argument("--target", default="en", help="只用日志：en/zh 等")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sdir = Path(args.src_dir).resolve()
    out = Path(args.out).resolve() if args.out else sdir
    out.mkdir(parents=True, exist_ok=True)

    rep = json.loads((sdir / "report.json").read_text(encoding="utf-8"))
    erased = sdir / "erased.mp4"
    meta = P.probe(erased)
    cues = [{"start": float(a), "end": float(b), "text": t}
            for a, b, t in rep.get("cue_debug") or []]
    cues = P.merge_overlaps(cues)
    # 单字语气词不值得上字幕（OCR 常把印地语行读成“奥/嗯”，翻出来像噪声）
    junk = set("嗯啊哦喔唉哎诶呃噢奥唔嘿哈呵")
    cues = [c for c in cues
            if not (len(c["text"].strip()) == 1 and c["text"].strip() in junk)]
    P.log(f"输入 {erased}  {meta['w']}x{meta['h']} {meta['dur']:.1f}s  "
          f"{len(cues)} 条字幕")

    t0 = time.time()
    for c in cues:
        c["zh"] = c["text"]
    en = P.translate(cues, cfg)

    max_chars = args.max_chars or (24 if meta["h"] > meta["w"] else 42)
    max_lines = 2
    final_cues, problems = [], []
    for c, t in zip(cues, en):
        planned, issues = P.plan_cues(t, c["start"], c["end"], max_chars=max_chars,
                                      max_lines=max_lines, cps=17.0)
        for x in planned:
            x["en"] = x["text"]
            final_cues.append(x)
        problems.extend(issues)
        problems.extend(P.check_cues(planned, max_chars=max_chars,
                                     max_lines=max_lines))
    P.log(f"断句后 {len(final_cues)} 条，问题 {len(problems)} 个：{problems[:8]}")

    ass = out / "final.ass"
    P.write_ass(final_cues, ass, meta, max_chars, max_lines,
                font_size=args.font_size, margin_v=args.margin_v, font=args.font)
    final = out / "final.mp4"
    if args.no_burn:
        P.log("跳过压制（--no-burn）")
    else:
        t1 = time.time()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(erased),
                        "-vf", f"ass={ass}", "-c:v", "h264_nvenc", "-preset", "p5",
                        "-cq", "21", "-b:v", "0", "-c:a", "copy",
                        "-movflags", "+faststart", str(final)], check=True)
        P.log(f"压制完成 {time.time()-t1:.1f}s")

    report = {
        "src": str(erased),
        "out": str(final),
        "duration": meta["dur"],
        "cues_in": len(cues),
        "cues_out": len(final_cues),
        "translated": sum(1 for x in en if x and x != "[?]"),
        "cue_issues": len(problems),
        "seconds": round(time.time() - t0, 1),
        "cues": [[round(c["start"], 2), round(c["end"], 2), c["text"]]
                 for c in final_cues],
    }
    (out / "report_translate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    P.log("完成：" + str(final))


if __name__ == "__main__":
    main()
