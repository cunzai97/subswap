#!/usr/bin/env python3
"""翻译策略 A/B：同一批 cue 跑多种 prompt/上下文方案，对人工参考打分。

用法：
  python3 scripts/translation_lab.py --variants baseline_norm,glossary,ctx,ctx_glossary
  python3 scripts/translation_lab.py --all --out /tmp/translab.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:1509/v1"
MODEL = "Hy-MT2-7B-Q4_K_M.gguf"
KEY = "local"
TEMP = 0.2
MAXTOK = 512
WORKERS = 4

GLOSSARY = (
    "固定译名（必须照用）：朵朵→ドウドウ；温妍→ウェン・イェン；"
    "容遇→容遇；阿渊→阿渊；姨姨→おばさん。"
)
BASE_JA = (
    "将以下文本翻译成自然、口语化的日语，保留人物称呼和语气，"
    "同一部剧的人称和语气保持一致。每条译文尽量控制在 24 个字符以内，"
    "能一行显示就一行。只输出译文：\n\n{text}"
)
STYLE = "这是竖屏短剧字幕，译文要口语化、短句，贴合人物关系。"

CJK = r"\u3400-\u9fff"
CJK_GAP = re.compile(rf"(?<=[{CJK}])\s+(?=[{CJK}])")


def norm_cn(s: str) -> str:
    s = s.replace("\u3000", " ")
    s = CJK_GAP.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def call(prompt: str, retries: int = 2) -> str:
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMP, "max_tokens": MAXTOK}
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"})
    last = ""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            out = (data["choices"][0]["message"]["content"] or "").strip()
            out = out.strip("`").strip().strip('"').strip()
            if out:
                return out
        except Exception as e:  # noqa: BLE001
            last = str(e)
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(last or "empty")


def load_data(state_path: Path, ref_path: Path):
    st = json.loads(state_path.read_text(encoding="utf-8"))
    cues = st["cues"]

    def sec(t):
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    ref = []
    for line in ref_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        p = line.split(",", 9)
        ref.append({"start": sec(p[1]), "text": p[9].strip().replace("\\N", "")})
    by_start = {round(r["start"], 2): r["text"] for r in ref}
    for c in cues:
        c["text"] = norm_cn(c["text"])
        c["ref"] = by_start.get(round(c["start"], 2), "")
    missing = [i for i, c in enumerate(cues) if not c["ref"]]
    if missing:
        raise SystemExit(f"参考字幕缺 {len(missing)} 条：{missing[:5]}")
    return cues


def ctx_lines(cues, i, n=2, oracle=False):
    lo = max(0, i - n)
    rows = []
    for j in range(lo, i):
        if oracle:
            rows.append(f"{j+1}. 原文：{cues[j]['text']}\n   日语：{cues[j]['ref']}")
        else:
            rows.append(f"{j+1}. {cues[j]['text']}")
    return "\n".join(rows)


GLOSS_MAP = [("朵朵", "ドウドウ"), ("温妍", "ウェン・イェン"),
             ("容遇", "容遇"), ("阿渊", "阿渊"), ("姨姨", "おばさん")]


def gloss_for(text):
    hits = [f"{s}→{t}" for s, t in GLOSS_MAP if s in text]
    return ("固定译名（必须照用）：" + "；".join(hits) + "。\n") if hits else ""


def strip_num(s):
    s = s.strip()
    while True:
        t = re.sub(r"^\s*\d+\s*[.．、)]\s*", "", s)
        if t == s:
            return s.strip()
        s = t


def run_one(name, cues, i):
    c = cues[i]
    text = c["text"]
    if name == "baseline_raw":
        return call(BASE_JA.format(text=c["text"]))
    if name == "baseline_norm":
        return call(BASE_JA.format(text=text))
    if name == "glossary":
        return call(BASE_JA.format(text=text) + "\n\n" + GLOSSARY + "\n" + STYLE)
    if name == "glossary_v2":
        return call(gloss_for(text) + BASE_JA.format(text=text))
    if name == "default_official":
        return call("将以下文本翻译为日语，注意只需要输出翻译后的结果，"
                    f"不要额外解释：\n\n{text}")
    if name == "glossary_official":
        hits = [f"{s} 翻译成 {t}" for s, t in GLOSS_MAP if s in text]
        prefix = ("参考下面的翻译：\n" + "\n".join(hits) + "\n") if hits else ""
        return call(prefix + "将以下文本翻译为日语，注意只需要输出翻译后的结果，"
                    f"不要额外解释：\n\n{text}")
    if name == "style":
        return call(BASE_JA.format(text=text) + " " + STYLE)
    if name in ("ctx", "ctx1", "ctx_glossary", "ctx_oracle"):
        prev = ctx_lines(cues, i, 1 if name == "ctx1" else 2,
                         oracle=(name == "ctx_oracle"))
        prompt = ("以下是同一部短剧的前文，只用于理解指代和语气，不要翻译：\n"
                  f"前文：\n{prev}\n\n"
                  "参考上面的信息，把下面的台词翻译成自然、口语化的日语，"
                  "保留人物称呼和语气，只输出当前台词的译文，不要翻译前文，不要解释：\n"
                  f"{text}")
        if name == "ctx_glossary":
            prompt = GLOSSARY + "\n" + STYLE + "\n\n" + prompt
        return call(prompt)
    if name in ("pair2_one", "pair2_one_gloss"):
        if i == 0:
            return call(BASE_JA.format(text=text))
        prev = cues[i - 1]["text"]
        prefix = gloss_for(text) if name.endswith("_gloss") else ""
        prompt = (prefix +
                  "下面是同一部短剧里连续的两句台词。请结合前一句，把后一句翻译成"
                  "自然、口语化的日语，保留人物称呼和语气，只输出后一句的译文，"
                  "不要翻译前一句，不要解释：\n"
                  f"前一句：{prev}\n后一句：{text}")
        return call(prompt)
    if name in ("pair2_clean", "pair2_gloss_clean"):
        block = cues[max(0, i - 1):i + 1]
        rows = "\n".join(f"{k+1}. {x['text']}" for k, x in enumerate(block))
        prefix = (gloss_for(" ".join(x["text"] for x in block))
                  if name.endswith("_gloss_clean") else "")
        prompt = (prefix + f"{STYLE}\n\n"
                  f"把下面 {len(block)} 条同一部短剧的台词依次翻译成自然、口语化的日语，"
                  "保持人物称呼和语气一致。输出同样行数，每行以“编号. ”开头，"
                  "只输出译文，不要解释：\n" + rows)
        raw = call(prompt)
        got = {}
        for ln in raw.splitlines():
            m = re.match(r"^\s*(\d+)\s*[.．、)]\s*(.+)$", ln)
            if m:
                k = int(m.group(1)) - 1
                if 0 <= k < len(block):
                    got[k] = strip_num(m.group(2))
        return got.get(len(block) - 1, "")
    if name == "pair2":
        block = cues[max(0, i - 1):i + 1]
        rows = "\n".join(f"{k+1}. {x['text']}" for k, x in enumerate(block))
        prompt = (f"{STYLE}\n\n"
                  f"把下面 {len(block)} 条同一部短剧的台词依次翻译成自然、口语化的日语，"
                  "保持人物称呼和语气一致。输出同样行数，每行以“编号. ”开头，"
                  "只输出译文，不要解释：\n" + rows)
        raw = call(prompt)
        got = {}
        for ln in raw.splitlines():
            m = re.match(r"^\s*(\d+)\s*[.．、)]\s*(.+)$", ln)
            if m:
                k = int(m.group(1)) - 1
                if 0 <= k < len(block):
                    got[k] = m.group(2).strip()
        return got.get(len(block) - 1, "")
    if name == "refine":
        prev = ctx_lines(cues, i, 2)
        draft = call(("以下是同一部短剧的前文，只用于理解指代和语气，不要翻译：\n"
                      f"前文：\n{prev}\n\n"
                      "参考上面的信息，把下面的台词翻译成自然、口语化的日语，"
                      "保留人物称呼和语气，只输出当前台词的译文，不要翻译前文，不要解释：\n"
                      f"{text}"))
        prompt = ("以下是同一部短剧的前文（不要翻译）：\n"
                  f"前文：\n{prev}\n\n"
                  f"当前中文台词：{text}\n当前日语译文：{draft}\n\n"
                  "请结合前文检查译文是否准确、语气是否连贯；如果有误，只输出修正后的日语译文；"
                  "如果正确，原样输出。不要解释：")
        return call(prompt)
    raise ValueError(name)


def run_block(cues, size=3):
    outs = [""] * len(cues)
    for bi in range(0, len(cues), size):
        block = cues[bi:bi + size]
        prev = ctx_lines(cues, bi, 2)
        lines = "\n".join(f"{k+1}. {c['text']}" for k, c in enumerate(block))
        prompt = (f"{GLOSSARY}\n{STYLE}\n\n"
                  f"前文（不要翻译）：\n{prev}\n\n"
                  f"把下面 {len(block)} 条同一部短剧的台词依次翻译成自然、口语化的日语，"
                  "保持人物称呼和语气一致。输出同样行数，每行以“编号. ”开头，"
                  "只输出译文，不要解释：\n" + lines)
        raw = call(prompt)
        got = {}
        for ln in raw.splitlines():
            m = re.match(r"^\s*(\d+)\s*[.．、)]\s*(.+)$", ln)
            if m:
                k = int(m.group(1)) - 1
                if 0 <= k < len(block):
                    got[k] = m.group(2).strip()
        for k, c in enumerate(block):
            outs[bi + k] = got.get(k, "")
    return outs


def norm_ja(s):
    s = s.replace("\\N", "")
    s = re.sub(r"\s+", "", s)
    return s.strip("。．.!！?？ ")


def score(pred, ref):
    a, b = norm_ja(pred), norm_ja(ref)
    if not a or not b:
        return 0.0, 0.0
    return (1.0 if a == b else 0.0), SequenceMatcher(None, a, b).ratio()


def key_report(cues, outs):
    rows = []
    for i, c in enumerate(cues):
        if c["text"] in ("我才走的", "就抛弃了我们"):
            rows.append({"i": i, "zh": c["text"], "ref": c["ref"],
                         "out": outs[i]})
    return rows


def main():
    global BASE, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(
        ROOT / "runtime/batch_auto_v1/fc66d5175314ee7c10219619f393eb94/state.json"))
    ap.add_argument("--ref", default=str(
        ROOT / "outputs/ja_batch/fc66d5175314ee7c10219619f393eb94_ja.ass"))
    ap.add_argument("--variants", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="/tmp/translab.json")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--base", default=os.environ.get("LAB_BASE", BASE))
    ap.add_argument("--model", default=os.environ.get("LAB_MODEL", MODEL))
    args = ap.parse_args()

    BASE, MODEL = args.base, args.model
    cues = load_data(Path(args.state), Path(args.ref))
    names = (["baseline_raw", "baseline_norm", "glossary", "ctx",
              "ctx_glossary", "ctx_oracle", "refine"]
             if args.all else [x.strip() for x in args.variants.split(",") if x.strip()])
    if not names:
        raise SystemExit("没有变体")
    results = {}
    for name in names:
        t0 = time.time()
        if name.startswith("block"):
            outs = run_block(cues, int(name[5:] or 3))
        else:
            outs = [""] * len(cues)
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(run_one, name, cues, i): i for i in range(len(cues))}
                done = 0
                for fu in as_completed(futs):
                    i = futs[fu]
                    try:
                        outs[i] = fu.result()
                    except Exception as e:  # noqa: BLE001
                        outs[i] = f"<ERR:{e}>"
                    done += 1
                    if done % 10 == 0 or done == len(cues):
                        print(f"  {name} {done}/{len(cues)}", flush=True)
        scores = [score(o, c["ref"]) for o, c in zip(outs, cues)]
        exact = sum(s[0] for s in scores) / len(scores)
        sim = sum(s[1] for s in scores) / len(scores)
        results[name] = {
            "seconds": round(time.time() - t0, 1),
            "exact": round(exact, 4),
            "sim": round(sim, 4),
            "empty": sum(1 for o in outs if not o),
            "key": key_report(cues, outs),
            "outs": outs,
        }
        print(f"{name}: exact={exact:.3f} sim={sim:.3f} "
              f"empty={results[name]['empty']} {results[name]['seconds']}s", flush=True)
        for k in results[name]["key"]:
            print(f"  {k['zh']}: {k['out']}  (ref {k['ref']})", flush=True)

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
