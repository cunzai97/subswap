#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""字幕合并 / 重断句 / 质检（纯标准库，可直接跑）

用途：
  1. merge_lines  : OCR 碎行 -> 可按语义翻译的 utterance
  2. plan_cues    : 目标语言长句 -> 满足行宽/CPS 的字幕 cue
  3. wrap_lines   : 单条 cue -> 1~2 行显示文本
  4. check_cues   : 确定性质检（长度、CPS、空行）

设计口径（竖屏 1080x1920）：
  每行 24 字符、最多 2 行、阅读速度 17 字符/秒、单条最短 0.8 秒
  横屏 16:9 把 max_chars 改成 42 即可。
"""

import re

CJK_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
NO_LINE_START = "。、，．！？：；）」』】…ー々"
NO_LINE_END = "（「『【("
_LATIN = re.compile(r"[A-Za-z0-9\u00c0-\u024f'’._-]")


def merge_lines(lines, gap=0.35, max_dur=6.0, sep=""):
    """把 OCR 的碎行合并成 utterance。

    lines: [{'start': float, 'end': float, 'text': str}, ...]
    - 同一时间轴上的多行（双语/折行）直接拼起来
    - 间隔小于 gap 且合并后不超过 max_dur 的相邻行合并
    """
    out = []
    for ln in sorted(lines, key=lambda x: (x["start"], x["end"])):
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        if out:
            cur = out[-1]
            same_cue = abs(ln["start"] - cur["start"]) < 0.05 and abs(ln["end"] - cur["end"]) < 0.05
            near = ln["start"] - cur["end"] <= gap and ln["end"] - cur["start"] <= max_dur
            if same_cue or near:
                cur["text"] = (cur["text"] + sep + text).strip()
                cur["end"] = max(cur["end"], ln["end"])
                continue
        out.append({"start": float(ln["start"]), "end": float(ln["end"]), "text": text})
    return out


_KANA_CHAR = re.compile(r"[\u3040-\u30ff]")


def _char_kind(c):
    if _KANA_CHAR.match(c):
        return "kana"
    if re.match(r"[\u3400-\u9fff]", c):
        return "kanji"
    return "other"


def _balanced_break(text, max_chars):
    """两行放得下时，在词/脚本边界处选一个平衡断点，避免只剩一个字"""
    if not (max_chars < len(text) <= 2 * max_chars):
        return None
    lo = max(4, max_chars // 2, len(text) - max_chars)
    best = None
    for p in range(lo, max_chars + 1):
        if text[p] in NO_LINE_START or text[p - 1] in NO_LINE_END:
            continue
        left, right = text[:p].rstrip(), text[p:].lstrip()
        if not left or not right:
            continue
        ka, kb = _char_kind(left[-1]), _char_kind(right[0])
        score = 0.0
        if {ka, kb} == {"kanji", "kana"}:
            score += 3.0
        if left[-1] in "、。，．！？：；":
            score += 4.0
        if ka == kb == "kanji":
            score -= 3.0
        if ka == kb == "kana":
            score -= 1.0
        if left[-1] in "がはをにでとものへやかねよなだてし" \
                and right[0] not in "ゃゅょっー":
            score += 2.5
        if len(left) >= 4 and len(right) >= 4:
            score += 2.0
        score -= abs(len(left) - len(right)) * 0.8
        score -= (max_chars - p) * 0.15
        if best is None or score > best[0]:
            best = (score, left, right)
    return [best[1], best[2]] if best else None


def _cjk_tokens(text):
    """拉丁词整体保留，CJK 逐字切，空白变成一个空格 token"""
    tokens, buf = [], ""
    for ch in text:
        if ch.isspace():
            if buf:
                tokens.append(buf)
                buf = ""
            if tokens and tokens[-1] != " ":
                tokens.append(" ")
        elif _LATIN.match(ch):
            buf += ch
        else:
            if buf:
                tokens.append(buf)
                buf = ""
            tokens.append(ch)
    if buf:
        tokens.append(buf)
    return tokens


def wrap_lines(text, max_chars=24):
    """折行：英文按词，中日韩按字（带简单避头尾），返回行列表"""
    text = (text or "").strip()
    if not text:
        return []
    if not CJK_CHAR.search(text):
        words = text.split()
        lines, cur = [], ""
        for w in words:
            cand = (cur + " " + w).strip()
            if len(cand) <= max_chars:
                cur = cand
                continue
            if cur:
                lines.append(cur)
            cur = w
            while len(cur) > max_chars:          # 超长单词硬切
                lines.append(cur[:max_chars])
                cur = cur[max_chars:]
        if cur:
            lines.append(cur)
        return lines

    balanced = _balanced_break(text, max_chars)
    if balanced:
        return balanced

    lines, cur = [], ""
    for tok in _cjk_tokens(text):
        if tok == " ":
            if cur and not cur.endswith(" "):
                cur += " "
            continue
        if len(cur) + len(tok) > max_chars and cur.strip():
            if tok in NO_LINE_START and len(cur) + len(tok) <= max_chars + 1:
                cur += tok
                lines.append(cur.rstrip())
                cur = ""
                continue
            if cur[-1] in NO_LINE_END and len(cur) > 1:
                last = cur[-1]
                head = cur[:-1].rstrip()
                if head:
                    lines.append(head)
                cur = last + tok
            else:
                lines.append(cur.rstrip())
                cur = tok
        else:
            cur += tok
        while len(cur) > max_chars + 1:
            lines.append(cur[:max_chars])
            cur = cur[max_chars:]
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def _units(text, cap):
    """先按句读切，再把超长片段按词/字切——保证每段不超过 cap"""
    parts = [p.strip() for p in re.split(r"(?<=[.!?;:。！？；：])\s*", text) if p.strip()]
    if CJK_CHAR.search(text):
        parts = [q for p in parts for q in re.split(r"(?<=[、，,])", p) if q.strip()]
    units = []
    for p in parts:
        if len(p) <= cap:
            units.append(p)
            continue
        if CJK_CHAR.search(p) and " " not in p:
            units.extend(p[i:i + cap] for i in range(0, len(p), cap))
            continue
        cur = []
        for w in p.split():
            cand = " ".join(cur + [w])
            if cur and len(cand) > cap:
                units.append(" ".join(cur))
                cur = [w]
            else:
                cur.append(w)
        if cur:
            units.append(" ".join(cur))
    return units


def _join_units(cur, u):
    if cur and u and _LATIN.match(cur[-1]) and _LATIN.match(u[0]):
        return cur + " " + u
    return cur + u


def _pack(units, cap):
    """把句读单元装进 cue，尽量在标点处断开"""
    cjk = any(CJK_CHAR.search(u or "") for u in units)
    bins, cur = [], ""
    for u in units:
        cand = _join_units(cur, u) if cjk else (cur + " " + u).strip()
        if cur and len(cand) > cap:
            bins.append(cur)
            cur = u
        else:
            cur = cand
    if cur:
        bins.append(cur)
    return bins


def plan_cues(text, start, end, max_chars=24, max_lines=2,
              cps=17.0, min_dur=0.8, max_dur=6.0):
    """把一段译文排成字幕 cue。

    返回 (cues, issues)。issues 非空表示这段需要重译/缩短/向空白处借时间。
    """
    text = " ".join((text or "").split())
    if not text:
        return [], ["空译文"]

    cap = max_chars * max_lines
    dur = max(0.4, end - start)
    chunks = _pack(_units(text, cap), cap)

    total_chars = sum(len(c) for c in chunks) or 1
    issues = []
    if total_chars / dur > cps:
        issues.append(f"时长不足：{dur:.2f}s 装不下 {total_chars} 字符（需 {total_chars / cps:.2f}s）")

    cues, t = [], start
    for ch in chunks:
        if t >= end - 0.05:
            # 时间窗用完：剩下的并回最后一条，绝不生成负时长 cue
            if cues:
                cues[-1]["text"] = (cues[-1]["text"] + " " + ch).strip()
            continue
        share = len(ch) / total_chars
        d = max(dur * share, min_dur, len(ch) / cps)
        d = min(d, max_dur, end - t)
        if d < 0.05:
            continue
        cues.append({
            "start": round(t, 3),
            "end": round(min(end, t + d), 3),
            "text": ch,
        })
        t += d

    if cues and cues[-1]["end"] < end - 0.05:
        cues[-1]["end"] = round(end, 3)
    for c in cues:
        if c["end"] - c["start"] < min_dur - 1e-3:
            issues.append(f"cue 过短 {c['end'] - c['start']:.2f}s")
        if wrap_lines(c["text"], max_chars) and len(wrap_lines(c["text"], max_chars)) > max_lines:
            issues.append("行数超限")
    return cues, issues


def check_cues(cues, max_chars=24, max_lines=2, cps=17.0, min_dur=0.8, lang="en"):
    """确定性质检，返回问题列表"""
    keep_cjk = str(lang or "en").lower().startswith(("zh", "cn", "ja", "ko"))
    problems = []
    for i, c in enumerate(cues):
        text = (c.get("text") or "").strip()
        if not text:
            problems.append(f"#{i} 空译文")
            continue
        lines = wrap_lines(text, max_chars)
        if len(lines) > max_lines:
            problems.append(f"#{i} 行数 {len(lines)} > {max_lines}")
        if any(len(x) > max_chars for x in lines):
            problems.append(f"#{i} 行宽超限")
        d = c["end"] - c["start"]
        if d < min_dur:
            problems.append(f"#{i} 时长 {d:.2f}s < {min_dur}s")
        if d > 0 and len(text) / d > cps * 1.2:
            problems.append(f"#{i} CPS {len(text) / d:.1f} 过高")
        if not keep_cjk and re.search(r"[\u4e00-\u9fff]", text):
            problems.append(f"#{i} 含未翻译中文")
    return problems


def _demo():
    print("== merge_lines ==")
    src = [
        {"start": 0.00, "end": 1.20, "text": "三年了"},
        {"start": 1.20, "end": 1.35, "text": "你终于回来了"},
        {"start": 2.60, "end": 3.80, "text": "我一直在等你"},
        {"start": 3.85, "end": 5.00, "text": "对不起"},
    ]
    for u in merge_lines(src):
        print(f"  {u['start']:.2f}-{u['end']:.2f}  {u['text']}")

    print("\n== plan_cues（竖屏 24x2） ==")
    text = ("Three years. You finally came back. Do you have any idea "
            "how long I have been waiting for you in this empty house?")
    cues, issues = plan_cues(text, 0.0, 5.4)
    for c in cues:
        lines = wrap_lines(c["text"], 24)
        print(f"  {c['start']:.2f}-{c['end']:.2f}  {' / '.join(lines)}")
    print("  issues:", issues or "none")

    print("\n== check_cues ==")
    bad = [
        {"start": 0.0, "end": 0.5, "text": "This line is far too long for half a second of screen time"},
        {"start": 0.6, "end": 1.6, "text": ""},
        {"start": 1.7, "end": 2.7, "text": "还没翻完"},
    ]
    print("  ", check_cues(bad))


if __name__ == "__main__":
    _demo()
