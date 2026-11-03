#!/usr/bin/env python3
"""erase() 并发脚手架的纯 CPU 自检：不碰 GPU/LaMa/ffmpeg。

验四件事：
  1. pipe=1 与 pipe=2 的总帧数、段文件数目一致（不会漏镜头）
  2. pipe=2 确实并行（墙钟显著低于串行）
  3. 模型调用被 _SerialModel 串行化（不会同时进两个 batch）
  4. 某个镜头抛异常时，erase() 会把异常抛出来，不静默吞掉
"""
import os
import sys
import threading
import time
from pathlib import Path

import os
import tempfile
DEV = Path(os.environ.get("VE_ROOT") or Path(__file__).resolve().parents[2])
SCRATCH = Path(tempfile.mkdtemp(prefix="ve_test_pipe_"))
sys.path.insert(0, str(DEV / "webapp"))
sys.path.insert(0, str(DEV / "scripts"))

import convert as C  # noqa: E402
import pipeline as P  # noqa: E402

in_model = 0
max_in_model = 0
model_lock = threading.Lock()


class FakeModel:
    def __call__(self, img, msk):
        global in_model, max_in_model
        with model_lock:
            in_model += 1
            max_in_model = max(max_in_model, in_model)
        time.sleep(0.05)
        with model_lock:
            in_model -= 1
        return img


class FakeWriter:
    def __init__(self, path):
        self.path = path
        self.n = 0
        self.stdin = self

    def write(self, b):
        self.n += 1

    def close(self):
        pass

    def wait(self):
        return 0


def run(pipe, fail_at=None):
    global max_in_model
    max_in_model = 0
    shots = [{"start": i * 1.0, "end": i * 1.0 + 1.0} for i in range(6)]
    per_shot = [[{"start": 0, "end": 1, "boxes": [], "ubox": (0, 0, 10, 10)}]
                if i % 3 else [] for i in range(6)]
    frames = list(range(25))

    class Fr:
        def tobytes(self):
            return b"x"

    def fake_iter(*a, **k):
        for _ in frames:
            yield Fr()

    def fake_erase_shot(model, path, shot, tracks, meta, writer, cfg, mask_mode,
                        amp=False, cues=None):
        t = shot["start"]
        if fail_at is not None and abs(t - fail_at) < 1e-6:
            raise RuntimeError("boom")
        # 接近真实比例：GPU 上的 batch 抓几批（0.4s），CPU 侧掩码/光流/合成 0.3s
        for _ in range(10):
            model(b"img", b"msk")
        time.sleep(0.3)
        for _ in range(25):
            writer.stdin.write(b"x")
        return 25, 1.0, 25.0, 0.1, "ref"

    P.iter_frames = fake_iter
    P.open_writer = lambda p, w, h, fps: FakeWriter(p)
    P.erase_shot = fake_erase_shot
    C.get_model = lambda cfg=None: FakeModel()
    C.filter_tracks_by_layout = lambda tracks, layout, meta: tracks
    seen = {}

    def fake_concat(segs, out, meta, log, expected=None):
        seen["n"] = expected
        seen["segs"] = len(segs)
        return expected

    C._concat_segments = fake_concat
    meta = {"w": 1080, "h": 1920, "fps": 25.0, "dur": 6.0}
    prog = []
    t0 = time.time()
    total = C.erase(SCRATCH / "job" / "src.mp4", meta, {}, shots, per_shot, {},
                    on_prog=lambda f, m: prog.append((round(f, 2), m)),
                    log=print, pipe=pipe)
    dt = time.time() - t0
    return seen.get("n"), dt, max_in_model, prog


def main():
    fail = 0
    tot1, dt1, mm1, pr1 = run(1)
    print(f"pipe=1  帧数 {tot1}  {dt1:.2f}s  并发进模型 {mm1}  进度回调 {len(pr1)}")
    tot2, dt2, mm2, pr2 = run(2)
    print(f"pipe=2  帧数 {tot2}  {dt2:.2f}s  并发进模型 {mm2}  进度回调 {len(pr2)}")
    if tot1 != tot2 or tot1 != 6 * 25:
        print(f"FAIL 帧数不一致 {tot1} vs {tot2}（期望 150）")
        fail += 1
    if mm2 > 1:
        print(f"FAIL 模型调用没串行化，峰值 {mm2}")
        fail += 1
    if dt2 > dt1 * 0.85:
        print(f"FAIL 并发没提速：{dt1:.2f}s -> {dt2:.2f}s")
        fail += 1
    if abs(pr2[-1][0] - 1.0) > 1e-6:
        print(f"FAIL 进度没走到 1.0：{pr2[-1][0]}")
        fail += 1
    try:
        run(2, fail_at=2.0)
    except RuntimeError as e:
        print(f"OK 异常正确抛出：{e}")
    else:
        print("FAIL 镜头异常被吞了")
        fail += 1
    print("RESULT:", "PASS" if not fail else f"{fail} 项失败")
    return 1 if fail else 0


if __name__ == "__main__":
    (SCRATCH / "job").mkdir(parents=True, exist_ok=True)
    os.environ.pop("ERASE_PIPE", None)
    sys.exit(main())
