# SubSwap

Replace **burned-in subtitles** in vertical short dramas: detect the original cues, erase them
frame by frame, translate, and re-burn the result in another language — one command, one GPU.

Built for short-drama localization where the source has hard-coded Chinese subtitles baked into
the pixels and there is no subtitle stream to work with.

```
video ──► shot split ──► OCR cue track ──► erase (inpaint) ──► residual repair ──► translate ──► re-burn ──► QC
                             │                  │                  │                  │
                        rapidocr on the     reference frame /    re-scan for       any OpenAI-compatible
                        subtitle band only  temporal fill /      leftovers and     endpoint or local
                                            LaMa                re-erase them      Hy-MT2 model
```

## Why the erase is the hard part

A subtitle pixel is only usable if some other frame shows what is behind it. The eraser
(`scripts/erase_ref.py`) works in three tiers, per shot:

1. **Reference-frame backfill** — find a frame in the same shot where that pixel is not covered,
   align it with DIS optical flow, verify photometrically, paste the real pixels back.
2. **Temporal fill** — neighbour frames usually expose part of the area; warp their real pixels
   into the current frame (nearest first, alignment-gated), so the result keeps real texture and
   moves with the background instead of being re-synthesised every frame.
3. **LaMa** — only the holes that no neighbour can fill go to the model.

Then a **residual scan** re-OCRs the erased video, and any leftover line triggers a repair pass
with a wider mask. A motion-compensated temporal filter smooths the erased region so it does not
shimmer against the moving background.

Measured on a 89.8 s / 2245-frame episode (single V100):

| | baseline (model only) | this pipeline |
|---|---|---|
| erased-region jitter¹ (median / p90) | 1.07 / 2.32 | **0.84 / 1.86** |
| leftover lines after first pass | 9 | **1** (0 after auto-repair) |
| erase time, 2 shots in flight | 281 s (serial) | **158 s** (227 s with temporal fill on) |

¹ frame-to-frame residual inside the erased region divided by the residual of the surrounding
moving background; 1.0 means "as stable as the background".

Batch throughput (4 clips, 390 s of source, one V100): **1130 clips/day / 32 source hours/day**
with 5 jobs in flight and a software x264 intermediate, vs 566 clips/day for the original
serial pipeline.

OCR runs only on the subtitle band with the detector's long side capped at 960 px, which alone is
a 3× speedup on cue extraction (`docs/OCR_FAST_2026-11-04.md`).

## Requirements

- Python 3.12, PyTorch with CUDA (`torch>=2.4`), `opencv-python`, `numpy`, `pyyaml`
- [rapidocr](https://github.com/RapidAI/RapidOCR) for cue detection
- `ffmpeg` with `h264_nvenc` (falls back to libx264)
- LaMa weights: `models/big-lama.pt`
- Optional: any OpenAI-compatible translation endpoint, or a local Hy-MT2 server

## Quick start

```bash
pip install torch opencv-python numpy pyyaml rapidocr_onnxruntime
mkdir -p models && curl -L -o models/big-lama.pt <big-lama-torchscript-url>

# 40 s probe first: writes cues + erased.mp4 + report.json, skips translation
python3 scripts/pipeline.py probe40.mp4 --out outputs/probe --no-burn

# full run: erase → translate → re-burn (target: en by default, ja/zh/ko/... supported)
python3 scripts/pipeline.py episode.mp4 --out outputs/ep01 --target ja

# batch, several videos at once, resumable
python3 scripts/batch_pipeline.py --src-dir ~/dramas --jobs 4 --target ja
```

Products land in `outputs/<name>/`: `erased.mp4`, `final.mp4`, `final.ass`, `report.json`,
`erased_v.mp4`. There is also a small FastAPI web UI (`webapp/app.py`) for drag-and-drop runs.

Configuration lives in `config.yaml` — band position, mask mode (`stroke` / `rect`), inpaint
backend, temporal-fill radius, stabilization strength, repair rounds, translation backend.

## Repository layout

| path | |
|---|---|
| `scripts/pipeline.py` | frame/track primitives: OCR boxes, masks, writers, stabilization |
| `scripts/erase_ref.py` | the v6 erase engine (reference frame → temporal fill → LaMa) |
| `webapp/convert.py` | shot-level pipeline: erase → residual repair → translate → burn → QC |
| `scripts/batch_pipeline.py` | multi-video scheduler with VRAM-aware concurrency |
| `bench/pipe/` | benchmarks: erase timing, jitter scan, residual comparison |
| `docs/` | architecture notes and per-change measurements (Chinese) |

## Limitations

- Source must have **monolingual** burned-in subtitles; bilingual hard subs leave the other
  language behind.
- White text on white background is the failure mode: glyph detection merges the text with the
  background, so those frames fall back to whole-line rectangles and lose real texture.
- Translation quality is not automatically verified — cue timing and text still need a human
  spot-check before delivery.
- Tuned and measured on 1080×1920 25 fps vertical video with a single GPU.

Architecture and measurement notes (Chinese) live in [`docs/`](docs/): erase engine
[`ERASE_ARCH_V6.md`](docs/ERASE_ARCH_V6.md), temporal fill
[`ERASE_TEMPORAL_FILL_2026-11-04.md`](docs/ERASE_TEMPORAL_FILL_2026-11-04.md), pipeline
parallelism [`PIPE_PARALLEL_2026-11-04.md`](docs/PIPE_PARALLEL_2026-11-04.md), OCR speedup
[`OCR_FAST_2026-11-04.md`](docs/OCR_FAST_2026-11-04.md). Operating runbook:
[`RUNBOOK.md`](RUNBOOK.md).
