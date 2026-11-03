# 运行手册（CLI + Webapp）

最后更新：2026-11-03  
对应代码：`scripts/pipeline.py`、`scripts/erase_ref.py`、`scripts/batch_pipeline.py`、`webapp/convert.py`、`webapp/app.py`

## 0. 当前状态

- **已能**：中文竖屏短剧 → 擦原字幕 → 本地翻译 → 重压 → 出成片。擦除可在**本地参考帧优先 / 本地 LaMa / 火山引擎云端**之间按任务切换；本地默认 `erase.engine: ref`（v6）：逐帧白字检测、无字帧原样、最近干净帧光流回填、失败才用模型兜底，并带自动残留复查。
- **日语已通**：CLI `--target ja`，网页目标语言选「日本語」。39s demo 实测 18 条日语、白字残留 0、OCR 残留 0，端到端 208.7s（V100，擦除约 9 fps）。
- **批量已通**：`scripts/batch_pipeline.py` 支持 K 路 prepare 并发 + 翻译模型按需装卸。三支正片 + 一支 103s 复制片 K=4 实测 **14.3 min**（源 538s），约 **404 支/日、15.1 源小时/日**（100% 满负荷口径；按 8 小时算约 135 支 / 5 源小时）；旧版 19.9 min / 291 支 / 10.9 源小时；串行基线 109 支/日。已含 native fp16 LaMa、分段返修、fp16 NaN 回退。
- **P0 可靠性已补（2026-11-03）**：网页路径恢复 fp16（103s prepare **379.3s**）；批量支持**断点续跑**（已完成阶段秒过，`--fresh` 强制重跑）、**阶段超时**（killpg）、QC 缩略图归档到 `outputs/<stem>_qc/`、**分段/成片帧数硬断言**、**压制后渲染验收**；K 按 3.1GB/路估算并封顶 4，`--force-jobs` 可跳过。方案与验收记录见 [docs/REVIEW_ACTIONS_2026-11-03.md](docs/REVIEW_ACTIONS_2026-11-03.md)。
- **未到**：翻译质量闸门（已知多义词/句式错误仍靠人工抽查）；模型兜底帧多时仍可能偏软（整镜头没有干净参考）；自动返修只覆盖 OCR 认得出的残留；云端只对 mock 验证过，还没有用真实 Key 跑过。批量不会再静默少交付，但“无人值守完工”仍差翻译验收。
- 结论：**能一键跑完，且批量不会静默少交付；翻译质量仍需人工抽查**。详见 §7。架构细节见 [docs/ERASE_ARCH_V6.md](docs/ERASE_ARCH_V6.md)。

### 工作原则

1. **先验源**：字幕是不是纯中文、在画面什么位置，先看 contact sheet，不确认不跑全片。
2. **先探针**：40 秒 `--no-burn`，看 cues + 抽帧，再决定是否跑全片。
3. **帧才是证据**：`residual_frames` / `cue_issues` 只是线索，必须抽 cue 前后帧看。
4. **先锁卡**：擦除用系统 `python3`（V100）；翻译 1509 必须 `--device CUDA0`，启动后查 `nvidia-smi` 和 `rocm-smi`。
5. **新片源先定 mask**：双行/粗描边用 `rect`，细单行用 `stroke`；不要盲信默认值。
6. **长跑后台 + /tmp 日志**：产物复制到 `outputs/`，不打断 7900 XTX 上的共享服务。

## 1. 素材要求（先过这一关）

- 必须是**原片自带、且只有中文**的硬字幕。中英双语硬字幕不能用：擦掉中文会留下英文行（例：庶女重生 `iIu-ELvhMhA`、冷王独宠妃 `QwJMaU0Xatc`）。
- 腾讯视频整集上传常常没有对白字幕，只有台标，不能拿来做效果验证（例：`YdbT4Iwv-ms`）。
- 推荐抖音/快手系竖屏短剧：白字黑描边、底部单行或双行。已验证样本：《我不是名媛》第五集（YouTube `5tiWo5Y-bEg`，184.32s，1080×1920，58 条字幕）。
- 跑之前先抽 contact sheet 看字幕语言和位置：

```bash
ffmpeg -i src.mp4 -vf "fps=1/6,scale=180:-1,tile=6x5" -frames:v 1 sheet.png
```

### 下载（YouTube）

```bash
export https_proxy=http://127.0.0.1:7900 http_proxy=http://127.0.0.1:7900

yt-dlp -F "URL"
yt-dlp -f 137 -o src_cn/xxx_video.mp4 "URL"   # 1080p 视频
yt-dlp -f 140 -o src_cn/xxx_audio.m4a "URL"   # 音轨（VAD 需要）
ffmpeg -i xxx_video.mp4 -i xxx_audio.m4a -c copy -shortest -movflags +faststart -y xxx.mp4
```

- yt-dlp 需 ≥ 2026.10.26；旧版会 403。升级：

```bash
/home/pan/.agent-reach-venv/bin/python3 -m pip install -U yt-dlp
```

- 只截片段时 `--download-sections` 会调用 ffmpeg，必须把代理传给它，否则直连 403：

```bash
yt-dlp -f "bv*[height<=1920]" --download-sections "*60-95" \
  --downloader-args "ffmpeg_i:-http_proxy http://127.0.0.1:7900" -o sample.%(ext)s "URL"
```

- 片源常无音轨（video-only 下载）；**跑 VAD 前先 mux 音轨**，否则 `vad_segments` 为 0，时间轴吸附失效。

## 2. 环境（两个必须知道的坑）

### 2.1 Python / torch

- 擦除用**系统 `python3`**：torch 2.6.0+cu124，支持 V100（sm_70）。
- `~/venvs/erase-cuda` 现为 torch 2.15.0+cu132，**不含 sm_70 kernel**，`torch.jit.load(..., "cuda")` 会直接报：

```
CUDA error: no kernel image is available for execution on the device
```

  不要用这个 venv 跑 LaMa。要修就按 PLAN.md §2.2 重建（torch 2.6.0+cu124）。
- 自检：

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
# 2.6.0+cu124 12.4 Tesla V100-SXM2-16GB
```

### 2.2 翻译服务（必须显式锁 CUDA0）

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/pan/work/turboquant/build-v100-cuda/bin/llama-server \
  -m /media/pan/未来/LLM/hy-mt2/Hy-MT2-7B-Q4_K_M.gguf \
  --device CUDA0 --host 127.0.0.1 --port 1509 \
  -ngl 99 -np 4 -c 32768 --jinja > /tmp/hymt2.log 2>&1 &
```

- 验证（必须两项都看）：

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # 应有 ~8.9GB
rocm-smi --showmeminfo vram                                     # used 不应变化
```

  7900 XTX 上的大占用是别的常驻服务，不要动；本项目的翻译和擦除都在 V100。
- 单条翻译冒烟：

```bash
curl -s http://127.0.0.1:1509/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Hy-MT2-7B-Q4_K_M.gguf","messages":[{"role":"user","content":"将以下文本翻译成英语，注意只需要输出翻译后的结果，不要额外解释：\n\n赵总监"}],"temperature":0,"max_tokens":32}'
# 期望 "Director Zhao"
```

- ⚠ 服务不在线或请求报错时，`pipeline.py` 会**静默退回内置 mock**并照常压制（日志写 `云端翻译失败，退回 mock` / `使用 mock`）。交付前必须 grep 日志确认没有 mock，不要只看“出片了”。

## 3. CLI 流程

### 3.1 40 秒探针（必做）

```bash
ffmpeg -ss 55 -t 40 -i src.mp4 -c:v libx264 -preset veryfast -crf 16 -c:a copy -y probe40.mp4
python3 scripts/pipeline.py assets/sources/probe40.mp4 --out outputs/cli/probe --no-burn
```

- 看 `report.json` 的 `cues`、`cue_issues`、`residual_frames`，再抽 cue 前后帧看擦除。
- 双行/大字号字幕先试 `rect`；细单行再考虑 `stroke`。

### 3.2 全片

```bash
python3 scripts/pipeline.py assets/sources/xxx.mp4 --out outputs/cli/xxx
```

- 产物：`erased.mp4`、`erased_v.mp4`、`final.mp4`、`final.ass`、`report.json`、`segments/`。
- 只改翻译/排版、不想重擦时：`python3 scripts/post_translate.py outputs/cli/xxx`（输入是含 `erased.mp4 + report.json` 的目录）。
- CLI 的英文字幕压在画面底部（`max_chars=24` 竖屏）；要“压回原字幕位置”，用 webapp。
- 耗参考：3 分钟片 ≈ 6.5 分钟；10 分钟片 ≈ 28.5 分钟（V100，擦除占 96%）。v6 参考帧引擎在 39s demo 上擦除约 9 fps，画质优先；纯模型帧越少越快。

### 3.4 日语字幕

```bash
# 翻译服务必须先按 §2.2 起在 1509 / CUDA0
CUDA_VISIBLE_DEVICES=0 python3 scripts/pipeline.py <src.mp4> \
  --out outputs/cli/ja_demo --target ja
```

- 字体 `Noto Sans CJK JP`；网页按原字幕位置压制，字号与原字幕同高、每行 12 字；CLI 用底部 18 字/行。
- 译文校验必须含假名；翻译服务不通时日语任务会直接报错，不会再用英语 mock 冒充。
- CLI 用固定字幕带，画面里的账本/招牌可能被 OCR 当台词；**交付优先走 webapp**（先探测版面并过滤场景文字）。

### 3.3 擦除引擎与 mask 模式（2026-11-02）
| 引擎/模式 | 当前用途 | 说明 |
|---|---|---|
| `erase.engine: ref`（默认） | 白字硬字幕 | 逐帧白字门控 + 最近干净参考帧光流回填；无字帧原样；无参考/光度失败才用模型。复杂背景优先 |
| `legacy` + `stroke` | 白字门控不可用/彩色字幕 | 旧逐帧 LaMa；白字门控检测不到时会自动回退，也可手动强制 |
| `rect` | 双行/粗描边 legacy | 整行矩形兜底；仅 legacy 引擎使用 |

> 参考帧优先的参数在 `config.yaml:erase`（`ref_photo/ref_feather/model_feather/model_stab`），白字门控阈值在 `config.yaml:mask`（`white_level/white_sat`）。不要按片子手调，先看日志里的“白字门控 / ref / model”比例。
> **LaMa 后端（2026-11-03）**：`inpaint.backend: native` + `fp16: true`（原生 PyTorch 重建 big-lama，fp32 权重 + 卷积 autocast fp16，FFT 分支内部 fp32）。CLI、网页、批量三条路径都直接读 `config.yaml`，不再各自覆盖。同一支 103s 片 prepare 538.3s → 425.2s（-21%），模型兜底帧约 6–8 → 9–12 fps。`backend: script` 仅作回退，脚本后端会自动关 fp16；native fp16 的 NaN 由首擦/返修整批回退 fp32 兜底。
| `rect` | 仅实验/对照 | 整行矩形擦除面积大，LaMa 几乎没有上下文，牛仔布/衣服上会补出黑块，不建议再用 |

关键修复（血泪）：

- **背景板不能用逐像素 min**：min 会系统性偏暗，60s 处整条字幕带发黑。已改为多帧均值。
- **最终掩码不能用整行矩形**：rect 会把 185px 高的字幕带几乎全盖住，LaMa 没有上下文，输出黑块/假字。
- **temporal 只在真正锁死的机位用**（`motion<=0.1`）；稍有漂移背景板和实拍帧对不上，字幕位置会出现方块接缝。
- **掩码时间提前量 0.8s→0.4s**：0.8s 会在台词出现前擦到干净画面，整块牛仔布被擦黑。

**不能只看 `residual_frames`**：stroke 和 rect 曾都是 `32/369`，但肉眼一个脏、一个干净。必须看帧。

### 3.3.1 自动残留返修（webapp 默认开启）

新素材不再靠人工调 `expand_ratio / bright_fallback`，流程自己兜底：

1. **首擦**保持 `stroke` + 现有默认（含白底白字自适应）。
2. **逐条复查**：对每条字幕轨，只 OCR 它自己的框（左右外扩 ~3.5% 宽、上下 25% 高）。识别到的文字必须和原台词有共同汉字（外语=共同单词）才算残留，避免把招牌/账本等画面文字误判。
3. **只返修失败镜头**：首擦按镜头写成 `segments/shot_XXXX.h264`（无 B 帧裸流，避免跨段 mux 丢帧），返修只重编码 suspect 镜头，其余分段字节拼接再 mux 成 MP4，不再整片重编码。103s 片 2 处残留，重擦+拼接约 10–18s（旧整片 patch 约 70s）。
4. **最多 2 轮**（`erase.repair_rounds`），每轮修完**只复查被改过的镜头**（其余镜头上一轮已证明干净，不再重复扫），第二轮扫描从 ~40s 降到几秒。任何一轮异常都回滚分段并保留上一轮结果。

`report.json` 的 `erase.repair` 给出 `suspects_before / suspects_after / repaired_shots / patched_frames`。实测 39s 片：首擦干净时复查 0 处、不触发返修；故意用“朴素首擦”留下 2 处白底白字，返修只重写 2 个镜头 124 帧，复查归零，画面无接缝。39s 片复查约 19s；发现残留时只重编码 suspect 镜头（秒级），不再整片重编码。

```yaml
erase:
  auto_repair: true   # 关掉就只做首擦
  repair_rounds: 2
```

注意：残留文字如果 OCR 完全认不出（只剩笔画碎片），复查抓不到；`qc.suspects` 缩略图仍保留给人眼复核。

**白字贴着白衣服/白墙**（2026-11-03 修）：逐帧白字门控按连通域找字，白衬衫袖口贴住首字时会粘成一个超尺寸亮块被整块丢掉，实测 1:43 片 1:15 处首字“朵”只残留 0.2–0.3s。现在：

- `_glyph_mask` 检测到这种“粘连大组件”时返回 `merged`；只有**保留下来的字形并集框比台词行框短了 >35% 字高**（真的缺首/尾字）才对这一帧按行框整行兜底。白背景片里大亮块很常见，字都找到了就不扩掩码（否则整行 rect 会让 LaMa 大面积糊）。参考帧填充仍用真实像素，模型兜底才走 LaMa。
- 残留复查从固定 3 点改为**字幕真实时间窗内的 5 点**（12/31/50/69/88%），0.12s 的短闪也能被 OCR 命中（实测 75.85s 能识别出残留“朵”）。每轮复查 OCR 调用约 +1/3，单轮多 ~10s。
- 这类兜底只影响“粘连且缺字”的少数帧；日志新增 `粘连兜底 N 帧`，正常片为 0。

**fp16 NaN 兜底**（2026-11-03）：native autocast fp16 在少数大矩形掩码上会整批输出 NaN；旧逻辑“坏像素不填”会让返修静默无效，留下灰字残影（实测 97.1s 两轮返修都修不掉）。现在首擦 `_lama_fill` 和返修 `_lama_patch_batch` 检测到非有限像素就整批用 fp32 权重重跑一次，日志打印一次“fp16 输出含 NaN，整批回退 fp32”。正常批次零开销。

### 3.3.2 擦除区域时域平滑（降抖）

症状：单帧看干净，视频里擦除区像“呼吸”一样闪，尤其纯色衣服/墙面上明显。原因是 LaMa 逐帧独立推理，同一个字位每帧补出的纹理都略有不同。

做法（`inpaint.temporal_ema: 0.4`，0 关闭）：

- 只在擦除掩码内混合（外扩 21px + 羽化），不动没改过的像素。
- 低运动镜头（`motion<=1.2`）直接和上一帧平均；高运动镜头用 DIS 光流把上一帧 warp 过来再混合。
- 掩码外光度一致性差的帧（切镜/遮挡）自动放弃混合，避免拖影。
- 首擦和自动返修的补擦都走同一套平滑。

实测 39s 片（19 条字幕，掩码内时域高频抖动）：

| 场景 | 降幅 |
|---|---|
| 纯色/低运动背景（白西装、墙面等） | 30–44% |
| 高运动/复杂纹理（金色流苏、旗袍花纹） | 5–9% |
| 全部字幕平均 | 21%（2.59 → 2.04） |

对比片：`outputs/local_compare/stabilize_check.mp4`（上旧/下新，裁字幕带）。注意高运动+复杂纹理处仍会有轻微游动，这是单帧 LaMa 的极限；这类镜头云端 pro 更稳。想关掉可把 `temporal_ema` 设 0。

### 3.3.3 多视频并发批处理（`scripts/batch_pipeline.py`，2026-11-03）

单视频时 GPU 大量时间在等 CPU（OCR 前后处理、光流、返修重编码），实测整批 GPU 均值只有 ~27–37%。做法是把流程拆成三段，用多视频把 GPU 喂满：

1. **prepare**（K 个进程并发）：归一化 → 版面 → OCR → 擦除 → 自动返修 → QC，产出 `erased.mp4` + `state.json`，不加载翻译模型。
2. **translate**（串行，几秒/支）：批处理自己的 `state.json`，临时拉起 Hy-MT2（9GB），翻完立即卸载。
3. **burn**（串行，10–20s/支）：排版 + NVENC 压制 + 复制成片/ASS/report/日志。

```bash
cd /media/pan/学习/work/video-erase
python3 scripts/batch_pipeline.py \
  --src /home/pan/视频/a.mp4 /home/pan/视频/b.mp4 \
  --out outputs/batch_auto --jobs 4 --target ja
# 目录模式：--src-dir /home/pan/视频；想保留中间文件加 --keep-work
# 续跑：默认复用已完成阶段；--fresh 强制重跑；--force-jobs 跳过显存自动降并发
```

实测（V100 16GB，三条 174/157/103s 正片，Hy-MT2 同卡但 prepare 阶段不驻留）：

| 调度 | 批量墙钟 | 源时长 | 实时倍率 | 支/日(100%) | 源小时/日 |
|---|---:|---:|---:|---:|---:|
| 串行（旧） | 39.6 min / 3 支 | 435s | 5.46× | 109 | 4.4 |
| K=3 | 19.8 min / 3 支 | 435s | 2.73× | 218 | 8.8 |
| K=4（旧：TorchScript + 整片 patch） | 19.9 min / 4 支 | 538s | 2.22× | 291 | 10.9 |
| **K=4（native fp16 + 分段返修 + NaN 回退）** | **14.3 min / 4 支** | **538s** | **1.59×** | **404** | **15.1** |

- 单支 prepare 从 959/1097/543s 降到 **767/751/413s**（v1/v2/v3，K=4 并发口径）。
- 峰值显存 12.5GB（NVENC 分段中间层），K=4 安全；K=5 预计贴 16GB 边缘，不默认开。
- GPU 采样：均值 36.8%、p50 28%、≥50% 占 32.5%。**注：20 集长跑（2026-11-04）更正过这句——擦除段也不是 90–100%，整段均值只有 34.9%（p50 25%），瓶颈是片内串行，见 §3.3.4。**
- 返修尾延迟从“整片重编码”改成**只重编码 suspect 镜头 + 裸流拼接**；返修后只复查被改过的镜头。v1 3 处残留：patch 87 帧、第二轮复查 0，整段返修 ~20s（旧整片 patch 约 2–3 min）。
- `fp16 NaN` 整批回退 fp32 后，v3 首擦后复查从 2 处降到 **0 处**（NaN 不再留下灰字）。
- 翻译每支 4.5–6.9s，压制 10–19s。
- 成功支的 `runtime/batch_auto/<stem>` 默认删除，只保留 `outputs/` 下的成片/ASS/report/三段日志；失败的会保留现场。
- 并发时用 `ERASE_WRITER=x264` 可把中间层强制回软编（显存更低、文件更小）；NVENC 中间层实测 PSNR 50.9dB/SSIM 0.995，速度快 3.6×。
- 返修掩码已从 `pad16/expand18%` 加宽到 `pad24/expand35%`，专门兜 OCR 漏掉的边缘字（实测片 2 行尾“点”）。
- 首擦按镜头写 `segments/shot_XXXX.h264` 裸流（**必须 `-bf 0`**，B 帧跨段 mux 实测丢 3/375 帧）；返修只重编码 suspect 段，最后按字节拼接再 mux。分段拼接后帧数/时长与单条 writer 口径一致（103s 片 2586 帧）。
- **断点续跑（2026-11-03）**：开批不再无条件删工作目录；source 指纹（大小 + 首尾 1MB sha256）一致且 `state.json` 阶段标记/产物齐全就跳过 prepare/translate/burn。`--fresh` 强制从头重跑。崩溃后重跑只补没完成的阶段。
- **阶段超时**：prepare `max(1800, 8×源时长)`、translate 900s、burn `max(600, 2×源时长)`；超时用 `killpg` 收掉整个进程组，标记该支失败，其余继续。
- **QC 归档**：`work/qc/` 拷到 `outputs/<stem>_qc/`（每支 12 张），复制后的 report 增加 `qc_dir`；不用再进 runtime 找图。
- **帧数硬断言**：每段裸流 `nb_read_packets` 求和必须等于期望帧数；拼接后 MP4 再验一次；burn 后验最终帧数（与擦除源 ±1）、时长、音轨。不再只比 ±0.3s（25fps 有 7 帧盲区）。
- **压制验收**：ASS Dialogue 事件数 >0、`fc-match` 精确命中 Noto Sans CJK JP、抽 3 条最长 cue 的中点比字幕带差异像素（<200 判没渲染）。任一硬项失败直接 error，不出片。
- **burn 不再用 `-shortest`**：音轨比画面短时会切掉尾部画面（实测 5ed 旧成片少 4 帧）；现在只有音轨更长时才用 `-t` 卡到画面时长。三支回归成片 4355/3936/2586 帧，与擦除源一致。
- **K 估算**：已改为按「每路常驻 + 每条流水通道的 NVENC 实例」算，见 §3.3.4；16GB 卡 K=4+pipe=2 峰值 14.3GB，K=5 需要 x264 中间层（14.3GB），K=6 不稳。

### 3.3.4 镜头级流水 + 压制线程池（2026-11-04）

擦除段 GPU 均值只有 34.9%，不是算力不够，是 `erase_shot_ref` 里两遍串行扫（先纯 CPU 建掩码，再 GPU+CPU 回填合成）。现在一路 prepare 内可以同时推进 N 个镜头：

```bash
# 批量：5 路 + 自动（pipe=2、x264 中间层、压制 3 并发）
python3 scripts/batch_pipeline.py --src-dir ~/视频 --jobs 5 --target ja

# 关掉流水/压制并发（老行为，A/B 或排障用）
python3 scripts/batch_pipeline.py --jobs 4 --erase-pipe -1 --burn-jobs 1 --writer nvenc

# 单跑（网页/CLI）默认 4 路；ERASE_PIPE=1 回到串行
ERASE_PIPE=1 python3 scripts/pipeline.py in.mp4 --out outputs/x --target ja
```

- `--erase-pipe 0`（默认）自动：K≤2 用 4 路，K≥3 用 2 路；`-1` = 关。
- `--writer auto`（默认）：`--jobs>=5` 自动改 x264 中间层（NVENC 每个实例占 372MB 显存，5 路就撑爆）。
- `--burn-jobs 3`（默认）：翻译完一支就开始压制，尾巴和后续翻译重叠。
- config.yaml 里也有 `worker.erase_pipe` / `worker.burn_jobs` 两个默认值。

实测（ep04/07/11/13，390s 源，V100 独占）：

| 配置 | 墙钟 | 支/日(100%) | 源小时/日 | GPU 均值 | 显存峰值 |
|---|---:|---:|---:|---:|---:|
| 老：pipe=1 + 压制串行 | 609.6s | 566.9 | 15.36 | 31.4% | 12493MB |
| 新：pipe=2 + 压制 3 并发 | 463.6s | 745.5 | 20.20 | 45.3% | 14287MB |
| 新：5 路 + x264 + 压制 3 并发 | 507.4s/5 支 | **851.4** | **24.31** | 49.5% | 14301MB |

单跑一支 89.8s 片只算擦除：pipe=1 281s → pipe=2 159s → pipe=4 94s；三份成片帧数一致（2245），PSNR 96dB。细节和踩坑见 `docs/PIPE_PARALLEL_2026-11-04.md`。

### 3.4 云端擦除（火山引擎 AI MediaKit，可选）

云端只做“擦字幕”这一件事；OCR、翻译、排版、压制仍然在本机，字幕位置/字号/音轨口径不变。

**价格**（2026-11 官方文档，按输出时长，失败不计费）：

| 服务 | 单价 | 3 分钟片 | 备注 |
|---|---|---|---|
| 火山 AI MediaKit · 标准版 | 0.4 元/分钟 | ≈¥1.2 | 快、便宜；复杂纹理会糊 |
| 火山 AI MediaKit · 精细化版 | 1 元/分钟 | ≈¥3 | 短剧竖屏优化，AIGC 重建纹理 |
| 阿里云 VIAPI 字幕擦除 | 0.4 元/分钟 | ≈¥1.2 | 要公网 URL + AK 签名；按区域整体擦 |
| 百度 VOD 智能去字幕 HD | 1.8 元/分钟 | ≈¥5.4 | 需先上传媒资 |
| 腾讯云 去字幕无痕 1080P | 3 元/分钟 | ≈¥9 | 要 COS 输出 |

默认选火山：API Key 直连、支持本地文件直传（`mediakit://`），不用先传 OSS/COS；精细化版对短剧的竖屏白字最对口。

**开通与配置**：

1. 火山引擎控制台 → AI MediaKit → API Key 管理，实名认证后创建 Key。
2. 只放环境变量，不要写进仓库：

```bash
export ERASE_CLOUD_API_KEY=...
# 或者写文件：
# mkdir -p ~/.config/video-erase
# printf '%s' '...' > ~/.config/video-erase/mediakit.key
# config.yaml: erase.cloud.api_key_file: ~/.config/video-erase/mediakit.key
```

3. 切换方式：

```bash
# CLI（不改配置，单次覆盖）
python3 scripts/pipeline.py src.mp4 --out outputs/cli/x \
  --erase-provider cloud --erase-quality pro

# 永久默认：config.yaml
# erase:
#   provider: cloud
#   cloud:
#     quality: pro
```

网页端在“擦除引擎”下拉里选“本地 / 云端精细化 / 云端标准版”；没有 Key 时云端选项自动置灰。

**限制与行为**：

- 输入最高 2K、输出最高 1080P；支持 mp4/mov/mkv 等主流容器，保留音轨。本地直传单文件 ≤5GB，文件在云端保留 30 天。
- `Subtitle` 模式默认只处理画面下半部；顶部字幕会自动改用 `Text` + 按版面框选（仅精细化版）。标准版遇到顶部字幕会**直接报错**，不会白扣费。
- 下载后自动对齐回源分辨率/帧率，并换回源音轨；`report.json` 的 `erase` 字段记录 `task_id`、档位和估算费用。
- 超时会保留 `task_id`，先按 ID 查询，不要重复提交。
- **未用真实 Key 验证**：当前只跑了本地 mock（上传→提交→轮询→下载→对齐全链路）。第一次用真实 Key 时先拿 40 秒探针跑 `cloud_standard`，确认字幕带和价格，再上全片。
- 代码：`scripts/erase_cloud.py`；Webapp 表单字段 `erase=local|cloud_pro|cloud_standard`。
- 官方文档：[字幕擦除](https://docs.volcengine.com/docs/6448/2371372?lang=zh) · [视频工具计费](https://docs.volcengine.com/docs/Intelligentprocessing/video-tool-billing?lang=zh) · [本地上传](https://docs.volcengine.com/docs/6448/2536893?lang=zh)。
- 本地 mock 联调：`ERASE_CLOUD_BASE_URL` 可覆盖 `base_url`（测试脚本不要提交 Key）。

## 4. Webapp 流程（`webapp/`）

```bash
cd /media/pan/学习/work/video-erase/webapp
/usr/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port 8899
# 打开 http://127.0.0.1:8899
```

- 接口：`POST /api/jobs`（`file`、`source=auto|zh|other`、`target=zh|en|...`、`erase=local|cloud_pro|cloud_standard`）、`GET /api/jobs/{id}`、`/video`。
- 目录：`runtime/web/jobs/<jid>/{input.mp4, work/, output.mp4, report.json}`，索引 `runtime/web/jobs/index.json`。
- 特点：
  - 版面探测自适应：从抽样 OCR 推字幕带、字高、边距、顶部/底部；重压回**原字幕位置**。
  - 翻译后端自动探测：1509（本地 Hy-MT2）→ 8790（qoder2api 网关）。
  - 单 worker 队列，长片排队；不要同时跑多个大任务抢 V100。
- 注意：
  - UI 默认 `target=English`（中文源→英文是主场景）；源语言下拉 `中文` / `英文/其他` / `自动`。
  - 字幕排版：字号/边距/行宽从素材实测；目标语言脚本不同会缩放（中→英 0.85，英→中 1.15）；长译文先缩字号，再按**句读**拆成连续 cue，绝不截断。
  - 擦除：默认 `stroke` + `mask_pad=14` + `dilate=20`；temporal 仅锁死机位启用（见 §3.3）。本地擦除后自动复查并只返修失败镜头（§3.3.1，`erase.auto_repair`），擦除区默认做时域平滑降抖（§3.3.2，`inpaint.temporal_ema: 0.4`）。云端擦除见 §3.4；无 Key 时下拉里的云端项会置灰。
  - 兼容性：旋转（display matrix）、VFR、无音轨、HEVC/10-bit、MOV/MKV 已过回归（`webapp/selftest.py`）。
  - 自动质检：擦除后按 cue 中点抽 12 帧（字幕带），OCR 检查源语言残留，结果和缩略图在任务页展示；`qc.suspects` 非空时要人工看红框帧。自动返修的结论在 `report.json` 的 `erase.repair`（`suspects_before → suspects_after`）。
  - 翻译探活只验证 1509 是否响应，不校验是否落在 CUDA0；启动服务按 §2.2。
- 兼容性回归（V100 空闲时，约 4–6 分钟）：

```bash
cd webapp && /usr/bin/python3 selftest.py
# 期望 7 个 PASS + RESULT: PASS（en_plain/en_noaudio/en_hevc/zh_rot90/zh_vfr/zh_mov/zh_to_ja）
```

API 冒烟：

```bash
curl -s http://127.0.0.1:8899/api/health
curl -s http://127.0.0.1:8899/api/translate/check
curl -s -F "file=@probe40.mp4" -F source=zh -F target=en -F erase=local \
  http://127.0.0.1:8899/api/jobs
# 云端：把 erase 换成 cloud_pro / cloud_standard，并先 export ERASE_CLOUD_API_KEY
```

## 5. 质检清单（人工 5–10 分钟）

- [ ] 原片只有中文；成片抽 5–10 个 cue 前后帧，无中文残影。
- [ ] 字幕带内无黑块/方块接缝（temporal 背景板失效的典型痕迹）；有则查 §3.3。
- [ ] 无 OCR 碎片（“苏/福/老/专”类）；单字台词（好/行/哟）应保留并翻译。
- [ ] 长译文不截断、不出现单词孤行（如 “…独自去” / “旅行。”）；英文不压脸。
- [ ] 时间轴与嘴型大致同步；3 分钟片 `vad_segments=27` 属正常。
- [ ] `residual_frames`、`cue_issues` 只当线索，必须看帧。
- [ ] 日志里没有 `退回 mock` / `使用 mock`；有就是翻译没跑成，成片作废重跑。
- [ ] 音轨保留、时长一致、能正常播放：

```bash
ffprobe -v error -show_entries stream=codec_type,width,height,channels \
  -show_entries format=duration -of default=nw=1 final.mp4
for t in 55 64 71 143; do ffmpeg -v error -ss $t -i final.mp4 -frames:v 1 -y f_$t.png; done
```

本次 3 分钟片抽查 55/64/68/71/105/143/171s 干净；10 分钟双语片 `373/1200` 残留标记**未逐帧核实**，属未结项风险。

## 6. 实测数据（V100，2026-10-30/31）

| 片源 | 时长/镜头/字幕 | 擦除 | 翻译 | 压制 | 端到端 | 残留标记 |
|---|---|---|---|---|---|---|
| 双语 10 分钟 | 600.0s / 306 / 339 | 1639.1s，11.4 fps | 14.5s | 53.4s | 28.5 min | 373/1200 |
| 纯中文 3 分钟（旧 stroke） | 184.3s / 42 / 58 | 373.7s，15.2 fps | 2.5s | 17s | ~6.6 min | 32/369（有残影） |
| 纯中文 3 分钟（旧 rect） | 同上 | 367.0s，15.5 fps | 2.5s | 17s | ~6.5 min | 32/369（肉眼干净） |
| 纯中文 3 分钟（2026-10-31 修复版，webapp） | 184.3s / 42 / 49 | 231s，~20 fps | 2.5s | 17.5s | 375s（6.25 min） | 自动抽帧 12 张，疑似残留 0 |
| 12s 竖屏基准 A（1024/b4） | 12.0s / 2 / 6 | 24.5s | 0.5s | 1.9s | 48.8s | — |
| 12s 竖屏基准 D（768/b4） | 同上 | 21.2s（-13%） | 0.5s | 1.9s | 42.5s | 肉眼无差 |
| 12s 竖屏 · 云端 mock（标准版） | 12.0s / 2 / 6 | 云端 mock 10.7s | mock | 3.5s | 29.1s | 自动抽帧 6 张，疑似 0 |
| 12s 竖屏 · 本地回归（同片） | 同上 | 本地 LaMa | mock | — | 87.7s | 自动抽帧 6 张，疑似 0 |

> 10 分钟片的翻译/压制由 `post_translate.py` 后处理完成；3 分钟片由 `pipeline.py` 一次跑完。
> 12s 基准里 batch=8 没有收益（48.5s）；竖屏自动用 768 画布，横屏仍用 1024。

- 10 分钟片掩码覆盖 49.5%，擦除模式 temporal 10 / lama 257；3 分钟 rect 版覆盖 41.0%，temporal 1 / lama 22。
- 真实片源的 11–15 fps 低于 PLAN §12 合成片的 21 fps，差额来自 OCR/VAD/分析开销和真实运动背景。
- 单卡 24h 满打满算约 50 条 10 分钟片（≈8.4 小时成片内容）；吞吐不是当前瓶颈，验收才是。

## 7. 已知问题与“一键完工”差距

| # | 问题 | 证据 | 修法 | 估时 |
|---|---|---|---|---|
| 1 | ~~stroke 双行字幕留残影 / LaMa 黑块~~ | 已修：`mask_pad=14`+`dilate=20`+均值背景板+笔画残差，60/61/68s 复测干净 | — | 已完成 |
| 2 | ~~质检不可定位~~ 已修（2026-11-03） | QC 缩略图 12 张/支归档到 `outputs/<stem>_qc/`；burn 后验帧数/时长/音轨/ASS 事件/字体/字幕带渲染差异，失败直接 error | 剩余：缩略图仍是抽样，最终看片仍建议抽 cue 前后帧 | 已完成 |
| 3 | 断句问题只记录不修 | 58 条报 90 个问题（pipeline.py:1215）；OCR 碎片直接进翻译 | 去碎片、合并被拆句、超行/CPS 重断；严重问题不出片 | 0.5–1 天 |
| 4 | 无预检/封装 | 服务、CUDA0、python 环境都要人工确认 | `run_episode.sh`：检查 1509+CUDA0，选对 python，跑完打包 | 2–3 h |
| 5 | ~~webapp 仍未接残留闸门~~ 已接自动返修 | 擦除后逐条 OCR 自己的字幕框，命中原台词才算残留；只重擦失败镜头，最多 2 轮，`erase.repair` 记录前后数量（§3.3.1）。实测 2 处白底白字 → 0 处 | 剩余：翻译探活仍不看 `CUDA_VISIBLE_DEVICES` | 已完成/0.5 h |
| 6 | 云端擦除只验过 mock | 上传→提交→轮询→下载→音轨/分辨率对齐全链路已通；真实云端的画质、延迟、计费未验证 | 用真实 Key 先跑 40s 探针，再跑 1 集；确认后把 `erase.provider` 设为 cloud | 0.5 天 + 云端费用 |
| 7 | 高运动+复杂纹理仍有轻微游动 | 金色流苏/旗袍花纹上逐帧 LaMa 抖动只降 5–9%（纯色背景降 30–44%）；光流误差吃掉了更强平滑的收益 | 参考帧重投影/真实干净帧补洞；或这类镜头走云端 pro | 后续迭代 |

合计约 1.5–2 天 + 3–5 集跨 2–3 部剧的验证。目标：同一部剧 0 人工；换剧只做一次 30 秒校准；人只看异常帧。

### 7.1 视频 inpainting 模型实测（2026-11-02）

在 39s 样片、同一字幕带上实测了 ProPainter / E2FGVI-HQ / FGT，完整报告见
`bench/video_inpaint/BENCH.md`。结论：

- 速度（1080×208，1168 帧）：LaMa+stab ~10.8 fps > E2FGVI-HQ fp16 5.45 fps（fp32 3.98）
  > ProPainter 2.64 fps（turbo 优化后 4.07 fps；单进程 4.91 fps 但 16GB 贴边）>> FGT ~0.22 fps。
  E2 fp16 峰值显存 4.1GB，与 fp32 掩码区 PSNR 58.8dB。
- 质量：复杂纹理/高运动镜头（27.3s 云纹布料、32.6s 编织绳）ProPainter 细节最好，但有暗部
  鬼影/鱼鳞 artifact；E2FGVI 更干净但偏软；LaMa 最平滑也最糊。平背景镜头 E2FGVI 观感最好。
- 叠加现有运动补偿时域稳定化后，PP/E2 的抖动可压到接近 LaMa，细节仍有 1.4–2.2 倍。
- 均需按镜头切块才能进 16GB 显存（PP ≤300 帧、E2 ≤160 帧）；E2 上游 `get_ref_index`
  有越界 bug，需修；FGT 1080 宽为分布外分辨率且慢 ~30 倍，不采用。
- 结论：**不整体替换**。LaMa+stab 继续做默认；后续可按镜头对“复杂纹理/抖动超标”的镜头
  走 ProPainter（或更干净的 E2FGVI）精修，复用现有 `auto_repair` 的逐镜头调度。
- LaMa 提速（2026-11-02）：native 后端（FP32 权重 + 卷积 autocast fp16，FFT 分支强制
  fp32）+ batch 8 + 稳定化定点优化，擦除阶段 146.3s → 126.1s（11.85 fps），抽样画质
  无差；按镜头缩画布会让 FFT 分支涂抹编织纹理，已关。详见
  `bench/video_inpaint/LAMA_OPT_STATUS.md`。
- 双模型分流 demo：按镜头纹理/运动把复杂镜头交给 ProPainter，并用“PP vs LaMa 掩码区
  分歧”挡住 PP 幻觉；本片 PP 只跑 29% 帧，预计在线 147.7s（整片 PP 287.2s）。
  v2 改为逻辑分段（cue+±0.5s，段外原样）+ 参考帧优先，PP 303 帧（26%）、预计 131.8s，
  成片与 v1 逐像素一致；本片 0 个静态段，参考帧档未触发。见
  `bench/video_inpaint/HYBRID_ROUTING.md`。

## 8. 目录与日志约定（2026-10-31 整理后）

```
config.yaml                配置（mask/inpaint/翻译后端）
scripts/                   核心流水线（pipeline / cues / post_translate / erase_cloud）+ 基准工具
webapp/                    FastAPI 网页、convert.py、selftest.py、vendor/
models/big-lama.pt         唯一运行模型（TorchScript）
assets/
  sources/                 正式片源（namenyuan_1080.mp4、en_1.mp4）
  fixtures/                兼容性回归固定片源（只读，en_cut/zh_cut/rot/vfr/mov/hevc/noaudio）
outputs/
  namenyuan_ep5/           交付成片 + sub.ass + report.json + qc/
  en_cut_ja/               今早 en→ja 测试成片 + 字幕 + 质检帧
  cli/                     CLI 运行产物（pipeline.py 默认 --out）
runtime/
  web/jobs/<jid>/          webapp 任务；index.json 是任务索引
  selftest/                兼容性回归临时 job（可随时删）
bench/video_inpaint/       视频 inpainting 模型评测（独立 venv/权重/数据，不参与生产）
archive/
  old_webapp_jobs/         已归档的旧网页任务
  experiments/             旧实验产物（10 分钟片、rect/stroke 对比、探针等）
  source_downloads/        原始下载/抽帧/其他剧集素材
  model_originals/         big-lama.zip 与原始 best.ckpt（运行不需要）
  debug/                   开发期抽帧和日志
/tmp/namenyuan_run.log     长跑日志（避免 ntfs3 直接写大日志）
```

- 交付示例：`outputs/namenyuan_ep5/output.mp4`（1080×1920，3:04，49 cue，音轨保留）。
- `outputs/` 里的文件与 `runtime/web/jobs/<jid>/output.mp4` 是硬链接，删网页任务不会丢成片。
- `archive/` 只存放历史数据，确认不再需要后可整体删除；不要删 `assets/`、`outputs/`。
- 长任务放后台跑，日志写 `/tmp`；不要在长跑时重启 1509 或动 7900 XTX 上的常驻服务。

## 9. 变更记录

- 2026-10-30：10 分钟双语片跑通；发现 bilingual 源不适合做效果验证。
- 2026-10-31：换纯中文 3 分钟片；stroke→rect 修双行残影；锁定 Hy-MT2 到 CUDA0；webapp 出现但未回归；记录 torch venv 漂移问题。
- 2026-10-31（晚）：webapp 排版重做（字号/边距/行宽实测 + 句读拆 cue 不截断）；修擦除黑块（均值背景板、笔画残差、pad14/dilate20、预擦 0.4s）；修旋转双重转正；兼容性 7 例全过；竖屏 768 画布提速 ~13%；3 分钟片复测 375s、49 cue、自动抽帧 12 张 0 疑似残留；webapp 增加自动质检缩略图。
- 2026-10-31（整理）：废弃 `work/` 平铺结构，重组为 `assets/`（源素材+回归片源）、`outputs/`（交付）、`runtime/`（网页任务/回归临时）、`archive/`（旧实验全部归档）；旧网页任务保留最新 2 个，其余归档；`models/` 只留 `big-lama.pt`。
- 2026-11-01：接入云端擦除（火山引擎 AI MediaKit，标准版 0.4 / 精细化版 1 元/分钟）；新增 `scripts/erase_cloud.py`，CLI `--erase-provider/--erase-quality`、网页“擦除引擎”下拉、`config.erase`；上传/轮询/下载/对齐音轨与分辨率全链路用本地 mock 验证，本地 LaMa 回归通过（12s 片 QC 0 疑似）；真实 Key 尚未验证。
- 2026-11-01（晚）：webapp/CLI 本地擦除加自动残留返修（`erase.auto_repair`）：逐条 OCR 字幕自己的框 + 原台词同字校验，只对失败镜头整行 rect+LaMa 重擦并逐帧拼回，最多 2 轮；39s 片 2 处白底白字残留 → 0，重写 124 帧无接缝；干净片复查 0 处不触发返修。新增 `erase.repair` 报告字段。
- 2026-11-01（字幕文本）：字幕文本只取“留下并擦除”的轨，账本/招牌文字不再混进翻译（23.5s 账本字曾并进台词）；同一句被 OCR 拆成左右两半时按 x 位置拼回（34.8s 曾丢“怕是藏”半句）；新增账本“单字簇”过滤，不误伤普通字幕。
- 2026-11-02：本地擦除加擦除区时域平滑（`inpaint.temporal_ema: 0.4`）：掩码内羽化混合，低运动直接平均、高运动 DIS 光流补偿，切镜/遮挡自动放弃；39s 片纯色背景抖动降 30–44%，全片平均 21%，复查残留仍 0。试过跨帧并集掩码，会把亮色流苏并进掩码补成黑块，已回退。
- 2026-11-02（模型评测）：`bench/video_inpaint/` 实测 ProPainter/E2FGVI-HQ/FGT（同字幕带、同掩码、同 39s 片）：PP 2.64 fps/9.7GB，E2 3.98 fps/7.5GB，FGT ~0.22 fps；均需按切镜分块防 OOM，E2 修了上游 ref 索引越界 bug。复杂纹理镜头 PP 细节最好但有 artifact，E2 更干净偏软，FGT 不采用。结论见 `bench/video_inpaint/BENCH.md` 与 §7.1。
- 2026-11-02（LaMa 提速 + 分流）：LaMa 改 native fp16（FFT fp32）擦除 146.3→126.1s；按镜头裁画布因 FFT 尺寸敏感被否。新增双模型分流 demo：LaMa 简单镜头 + PP 复杂镜头 + 分歧安全闸，PP 只跑 29% 帧，预计 147.7s。见 `bench/video_inpaint/LAMA_OPT_STATUS.md`、`HYBRID_ROUTING.md`。
- 2026-11-03（吞吐 + 质量）：webapp 接入 native fp16 LaMa（此前只有 CLI 用），单支 103s prepare 538→425s；首擦改逐镜头 H.264 裸流分段（`-bf 0`，B 帧跨段 mux 会丢帧），返修只重编码 suspect 镜头、其余流复制拼接，返修后只复查被改镜头；K=4 四支 14.3min / 404 支/日 / 15.1 源小时/日（旧 19.9min / 291 / 10.9）。修白字贴白衣服首字残留（行框缺字兜底）与复查 5 点采样；修 native fp16 大 rect 掩码 NaN 导致返修静默失效（整批 fp32 重跑，97.1s 灰字残影消失）。
- 2026-11-03（P0 可靠性）：修 `webapp/app.py` 硬关 fp16（此前网页实际跑 fp32，与 config/RUNBOOK 相反）；批量加断点续跑（source 指纹+阶段复用，`--fresh` 强制）、阶段超时 killpg、QC 缩略图归档、分段/成片帧数硬断言、压制后渲染验收、K 按 3.1GB/路封顶 4（`--force-jobs`）。修 burn `-shortest` 切尾帧（5ed 旧成片少 4 帧；三支回归 4355/3936/2586 帧与擦除源一致）。回归：K=3 三支 14.4min / 300 支/日 / 12.1 源小时/日，峰值 9.2GB；网页路径 103s prepare 379.3s。方案见 `docs/REVIEW_ACTIONS_2026-11-03.md`。
