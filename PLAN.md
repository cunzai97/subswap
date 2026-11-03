# 短剧硬字幕逐帧擦除 · 实施方案 v1

目标：40 部/天 × 75 分钟（3000 分钟素材），本地逐帧擦除，质量达到"盖上新字幕后看不出"。
硬件：V100 16G（CUDA 12.4）+ RX 7900 XTX 24G（ROCm 10.0 / gfx1100），同机，36 核 / 125G 内存 / 399G 可用磁盘。

> 运行手册见 [RUNBOOK.md](RUNBOOK.md)：CLI + webapp、GPU 绑定、片源要求、质检清单与已知坑（2026-10-31 实测）。

---

## 0. 选型结论

| 决策项 | 选择 | 理由 |
|---|---|---|
| 修复模型 | LaMa（big-lama TorchScript） | 纯 PyTorch、没有自定义 CUDA 算子，ROCm 能直接跑；fp16 吞吐高；大掩码纹理补全够用 |
| 输入形态 | 字幕带裁剪（1080×约 256），等比缩到 ≤1024 宽 | 计算面积约等于 512×512，吞吐是整帧的 3–4 倍，质量损失可忽略 |
| 掩码 | 镜头级固定矩形：OCR 抽样 → 跨帧并集 → 外扩 | 掩码稳定是抑制闪烁最有效的手段，比任何后处理都管用 |
| 运行分工 | 7900XTX 单 worker 擦除；V100 做 OCR / 编码 / 兜底 | 生态互补；同一镜头只在一张卡上处理，不做帧级跨卡 |
| 质量兜底 | 重跑 → 底板覆盖 → 云 API | 有降级链，不阻塞交付 |

不选 ProPainter / 扩散类：1080p 在 16G 上会 OOM，速度 1–8 fps，按当前量需要 10–40 张卡，不成立。

---

## 1. 验收指标（M0 先钉死）

| 指标 | 目标 | 测法 |
|---|---|---|
| 吞吐 | ≥ 40 fps（1080p 字幕带，fp16，batch 4） | `bench_lama.py` |
| 漏检 | < 2% 的帧在擦除区仍有可识别文字 | 擦除后抽样跑检测 |
| 闪烁 | 掩码区帧间差 / 环带帧间差 < 1.5 | 自动指标，见 §7 |
| 单部耗时 | ≤ 45 分钟（75 分钟剧） | 端到端 |
| 观感 | 抽检 5% 的剧，≥ 4 分（5 分制） | 人眼 |

---

## 2. 环境部署

### 2.1 机器现状（已确认）

- V100：驱动 580.178.04，系统 Python 3.12 + torch 2.6.0+cu124；ffmpeg 6.1.1，带 `h264_nvenc` / `hevc_nvenc`
- 7900 XTX：ROCm 10.0.0，gfx1100；已有 `~/work/ComfyUI-ROCm714/.venv`（torch 2.12.0+rocm7.14.0 可用），说明这条链路是通的
- 网络：`github.com`、`huggingface.co` 不通；`pypi.org`、`hf-mirror.com`、`download.pytorch.org` 可用 —— 权重和 wheel 都走镜像

### 2.2 V100 节点（OCR + 编码 + 兜底）

```bash
python3 -m venv ~/venvs/erase-cuda
~/venvs/erase-cuda/bin/pip install -U pip
~/venvs/erase-cuda/bin/pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
~/venvs/erase-cuda/bin/pip install opencv-python-headless numpy scipy pyyaml \
  rapidocr_onnxruntime "scenedetect[opencv]" faster-whisper
```

V100 是 Volta（sm_70），CUDA 13 已不支持它，torch 锁死在这个版本，别跟着系统升级。

> 2026-10-31 状态：`~/venvs/erase-cuda` 已被升级到 torch 2.15.0+cu132，实测不含 sm_70 kernel，不能跑 LaMa（`CUDA error: no kernel image is available`）。当前擦除用系统 `python3`（torch 2.6.0+cu124）；要恢复 venv 请按本节重建。详见 [RUNBOOK.md](RUNBOOK.md) §2.1。

### 2.3 AMD 节点（擦除主力）

```bash
python3 -m venv ~/venvs/erase-rocm
~/venvs/erase-rocm/bin/pip install -U pip
~/venvs/erase-rocm/bin/pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/rocm10.0
~/venvs/erase-rocm/bin/pip install --no-deps simple-lama-inpainting
~/venvs/erase-rocm/bin/pip install opencv-python-headless numpy pillow
```

验证：

```bash
~/venvs/erase-rocm/bin/python -c \
"import torch;print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.x.x+rocm10.0 ... True AMD Radeon RX 7900 XTX
```

如果 rocm10.0 的 wheel 装不上，退回 `rocm7.2` 索引；本机已有 torch 2.12.0+rocm7.14.0 在跑，向后兼容没问题。

两张卡在同一台机器上不冲突：CUDA 版 torch 只认 `CUDA_VISIBLE_DEVICES`，ROCm 版只认 `HIP_VISIBLE_DEVICES`，各管各的。

### 2.4 权重

```bash
mkdir -p ~/models/lama && cd ~/models/lama
curl -L -o big-lama.zip \
  https://hf-mirror.com/smartywu/big-lama/resolve/main/big-lama.zip
unzip -o big-lama.zip
find . -maxdepth 3 \( -name "*.pt" -o -name "*.ckpt" \) -exec ls -lh {} \;
```

拿到 `big-lama.pt` 后先验证能否在 ROCm 上加载：

```bash
~/venvs/erase-rocm/bin/python -c \
"import torch;m=torch.jit.load('big-lama.pt',map_location='cuda');m.eval();print('JIT ok on',next(m.parameters()).device)"
```

三种情况分别处理：

- 加载成功 → 直接用
- 报 CUDA 特有算子 → 改用原始 `saicinpainting` 结构 + `best.ckpt` 加载（只改 10 行）
- 权重不是 TorchScript（zip 里只有 `best.ckpt` + `config.yaml`）→ 同上，走原始结构

备选来源：ModelScope 搜 `big-lama`；有代理时用官方 GitHub release。

### 2.5 M0 基准（第一件事，不做完不写业务代码）

```bash
~/venvs/erase-rocm/bin/python scripts/bench_lama.py sample.mp4 out.mp4 \
  --start 0 --dur 20 --rect 0.06,0.72,0.94,0.90 --batch 4
```

输出：fps、单帧耗时、显存峰值、按 75 分钟/集折算的日产能。**这一个数字决定后面所有排期**，低于 30 fps 就先做 §8 的调优，不要硬上。

---

## 3. 工作流

### 3.1 任务 DAG

```
ingest → probe → shot_split
   └─ 每个镜头：ocr_mask → erase → qa ──失败──> degrade(重跑/底板/云)
                     ↓
             concat → 音轨回填 → 翻译/ASS → 最终压制(NVENC) → 交付
```

擦除阶段只产出"无原字幕的视频"，翻译和压制是后续阶段，接口就是文件路径 + 字幕时间轴 JSON。

### 3.2 拆条

```bash
scenedetect -i ep.mp4 detect-content --threshold 27 list-scenes -o work/ep01/
```

- 阈值 27 对短剧够用；合并 < 0.5 秒的碎镜头
- 可变帧率源先归一：`ffmpeg -i in.mp4 -vf fps=25 -c:v libx264 -crf 14 ...`
- 输出 `shots.json`：`[{"idx":0,"start":0.0,"end":3.6}, ...]`

### 3.3 掩码生成

用 OCR 检测框，**不用** ASR 时间轴（短剧 BGM 重，ASR 打轴偏）。

```python
# mask.py 核心逻辑
from rapidocr_onnxruntime import RapidOCR

OCR = RapidOCR()          # 进程内复用，别每帧新建

def detect(frame_bgr, band=(0.70, 0.95), conf=0.5):
    res, _ = OCR(frame_bgr)
    h, w = frame_bgr.shape[:2]
    out = []
    for box, text, score in (res or []):
        if score < conf:
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        x0, x1 = max(0, int(min(xs))), min(w, int(max(xs)))
        y0, y1 = max(0, int(min(ys))), min(h, int(max(ys)))
        if y1 < h * band[0] or y0 > h * band[1]:   # 只要底部字幕带
            continue
        if x1 - x0 < 0.05 * w:                     # 太窄的不要
            continue
        out.append({"box": (x0, y0, x1, y1), "text": text, "score": float(score)})
    return out
```

镜头级掩码的生成规则：

1. 每 12 帧（25fps 下约 2fps）抽样检测一次
2. 按 y 中心把框聚成 1–2 行，每行取并集矩形
3. 外扩 8px 吃掉描边和阴影；**上沿再多扩一个字高**（给英文两行留位置）
4. 静态场景文字过滤：同一文本在镜头内持续 > 2 秒且宽度 > 40% 画面 → 判定为招牌/贴片，不擦
5. 输出 `masks.json`：`{shot_idx: [[x0,y0,x1,y1], ...]}`，后续每帧复用

第 3 条很关键：中文一行，英文通常两行。擦除区预留不够，后面压制时新字幕会压到没擦干净的边上。

### 3.4 逐帧修复

单帧流程：裁剪 → 补齐到 8 的倍数 → fp16 LaMa → 还原尺寸 → 颜色匹配 → 羽化合成。

```python
import cv2, numpy as np, torch

def inpaint_crop(model, crop_bgr, mask, max_w=1024, fp16=True, dev="cuda"):
    h, w = crop_bgr.shape[:2]
    scale = min(1.0, max_w / w)
    cw, ch = max(8, int(w * scale) // 8 * 8), max(8, int(h * scale) // 8 * 8)

    crop = cv2.resize(crop_bgr, (cw, ch), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    m = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)

    img = torch.from_numpy(crop[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().div_(255).to(dev)
    msk = torch.from_numpy((m > 0).astype(np.float32))[None, None].to(dev)
    if fp16:
        img = img.half()

    with torch.inference_mode():
        out = model(img, msk)                       # LaMa 返回整张图
    out = out.float().clamp_(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    out = (out * 255).round().astype(np.uint8)[:, :, ::-1]   # RGB -> BGR
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
```

颜色匹配（防止修复区比周围亮/暗一个色阶，这是"一眼假"的主要来源）：

```python
def match_color(src, dst, mask):
    """把 dst 掩码区的均值/方差对齐到 src 的环带"""
    ring = (cv2.dilate(mask, np.ones((11, 11), np.uint8)) > 0) & (mask == 0)
    if ring.sum() < 50:
        return dst
    for c in range(3):
        s_m, s_s = src[..., c][ring].mean(), src[..., c][ring].std() + 1e-6
        hole = mask > 0
        d_m, d_s = dst[..., c][hole].mean(), dst[..., c][hole].std() + 1e-6
        dst[..., c][hole] = (dst[..., c][hole] - d_m) * (s_s / d_s) + s_m
    return dst
```

合成：掩码高斯羽化 2–3px 后 alpha 混合，避免硬边。

时序处理（抑制闪烁，按需开）：

- 默认：镜头级固定掩码 + 颜色匹配，已经能压掉大部分闪烁
- 增强：掩码区做指数平滑 `out_t = 0.65*out_t + 0.35*out_{t-1}`，只对运动量低的镜头开，否则拖影

### 3.5 编码与拼接

逐镜头处理，输出段落文件：

```bash
# 解码（Python 侧 subprocess 管道，bgr24）
ffmpeg -v error -ss {start} -t {dur} -i ep.mp4 -an -f rawvideo -pix_fmt bgr24 -

# 编码段落（所有段落参数必须完全一致）
ffmpeg -v error -y -f rawvideo -pix_fmt bgr24 -s {W}x{H} -r 25 -i - -an \
  -c:v libx264 -crf 14 -preset veryfast -pix_fmt yuv420p seg_{idx:05d}.mp4
```

```bash
# 拼接 + 回填原音轨
ffmpeg -v error -f concat -safe 0 -i list.txt -c copy -an erased_v.mp4
ffmpeg -v error -i erased_v.mp4 -i ep.mp4 -map 0:v:0 -map 1:a? -c copy -shortest erased.mp4
```

段落用 CRF 14 是刻意的：后面烧字幕还要再编码一次，中间片必须接近无损。短剧源码本身码率低（1–3 Mbps），一集段落总量也就 2–3GB，磁盘不是瓶颈。

### 3.6 最终压制（V100）

```bash
ffmpeg -v error -y -i erased.mp4 -vf "ass=subs/final.ass" \
  -c:v h264_nvenc -preset p5 -cq 21 -b:v 0 -maxrate 8M -bufsize 16M \
  -c:a copy out.mp4
```

字幕样式建议：底部居中、两行上限、描边 + 20% 半透明底板。底板既是可读性设计，也顺手盖住少量残影。

---

## 4. 调度与并发（单机版）

不要上 Celery/Temporal，单机两个 worker 用 `multiprocessing` + SQLite 就够，复杂度低一个数量级。

```sql
CREATE TABLE IF NOT EXISTS shots (
  id TEXT PRIMARY KEY,          -- ep01/00023
  ep TEXT NOT NULL,
  idx INTEGER NOT NULL,
  start REAL, end REAL,
  status TEXT DEFAULT 'queued', -- queued/running/done/failed/skip
  attempts INTEGER DEFAULT 0,
  fps REAL, ms INTEGER, err TEXT
);
CREATE INDEX IF NOT EXISTS idx_status ON shots(status, ep, idx);
```

领取任务（SQLite 3.35+ 支持 RETURNING）：

```sql
UPDATE shots SET status='running', attempts=attempts+1
WHERE id = (SELECT id FROM shots WHERE status='queued' ORDER BY ep, idx LIMIT 1)
RETURNING id, ep, idx, start, end;
```

进程模型：

| 进程 | 卡 | 职责 |
|---|---|---|
| `masker` | CPU（8–12 线程） | 拆条、OCR 抽样、生成 masks.json |
| `eraser` | 7900XTX | 消费 shot 队列，逐帧修复，写段落 |
| `encoder` | V100 | 拼接、烧字幕、NVENC 成品 |
| `qa` | CPU | 残影检测、闪烁指标 |

幂等：任务 id = `集/镜头号`，输出路径确定；重跑前先删旧段落。失败重试 3 次，仍失败进降级链。夜里跑批时把 AMD 卡的 runpm 休眠关掉。

---

## 5. 质量闸门与降级

每个镜头擦完立即检查，不过关的当场打回，不要等到整集结束。

| 检查 | 方法 | 阈值 |
|---|---|---|
| 残影 | 擦除后抽 5 帧，在掩码区跑一次检测 | 任一框置信度 > 0.5 → 失败 |
| 闪烁 | 掩码区帧间差 ÷ 环带帧间差 | > 1.5 且持续 > 20% 帧对 → 警告 |
| 结构异常 | 掩码区与环带的色差 | 均值差 > 12/255 → 重做颜色匹配 |

降级链（按成本从低到高）：

1. **重跑**：掩码外扩 +6px，换 fp32（fp16 偶发溢出会出现色块）
2. **底板覆盖**：用掩码周边中值色画圆角底板，直接盖住。零成本，观感是"字幕板"
3. **云 API**：把该镜头单独切出来送火山 AI MediaKit 精细化版（1 元/分钟，一个 4 秒镜头几分钱），失败镜头通常 < 5%，成本可忽略
4. **人工**

---

## 6. 容量与成本

| 项 | 数字 |
|---|---|
| 单集帧数 | 75 min × 60 × 25 = 112,500 帧 |
| 单集 GPU 时间 | 112,500 ÷ fps |
| 全天 40 部 | 450 万帧 |
| @50 fps | 25 GPU·小时（一张 7900XTX 跑得下） |
| @30 fps | 41.7 GPU·小时（要开 §8 的优化或分出 20% 给云） |
| 电费 | 7900XTX 满载约 355W，一天 8.5 度，约 5 元 |

> 以上是设计估算，实测见 §12：端到端 21 fps，单部 89 分钟 GPU 时间，两张卡约 32 部/天。

V100 那侧的 OCR（2fps 抽样 ≈ 36 万帧/天）和 NVENC 编码（≈ 6–8 GPU·小时/天）都在余量内。翻译阶段不在这条链路里，后续接 LLM。

---

## 7. 参数表（config.yaml）

见同目录 `config.yaml`。几个会反复调的：

| 参数 | 默认 | 说明 |
|---|---|---|
| `band` | `[0.70, 0.95]` | 字幕带纵向范围（相对高度） |
| `mask_pad` | `8` | 掩码外扩像素；调到 12 可吃掉更多描边 |
| `mask_extra_top` | `48` | 上沿额外扩，给英文两行留位 |
| `max_w` | `1024` | LaMa 输入最大宽度，降到 768 提速约 1.7× |
| `batch` | `4` | 8 可再提 10–20%，显存够就上 |
| `fp16` | `true` | 出现色块就关 |
| `temporal_ema` | `0.0` | 0 关闭；0.35 开启平滑 |

---

## 8. 吞吐不够时的调优顺序

1. `max_w` 1024 → 768（质量损失很小，因为新字幕会盖住）
2. `batch` 4 → 8
3. 静态镜头背景板复用：镜头内背景帧间差 < 阈值时，只修第一帧，其余帧贴回。短剧对话戏大量固定机位，通常能省 30–50% 计算
4. 只处理字幕实际出现的帧段（短剧字幕覆盖率约 70–90%，收益有限）
5. 仍不够 → 把 20–30% 的镜头分给云 API，或加一张卡

---

## 9. 里程碑

| 阶段 | 内容 | 验收 | 预估 |
|---|---|---|---|
| M0 | 环境 + 权重 + `bench_lama.py` 跑通 20 秒样片 | 拿到 fps / 显存 / 日产能 | 半天 |
| M1 | 单集端到端：拆条 → 掩码 → 擦除 → 拼接 → 烧字幕 | 一集 75 分钟完整出片 | 1.5 天 |
| M2 | 批量：SQLite 队列 + 重试 + 日志 + 报告 | 10 集无人值守跑完 | 1 天 |
| M3 | 质量闸门 + 降级链 + 校对台（最小可用） | 失败镜头自动进入降级 | 1.5 天 |
| M4 | 接翻译/压制流水线 + 对外 API（形状照竞品文档） | 客户可下单 | 3–5 天 |

---

## 10. 风险

| 风险 | 影响 | 对策 |
|---|---|---|
| LaMa JIT 在 ROCm 上跑不起来 | 方案要换 | M0 第一天验证；备选 ONNX Runtime + MIGraphX，或退回 V100（慢但能跑） |
| 权重下载路径失效 | 部署卡住 | 三个来源备选，M0 就把权重落到本地 |
| 人物手部穿过字幕区被抹掉 | 局部穿帮 | 镜头级 QA 检出高运动镜头，走底板或云 |
| 静态招牌被误擦 | 画面信息丢失 | 文本稳定性过滤 + 人工白名单 |
| 闪烁仍明显 | 观感差 | 镜头级固定掩码 + 颜色匹配 + EMA；极端镜头走云 |
| V100 NVENC 质量一般 | 成品发糊 | 提高码率（cq 19）或授权时用 CPU 慢压 |
| 40 部/天的量不成立 | 成本模型变化 | 单集成本与并发解耦，量小就全本地慢慢跑 |

---

## 11. 部署完成后的自检清单

- [ ] `torch.cuda.is_available()` 在 ROCm 环境为 True，设备名是 7900 XTX
- [ ] `torch.jit.load` 加载 `big-lama.pt` 成功并能在 GPU 上前向
- [ ] 20 秒样片擦除后肉眼无残影，画面不抖
- [ ] 段落拼接处无花屏（参数一致、关键帧对齐）
- [ ] NVENC 输出能被目标平台正常上传播放
- [ ] 一集跑完的临时文件自动清理，磁盘不涨


---

## 12. 实测记录（2026-10-30，本机）

样本：24 秒合成竖屏片（4 种背景 + 9 条中文硬字幕，两条为两行排版）。
翻译阶段用的是内置 mock 英文（无 API key），只验证链路；云端接口已实现但未实跑。

> 真实片源实测（10 分钟双语、3 分钟纯中文、rect 返工）见 [RUNBOOK.md](RUNBOOK.md) §6。真实片源 11–15 fps，低于本节的合成片 21 fps。
配置：LaMa TorchScript，输入固定 1024×256，batch 4，fp16。

| 指标 | 7900XTX（ROCm 10.0） | V100 16G（CUDA 12.4） |
|---|---|---|
| LaMa 纯前向 | 56.5 fps | 50.2 fps |
| 端到端（OCR/解码/贴合/编码） | 20.8 fps | 22.5 fps |
| 显存峰值 | 1.8 GB | 0.5 GB |
| 预热（MIOpen / cuDNN） | 26 s | 5 s |
| 24 秒样片总耗时 | ~100 s | ~76 s |
| 残留检测 | 0/48 帧 | 0/48 帧 |

外推 75 分钟一集：约 89 分钟 GPU 时间。两张卡 24 小时 ≈ 32 部/天；要到 40 部/天还需要下面三条之一。

### 三个必须记住的实测结论

1. **fp16 只对 2 的幂尺寸可用**。LaMa 的 FFC 用 cuFFT/hipFFT，半精度要求 FFT 尺寸是 2 的幂。
   输入高 288 直接报错，固定成 256 后可跑，速度约为 fp32 的 1.9 倍。
   所以所有镜头统一缩放到 1024×256，不再按实际字幕带高度跑。
2. **尾批不足一个 batch 会触发 MIOpen 重新选算法**，单次 20 秒以上。
   150 帧（37×4+2）比 300 帧（75×4）还慢，就是这个原因。flush 里把尾批补齐到固定 batch 即可。
3. **不同输入形状也会反复调优**。固定形状 + 启动预热 3 次，把一次性成本放在 worker 启动时。

### 当前瓶颈与下一步优化

- 端到端 21 fps 里，模型只占约一半；后处理（颜色匹配 + 羽化合成 + 6MB/帧管道写入）约 19 ms/帧。
  下一步把颜色匹配和合成搬到 GPU，预计能到 28–32 fps。
- 真实短剧大量固定机位对话镜头，背景板复用能省 30–50% 计算，这是冲 40 部/天最有效的一条。
- 合成色条背景上能看到 LaMa 的轻微重建痕迹；加半透明字幕底板后基本不可见（已默认开启）。

---

## 13. 模型选型调研（2026-10）

结论先说：**开源视频修复在 2026 年仍然是"最新"和"吞吐"不可兼得**，差距 5–10 倍。
当前量产链路继续用 LaMa 不是因为它新，而是因为它是唯一同时满足「1080p 带裁 ≥20 fps + Apache 2.0 可商用 + 显存 <2G」的。

| 模型 | 时间 | 架构 | 1080p 实测速度 | 显存 | 许可证 | 现状 |
|---|---|---|---|---|---|---|
| LaMa（当前） | 2021 | GAN/UNet | 21 fps 端到端（字幕带） | 0.5–1.8 G | Apache 2.0 | 已落地 |
| ProPainter | 2023 | 光流 + Transformer | 1–5 fps | 16 G 会 OOM | S-Lab 非商用 | 不可商用 |
| DiffuEraser | 2024 | SD UNet + ProPainter 先验 | <1 fps 级 | 24 G | 需逐个确认 | 太慢 |
| MiniMax-Remover | 2025 | 1.3B DiT，6 步无 CFG | 2 fps（4090，全帧） | 12–20 G | **CC-BY-NC 4.0** | 能跑，禁商用 |
| Netflix VOID | 2026-04 | 5B CogVideoX，两阶段 | 未公开 | **40 G+** | Apache 2.0 | 卡不够 |
| SEDiT | 2026-05 | 一步 DiT，**免掩码**，字幕专用 | 65 帧 / 4 s（A800）≈16 fps | 未公开 | 未放代码 | 等权重 |
| 火山 AI MediaKit ReFM | 2026 | 商用 DiT | 按分钟 | — | 商用 API | 1 元/分钟 |

几个关键事实：

- **SEDiT 是唯一"为字幕擦除而生"的模型**：免掩码、一步生成、支持无限长。论文口径 65 帧 4 秒（A800），
  折合 16 fps；按 A800 与 7900XTX 的算力差估，本机大概 5–8 fps。项目页暂无代码和权重，只能等。
- **MiniMax-Remover 权重已开源（2.8G）**，但许可证是 CC-BY-NC 4.0，**不能用于付费服务**；
  且 1080p 全帧只有 2 fps，比 LaMa 慢一个数量级。拿它做全量，40 部/天需要 25 张 4090，不成立。
- **Netflix VOID 质量最强**（人类偏好 64.8% vs Runway 18.4%），Apache 2.0 可商用，但要 40G+ 显存，16/24G 跑不动。
- **ProPainter 是 S-Lab 非商用许可**，即使速度能接受也不能直接商业化。
- 所以 2026 年真正可商用的开源快模型，仍然是 LaMa。

### 建议：双层，而不是换模型

| 档位 | 引擎 | 覆盖 | 成本 |
|---|---|---|---|
| 标准档 | LaMa（本地，带裁） | 90–95% 镜头 | 电费 |
| 精修档 | 火山 API / 等 SEDiT | 手穿字幕、复杂纹理、客户加价 | 1 元/分钟；一个 4 秒镜头几分钱 |

### 比换模型更划算的一件事

现在的掩码是**矩形**，LaMa 要凭空补整块面积。改成**笔画级掩码**（OCR 框内按文字颜色/边缘做二值化 + 膨胀），
掩码面积能降 50–70%：残影更少、速度更快（面积小了）、对 LaMa 这种纹理补全模型提升最直接。
这件事的收益大于把 LaMa 换成任何一个慢 5–10 倍的扩散模型。
