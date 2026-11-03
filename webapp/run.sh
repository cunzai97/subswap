#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# 擦除/翻译都固定在 V100；7900 XTX 上的常驻服务不要动
export CUDA_VISIBLE_DEVICES=0
PY=${PY:-/usr/bin/python3}

"$PY" - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA 不可用：请用系统 python3（torch 2.6.0+cu124，支持 V100 sm_70）")
print(f"GPU: {torch.cuda.get_device_name(0)} · torch {torch.__version__}")
PY

if ! curl -sf --max-time 3 http://127.0.0.1:1509/v1/models >/dev/null; then
  echo "警告：127.0.0.1:1509 翻译服务不通；任务会在翻译阶段失败。" >&2
fi

exec "$PY" app.py
