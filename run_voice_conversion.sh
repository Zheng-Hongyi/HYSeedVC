#!/usr/bin/env bash
# 在 Seed-VC 项目根目录执行：把「源朗读」换成「参考音频」的音色。
# 用法：
#   ./run_voice_conversion.sh
#   ./run_voice_conversion.sh /path/to/source.mp3 /path/to/reference.mp3 ./output_dir
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif [[ -n "${CONDA_DEFAULT_ENV:-}" ]] && command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "未找到 Python 环境。任选其一："
  echo "  1) venv: /opt/homebrew/bin/python3.10 -m venv .venv && source .venv/bin/activate && pip install -r requirements-macos-arm64.txt"
  echo "  2) conda: conda env create -f environment.yml && conda activate seed-vc"
  exit 1
fi

SOURCE_MP3="${1:-/Users/tal/Desktop/Workbench/BigClassCloud/Code/Others/data/output1.mp3}"
TARGET_MP3="${2:-/Users/tal/Desktop/Workbench/BigClassCloud/Code/Others/data/train/myAudio/output1.mp3}"
OUT_DIR="${3:-$ROOT/output_vc_run}"

WORK="$ROOT/_work"
mkdir -p "$WORK" "$OUT_DIR"

echo "源（内容）: $SOURCE_MP3"
echo "参考（音色）: $TARGET_MP3"
echo "输出目录: $OUT_DIR"

ffmpeg -y -i "$SOURCE_MP3" -ac 1 -ar 22050 "$WORK/source.wav"
ffmpeg -y -i "$TARGET_MP3" -ac 1 -ar 22050 "$WORK/target.wav"

# 首次运行会从 Hugging Face 下载权重到 ./checkpoints/hf_cache
export HF_HUB_CACHE="${HF_HUB_CACHE:-$ROOT/checkpoints/hf_cache}"
mkdir -p "$HF_HUB_CACHE"

# Apple Silicon 上 MPS + fp16 易触发崩溃时：改用下面两行之一
# export SEED_VC_FORCE_CPU=1
# FP16_EXTRA=(--fp16 True)
FP16_EXTRA=(--fp16 False)

# inference F0 相关（勿插在反斜杠续行中间，否则会打断续行并报 command not found）：
# --f0-condition True：44k F0 模型；False 为默认 DiT 非 F0。
# --f0-scale：仅 F0 模式下有声段 F0 乘数（如 0.75 略降调，1.0 不变）。

# 中断（Ctrl+C）/ 关终端 / SIGTERM 时结束 inference 子进程，避免 Python+MPS 占着 Metal 显存导致下次运行卡顿。
# 若仍卡：活动监视器结束 python，或执行 pkill -f "Seed-VC/inference.py"（按本机路径调整）。
INF_PID=""
_cleanup_infer_child() {
  if [[ -n "${INF_PID:-}" ]] && kill -0 "$INF_PID" 2>/dev/null; then
    echo "正在终止 inference 子进程（释放 GPU/MPS）…" >&2
    kill -TERM "$INF_PID" 2>/dev/null || true
    local _i=0
    while kill -0 "$INF_PID" 2>/dev/null && [[ $_i -lt 25 ]]; do
      sleep 0.2
      _i=$((_i + 1))
    done
    kill -KILL "$INF_PID" 2>/dev/null || true
  fi
}
_on_infer_signal() {
  _cleanup_infer_child
  trap - INT TERM HUP
  exit 130
}
trap _on_infer_signal INT TERM HUP

"$PY" inference.py \
  --source "$WORK/source.wav" \
  --target "$WORK/target.wav" \
  --output "$OUT_DIR" \
  --diffusion-steps 30 \
  --length-adjust 1.0 \
  --inference-cfg-rate 0.7 \
  --f0-condition True \
  --f0-scale 0.75 \
  "${FP16_EXTRA[@]}" &
INF_PID=$!

wait_ec=0
wait "$INF_PID" || wait_ec=$?
INF_PID=""
trap - INT TERM HUP
if [[ "$wait_ec" -eq 0 ]]; then
  echo "完成。查看目录: $OUT_DIR"
fi
exit "$wait_ec"
