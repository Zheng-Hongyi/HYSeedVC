#!/usr/bin/env bash
# 在 Seed-VC 项目根目录执行：V2 管线（inference_v2.py），口语音色/口音转换，无 F0/SVC 参数。
# 调用方式与 run_voice_conversion.sh 相同，仅推理入口与参数集不同。
# 用法：
#   ./run_voice_conversion_v2.sh
#   ./run_voice_conversion_v2.sh /path/to/source.mp3 /path/to/reference.mp3 ./output_dir
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
OUT_DIR="${3:-$ROOT/output_vc_run_v2}"

WORK="$ROOT/_work"
mkdir -p "$WORK" "$OUT_DIR"

echo "源（内容）: $SOURCE_MP3"
echo "参考（音色）: $TARGET_MP3"
echo "输出目录: $OUT_DIR"

ffmpeg -y -i "$SOURCE_MP3" -ac 1 -ar 22050 "$WORK/source.wav"
ffmpeg -y -i "$TARGET_MP3" -ac 1 -ar 22050 "$WORK/target.wav"

export HF_HUB_CACHE="${HF_HUB_CACHE:-$ROOT/checkpoints/hf_cache}"
mkdir -p "$HF_HUB_CACHE"

# V2 脚本内部默认使用 float16；无需 --fp16。若需加速可研究 inference_v2.py 的 --compile（CUDA/Triton）。

# 中断时结束子进程，避免占着 GPU/MPS。
INF_PID=""
_cleanup_infer_child() {
  if [[ -n "${INF_PID:-}" ]] && kill -0 "$INF_PID" 2>/dev/null; then
    echo "正在终止 inference_v2 子进程（释放 GPU/MPS）…" >&2
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

# 要口音/情绪转换，把脚本里的 --convert-style False 改成 True 即可
"$PY" inference_v2.py \
  --source "$WORK/source.wav" \
  --target "$WORK/target.wav" \
  --output "$OUT_DIR" \
  --diffusion-steps 50 \
  --length-adjust 1.0 \
  --intelligibility-cfg-rate 0.6 \
  --similarity-cfg-rate 0.8 \
  --convert-style True \
  --anonymization-only False \
  --top-p 0.9 \
  --temperature 1.0 \
  --repetition-penalty 1.0 &
INF_PID=$!

wait_ec=0
wait "$INF_PID" || wait_ec=$?
INF_PID=""
trap - INT TERM HUP
if [[ "$wait_ec" -eq 0 ]]; then
  echo "完成。查看目录: $OUT_DIR"
fi
exit "$wait_ec"
