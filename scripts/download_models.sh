#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/wan_models}"
HF_CACHE="${HF_CACHE:-$ROOT/.cache/huggingface}"

mkdir -p "$MODEL_ROOT" "$HF_CACHE"

hf download Wan-AI/Wan2.1-T2V-1.3B \
  --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
  --cache-dir "$HF_CACHE" \
  --local-dir "$MODEL_ROOT/Wan2.1-T2V-1.3B"

hf download Wan-AI/Wan2.1-T2V-14B \
  --revision a064a6c71f5be440641209c07bf2a5ce7a2ff5e4 \
  --cache-dir "$HF_CACHE" \
  --local-dir "$MODEL_ROOT/Wan2.1-T2V-14B"

hf download facebook/vjepa2-vith-fpc64-256 \
  --revision b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5 \
  --cache-dir "$HF_CACHE"
