#!/usr/bin/env bash
# Build a GGUF at a given quant from a Hugging Face model and import it into ollama.
#   scripts/quantize.sh Qwen/Qwen3-4B Q5_K_M qwen3-4b-local:q5_K_M
# Needs: git, python, cmake, a C++ compiler, huggingface-cli. Uses ./models/ as scratch.
set -euo pipefail
hf_model="$1"; quant="$2"; tag="$3"
work="models"; mkdir -p "$work"

if [ ! -d "$work/llama.cpp" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$work/llama.cpp"
  cmake -S "$work/llama.cpp" -B "$work/llama.cpp/build" -DGGML_CUDA="${GGML_CUDA:-OFF}"
  cmake --build "$work/llama.cpp/build" --target llama-quantize -j
  pip install -q -r "$work/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
fi

name="$(basename "$hf_model")"
src="$work/hf/$name"
f16="$work/$name-F16.gguf"
out="$work/$name-$quant.gguf"

[ -d "$src" ] || huggingface-cli download "$hf_model" --local-dir "$src"
[ -f "$f16" ] || python "$work/llama.cpp/convert_hf_to_gguf.py" "$src" --outtype f16 --outfile "$f16"
[ -f "$out" ] || "$work/llama.cpp/build/bin/llama-quantize" "$f16" "$out" "$quant"

# reuse the chat template + params of the library model so only the weights differ
base="${BASE_TAG:-qwen3:4b}"
ollama show "$base" --modelfile | sed "s|^FROM .*|FROM $(realpath "$out")|" > "$work/Modelfile.$quant"
ollama create "$tag" -f "$work/Modelfile.$quant"
echo "created $tag from $out"
