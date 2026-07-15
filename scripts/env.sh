#!/bin/bash
# Hallo3D — common environment for all runs.
# Usage: source scripts/env.sh
# Every variable can be pre-set in the shell to override the default.

# path to your threestudio checkout (with custom/threestudio-hallo3d linked in)
export THREESTUDIO_DIR=${THREESTUDIO_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/threestudio"}
# this repository
export HALLO3D_DIR=${HALLO3D_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
# python interpreters (training/eval vs. LMM server may live in different envs)
export PYTHON=${PYTHON:-python}
export LMM_PYTHON=${LMM_PYTHON:-python}

# Optional: fully-offline HuggingFace loading once models are cached.
# export HF_HOME=/path/to/huggingface/cache
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# keep the local LMM server reachable if you use an http(s) proxy
export no_proxy=${no_proxy:-127.0.0.1,localhost}
export TOKENIZERS_PARALLELISM=false
