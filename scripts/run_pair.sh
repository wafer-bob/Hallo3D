#!/bin/bash
# Run one prompt through a baseline and its Hallo3D-enhanced counterpart.
#
# Usage:
#   bash scripts/run_pair.sh <framework> <gpu> "<prompt>" [extra args...]
#   framework in {gaussiandreamer, dreamfusion-sd, sjc, magic3d-coarse-sd}
#
# Baseline uses the original threestudio config (batch_size forced to 4 and
# iterations 1/4, per paper Appendix B); Hallo3D uses the extension config.

set -e
source "$(dirname "$0")/env.sh"

FRAMEWORK=$1
GPU=$2
PROMPT=$3
shift 3 || true

cd "$THREESTUDIO_DIR"

case $FRAMEWORK in
gaussiandreamer)
    BASE_CFG=custom/threestudio-gaussiandreamer/configs/gaussiandreamer.yaml
    HALLO_CFG=custom/threestudio-hallo3d/configs/hallo3d-gaussiandreamer.yaml
    BASE_EXTRA=(data.batch_size=4 "system.geometry.geometry_convert_from=shap-e:${PROMPT}")
    HALLO_EXTRA=("system.geometry.geometry_convert_from=shap-e:${PROMPT}")
    ;;
dreamfusion-sd)
    BASE_CFG=configs/dreamfusion-sd.yaml
    HALLO_CFG=custom/threestudio-hallo3d/configs/hallo3d-dreamfusion-sd.yaml
    BASE_EXTRA=(data.batch_size=4 trainer.max_steps=2500)
    HALLO_EXTRA=()
    ;;
sjc)
    BASE_CFG=configs/sjc.yaml
    HALLO_CFG=custom/threestudio-hallo3d/configs/hallo3d-sjc.yaml
    BASE_EXTRA=(data.batch_size=4 trainer.max_steps=2500)
    HALLO_EXTRA=()
    ;;
magic3d-coarse-sd)
    BASE_CFG=configs/magic3d-coarse-sd.yaml
    HALLO_CFG=custom/threestudio-hallo3d/configs/hallo3d-magic3d-coarse-sd.yaml
    BASE_EXTRA=(data.batch_size=4 trainer.max_steps=2500)
    HALLO_EXTRA=()
    ;;
*)
    echo "unknown framework: $FRAMEWORK" && exit 1
    ;;
esac

echo "=== [$FRAMEWORK][gpu $GPU] baseline: $PROMPT"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON launch.py --config "$BASE_CFG" --train --gpu 0 \
    system.prompt_processor.prompt="$PROMPT" "${BASE_EXTRA[@]}" "$@"

echo "=== [$FRAMEWORK][gpu $GPU] hallo3d: $PROMPT"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON launch.py --config "$HALLO_CFG" --train --gpu 0 \
    system.prompt_processor.prompt="$PROMPT" "${HALLO_EXTRA[@]}" "$@"
