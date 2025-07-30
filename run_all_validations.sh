#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

NUM_QUERY_POINTS=50000
TOP_K=5000
NUM_POINTS=5000
VARIABLE=2
NUM_TREES_TOTAL=50
DECIDUOUS=true

declare -A ENV_NAMES
ENV_NAMES["shadow_dsm"]="Shadow DSM"
ENV_NAMES["shadow_ortho"]="Shadow Ortho"
ENV_NAMES["shadow"]="Shadow DSM+Ortho"
ENV_NAMES["silhouettes_dsm"]="Silh. DSM"
ENV_NAMES["silhouettes_ortho"]="Silh. Ortho"
ENV_NAMES["silhouettes"]="Silh. DSM+Ortho"
ENV_NAMES["shadow_silhouettes_dsm"]="Sh+Si DSM"
ENV_NAMES["shadow_silhouettes_ortho"]="Sh+Si Ortho"
ENV_NAMES["shadow_silhouettes"]="Sh+Si DSM+Ortho"
ENV_NAMES["bce_dsm"]="BCE DSM"
ENV_NAMES["bce_new_ortho"]="BCE Ortho"
ENV_NAMES["bce"]="BCE DSM+Ortho"
ENV_NAMES["bce_shadow_dsm"]="BCE+Sh DSM"
ENV_NAMES["bce_shadow_new_ortho"]="BCE+Sh Ortho"
ENV_NAMES["bce_shadow"]="BCE+Sh DSM+Ortho"
ENV_NAMES["bce_silhouettes_dsm"]="BCE+Si DSM"
ENV_NAMES["bce_silhouettes_ortho"]="BCE+Si Ortho"
ENV_NAMES["bce_silhouettes"]="BCE+Si DSM+Ortho"
ENV_NAMES["mixed_dsm"]="Mixed DSM"
ENV_NAMES["mixed_new_ortho"]="Mixed Ortho"
ENV_NAMES["mixed"]="Mixed DSM+Ortho"

ENV_KEYS=(
  "shadow_dsm" "shadow_ortho" "shadow"
  "silhouettes_dsm" "silhouettes_ortho" "silhouettes"
  "shadow_silhouettes_dsm" "shadow_silhouettes_ortho" "shadow_silhouettes"
  "bce_dsm" "bce_ortho" "bce"
  "bce_shadow_dsm" "bce_shadow_ortho" "bce_shadow"
  "bce_silhouettes_dsm" "bce_silhouettes_ortho" "bce_silhouettes"
  "mixed_dsm" "mixed_ortho" "mixed"
)

for ENV_KEY in "${ENV_KEYS[@]}"; do
  DISPLAY_NAME="${ENV_NAMES[$ENV_KEY]}"
  if [ -d "log/$ENV_KEY" ]; then
    if [[ "$ENV_KEY" == *_dsm ]]; then
      MODEL=4
    elif [[ "$ENV_KEY" == *_ortho ]]; then
      MODEL=5
    else
      MODEL=3
    fi

    echo "→ $DISPLAY_NAME (env=$ENV_KEY, model=$MODEL)"
    OUTPUT=$(python validation/validation_pipeline.py \
      --env "$ENV_KEY" \
      --num_query_points $NUM_QUERY_POINTS \
      --top_k $TOP_K \
      --num_points $NUM_POINTS \
      --model $MODEL \
      --variable $VARIABLE \
      --num_trees_total $NUM_TREES_TOTAL \
      --deciduous $DECIDUOUS)

    CD=$(echo "$OUTPUT" | grep "Chamfer Distance" | grep -oE '[0-9]+\.[0-9]+')
    NCD=$(echo "$OUTPUT" | grep "Normalized Chamfer Distance" | grep -oE '[0-9]+\.[0-9]+')
    F1=$(echo "$OUTPUT" | grep "F1 Score" | grep -oE '[0-9]+\.[0-9]+')
    COV=$(echo "$OUTPUT" | grep "Variance Score" | grep -oE '[0-9]+\.[0-9]+')
    
    # printf "%-25s  CD=%.4f  NCD=%.4f  F1=%.4f  COV=%.0f%%\n" "$DISPLAY_NAME:" "$CD" "$NCD" "$F1" "$COV"
    printf "%-25s & %.4f & %.4f & %.4f & %.0f%% \n" "$DISPLAY_NAME:" "$CD" "$NCD" "$F1" "$COV"
  else
    echo "⚠ Skipping $ENV_KEY (log not found)"
  fi
done