#!/bin/bash

echo "🚀 Starting Round 2 Training..."
echo "Config: 4x GPU, Duration Head, Text Weighting, Speaker Encoder Training"

# Ensure we are using the FIXED script
# Added --train_speaker_encoder to enable the voice cloning adaptation
accelerate launch --num_processes=4 --mixed_precision=bf16 sft_12hz_complete_FIXED.py \
  --init_model_path "simone00/it2" \
  --train_jsonl "train_with_codes.jsonl" \
  --output_model_path "output_round2_complete" \
  --speaker_name "italian_multi" \
  --batch_size 10 \
  --gradient_accumulation_steps 3 \
  --lr 3e-6 \
  --weight_decay 0.015 \
  --max_grad_norm 0.5 \
  --warmup_steps 200 \
  --duration_loss_weight 0.12 \
  --duration_hidden_size 2048 \
  --use_text_weighting \
  --weight_power 0.5 \
  --max_sample_weight 3.5 \
  --min_sample_weight 1.0 \
  --num_epochs 6 \
  --val_split 0.02 \
  --save_steps 1000 \
  --eval_steps 500 \
  --early_stopping \
  --early_stopping_patience 3 \
  --early_stopping_min_delta 0.001 \
  --train_speaker_encoder

echo "=================================================="
echo "Training complete! Check output_round2_complete/"
echo "Best checkpoint: output_round2_complete/checkpoint-best/"

echo "=================================================="
