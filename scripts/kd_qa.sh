#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

# em: 82.8, f1: 85.7
python question-answering/ted_qa_no_trainer.py \
    --model_name_or_path microsoft/deberta-v3-xsmall --teacher_model_name_or_path cliang1453/deberta-v3-base-squadv2 \
    --model_type ted-deberta-v2 \
    --dataset_name squad_v2 --version_2_with_negative \
    --per_device_train_batch_size 8 \
    --max_seq_length 384 --doc_stride 128 \
    --learning_rate 5e-5 --num_train_epochs 3 \
    --kl_alpha 10 --mse_alpha 0 --filter_disabled \
    --output_dir /home/ubuntu/ted_output/squadv2/kd/ --seed 42 --mixed_precision fp16 --save_best