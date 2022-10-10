#!/usr/bin/env python

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA_DIR=YOUR/PATH
# plain text data
TRAIN_F=${DATA_DIR}/cc100/
VAL_F=${DATA_DIR}/val/cc100.eval.txt

# code switched data
K_TRAIN_F=${DATA_DIR}/kg/
K_VAL_F=${DATA_DIR}/val/kg.eval.txt

# reasoning based data
R_TRAIN_F=${DATA_DIR}/kg.cycle/
R_VAL_F=${DATA_DIR}/val/r.eval.txt

# path to official checkpoint
PRETRAIN_MODEL=PATH/TO/model/xlmr_base/
OUTPUT=PATH/TO/output/xlmr_base/

accelerate launch --config_file acc_conf.yaml run_k_mlm_no_trainer.py \
    --train_file $TRAIN_F \
    --validation_file $VAL_F \
    --k_train_file $K_TRAIN_F \
    --k_validation_file $K_VAL_F \
    --reason_train_file $R_TRAIN_F \
    --reason_validation_file $R_VAL_F \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --per_device_k_train_batch_size 4 \
    --per_device_k_eval_batch_size 4 \
    --per_device_reason_train_batch_size 4 \
    --per_device_reason_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 300 \
    --mlm_probability 0.15 \
    --model_name_or_path $PRETRAIN_MODEL \
    --output_dir $OUTPUT \
    --max_seq_length 128 \
    --num_warmup_steps 100 \
    --save_model_step 1500 \
    --line_by_line true \

