#!/bin/sh

export RoBERTa_Type = "roberta-base"
export RoBERTa_Path = "roberta-base"

# create directories that contain predicted labels (as well as actual labels)
mkdir outputs/predictions/MNLI
mkdir outputs/predictions/MNLI/RoBERTa/$RoBERTa_Path
mkdir outputs/predictions/MNLI/RoBERTa/$RoBERTa_Path/original_dev

# create directories that contain fine-tuned models
mkdir outputs/models/MNLI
mkdir outputs/models/MNLI/RoBERTa/$RoBERTa_Path

export GLUE_DIR=./data/GLUE/
export TASK_NAME=MNLI

# Using RoBERTa 
#Train on the original training split and evaluate on the original dev split:  
# export PRED_DIR=./outputs/predictions/MNLI/RoBERTa/original_dev/
export PRED_DIR=./outputs/predictions/MNLI/RoBERTa/$RoBERTa_Path/original_dev

# python ./transformers/examples/run_glue.py --model_type roberta --model_name_or_path roberta-base --task_name $TASK_NAME --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 32 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./outputs/models/$TASK_NAME/RoBERTa/
python ./transformers/examples/run_glue.py --model_type roberta --model_name_or_path $RoBERTa_Path --task_name $TASK_NAME --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 32 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./outputs/models/$TASK_NAME/RoBERTa/$RoBERTa_Path