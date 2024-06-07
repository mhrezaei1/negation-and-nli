#!/bin/bash

# Define your inputs here. For example:
inputs=(
    "roberta-base mhr2004/roberta-base-dual-1000000-1e-06-128"
)

# Maximum number of concurrent jobs, equals to the number of GPUs
MAX_JOBS=4

# Function to check and wait for available job slot
check_jobs() {
  while [ $(jobs -p | wc -l) -ge $MAX_JOBS ]; do
    sleep 2
  done
}

# Function to download model from Hugging Face
download_model() {
  model_path=$1
  output_dir=$2

  if [ ! -d "$output_dir" ]; then
    mkdir -p $output_dir
  fi

  python -c "
import os
from transformers import AutoModel, AutoTokenizer

model_path = '$model_path'
output_dir = '$output_dir'

model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
"
}

# Loop through inputs and execute them
for i in "${!inputs[@]}"; do
    # Check if we need to wait for a job slot to become available
    check_jobs
    
    # Calculate GPU index: i % MAX_JOBS ensures cycling through GPUs 0 to MAX_JOBS-1
    gpu_index=$((i % MAX_JOBS))

    # Split input into variables
    roberta_type=$(echo ${inputs[$i]} | awk '{print $1}')
    roberta_path=$(echo ${inputs[$i]} | awk '{print $2}')

    echo "Starting job $i on GPU $gpu_index"
    echo "CUDA_VISIBLE_DEVICES=$gpu_index python3 nlutrainer.py --model_type $roberta_type --model_path $roberta_path"

    # Set environment variables
    export RoBERTa_Type=$roberta_type
    export RoBERTa_Path=$roberta_path

    # Create directories that contain predicted labels (as well as actual labels)
    pred_dir=outputs/predictions/MNLI/RoBERTa/$RoBERTa_Path/original_dev
    mkdir -p $pred_dir

    # Create directories that contain fine-tuned models
    model_dir=outputs/models/MNLI/RoBERTa/$RoBERTa_Path
    mkdir -p $model_dir

    # Download model from Hugging Face
    download_model $roberta_path $model_dir

    # Verify the directories are created and populated
    if [ ! -d "$pred_dir" ]; then
        echo "Error: Prediction directory $pred_dir not created."
        exit 1
    fi

    if [ ! "$(ls -A $model_dir)" ]; then
        echo "Error: Model files were not downloaded to $model_dir."
        exit 1
    fi

    export GLUE_DIR=./data/GLUE/
    export TASK_NAME=MNLI

    # Using RoBERTa 
    # Train on the original training split and evaluate on the original dev split:
    export PRED_DIR=$pred_dir

    # Execute the training script with CUDA_VISIBLE_DEVICES set for the specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_index python ./transformers/examples/run_glue.py --model_type roberta --model_name_or_path $model_dir --task_name $TASK_NAME --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 32 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir $model_dir &

done

# Wait for all background jobs to finish
wait

echo "All processes completed"

curl -d "condaqa trainings finished" ntfy.sh/mhrnlpmodels 
