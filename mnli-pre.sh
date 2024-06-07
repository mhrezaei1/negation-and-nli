#!/bin/bash

# Define your inputs here. For example:
inputs=(
    "roberta-base roberta-base-dual-1000000-1e-06-128"
    "roberta-base roberta-base-pp-500000-1e-06-128"
    "roberta-base roberta-base-nsp-1000000-1e-06-32"
    "roberta-large roberta-large-pp-500000-1e-06-128"
    "roberta-large roberta-large-nsp-1000000-1e-06-32-pp-1000000-1e-06-128"
    "roberta-large roberta-large-dual-500000-1e-06-128"
)

# Maximum number of concurrent jobs, equals to the number of GPUs
MAX_JOBS=4

# Directory to store the downloaded models
MODEL_DIR=./downloaded_models

# Function to check and wait for available job slot
check_jobs() {
  while [ $(jobs -p | wc -l) -ge $MAX_JOBS ]; do
    sleep 2
  done
}

# Function to download model from Hugging Face if not already downloaded
download_model() {
  local model_name=$1
  local model_path="$MODEL_DIR/$model_name"
  if [ ! -d "$model_path" ]; then
    echo "Downloading model: $model_name"
    mkdir -p "$model_path"
    python -c "from transformers import AutoModel; AutoModel.from_pretrained('$model_name').save_pretrained('$model_path')"
  else
    echo "Model $model_name already downloaded."
  fi
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

    # Download the model locally
    download_model "mhr2004/$roberta_path"

    echo "Starting job $i on GPU $gpu_index"

    # Set environment variables
    export RoBERTa_Type=$roberta_type
    export RoBERTa_Path=$roberta_path

    # Create directories that contain predicted labels (as well as actual labels)
    mkdir -p outputs/predictions/MNLI/RoBERTa/$RoBERTa_Path/original_dev

    # Create directories that contain fine-tuned models
    mkdir -p outputs/models/MNLI/RoBERTa/$RoBERTa_Path

    export GLUE_DIR=./data/GLUE/
    export TASK_NAME=MNLI

    # Using RoBERTa 
    # Train on the original training split and evaluate on the original dev split:
    export PRED_DIR=./outputs/predictions/MNLI/RoBERTa/$RoBERTa_Path/original_dev

    # Execute the training script with CUDA_VISIBLE_DEVICES set for the specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_index python ./transformers/examples/run_glue.py --model_type roberta --model_name_or_path $MODEL_DIR/mhr2004/$roberta_path --task_name $TASK_NAME --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 32 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./outputs/models/$TASK_NAME/RoBERTa/$RoBERTa_Path &

done

# Wait for all background jobs to finish
wait

echo "All processes completed"

curl -d "MNLI trainings finished" ntfy.sh/mhrnlpmodels
