#!/bin/bash

# Define your inputs here. For example:
inputs=(
  "bert bert-large-uncased"
  "bert bert-large-uncased-nsp-1000000-1e-06-64"
  "bert bert-large-uncased-pp-1000000-1e-06-32"
  "bert bert-large-uncased-dual-1000000-1e-06-32"
  "bert bert-base-uncased"
  "bert bert-base-uncased-pp-1000000-1e-06-32"
  "bert bert-base-uncased-nsp-1000000-1e-06-32"
  "bert bert-base-uncased-dual-1000000-1e-06-32"
)

# Maximum number of concurrent jobs, equals to the number of GPUs
MAX_JOBS=8

# Directory to store the downloaded models
MODEL_DIR=./downloaded_models

# remove old dev files 
rm -rf ./data/GLUE/SNLI/cached_dev_*
rm -rf ./data/GLUE/SNLI/dev*

# copy the new benchmark and save to appropriate location 
cp ./data/new_benchmarks/processed_for_run/SNLI/dev.tsv ./data/GLUE/SNLI/dev.tsv

mkdir outputs/predictions/SNLI/RoBERTa/new_dev

# export task directory and name
export GLUE_DIR=./data/GLUE/
export TASK_NAME=SNLI


# Function to check and wait for available job slot
check_jobs() {
  while [ $(jobs -p | wc -l) -ge $MAX_JOBS ]; do
    sleep 2
  done
}


# Loop through inputs and execute them
for i in "${!inputs[@]}"; do
    # Check if we need to wait for a job slot to become available
    check_jobs
    sleep 60
    # Calculate GPU index: i % MAX_JOBS ensures cycling through GPUs 0 to MAX_JOBS-1
    gpu_index=$((i % 4))

    # Split input into variables
    roberta_type=$(echo ${inputs[$i]} | awk '{print $1}')
    roberta_path=$(echo ${inputs[$i]} | awk '{print $2}')


    echo "Starting job $i on GPU $gpu_index"

    # Set environment variables
    export RoBERTa_Type=$roberta_type
    export RoBERTa_Path=$roberta_path

    # Create directories that contain predicted labels (as well as actual labels)
    mkdir -p outputs/predictions/SNLI/RoBERTa/$RoBERTa_Path/new_dev

    export GLUE_DIR=./data/GLUE/
    export TASK_NAME=SNLI

    # Using RoBERTa 
    # Train on the original training split and evaluate on the original dev split:
    export PRED_DIR=./outputs/predictions/SNLI/RoBERTa/$RoBERTa_Path/new_dev/

    # Execute the training script with CUDA_VISIBLE_DEVICES set for the specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_index python ./transformers/examples/run_glue.py --model_type bert --model_name_or_path $MODEL_DIR/mhr2004/$roberta_path --task_name $TASK_NAME --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 32 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./outputs/models/$TASK_NAME/RoBERTa/$RoBERTa_Path &

done

# Wait for all background jobs to finish
wait

echo "All processes completed"

curl -d "SNLI evaluations finished" ntfy.sh/mhrnlpmodels
