#!/bin/bash

# Define your inputs here. For example:
inputs=(
    "roberta-large roberta-large-nsp-1000000-1e-06-32"
    "roberta-large roberta-large-pp-500000-1e-06-128"
    "roberta-large roberta-large-dual-500000-1e-06-128"

    "roberta-base roberta-base-pp-1000000-1e-06-128"
    "roberta-base roberta-base-nsp-1000000-1e-06-32"
    "roberta-base roberta-base-dual-1000000-1e-06-128"

    "roberta-base roberta-base"
    "roberta-large roberta-large"
)

# Maximum number of concurrent jobs, equals to the number of GPUs
MAX_JOBS=4

# Directory to store the downloaded models
MODEL_DIR=./downloaded_models

# remove old dev files 
rm -rf ./data/GLUE/MNLI/cached_dev_*
rm -rf ./data/GLUE/MNLI/dev*

# copy the new benchmark and save to appropriate location 
cp ./data/new_benchmarks/processed_for_run/MNLI/dev_matched.tsv ./data/GLUE/MNLI/dev_matched.tsv
# We haven't worked with mismatched genres. Just just made a duplicate copy of dev_matched.tsv file and named as dev_mismatched.tsv for running the systems successfully.
cp ./data/new_benchmarks/processed_for_run/MNLI/dev_mismatched.tsv ./data/GLUE/MNLI/dev_mismatched.tsv

mkdir outputs/predictions/MNLI/RoBERTa/new_dev
# export task directory and name
export GLUE_DIR=./data/GLUE/
export TASK_NAME=MNLI


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
    
    # Calculate GPU index: i % MAX_JOBS ensures cycling through GPUs 0 to MAX_JOBS-1
    gpu_index=$((i % MAX_JOBS))

    # Split input into variables
    roberta_type=$(echo ${inputs[$i]} | awk '{print $1}')
    roberta_path=$(echo ${inputs[$i]} | awk '{print $2}')


    echo "Starting job $i on GPU $gpu_index"

    # Set environment variables
    export RoBERTa_Type=$roberta_type
    export RoBERTa_Path=$roberta_path

    # Create directories that contain predicted labels (as well as actual labels)
    mkdir -p outputs/predictions/MNLI/RoBERTa/$RoBERTa_Path/new_dev

    export GLUE_DIR=./data/GLUE/
    export TASK_NAME=MNLI

    # Using RoBERTa 
    # Train on the original training split and evaluate on the original dev split:
    export PRED_DIR=./outputs/predictions/MNLI/RoBERTa/$RoBERTa_Path/new_dev/

    # Execute the training script with CUDA_VISIBLE_DEVICES set for the specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_index python ./transformers/examples/run_glue.py --model_type roberta --model_name_or_path ./outputs/models/MNLI/RoBERTa/$roberta_path --task_name $TASK_NAME --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 32 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./outputs/models/$TASK_NAME/RoBERTa/$RoBERTa_Path &

done

# Wait for all background jobs to finish
wait

echo "All processes completed"

curl -d "MNLI evaluations finished" ntfy.sh/mhrnlpmodels
