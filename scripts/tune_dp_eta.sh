#!/bin/sh

PYTHON_PATH='python'  # Change this to the path of your Python executable.

# The following settings are for a server with 8 NVIDIA RTX 4090 GPUs. Adjust them as needed for your hardware.

nohup python -u tune_dp_eta.py \
  --python_path $PYTHON_PATH \
  --dp_etas 1,50,100 \
  --model_name llama3.2-1b \
  --client_dataset_name snli \
  --seeds 42,62,82 \
  --gpus 0,0,1,1 \
  > tune_dp_eta_snli.log 2>&1 &

nohup python -u tune_dp_eta.py \
  --python_path $PYTHON_PATH \
  --dp_etas 1,50,100 \
  --model_name llama3.2-1b \
  --client_dataset_name dolly-15k \
  --seeds 42,62,82 \
  --gpus 2,2,3,3,4,4 \
  > tune_dp_eta_dolly.log 2>&1 &

nohup python -u tune_dp_eta.py \
  --python_path $PYTHON_PATH \
  --dp_etas 1,50,100 \
  --model_name llama3.2-1b \
  --client_dataset_name natural-instruct \
  --seeds 42,62,82 \
  --gpus 5,5,6,6,7,7 \
  > tune_dp_eta_ni.log 2>&1 &