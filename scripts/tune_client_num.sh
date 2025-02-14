#!/bin/sh

PYTHON_PATH='python'  # Change this to the path of your Python executable.

# The following settings are for a server with 8 NVIDIA RTX 4090 GPUs. Adjust them as needed for your hardware.

nohup python -u tune_client_num.py \
  --python_path $PYTHON_PATH \
  --algorithms fed_avg,fdlora \
  --client_nums 20,30,40,50 \
  --model_name llama3.2-1b \
  --client_dataset_name snli \
  --gpus 0,0,1,1 \
  > tune_client_num_snli_baselines.log 2>&1 &

nohup python -u tune_client_num.py \
  --python_path $PYTHON_PATH \
  --algorithms fed_moe \
  --client_nums 20,30,40,50 \
  --model_name llama3.2-1b \
  --client_dataset_name snli \
  --gpus 2,2,3,3 \
  > tune_client_num_snli_moe.log 2>&1 &

nohup python -u tune_client_num.py \
  --python_path $PYTHON_PATH \
  --algorithms fed_avg,fdlora \
  --client_nums 20,30,40,50 \
  --model_name llama3.2-1b \
  --client_dataset_name natural-insturct \
  --gpus 4,5 \
  > tune_client_num_ni_baselines.log 2>&1 &

nohup python -u tune_client_num.py \
  --python_path $PYTHON_PATH \
  --algorithms fed_moe \
  --client_nums 20,30,40,50 \
  --model_name llama3.2-1b \
  --client_dataset_name natural-instruct \
  --gpus 6,7 \
  > tune_client_num_ni_moe.log 2>&1 &