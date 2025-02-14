#!/bin/sh

PYTHON_PATH='python'  # Change this to the path of your Python executable.

# The following settings are for a server with 8 NVIDIA RTX 4090 GPUs. Adjust them as needed for your hardware.

nohup python -u tune_hyper_params.py \
  --python_path $PYTHON_PATH \
  --model_name llama3.2-1b \
  --client_dataset_name snli \
  --data_hes dir1.0 \
  --max_experts 8 \
  --expert_nums 15 \
  --gpus 0,1,2,3,0,1,2,3 \
  > tune_lr_expert_choices.log 2>&1 &

nohup python -u tune_hyper_params.py \
  --python_path $PYTHON_PATH \
  --model_name llama3.2-1b \
  --client_dataset_name snli \
  --data_hes dir1.0 \
  --lrs 5e-5 \
  --expert_choices 2 \
  --gpus 4,5,6,7,4,5,6,7 \
  > tune_expert_num_max_experts.log 2>&1 &