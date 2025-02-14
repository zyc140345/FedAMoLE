#!/bin/sh

PYTHON_PATH='python'  # Change this to the path of your Python executable.

# The following settings are for a server with 8 NVIDIA RTX 4090 GPUs. Adjust them as needed for your hardware.

nohup python -u run.py \
  --python_path $PYTHON_PATH \
  --algorithms fed_avg,fed_prompt,fed_ptuning \
  --do_ft true \
  --model_name llama3.2-1b \
  --precision fp16 \
  --lr 5e-5 \
  --client_dataset_name snli \
  --client_step 200 \
  --data_hes dir0.1,dir1.0,dir100.0 \
  --rounds 30 \
  --seeds 42,62,82 \
  --gpus 0,0,1 \
  > train_snli_1.log 2>&1 &

nohup python -u run.py \
  --python_path $PYTHON_PATH \
  --algorithms fdlora \
  --model_name llama3.2-1b \
  --precision fp16 \
  --lr 5e-5 \
  --client_dataset_name snli \
  --client_step 200 \
  --data_hes dir0.1,dir1.0,dir100.0 \
  --rounds 30 \
  --seeds 42,62,82 \
  --gpus 1,2 \
  > train_snli_2.log 2>&1 &

nohup python -u run.py \
  --python_path $PYTHON_PATH \
  --algorithms fed_avg,fed_prompt,fed_ptuning \
  --do_ft true \
  --model_name llama3.2-1b \
  --precision fp16 \
  --lr 5e-5 \
  --client_dataset_name dolly-15k \
  --client_step 200 \
  --data_hes dir0.1,dir1.0,dir100.0 \
  --rounds 30 \
  --seeds 42,62,82 \
  --gpus 2,3,3 \
  > train_dolly_1.log 2>&1 &

nohup python -u run.py \
  --python_path $PYTHON_PATH \
  --algorithms fdlora \
  --model_name llama3.2-1b \
  --precision fp16 \
  --lr 5e-5 \
  --client_dataset_name dolly-15k \
  --client_step 200 \
  --data_hes dir0.1,dir1.0,dir100.0 \
  --rounds 30 \
  --seeds 42,62,82 \
  --gpus 4,4 \
  > train_dolly_2.log 2>&1 &

nohup python -u run.py \
  --python_path $PYTHON_PATH \
  --algorithms fed_avg,fed_prompt,fed_ptuning \
  --model_name llama3.2-1b \
  --precision fp16 \
  --lr 5e-5 \
  --client_dataset_name natural-instruct \
  --client_step 200 \
  --data_hes meta1 \
  --rounds 30 \
  --seeds 42,62,82 \
  --gpus 5,6 \
  > train_ni_1.log 2>&1 &

nohup python -u run.py \
  --python_path $PYTHON_PATH \
  --algorithms fed_avg,fed_prompt,fed_ptuning \
  --model_name llama3.2-1b \
  --precision fp16 \
  --lr 5e-5 \
  --client_dataset_name natural-instruct \
  --client_step 200 \
  --data_hes meta1 \
  --rounds 30 \
  --seeds 42,62,82 \
  --gpus 7 \
  > train_ni_2.log 2>&1 &