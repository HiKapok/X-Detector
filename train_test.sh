#!/bin/bash
export CUDA_VISIBLE_DEVICES=''
cp light_head_rfcn_train.py .fuck.py
source /home/kapok/pyenv35/bin/activate && python .fuck.py --run_on_cloud=False --data_format=channels_last --batch_size=1 --log_every_n_steps=1
