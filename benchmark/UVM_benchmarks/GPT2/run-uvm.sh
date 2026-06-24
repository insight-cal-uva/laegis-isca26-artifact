#!/bin/bash

./train_gpt2_fp32-uvm \
    -i "dev/data/fineweb10B/fineweb_train_*.bin" \
    -j "dev/data/fineweb10B/fineweb_val_*.bin" \
    -o log-rtx-4090-fp32-uvm \
    -b 2 \
    -t 1024 \
    -l 0.0003 \
    -v 50 \
    -m 25 \
    -s 100 \
    -g 128