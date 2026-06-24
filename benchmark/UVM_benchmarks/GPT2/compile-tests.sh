#!/bin/bash

HOME=/hy-tmp/ make train_gpt2_fp32-uvm  USE_CUDNN=1
HOME=/hy-tmp/ make train_gpt2fp32cu  USE_CUDNN=1