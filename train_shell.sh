#!/bin/bash -l


export SUBMIT_DIR=/home/nghiemb/RMC_repos/MoCo_cGAN_Hewlett_NMR2024

python ${SUBMIT_DIR}/train.py > ${SUBMIT_DIR}/train_log_2025-03-21.txt & disown
