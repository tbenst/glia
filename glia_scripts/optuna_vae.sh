#!/usr/bin/env bash
. activate py38
gpus=( 2 3 4 5 6 7 8 9 )

run_optuna() {
    while true; do
        python glia/reconstructions/train.py "/mnt/fs1/tbenst/200623_faces/R1_E3_AMES_200min_200f_14l_rgb.h5" "/mnt/fs1/tbenst/models/3brain/" "$1"
        sleep 1
        echo "======== Restarting GPU $1 ========"
    done
}
export -f run_optuna
parallel -u run_optuna ::: "${gpus[@]}"