#!/bin/bash

export TF_ENABLE_ONEDNN_OPTS=0


for batch_size in 4 8
do
    for lr in 0.005 0.001 0.0005 0.00005 0.000005
    do
        for data in RIES CF CON PN ALL
        do
            echo "Running with lr=$lr and data=$data and batch_size=$batch_size"
            python vivit-run.py --lr $lr --data $data --batch_size $batch_size
        done
    done
done