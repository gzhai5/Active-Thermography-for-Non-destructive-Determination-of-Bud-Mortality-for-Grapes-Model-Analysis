#!/bin/bash


for data in RIES CF CON PN ALL
do
    for seed in 947 1375 6031 1480 8815 702 3977 4060 8105 8473 9844 1468 2886 9821 3895 4448 6084 349 5861 3948 594 371 5333 4975 7244 1934 3985 4633 9840 8226
    do
        python linear.py --data $data --seed $seed
        python random_forest.py --data $data --seed $seed
        python svm.py --data $data --seed $seed
    done
done