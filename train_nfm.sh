#!/usr/bin/env bash
lr_arrays=("0.001" "0.01" "0.1" "0.2")
for lr in ${lr_arrays[@]}
do
    echo $idx
    test_file_name="../Data/baseModel_test.txt"
    CUDA_VISIBLE_DEVICES=$1 python MF.py --test 0 --lamda $2 --lr $lr --test_file_path $test_file_name
    break
done