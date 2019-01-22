#***********************************************************
#Copyright 2018 eBay Inc.
 
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
#!/bin/bash

data_dir='../data/output'
predicate_name='rate'
mean_model=3
model_path=$1
out_file=$2
gpu_id=$3
test_list_file=$4
top_k=$5

#this will output score file
th test_from_checkpoint.lua -input_dir $data_dir  -out_file $out_file -predicate_name $predicate_name -meanModel \
    $mean_model -model_path $model_path -test_list $test_list_file -gpu_id $gpu_id -top_k $5