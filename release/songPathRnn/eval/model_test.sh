#***********************************************************
#Copyright 2018 eBay Inc.
 
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
#!/usr/bin/env bash
model_path=$1
gpu_id=$2
result_path=$3
top_k=$4

echo "test"
bash model_test_one_list.sh $1 $result_path'/test.res' $2 'test.list' $top_k

# for i in {1..13}
# do
#     echo $i
#     bash model_test_one_list.sh $1 $result_path'/test_'$i'.res' $2 'test_'$i'.list'
# done