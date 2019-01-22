#***********************************************************
#Copyright 2018 eBay Inc.
 
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
#!/usr/bin/env bash
mode_path=$1
result_path=$2
top_k=$3

bash model_test.sh $mode_path "0" $result_path $top_k
bash combine_result.sh $result_path
python resort.py $result_path"/test_combine.txt" $result_path"/test_combine_sorted.txt"
bash eval_score.sh $result_path"/test_combine_sorted.txt" $result_path
