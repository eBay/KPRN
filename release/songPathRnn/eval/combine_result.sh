#***********************************************************
#Copyright 2018 eBay Inc.
 
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
#!/usr/bin/env bash
result_path=$1
echo "test"
python combine_result.py "test.list.entity" $result_path'/test.res' $result_path'/test'

# for i in {1..13}
# do
#     echo $i
#     python combine_result.py "test_"$i".list.entity" $result_path'/test_'$i'.res' $result_path'/test_'$i
# done