#!/usr/bin/env bash
for i in {0..10}
do
    test_file_name="tuples/test_samples/"user_song_test_part_$i".txt"
    test_res_name="test_result/test_result_"$i".res"
    echo "test file:"$test_file_name
    python3 movie_run_exp.py config/song.config -reg 0.5 -test_file_path $test_file_name \
    -test_res_save_path $test_res_name
done

echo "combine result"
total_result_name="test_result/total.res"
part_res_name=""
for i in {0..10}
do
    part_res_name=$part_res_name"test_result/test_result_"$i".res "
done
cat $part_res_name > $total_result_name

python3 res_combine.py test_result/total.res data/song/tuples/user_song_test_fmg.txt test_result/total_combine.txt

python3 eval.py test_result/total_combine.txt test_result/test_score.txt