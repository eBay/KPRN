#!/usr/bin/env bash
source path_config.sh

#echo "find depth 3 paths..."
python path_find_depth_3.py $movie_person_dict_path $person_movie_dict_path $movie_type_dict_path $type_movie_dict_path \
       $movie_user_dict_path $user_movie_dict_path $depth_3_person_sample_nums $depth_3_type_sample_nums $depth_3_user_sample_nums \
       $user_rate_movie_path $depth_3_output_file_path

#echo "find depth 5 paths..."
python path_find_depth_5.py $movie_person_dict_path $person_movie_dict_path $movie_type_dict_path $type_movie_dict_path \
       $movie_user_dict_path $user_movie_dict_path $depth_5_person_sample_nums $depth_5_type_sample_nums $depth_5_user_sample_nums \
       $user_rate_movie_path $depth_5_output_file_path

#echo "depth 3 clustering..."
python3 clustering.py $depth_3_output_file_path $depth_3_clustering_path

#echo "depth 5 clustering..."
python3 clustering.py $depth_5_output_file_path $depth_5_clustering_path

#echo "combine depth 3 and 5..."
python3 combine.py $user_id_file_path $depth_3_clustering_path $depth_5_clustering_path $depth_3_5_combine_output_path


#echo "add relationships and labels..."
python3 add_relation_label.py $user_rate_movie_path $depth_3_5_combine_output_path $pos_path_out_path $neg_path_out_path

#echo "split train and test..."
python3 split_train_test.py $pos_split_num $neg_split_num $pos_path_out_path $neg_path_out_path $train_pos_file_path $train_neg_file_path $test_file_path

echo "get positive and negative tuples in test data..."
python3 get_test_data_tuples.py $user_id_file_path $test_file_path $test_pos_tuple_path $test_neg_tuple_path

echo "generate items frequency dict..."
python3 generate_item_frequency_dict.py $user_rate_movie_path $movie_id_path $movie_fq_path


echo "format data for fmg..."
python3 format_fmg_data.py $pos_path_out_path $neg_path_out_path $fmg_pos_file_path $fmg_neg_file_path $train_pos_file_path \
        $train_neg_file_path $fmg_train_file_path

echo "sample test..."
alpha_array=("0.0")
for alpha in ${alpha_array[@]}
do
  echo "sample on "alpha
  path_rnn_sample_out_file_path="./data/output/path_rnn_test_samples_"$alpha".txt"
  fmg_sample_out_file_path="./data/output/fmg_data/fmg_test_samples_"$alpha".txt"
  python3 sample.py $alpha $user_id_file_path $movie_fq_path $test_pos_tuple_path $test_neg_tuple_path \
                    $path_rnn_sample_out_file_path $fmg_sample_out_file_path
done
