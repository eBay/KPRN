#!/usr/bin/env bash

# positive的(user, movie)
user_rate_movie_path="./data/kg_data/user_song_tuple.txt"
# user id的输入路径
user_id_file_path="./data/kg_data/user_id.txt"
# movie id的路径
movie_id_path="./data/kg_data/song_id.txt"

# 三种实体的dict路径
movie_person_dict_path="./data/kg_data/song_person.dict"
movie_type_dict_path="./data/kg_data/song_type.dict"
movie_user_dict_path="./data/kg_data/song_user.dict"
person_movie_dict_path="./data/kg_data/person_song.dict"
type_movie_dict_path="./data/kg_data/type_song.dict"
user_movie_dict_path="./data/kg_data/user_song.dict"

# 深度为3的每种实体的随机采样个数，即输出路径
depth_3_person_sample_nums=2
depth_3_type_sample_nums=2
depth_3_user_sample_nums=2
depth_3_output_file_path="./data/path/path_depth3.txt"

# 深度为5的每种实体的随机采样个数，即输出路径
depth_5_person_sample_nums=1
depth_5_type_sample_nums=1
depth_5_user_sample_nums=1
depth_5_output_file_path="./data/path/path_depth5.txt"

# 深度为3，5的路径的聚类输出路径
depth_3_clustering_path="./data/output/path_depth_3_cluster.txt"
depth_5_clustering_path="./data/output/path_depth_5_cluster.txt"


# 所有路径格式化后的输出路径
depth_3_5_combine_output_path="./data/output/path_combine.txt"

# positive 和negative的path的输出文件
pos_path_out_path="./data/output/pos_path.txt"
neg_path_out_path="./data/output/neg_path.txt"

# positive和negative的划分训练测试集时，训练集的比例
pos_split_num=0.8
neg_split_num=0.2

# path rnn的三个输出文件
train_pos_file_path="./data/output/positive_matrix.tsv.translated" #pos训练集
train_neg_file_path="./data/output/negative_matrix.tsv.translated" #neg训练集
test_file_path="./data/output/test_matrix.tsv.translated" #测试集

# 测试集中的pos和neg的tuples的输出路径
test_pos_tuple_path="./data/output/test_pos_tuples.txt"
test_neg_tuple_path="./data/output/test_neg_tuples.txt"

# movie frequency的输出路径
movie_fq_path="./data/output/song_fq.txt"

# fmg data path
fmg_pos_file_path="./data/output/fmg_data/user_pos_song.txt"
fmg_neg_file_path="./data/output/fmg_data/user_neg_song.txt"
fmg_train_file_path="./data/output/fmg_data/user_song_train.txt"
