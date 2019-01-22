#***********************************************************
#Copyright 2018 eBay Inc.
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
# -*- coding:utf-8 -*-
import codecs
import pickle
import os
import time
import random
import sys



movie_person_dict = pickle.load(open(sys.argv[1], mode="rb"))
person_movie_dict = pickle.load(open(sys.argv[2], mode="rb"))
movie_type_dict = pickle.load(open(sys.argv[3], mode="rb"))
type_movie_dict = pickle.load(open(sys.argv[4], mode="rb"))
movie_user_dict = pickle.load(open(sys.argv[5], mode="rb"))
user_movie_dict = pickle.load(open(sys.argv[6], mode="rb"))


def get_random_index(nums, max_length):
    index_list = list(range(max_length))
    random.shuffle(index_list)
    return index_list[:nums]


person_sample_nums = int(sys.argv[7])
type_sample_nums = int(sys.argv[8])
user_sample_nums = int(sys.argv[9])


path_reader = codecs.open(sys.argv[10], mode="r", encoding="utf-8")
path_writer = codecs.open(sys.argv[11], mode="w", encoding="utf-8")


line = path_reader.readline()
nums = 0
start_time = time.time()

while line:
    line_list = line.strip().split()

    path_set = set(line_list)
    movie_node = line_list[-1]

    person_node_list = movie_person_dict.get(movie_node)
    if person_node_list and person_sample_nums>0:
        list_length = len(person_node_list)
        if list_length <= person_sample_nums:
            sample_index_list = list(range(list_length))
        else:
            sample_index_list = get_random_index(person_sample_nums, list_length)
        for index in sample_index_list:
            mid_node = person_node_list[index]
            if mid_node not in path_set:
                end_movie_node_list = person_movie_dict[mid_node]
                end_node_index = random.randint(0, len(end_movie_node_list) - 1)
                end_movie_node = end_movie_node_list[end_node_index]
                path_writer.write("\t".join(line_list) + "\t" + mid_node + "\t" + end_movie_node + "\n")


    type_node_list = movie_type_dict.get(movie_node)
    if type_node_list and type_sample_nums>0:
        list_length = len(type_node_list)
        if list_length <= type_sample_nums:
            sample_index_list = list(range(list_length))
        else:
            sample_index_list = get_random_index(type_sample_nums, list_length)
        for index in sample_index_list:
            mid_node = type_node_list[index]
            if mid_node not in path_set:
                end_movie_node_list = type_movie_dict[mid_node]
                end_node_index = random.randint(0, len(end_movie_node_list) - 1)
                end_movie_node = end_movie_node_list[end_node_index]
                path_writer.write("\t".join(line_list) + "\t" + mid_node + "\t" + end_movie_node + "\n")

    user_node_list = movie_user_dict.get(movie_node)
    if user_node_list and user_sample_nums>0:
        list_length = len(user_node_list)
        if list_length <= user_sample_nums:
            sample_index_list = list(range(list_length))
        else:
            sample_index_list = get_random_index(user_sample_nums, list_length)
        for index in sample_index_list:
            mid_node = user_node_list[index]
            if mid_node not in path_set:
                end_movie_node_list = user_movie_dict[mid_node]
                end_node_index = random.randint(0, len(end_movie_node_list) - 1)
                end_movie_node = end_movie_node_list[end_node_index]
                path_writer.write("\t".join(line_list) + "\t" + mid_node + "\t" + end_movie_node + "\n")
    # count
    nums += 1
    if nums % 1000 == 0:
        print(nums, (time.time() - start_time) / (nums / 1000))
        path_writer.flush()
        # break

    # read next entity pairs
    line = path_reader.readline()
    # break

path_writer.close()
path_reader.close()
