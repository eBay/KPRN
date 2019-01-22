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


def sample_nodes(node_list, sample_nums):
    sample_node_res = []
    list_length = len(node_list)
    if list_length <= sample_nums:
        sample_index_list = list(range(list_length))
    else:
        sample_index_list = get_random_index(sample_nums, list_length)
    for index in sample_index_list:
        sample_node_res.append(node_list[index])
    return sample_node_res


def get_mid_path_list(movie_node, person_sample_nums=2, type_sample_nums=1, user_sample_nums=3):
    mid_path_list = []
    person_node_list = movie_person_dict.get(movie_node)
    if person_node_list and person_sample_nums>0:
        person_sample_res = sample_nodes(person_node_list, person_sample_nums)
        for person_node in person_sample_res:
            end_movie_node_list = person_movie_dict[person_node]
            end_node_index = random.randint(0, len(end_movie_node_list) - 1)
            end_movie_node = end_movie_node_list[end_node_index]
            if end_movie_node != movie_node:
                mid_path_list.append([person_node, end_movie_node])
    else:
        pass

    type_node_list = movie_type_dict.get(movie_node)
    if type_node_list and type_sample_nums>0:
        type_sample_res = sample_nodes(type_node_list, type_sample_nums)
        for type_node in type_sample_res:
            end_movie_node_list = type_movie_dict[type_node]
            end_node_index = random.randint(0, len(end_movie_node_list) - 1)
            end_movie_node = end_movie_node_list[end_node_index]
            if end_movie_node != movie_node:
                mid_path_list.append([type_node, end_movie_node])
    else:
        pass

    user_node_list = movie_user_dict.get(movie_node)
    if user_node_list and user_sample_nums>0:
        user_sample_res = sample_nodes(user_node_list, user_sample_nums)
        for user_node in user_sample_res:
            end_movie_node_list = user_movie_dict[user_node]
            end_node_index = random.randint(0, len(end_movie_node_list) - 1)
            end_movie_node = end_movie_node_list[end_node_index]
            if end_movie_node != movie_node:
                mid_path_list.append([user_node, end_movie_node])
    else:
        pass

    return mid_path_list



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

    movie_node = line_list[-1]

    mid_path_list = get_mid_path_list(movie_node=movie_node, person_sample_nums=person_sample_nums,
                                      type_sample_nums=type_sample_nums, user_sample_nums=user_sample_nums)

    for mid_path in mid_path_list:
        current_path = []
        current_path.extend(line_list)
        current_path.extend(mid_path)
        if len(current_path) == len(set(current_path)):
            _movie_node = current_path[-1]
            _mid_path_list = get_mid_path_list(movie_node=_movie_node, person_sample_nums=person_sample_nums,
                                               type_sample_nums=type_sample_nums, user_sample_nums=user_sample_nums)
            for _mid_path in _mid_path_list:
                _current_path = []
                _current_path.extend(current_path)
                _current_path.extend(_mid_path)
                if len(_current_path) == len(set(_current_path)):
                    path_writer.write("\t".join(_current_path) + "\n")
                else:
                    pass
        else:
            pass

    # read next line
    line = path_reader.readline()
    nums += 1
    if nums % 1000 == 0:
        print(nums, (time.time() - start_time) / (nums / 100))
        path_writer.flush()
    # break

path_reader.close()
path_writer.close()