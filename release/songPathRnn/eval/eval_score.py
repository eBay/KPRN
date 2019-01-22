#***********************************************************
#Copyright 2018 eBay Inc.
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
# -*- coding:utf-8 -*-
from __future__ import print_function
import codecs
# import numpy as np
import math
import heapq
import random
import sys
import os

negNum = 100


def get_hit_ratio(rank_list, target_item):
    for item in rank_list:
        if item == target_item:
            return 1
    return 0


def get_ndcg(rank_list, target_item):
    for i, item in enumerate(rank_list):
        if item == target_item:
            return math.log(2) / math.log(i + 2)
    return 0


def eval_one_rating(i_gnd, i_pre, K):
    if sum(i_pre) == 0:
        return 0, 0
    map_score = {}
    for item, score in enumerate(i_pre):
        map_score[item] = score

    target_item = i_gnd.index(1.0)

    rank_list = heapq.nlargest(K, map_score, key=map_score.get)
    hit = get_hit_ratio(rank_list, target_item)
    ndcg = get_ndcg(rank_list, target_item)
    return hit, ndcg


if __name__ == "__main__":
    # read pos user ids
    user_id_list = []
    user_id_reader = codecs.open("test_user_id.txt", mode="r", encoding="utf-8")
    for line in user_id_reader:
        user_id_list.append(line.strip())
    user_id_reader.close()

    alpha = float(sys.argv[1])
    print("eval with alpha:", alpha)

    not_enough_num = 0

    # generate test samples
    score_reader = codecs.open(sys.argv[2], mode="r", encoding="utf-8")
    sample_reader = codecs.open("test_samples/test_samples_"+str(alpha)+".txt", mode="r", encoding="utf-8")
    score_line = score_reader.readline()
    sample_line = sample_reader.readline()

    debug_num = 0
    hit_k_score = []
    ndcg_k_score = []
    pos_error_pairs = []
    neg_error_pairs = []
    for user_id in user_id_list:
        print("user id:", user_id)
        user_item_score_dict = dict()
        while score_line:
            score_line_list = score_line.strip().split("\t")
            if score_line_list[0] == user_id:
                score_line = score_reader.readline()
                user_item_score_dict[(score_line_list[0], score_line_list[1])] = (score_line_list[-2], score_line_list[-1])
            else:
                break

        print("user item len:", len(user_item_score_dict))

        test_sample_list = []
        # read test samples
        while sample_line:
            sample_line_list = sample_line.strip().split("\t")
            if sample_line_list[0] == user_id:
                test_sample_list.append(sample_line_list)
                sample_line = sample_reader.readline()
            else:
                break

        # samples
        for test_object in test_sample_list:
            pos_pair = (user_id, test_object[1])
            neg_pair_list = []
            _flag = False
            for neg_idx in test_object[2].split("#"):
                if (user_id, neg_idx) in user_item_score_dict:
                    neg_pair_list.append((user_id, neg_idx))
                else:
                    _flag = True
                    break
            
            if _flag:
                continue
            
            ground_truth_labels = []
            predict_labels = []
            ground_truth_labels.append(1.0)
            predict_labels.append(float(user_item_score_dict[pos_pair][1]))

            for neg_pair in neg_pair_list:
                _scores = user_item_score_dict[neg_pair]
                ground_truth_labels.append(0.0)
                predict_labels.append(float(_scores[1]))

            temp_hit = []
            temp_ndcg = []
            for k in range(1, 16):
                hit, ndcg = eval_one_rating(i_gnd=ground_truth_labels, i_pre=predict_labels, K=k)
                temp_hit.append(hit)
                temp_ndcg.append(ndcg)

            hit_k_score.append(temp_hit)
            ndcg_k_score.append(temp_ndcg)

    sample_reader.close()
    score_reader.close()

#     total_hit_res_array = np.asarray(hit_k_score)
#     total_ndcg_res_array = np.asarray(ndcg_k_score)
#     print(total_hit_res_array.shape)
#     print(total_ndcg_res_array.shape)
    print(len(hit_k_score), len(hit_k_score[0]))
    
    hit_average = []
    ndcg_average = []
    for i in range(15):
        _temp_hit = []
        for item in hit_k_score:
            _temp_hit.append(float(item[i]))
        hit_average.append("%.5f" % (sum(_temp_hit)/len(_temp_hit)))
        
        _temp_ndcg = []
        for item in ndcg_k_score:
            _temp_ndcg.append(float(item[i]))
        ndcg_average.append("%.5f" % (sum(_temp_ndcg)/len(_temp_ndcg)))
    
    print("hit score:", hit_average)
    print("ndcg score:", ndcg_average)

    score_dir = sys.argv[3]
    score_writer = codecs.open(os.path.join(score_dir, "eval_res_%.1f.txt" % alpha), mode="w", encoding="utf-8")
    score_writer.write("hit scores\t:" + "\t".join(hit_average) + "\n")
    score_writer.write("ndcg scores\t:" + "\t".join(ndcg_average) + "\n")
    score_writer.close()
