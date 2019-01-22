#***********************************************************
#Copyright 2018 eBay Inc.
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
# -*- coding:utf-8 -*-
from __future__ import print_function
import codecs
import numpy as np
import random
import sys

negNum = 100
maxSampleTimes = 100


def generate_fq_dict(alpha, origin_fq_dict):
    sum_pros = 0.0
    for item_id, item_fq in origin_fq_dict.items():
        sum_pros += item_fq**alpha

    new_item_fq_dict = dict()
    for item_id, item_fq in origin_fq_dict.items():
        new_item_fq_dict[item_id] = item_fq**alpha/sum_pros
    return new_item_fq_dict


def sample_with_fq(sample_num, item_fq_list):
    sample_id_list = []
    sample_times = 0
    while True:
        temp_res = list(np.random.multinomial(sample_num * 5, item_fq_list, 1)[0])
        if temp_res.count(0) <= len(item_fq_list) - sample_num:
            for idx, res in enumerate(temp_res):
                if res > 0:
                    sample_id_list.append((idx, res))
            sample_id_list = sorted(sample_id_list, key=lambda x: x[1], reverse=True)
            return True, [item[0] for item in sample_id_list[:100]]
        sample_times += 1
        if sample_times > maxSampleTimes:
            return False, None


if __name__ == "__main__":
    # read pos user ids
    user_id_list = []
    user_id_reader = codecs.open(sys.argv[2], mode="r", encoding="utf-8")
    for line in user_id_reader:
        user_id_list.append(line.strip())
    user_id_reader.close()

    # read origin fq dict
    # movie frequency file
    item_fq_reader = codecs.open(sys.argv[3], mode="r", encoding="utf-8")
    item_fq_dict = dict()
    for line in item_fq_reader.readlines():
        line_list = line.strip().split("\t")
        item_fq_dict[line_list[0]] = float(line_list[1])
    item_fq_reader.close()

    alpha = float(sys.argv[1])
    print("eval with alpha:", alpha)
    itemFqDict = generate_fq_dict(alpha=alpha, origin_fq_dict=item_fq_dict)

    # generate test samples
    pos_reader = codecs.open(sys.argv[4], mode="r", encoding="utf-8")
    neg_reader = codecs.open(sys.argv[5], mode="r", encoding="utf-8")
    pos_line = pos_reader.readline()
    neg_line = neg_reader.readline()

    path_rnn_sample_writer = codecs.open(sys.argv[6], mode="w", encoding="utf-8")
    fmg_sample_writer = codecs.open(sys.argv[7], mode="w", encoding="utf-8")

    failed_num = 0

    for user_id in user_id_list:
        print("user id:", user_id)

        user_pos_list = []
        # read positive items
        while pos_line:
            pos_line_list = pos_line.strip().split("\t")
            if pos_line_list[0] == user_id:
                user_pos_list.append(pos_line_list)
                pos_line = pos_reader.readline()
            else:
                break

        user_neg_list = []
        # read negative items
        while neg_line:
            neg_line_list = neg_line.strip().split("\t")
            if neg_line_list[0] == user_id:
                user_neg_list.append(neg_line_list)
                neg_line = neg_reader.readline()
            else:
                break

        # samples
        for pos_item in user_pos_list:
            if len(user_neg_list) > negNum:
                if alpha == 0.0:
                    neg_sample_list = random.sample(user_neg_list, negNum)
                else:
                    item_fq_list = [itemFqDict[item_object[1]] for item_object in user_neg_list]
                    flag, neg_sample_idx_list = sample_with_fq(sample_num=negNum, item_fq_list=item_fq_list)
                    if flag:
                        neg_sample_list = [user_neg_list[idx] for idx in neg_sample_idx_list]
                    else:
                        neg_sample_list = random.sample(user_neg_list, negNum)
                        failed_num += 1
            else:
                neg_sample_list = user_neg_list
                while len(neg_sample_list) < negNum:
                    idx = random.randint(0, len(user_neg_list)-1)
                    neg_sample_list.append(user_neg_list[idx])

            neg_item_idx = [_item[1] for _item in neg_sample_list]
            path_rnn_sample_writer.write(pos_item[0]+"\t"+pos_item[1]+"\t"+"#".join(neg_item_idx)+"\n")
            fmg_sample_writer.write(pos_item[0]+"\t"+pos_item[1]+"\t1.0\n")
            for neg_item in neg_item_idx:
                fmg_sample_writer.write(pos_item[0] + "\t" + neg_item + "\t0.0\n")

    pos_reader.close()
    neg_reader.close()

    path_rnn_sample_writer.close()
    fmg_sample_writer.close()
