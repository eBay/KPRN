#***********************************************************
#Copyright 2018 eBay Inc.
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
# -*- coding:utf-8 -*-
import codecs
import sys


if __name__ == "__main__":
    # read user id
    user_id_list = []
    user_id_reader = codecs.open(sys.argv[1], mode="r")
    for line in user_id_reader.readlines():
        user_id_list.append(line.strip())
    user_id_reader.close()

    # build dict
    user_tuple_dict = dict()
    for user_id in user_id_list:
        # print(user_id)
        user_tuple_dict[user_id] = [[], []]

    # print(len(user_tuple_dict))

    test_file_reader = codecs.open(sys.argv[2], mode="r")
    line = test_file_reader.readline()
    while line:
        line_list = line.strip().split("\t")
        user_id = line_list[0]
        if line_list[-1] == "1":
            user_tuple_dict[user_id][0].append(line_list[1])
        else:
            user_tuple_dict[user_id][1].append(line_list[1])
        line = test_file_reader.readline()
    test_file_reader.close()

    pos_writer = codecs.open(sys.argv[3], mode="w", encoding="utf-8")
    neg_writer = codecs.open(sys.argv[4], mode="w", encoding="utf-8")

    for user_id in user_id_list:
        item = user_tuple_dict[user_id]
        if len(item[0]) != 0 and len(item[1]) != 0:
            for pos_item in item[0]:
                pos_writer.write(user_id+"\t"+pos_item+"\t1\n")
            for neg_item in item[1]:
                neg_writer.write(user_id+"\t"+neg_item+"\t-1\n")
    pos_writer.close()
    neg_writer.close()

