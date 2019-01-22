#***********************************************************
#Copyright 2018 eBay Inc.
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
# -*- coding:utf-8 -*-
import codecs
import sys
import random

if __name__ == "__main__":
    pos_split_num = float(sys.argv[1])
    neg_split_num = float(sys.argv[2])
    pos_file_path = sys.argv[3]
    neg_file_path = sys.argv[4]

    print("read positive lines...")
    pos_reader = codecs.open(pos_file_path, mode="r", encoding="utf-8")
    pos_list = []
    line = pos_reader.readline()
    while line:
        pos_list.append(line.strip())
        line = pos_reader.readline()
    pos_reader.close()
    random.shuffle(pos_list)

    print("read negative lines...")
    neg_reader = codecs.open(neg_file_path, mode="r", encoding="utf-8")
    neg_list = []
    line = neg_reader.readline()
    while line:
        neg_list.append(line.strip())
        line = neg_reader.readline()
    neg_reader.close()
    random.shuffle(neg_list)

    train_pos_len = int(len(pos_list)*pos_split_num)
    train_neg_len = int(len(neg_list)*neg_split_num)

    train_pos_file_writer = codecs.open(sys.argv[5], mode="w", encoding="utf-8")
    for line in pos_list[:train_pos_len]:
        train_pos_file_writer.write(line+"\n")
    train_pos_file_writer.close()

    train_neg_file_writer = codecs.open(sys.argv[6], mode="w", encoding="utf-8")
    for line in neg_list[:train_neg_len]:
        train_neg_file_writer.write(line + "\n")
    train_neg_file_writer.close()

    test_file_writer = codecs.open(sys.argv[7], mode="w", encoding="utf-8")
    for line in pos_list[train_pos_len:]:
        test_file_writer.write(line+"\n")
    for line in neg_list[train_neg_len:]:
        test_file_writer.write(line+"\n")
    test_file_writer.close()

