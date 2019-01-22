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
    pos_file_path = sys.argv[1]
    neg_file_path = sys.argv[2]

    pos_reader = codecs.open(pos_file_path, mode="r")
    fmg_pos_writer = codecs.open(sys.argv[3], mode="w", encoding="utf-8")
    line = pos_reader.readline()
    while line:
        line_list = line.strip().split("\t")
        fmg_pos_writer.write(line_list[0]+"\t"+line_list[1]+"\n")
        line = pos_reader.readline()
    pos_reader.close()
    fmg_pos_writer.close()

    neg_reader = codecs.open(neg_file_path, mode="r")
    fmg_neg_writer = codecs.open(sys.argv[4], mode="w", encoding="utf-8")
    line = neg_reader.readline()
    while line:
        line_list = line.strip().split("\t")
        fmg_neg_writer.write(line_list[0] + "\t" + line_list[1] + "\n")
        line = neg_reader.readline()
    neg_reader.close()
    fmg_neg_writer.close()

    fmg_item_list = []

    train_pos_reader = codecs.open(sys.argv[5], mode="r")
    line = train_pos_reader.readline()
    while line:
        line_list = line.strip().split("\t")
        fmg_item_list.append((line_list[0], line_list[1], "1.0"))
        line = train_pos_reader.readline()
    train_pos_reader.close()

    train_neg_reader = codecs.open(sys.argv[6], mode="r")
    line = train_neg_reader.readline()
    while line:
        line_list = line.strip().split("\t")
        fmg_item_list.append((line_list[0], line_list[1], "0.0"))
        line = train_neg_reader.readline()
    train_neg_reader.close()

    random.shuffle(fmg_item_list)

    fmg_train_file_path = sys.argv[7]
    fmg_train_file_writer = codecs.open(fmg_train_file_path, mode="w", encoding="utf-8")

    for item in fmg_item_list:
        fmg_train_file_writer.write("\t".join(item)+"\n")
    fmg_train_file_writer.close()

