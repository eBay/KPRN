#***********************************************************
#Copyright 2018 eBay Inc.
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
# -*- coding:utf-8 -*-
from __future__ import print_function
import codecs

if __name__ == "__main__":
    file_writer = codecs.open("data/output/baseModel/baseModel_train.txt", mode="w", encoding="utf-8")
    file_reader = codecs.open("data/output/fmg_data/user_song_train.txt", mode="r")
    line = file_reader.readline()

    while line:
        line_list = line.strip().split("\t")
        file_writer.write(line_list[0].replace("u", "")+"\t"+line_list[1].replace("s", "")+"\t"+line_list[2]+"\n")
        line = file_reader.readline()
    file_writer.close()
    file_reader.close()

    file_writer = codecs.open("data/output/baseModel/baseModel_test.txt", mode="w", encoding="utf-8")
    file_reader = codecs.open("data/output/fmg_test_samples_0.0.txt", mode="r")
    line = file_reader.readline()

    while line:
        line_list = line.strip().split("\t")
        file_writer.write(
            line_list[0].replace("u", "") + "\t" + line_list[1].replace("s", "") + "\t" + line_list[2] + "\n")
        line = file_reader.readline()
    file_writer.close()
    file_reader.close()
