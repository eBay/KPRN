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
    title_dict = dict()
    file_reader = codecs.open(sys.argv[1], mode="r", encoding="utf-8")
    head_line = file_reader.readline()
    line = file_reader.readline()
    total_num = 0
    while line:
        line_list = line.strip().split("\t")
        movie_id = line_list[-1]
        if movie_id not in title_dict:
            title_dict[movie_id] = 1
        else:
            title_dict[movie_id] += 1

        total_num += 1
        line = file_reader.readline()
    file_reader.close()

    print("dict len:", len(title_dict), total_num)
    fq_writer = codecs.open(sys.argv[3], mode="w", encoding="utf-8")
    temp = 0.0
    movie_file_reader = codecs.open(sys.argv[2], mode="r", encoding="utf-8")
    movie_set = []
    for line in movie_file_reader.readlines():
        movie_set.append(line.strip())
    movie_file_reader.close()
    print("item numbers:", len(movie_set))
    for item_id in movie_set:
        if item_id in title_dict:
            fq = float(title_dict[item_id]) / total_num
            fq_writer.write("%s\t%.8f\n" % (item_id, fq))
            temp += fq
        else:
            fq_writer.write("%s\t%f\n" % (item_id, 0.0))
    fq_writer.close()
    print(temp)
