#***********************************************************
#Copyright 2018 eBay Inc.
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
# -*- coding:utf-8 -*-
import codecs
import gc
import time
import sys


if __name__ == "__main__":
    # read user id
    user_id_list = []
    user_id_reader = codecs.open(sys.argv[1], mode="r")
    for line in user_id_reader.readlines():
        user_id_list.append(line.strip())
    user_id_reader.close()

    # path 3 and 5 files
    depth_3_reader = codecs.open(sys.argv[2], mode="r", encoding="utf-8")
    depth_5_reader = codecs.open(sys.argv[3], mode="r", encoding="utf-8")
    depth_3_line = depth_3_reader.readline()
    depth_5_line = depth_5_reader.readline()
    # output path
    file_writer = codecs.open(sys.argv[4], mode="w", encoding="utf-8")
    entity_pair_num = 0
    path_num = 0
    count_num = 0
    for user_id in user_id_list:
        user_dict = dict()

        # read depth 3
        while depth_3_line:
            line_list = depth_3_line.strip().split("\t")
            if line_list[0] == user_id:
                # print(line_list)
                entity_pair = (line_list[0], line_list[1])
                # print(entity_pair)
                if entity_pair not in user_dict:
                    user_dict[entity_pair] = line_list[2].split("###")
                else:
                    user_dict[entity_pair].extend(line_list[2].split("###"))
            else:
                break
            depth_3_line = depth_3_reader.readline()

        # read depth 5
        while depth_5_line:
            line_list = depth_5_line.strip().split("\t")
            if line_list[0] == user_id:
                entity_pair = (line_list[0], line_list[1])
                if entity_pair not in user_dict:
                    user_dict[entity_pair] = line_list[2].split("###")
                else:
                    user_dict[entity_pair].extend(line_list[2].split("###"))
            else:
                break
            depth_5_line = depth_5_reader.readline()

        # write
        for k, v in user_dict.items():
            file_writer.write(k[0] + "\t" + k[1] + "\t")
            file_writer.write("###".join(v) + "\n")
            path_num += len(v)
            entity_pair_num += 1
        file_writer.flush()

        del user_dict
        gc.collect()

        count_num += 1
        if count_num % 100 == 0:
            print(count_num)

    print("path nums", path_num, "entity pair nums", entity_pair_num)
    depth_3_reader.close()
    depth_5_reader.close()
    file_writer.close()
