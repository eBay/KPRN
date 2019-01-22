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
    # input path
    file_reader = codecs.open(sys.argv[1], mode="r", encoding="utf-8")
    # output path
    file_writer = codecs.open(sys.argv[2], mode="w", encoding="utf-8")

    line = file_reader.readline()
    line_list = line.strip().split("\t")
    current_user_id = line_list[0]
    current_user_dict = dict()
    count_num = 0
    path_num = 0
    entity_pair_num = 0
    start_time = time.time()
    while line:
        line_list = line.strip().split("\t")
        user_id = line_list[0]

        if user_id == current_user_id:
            entity_tuple = (line_list[0], line_list[-1])
            if entity_tuple not in current_user_dict:
                current_user_dict[entity_tuple] = ["/".join(line_list[1:-1])]
                # print("/".join(line_list[1:-1]))
            else:
                current_user_dict[entity_tuple].append("/".join(line_list[1:-1]))
                # print("/".join(line_list[1:-1]))
        else:
            for k, v in current_user_dict.items():
                file_writer.write(k[0]+"\t"+k[1]+"\t")
                file_writer.write("###".join(v)+"\n")
                path_num += len(v)
                entity_pair_num += 1
            file_writer.flush()

            del current_user_dict
            gc.collect()

            current_user_dict = dict()
            current_user_id = line_list[0]

            entity_tuple = (line_list[0], line_list[-1])
            if entity_tuple not in current_user_dict:
                current_user_dict[entity_tuple] = ["/".join(line_list[1:-1])]
            else:
                current_user_dict[entity_tuple].append("/".join(line_list[1:-1]))

        count_num += 1
        if count_num % 10000 == 0:
            print(count_num, (time.time()-start_time)/(count_num/10000))
        # read next line
        line = file_reader.readline()

    # write last batch
    for k, v in current_user_dict.items():
        file_writer.write(k[0] + "\t" + k[1] + "\t")
        file_writer.write("###".join(v) + "\n")
        path_num += len(v)
        entity_pair_num += 1
    file_writer.flush()

    file_writer.close()
    file_reader.close()
    print("total cost time:", time.time()-start_time)
    print("path nums:", path_num, "entity pair nums:", entity_pair_num)
