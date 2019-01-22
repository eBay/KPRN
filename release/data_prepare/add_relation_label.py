#***********************************************************
#Copyright 2018 eBay Inc.
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
# -*- coding:utf-8 -*-
import codecs
import time
import sys

# relation dict
relation_dict = {"rate": "r1", "belong": "r2", "category": "r3",
                 "_rate": "r4", "_belong": "r5", "_category": "r6"}


# Find Paths between head entity and tail entity

def get_relation(head_entity, end_entity):
    if "s" in head_entity:
        if "u" in end_entity:
            return relation_dict["_rate"]
        elif "p" in end_entity:
            return relation_dict["_category"]
        elif "t" in end_entity:
            return relation_dict["_belong"]
        else:
            pass
    elif "u" in head_entity:
        if "s" in end_entity:
            return relation_dict["rate"]
        else:
            pass
    elif "p" in head_entity:
        if "s" in end_entity:
            return relation_dict["category"]
        else:
            pass
    elif "t" in head_entity:
        if "s" in end_entity:
            return relation_dict["belong"]
        else:
            pass
    else:
        pass


if __name__ == "__main__":
    # input of positive（user，movie）file
    user_rate_reader = codecs.open(sys.argv[1], mode="r", encoding="utf-8")
    head_line = user_rate_reader.readline()
    ground_truth_list = []
    line = user_rate_reader.readline()
    while line:
        line_list = line.strip().split("\t")
        ground_truth_list.append((line_list[0], line_list[1]))
        line = user_rate_reader.readline()
    user_rate_reader.close()

    ground_truth_list = set(ground_truth_list)
    print(len(ground_truth_list))

    # input and output path 
    path_reader = codecs.open(sys.argv[2], mode="r", encoding="utf-8")
    pos_writer = codecs.open(sys.argv[3], mode="w", encoding="utf-8")
    neg_writer = codecs.open(sys.argv[4], mode="w", encoding="utf-8")

    line = path_reader.readline()
    count_num = 0
    start_time = time.time()
    pos_path_num = 0
    neg_path_num = 0
    pos_pair_num = 0
    neg_pair_num = 0
    while line:
        line_list = line.strip().split("\t")
        entity_pair = (line_list[0], line_list[1])
        start_node = line_list[0]
        end_node = line_list[1]
        # add relation
        path_with_relation_list = []
        path_list = line_list[2].split("###")
        for path in path_list:
            temp_path = []
            node_list = path.split("/")
            # node_list.index(0, start_node)
            pre_node = start_node
            for node in node_list:
                re_id = get_relation(pre_node, node)
                temp_path.append(re_id)
                temp_path.append(node)
                pre_node = node
            re_id = get_relation(pre_node, end_node)
            temp_path.append(re_id)
            path_with_relation_list.append("-".join(temp_path))

        # add label
        if entity_pair in ground_truth_list:
            pos_writer.write("\t".join(entity_pair)+"\t"+"###".join(path_with_relation_list)+"\t1\n")
            pos_pair_num += 1
            pos_path_num += len(path_with_relation_list)
        else:
            neg_writer.write("\t".join(entity_pair)+"\t"+"###".join(path_with_relation_list)+"\t-1\n")
            neg_pair_num += 1
            neg_path_num += len(path_with_relation_list)
        # read next line
        line = path_reader.readline()

        count_num += 1
        if count_num % 10000 == 0:
            print(count_num, (time.time()-start_time)/(count_num/10000))

        # break

    path_reader.close()
    pos_writer.close()
    neg_writer.close()

    print("total cost time:", time.time()-start_time)
    print("pos pair nums:", pos_pair_num, "pos path nums:", pos_path_num)
    print("neg pair nums:", neg_pair_num, "neg path nums:", neg_path_num)
