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
    file_reader = codecs.open(sys.argv[1], mode="r", encoding="utf-8")
    item_list = []
    line = file_reader.readline()
    while line:
        line_list = line.strip().split("\t")
        user_id = line_list[0].replace("u", "")
        item_list.append((int(user_id), line_list[1:]))
        line = file_reader.readline()
    file_reader.close()
    print("resort...", len(item_list))

    new_item_list = sorted(item_list, key=lambda x:x[0])

    print("sorted done")
    file_writer = codecs.open(sys.argv[2], mode="w", encoding="utf-8")
    for item in new_item_list:
        file_writer.write("u"+str(item[0])+"\t"+"\t".join(item[1])+"\n")
    file_writer.close()
