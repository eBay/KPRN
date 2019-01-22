#***********************************************************
#Copyright 2018 eBay Inc.
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
# -*- coding:utf-8 -*-
import codecs
import os
import sys

if __name__ == "__main__":
    data_dir = sys.argv[1]
    for type_str in ["train", "test"]:
        path = os.path.join(data_dir, type_str)
        file_writer = codecs.open(os.path.join(data_dir, type_str+".list"), mode="w", encoding="utf-8")
        for file_name in os.listdir(path):
            if ".torch" in file_name:
                file_writer.write(type_str+"/"+file_name+"\n")
        file_writer.close()
