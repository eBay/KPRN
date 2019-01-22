#***********************************************************
#Copyright 2018 eBay Inc.
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
# -*- coding:utf-8 -*-
from __future__ import print_function
import codecs
import sys

if __name__ == "__main__":
    file_name = sys.argv[1]
    file_reader = codecs.open(file_name, mode="r", encoding="utf-8")
    num_lines = int(sys.argv[2])
    pre_str = sys.argv[3]
    count = 0

    file_writer = codecs.open(pre_str+"_part_%d.int" % count, mode="w", encoding="utf-8")
    line = file_reader.readline()
    line_count = 0
    while line:
        file_writer.write(line)
        line_count += 1
        if line_count % num_lines == 0:
            print(count)
            file_writer.close()
            count += 1
            file_writer = codecs.open(pre_str+"_part_%d.int" % count, mode="w", encoding="utf-8")
        line = file_reader.readline()
    file_writer.close()
