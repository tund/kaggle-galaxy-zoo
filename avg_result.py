# Copyright (c) 2014, Tu Dinh Nguyen (nguyendinhtu@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
average several results
"""

import sys
import os
import csv
import cPickle as pickle
import numpy as np
import gzip
from numpy import isnan


RESULT_FILE = ["./RUN/avg_res/0.07889.csv",
               "./RUN/avg_res/0.07895.csv",
               "./RUN/avg_res/0.07911.csv",
               "./RUN/avg_res/0.07939.csv"]

OUTPUT_FILE = "./RUN/avg_res/final_submission.csv"


def refine_result(res):
    # all values > 1 should be = 1
    res[res > 1] = 1

    # al values < 0 should = 0
    res[res < 0] = 0


def main():
    res_total = np.zeros((79975, 37))
    for i in xrange(len(RESULT_FILE)):
        result = np.genfromtxt(RESULT_FILE[i], dtype=np.float32, delimiter=',', skip_header=1)
        result = result[:, 1:]
        res_total += result

    res_total /= len(RESULT_FILE)

    first_col = np.genfromtxt("./raw_data/kaggle_submission.csv",
                              dtype=np.int32, delimiter=',', skip_header=1, usecols=0)
    first_col = first_col.reshape(len(first_col), 1)

    r = csv.reader(open("./raw_data/kaggle_submission.csv", 'rb'), delimiter=",")
    h = r.next()

    refine_result(res_total)

    with open(OUTPUT_FILE, 'wb') as f_out:
        w = csv.writer(f_out, delimiter=",")
        w.writerow(h)
        for i in range(res_total.shape[0]):
            w.writerow(np.hstack([first_col[i, 0], res_total[i, :]]).astype(np.single))

if __name__ == '__main__':
    main()
