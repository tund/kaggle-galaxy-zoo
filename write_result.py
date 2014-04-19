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
write results produced by ConvNet to CSV file to submit
"""

import sys
import os
import csv
import cPickle as pickle
import numpy as np


INPUT_PATH = "./RUN/model/ConvNet__2014-03-29_07.47.35_182.59/feat_62_140_fc37"


def refine_result(res):
    # all values > 1 should be = 1
    res[res > 1] = 1
    # al values < 0 should = 0
    res[res < 0] = 0


def main():
    start_batch = int(sys.argv[1])
    end_batch = int(sys.argv[2])

    res_total = np.zeros((0, 37), dtype=np.float32)

    label_data = np.genfromtxt("./raw_data/kaggle_submission.csv",
                               dtype=np.int32, delimiter=',', skip_header=1, usecols=0)
    label_data = label_data.reshape(len(label_data), 1)

    r = csv.reader(open("./raw_data/kaggle_submission.csv", 'rb'), delimiter=",")
    h = r.next()

    for inp in INPUT_PATH:
        res_each_inp = np.zeros((0, 37), dtype=np.float32)
        for i in range(start_batch, end_batch+1):
            res_each = pickle.load(open(os.path.join(inp, "data_batch_{}".format(i)), 'rb'))
            res_each_inp = np.vstack((res_each_inp, res_each['data'].T))
        if res_total.shape[0] == 0:
            res_total = res_each_inp
        else:
            res_total += res_each_inp

    res_total /= len(INPUT_PATH)

    refine_result(res_total)

    with open('./RUN/avg_res/0.07939.csv', 'wb') as f_out:
        w = csv.writer(f_out, delimiter=",")
        w.writerow(h)
        for i in range(label_data.shape[0]):
            w.writerow(np.hstack([label_data[i, 0], res_total[i, :]]).astype(np.single))


if __name__ == '__main__':
    main()
