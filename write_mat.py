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
write features extracted by ConvNet to MATLAB format file for further processes
"""

import sys
import os
import cPickle as pickle
import scipy.io as sio
import numpy as np


def main():
    batch_range = xrange(1, 141)
    input_dir = sys.argv[1]
    matlab_file = sys.argv[2]
    data = np.zeros((0, 2048), dtype=np.single)
    label = np.zeros((0, 37), dtype=np.single)
    for i in batch_range:
        print 'process data_batch_{}'.format(i)
        batch = pickle.load(open(os.path.join(input_dir, 'data_batch_{}'.format(i)), 'rb'))
        data = np.vstack([data, batch['data'].T])
        label = np.vstack([label, batch['labels'].T])

    sio.savemat(matlab_file, {'data': data, 'label': label})

if __name__ == '__main__':
    main()
