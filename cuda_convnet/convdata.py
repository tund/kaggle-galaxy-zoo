# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
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

from data import *
import numpy as np
import numpy.random as npr
from scipy.misc import imresize
from scipy.misc import imsave
import time


# 128x128x3 images
# train: x1: 1 random crop | horizontal reflection | +90, +180, +270 rotation | 1 zoom-in;
# test: x1: center;
class CroppedKaggleGalaxyZoo128DataProviderX90rotZoomMemory(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1,
                 init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch,
                                           init_batchnum, dp_params, test)

        self.image_size = 128
        self.border_size = dp_params['crop_border']
        self.inner_size = self.image_size - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_trans = 5
        self.num_views = 5*self.num_trans
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3

        for d in self.data_dic:
            d['data'] = np.require(d['data'], requirements='C')
            d['labels'] = np.require(np.tile(d['labels'], (1, self.data_mult)),
                                     requirements='C')

        self.cropped_data = [np.zeros((self.get_data_dims(),
                                      self.batch_meta['num_cases_per_batch']*self.data_mult),
                                      dtype=np.single) for x in xrange(2)]

        self.batches_generated = 0
        self.data_mean = (
            self.batch_meta['data_mean'].reshape((3, self.image_size, self.image_size))
            [:, self.border_size:self.border_size+self.inner_size,
             self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))
        )

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        cropped = self.cropped_data[self.batches_generated % 2]
        if datadic['data'].shape[1] == cropped.shape[1]:
            self.batches_generated += 1
        else:
            cropped = np.zeros((self.get_data_dims(), datadic['data'].shape[1]*self.data_mult),
                               dtype=np.single)

        self.__trim_borders(datadic['data'], cropped)
        cropped -= self.data_mean

        return epoch, batchnum, [cropped, datadic['labels']]

    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 37

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return np.require((data + self.data_mean).T.
                          reshape(data.shape[1], 3, self.inner_size, self.inner_size)
                          .swapaxes(1, 3).swapaxes(1, 2) / 255.0, dtype=np.single)

    def __trim_borders(self, x, target):
        y = x.reshape(3, self.image_size, self.image_size, x.shape[1])

        if self.test:   # don't need to loop over cases
            if self.multiview:
                start_positions = [(0, 0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                   (self.border_size*2, 0),
                                   (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size)
                                 for (sy, sx) in start_positions]
                for i in xrange(self.num_views/self.num_trans):
                    pic = y[:, start_positions[i][0]:end_positions[i][0],
                            start_positions[i][1]:end_positions[i][1], :]

                    # original image
                    target[:, i*x.shape[1]:(i+1)*x.shape[1]] = pic.reshape((self.get_data_dims(),
                                                                            x.shape[1]))
                    # horizontal reflection
                    target[:, (self.num_views/self.num_trans + i)*x.shape[1]:
                           (self.num_views/self.num_trans + i + 1)*x.shape[1]] = (
                        pic[:, :, ::-1, :].reshape((self.get_data_dims(), x.shape[1])))

                    pic = np.rollaxis(pic, 0, 3)
                    # rotate +90
                    pic = np.rot90(pic)
                    target[:, (2*self.num_views/self.num_trans + i)*x.shape[1]:
                           (2*self.num_views/self.num_trans + i + 1)*x.shape[1]] = (
                        np.rollaxis(pic, 2, 0).reshape((self.get_data_dims(), x.shape[1])))
                    # rotate +180
                    pic = np.rot90(pic)
                    target[:, (3*self.num_views/self.num_trans + i)*x.shape[1]:
                           (3*self.num_views/self.num_trans + i + 1)*x.shape[1]] = (
                        np.rollaxis(pic, 2, 0).reshape((self.get_data_dims(), x.shape[1])))
                    # rotate +270
                    pic = np.rot90(pic)
                    target[:, (4*self.num_views/self.num_trans + i)*x.shape[1]:
                           (4*self.num_views/self.num_trans + i + 1)*x.shape[1]] = (
                        np.rollaxis(pic, 2, 0).reshape((self.get_data_dims(), x.shape[1])))
            else:
                # just take the center for now
                pic = y[:, self.border_size:self.border_size+self.inner_size,
                        self.border_size:self.border_size+self.inner_size, :]
                target[:, :] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]):    # loop over cases
                startY, startX = (npr.randint(0, self.border_size*2 + 1),
                                  npr.randint(0, self.border_size*2 + 1))
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:, startY:endY, startX:endX, c]
                # 0: unchanged
                # 1, 2, 3: +90, +180, +270 rotation
                # 4: horizontal reflection
                # 5: zoom
                tmp = npr.randint(0, 6)
                if tmp == 4:
                    pic = pic[:, :, ::-1]
                elif tmp == 5:
                    ratio = npr.uniform(2.0, 3.0)
                    crop_size = np.ceil(self.image_size / ratio)
                    left = np.ceil((self.image_size-crop_size) / 2)
                    top = np.ceil((self.image_size-crop_size) / 2)
                    right = left + crop_size
                    bottom = top + crop_size
                    pic = pic[:, top:bottom, left:right]
                    pic = np.rollaxis(pic, 0, 3)
                    pic = imresize(pic, (self.inner_size, self.inner_size), interp='bicubic')
                    pic = np.rollaxis(pic, 2, 0)
                elif tmp != 0:
                    pic = np.rollaxis(pic, 0, 3)
                    pic = np.rot90(pic, tmp)
                    pic = np.rollaxis(pic, 2, 0)
                target[:, c] = pic.reshape((self.get_data_dims(),))
