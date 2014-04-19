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
create batches data for training and validation and testing using raw images
+ training data: batches 1-59
+ validation data: batches 60-61
+ testing data: batches 62-140
"""

import os
import sys
import numpy as np
import cPickle as pickle
from natsort import natsorted
from PIL import Image


TRAIN_IMAGE_DIR = "./raw_data/images_training_rev1/"
TEST_IMAGE_DIR = "./raw_data/images_test_rev1/"
TRAIN_OUTPUT_DIR = "./RUN/data/"
TEST_OUTPUT_DIR = TRAIN_OUTPUT_DIR
LABEL_FILE = "./raw_data/training_solutions_rev1.csv"

IMAGE_SIZE = 128                # final size of the image
CROP_SIZE = 200                 # set = 0 if you dont want to crop image first
IMAGE_NUM_CHANNELS = 3
GRAY_SCALE = False              # convert to grayscale or not
NORMALIZE_PIXEL_VALUE = False   # normalize the pixel value into [0, 1] or not
NUM_LABELS = 37
BATCH_SIZE = 1024


def process(image, zoom=0):
    if CROP_SIZE > 0:
        (width, height) = image.size
        left = (width-CROP_SIZE) / 2
        top = (height-CROP_SIZE) / 2
        right = left + CROP_SIZE
        bottom = top + CROP_SIZE
        image = image.crop((left, top, right, bottom))
    if not (IMAGE_SIZE, IMAGE_SIZE) == image.size:
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    image = np.array(image)
    image = np.rollaxis(image, 2)
    image = image.reshape(-1)
    return image


def get_batch_path(output_dir, number):
    filename = "data_batch_{}".format(number)
    return os.path.join(output_dir, filename)


def get_empty_batch():
    return np.zeros((IMAGE_SIZE*IMAGE_SIZE*IMAGE_NUM_CHANNELS, 0), dtype=np.uint8)


def get_empty_batch_label():
    return np.zeros((NUM_LABELS, 0), dtype=np.float32)


def write_train_batch(path, batch, label_prob, idx):
    print "writing {}...\n".format(path)
    d = {'labels': label_prob, 'data': batch, 'idx': idx}
    pickle.dump(d, open(path, "wb"))


def write_test_batch(path, batch):
    print "writing {}...\n".format(path)
    label = np.zeros((NUM_LABELS, batch.shape[1]), dtype=np.float32)
    d = {'data': batch, 'labels': label}
    pickle.dump(d, open(path, "wb"))


def write_batch_meta(output_dir, train_idx, data_mean):
    batches_meta = {}   # meta = dict()
    batches_meta['num_cases_per_batch'] = BATCH_SIZE
    batches_meta['num_vis'] = IMAGE_SIZE*IMAGE_SIZE*IMAGE_NUM_CHANNELS
    batches_meta['data_mean'] = data_mean
    batches_meta['train_idx'] = train_idx
    with open(os.path.join(output_dir, 'batches.meta'), 'wb') as f:
        pickle.dump(batches_meta, f)


def main():
    np.random.seed(6789)

    print "============== Create train and test batches ================"

    # print configuration
    print "train_image_dir = " + TRAIN_IMAGE_DIR
    print "test_image_dir = " + TEST_IMAGE_DIR
    print "label_file = " + LABEL_FILE

    print "image_size = {}".format(IMAGE_SIZE)
    print "crop_size = {}".format(CROP_SIZE)
    print "num_channels = {}".format(IMAGE_NUM_CHANNELS)
    print "num_labels = {}".format(NUM_LABELS)
    print "batch_size = {}".format(BATCH_SIZE)

    print "train_output_dir = " + TRAIN_OUTPUT_DIR
    if not os.path.exists(TRAIN_OUTPUT_DIR):
        print "    This path does not exist. Created one."
        os.makedirs(TRAIN_OUTPUT_DIR)
    print "test_output_dir = " + TEST_OUTPUT_DIR
    if not os.path.exists(TEST_OUTPUT_DIR):
        print "    This path does not exist. Created one."
        os.makedirs(TEST_OUTPUT_DIR)

    train_start_batch_idx = int(sys.argv[1])
    train_end_batch_idx = int(sys.argv[2])
    test_start_batch_idx = int(sys.argv[3])
    test_end_batch_idx = int(sys.argv[4])

    label_data = np.genfromtxt(LABEL_FILE, dtype=np.float32, delimiter=',', skip_header=1)
    # sort labels
    sorted_idx = label_data[:, 0].argsort()
    label_data = label_data[sorted_idx, 1:]
    label_data = label_data.T

    print "retrieving train image filenames..."
    train_names = [d for d in os.listdir(TRAIN_IMAGE_DIR) if d.endswith('.jpg')]
    train_names = natsorted(train_names)
    num_trains = len(train_names)

    train_batch_counter = train_start_batch_idx - 1
    data_mean = np.zeros((IMAGE_SIZE*IMAGE_SIZE*IMAGE_NUM_CHANNELS, 1), dtype=np.float32)

    # CREATE TRAIN DATA
    train_current_batch = get_empty_batch()
    train_current_batch_label = get_empty_batch_label()
    train_current_idx = []
    train_counter = 0
    train_order = np.random.permutation(num_trains)     # randomly take training images

    for i in train_order:
        image_file_name = train_names[i]
        image = Image.open(os.path.join(TRAIN_IMAGE_DIR, image_file_name))
        try:
            image = process(image)
        except ValueError:
            print "problem with train image {}".format(image_file_name)
            sys.exit(1)

        image = image.reshape(-1, 1)
        data_mean += image
        train_current_batch = np.hstack((train_current_batch, image))
        train_current_batch_label = np.hstack((train_current_batch_label,
                                               label_data[:, i].reshape(-1, 1)))
        train_current_idx += [i]

        train_counter += 1
        if train_counter % int(BATCH_SIZE/10) == 0:
            print image_file_name

        if train_current_batch.shape[1] == BATCH_SIZE:
            train_batch_counter += 1
            batch_path = get_batch_path(TRAIN_OUTPUT_DIR, train_batch_counter)
            write_train_batch(batch_path, train_current_batch, train_current_batch_label,
                              train_current_idx)
            train_current_batch = get_empty_batch()
            train_current_batch_label = get_empty_batch_label()
            train_current_idx = []
            if train_batch_counter == train_end_batch_idx:
                break

    if train_current_batch.shape[1] > 0:
        train_batch_counter += 1
        batch_path = get_batch_path(TRAIN_OUTPUT_DIR, train_batch_counter)
        write_train_batch(batch_path, train_current_batch, train_current_batch_label,
                          train_current_idx)

    test_counter = 0

    if test_start_batch_idx != -1:
        if test_start_batch_idx == 0:
            test_batch_counter = train_batch_counter
        else:
            test_batch_counter = test_start_batch_idx - 1
        print "retrieving test image filenames"
        test_names = [d for d in os.listdir(TEST_IMAGE_DIR) if d.endswith('.jpg')]
        test_names = natsorted(test_names)

        # CREATE TEST DATA
        test_current_batch = get_empty_batch()
        for image_file_name in test_names:
            image = Image.open(os.path.join(TEST_IMAGE_DIR, image_file_name))
            try:
                image = process(image)
            except ValueError:
                print "problem with train image {}".format(image_file_name)
                sys.exit(1)

            image = image.reshape(-1, 1)
            data_mean += image
            test_current_batch = np.hstack((test_current_batch, image))

            test_counter += 1
            if test_counter % int(BATCH_SIZE/10) == 0:
                print image_file_name

            if test_current_batch.shape[1] == BATCH_SIZE:
                test_batch_counter += 1
                batch_path = get_batch_path(TEST_OUTPUT_DIR, test_batch_counter)
                write_test_batch(batch_path, test_current_batch)
                test_current_batch = get_empty_batch()
                if test_batch_counter == test_end_batch_idx:
                    break

        if test_current_batch.shape[1] > 0:
            test_batch_counter += 1
            batch_path = get_batch_path(TEST_OUTPUT_DIR, test_batch_counter)
            write_test_batch(batch_path, test_current_batch)

    data_mean /= (train_counter + test_counter)
    write_batch_meta(TRAIN_OUTPUT_DIR, train_order, data_mean)


if __name__ == '__main__':
    main()
