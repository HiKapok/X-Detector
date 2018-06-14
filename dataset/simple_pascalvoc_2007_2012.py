# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import tensorflow as tf
from . import dataset_common

slim = tf.contrib.slim

FILE_PATTERN = 'voc_20??_%s_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}
# (Images, Objects) statistics on every class.
TRAIN_STATISTICS = {
    'cow': (444, 847),
    'car': (1874, 3267),
    'pottedplant': (772, 1487),
    'none': (0, 0),
    'person': (6095, 13256),
    'bicycle': (795, 1064),
    'bottle': (950, 1764),
    'dog': (1707, 2025),
    'motorbike': (771, 1052),
    'boat': (689, 1140),
    'train': (805, 925),
    'total': (16551, 40058),
    'diningtable': (738, 824),
    'sheep': (421, 1070),
    'bus': (607, 822),
    'aeroplane': (908, 1171),
    'sofa': (736, 814),
    'chair': (1564, 3152),
    'tvmonitor': (831, 1108),
    'horse': (769, 1072),
    'cat': (1417, 1593),
    'bird': (1095, 1605)
}

TEST_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (1, 1),
    'bicycle': (1, 1),
    'bird': (1, 1),
    'boat': (1, 1),
    'bottle': (1, 1),
    'bus': (1, 1),
    'car': (1, 1),
    'cat': (1, 1),
    'chair': (1, 1),
    'cow': (1, 1),
    'diningtable': (1, 1),
    'dog': (1, 1),
    'horse': (1, 1),
    'motorbike': (1, 1),
    'person': (1, 1),
    'pottedplant': (1, 1),
    'sheep': (1, 1),
    'sofa': (1, 1),
    'train': (1, 1),
    'tvmonitor': (1, 1),
    'total': (20, 20),
}
SPLITS_TO_SIZES = {
    'train': 22136,
    'test': 4952,
}
SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
    'test': TEST_STATISTICS,
}
NUM_CLASSES = 20


def get_split(split_name, dataset_dir, image_preprocessing_fn, dataset_name, file_pattern=None, reader=None, **kwargs):
    """Gets a dataset tuple with instructions for reading ImageNet.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if not file_pattern:
        file_pattern = FILE_PATTERN
    return dataset_common.simple_slim_get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      image_preprocessing_fn,
                                      dataset_name,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,
                                      NUM_CLASSES,
                                      **kwargs)
# for k, v in TRAIN_STATISTICS2.items():
#     print("'{}': ({}, {})".format(k ,v[0]+TRAIN_STATISTICS1[k][0],v[1]+TRAIN_STATISTICS1[k][1]))
