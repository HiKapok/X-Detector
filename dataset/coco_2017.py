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
from pycocotools import coco

slim = tf.contrib.slim

FILE_PATTERN = 'coco_%s_*.tfrecord'

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}


TRAIN_STATISTICS = {
    "giraffe":  (2464, 5131) ,
    "pizza":  (2707, 5821) ,
    "cow":  (2041, 8147) ,
    "teddy bear":  (2567, 4793) ,
    "cat":  (1294, 4768) ,
    "tennis racket":  (3720, 4812) ,
    "sandwich":  (3053, 4373) ,
    "knife":  (2903, 7770) ,
    "dining table":  (13825, 15714) ,
    "handbag":  (7005, 12354) ,
    "microwave":  (1243, 1673) ,
    "frisbee":  (2230, 2682) ,
    "baseball bat":  (899, 3276) ,
    "dog":  (1211, 5508) ,
    "surfboard":  (4383, 6126) ,
    "broccoli":  (2205, 7308) ,
    "motorcycle":  (1880, 8725) ,
    "hot dog":  (1682, 2917) ,
    "train":  (1938, 4571) ,
    "truck":  (5892, 9973) ,
    "bench":  (5285, 9838) ,
    "bird":  (3162, 10806) ,
    "hair drier":  (268, 198) ,
    "bowl":  (6382, 14358) ,
    "boat":  (2636, 10759) ,
    "toilet":  (2156, 4157) ,
    "bear":  (965, 1294) ,
    "umbrella":  (1978, 11431) ,
    "parking meter":  (646, 1285) ,
    "skateboard":  (3111, 5543) ,
    "toaster":  (30, 225) ,
    "person":  (80455, 262464) ,
    "baseball glove":  (2345, 3747) ,
    "chair":  (15916, 38491) ,
    "banana":  (2367, 9458) ,
    "fork":  (1907, 5479) ,
    "vase":  (5109, 6613) ,
    "kite":  (2351, 9076) ,
    "horse":  (1374, 6587) ,
    "bottle":  (7532, 24342) ,
    "oven":  (2995, 3334) ,
    "clock":  (3597, 6334) ,
    "zebra":  (1825, 5303) ,
    "elephant":  (1893, 5513) ,
    "donut":  (1915, 7179) ,
    "sink":  (5175, 5610) ,
    "cell phone":  (2832, 6434) ,
    "snowboard":  (1617, 2685) ,
    "none":  (0, 0) ,
    "couch":  (4089, 5779) ,
    "wine glass":  (1994, 7913) ,
    "sheep":  (1742, 9509) ,
    "scissors":  (986, 1481) ,
    "bed":  (3035, 4192) ,
    "spoon":  (2775, 6165) ,
    "bicycle":  (2283, 7113) ,
    "mouse":  (1080, 2262) ,
    "laptop":  (2288, 4970) ,
    "bus":  (1436, 6069) ,
    "fire hydrant":  (1427, 1865) ,
    "stop sign":  (1766, 1983) ,
    "tv":  (2417, 5805) ,
    "remote":  (3740, 5703) ,
    "airplane":  (1967, 5135) ,
    "cup":  (11417, 20650) ,
    "potted plant":  (4256, 8652) ,
    "orange":  (2140, 6399) ,
    "keyboard":  (2424, 2855) ,
    "sports ball":  (1756, 6347) ,
    "suitcase":  (2527, 6192) ,
    "car":  (13421, 43867) ,
    "toothbrush":  (1053, 1954) ,
    "skis":  (4733, 6646) ,
    "carrot":  (2821, 7852) ,
    "backpack":  (4873, 8720) ,
    "apple":  (1998, 5851) ,
    "tie":  (1580, 6496) ,
    "traffic light":  (4633, 12884) ,
    "cake":  (2988, 6353) ,
    "book":  (13210, 24715) ,
    "refrigerator":  (1175, 2637) ,
    "total":  (118287, 859999)}
VAL_STATISTICS = {
    "bench":  (212, 413) ,
    "skateboard":  (126, 179) ,
    "toothbrush":  (38, 57) ,
    "person":  (3380, 11004) ,
    "donut":  (69, 338) ,
    "none":  (0, 0) ,
    "refrigerator":  (54, 126) ,
    "horse":  (48, 273) ,
    "elephant":  (84, 255) ,
    "book":  (568, 1161) ,
    "car":  (554, 1932) ,
    "keyboard":  (112, 153) ,
    "cow":  (93, 380) ,
    "microwave":  (62, 55) ,
    "traffic light":  (260, 637) ,
    "tie":  (48, 254) ,
    "dining table":  (572, 697) ,
    "toaster":  (1, 9) ,
    "baseball glove":  (90, 148) ,
    "giraffe":  (92, 232) ,
    "cake":  (164, 316) ,
    "handbag":  (254, 540) ,
    "scissors":  (20, 36) ,
    "bowl":  (291, 626) ,
    "couch":  (190, 261) ,
    "chair":  (749, 1791) ,
    "boat":  (110, 430) ,
    "hair drier":  (13, 11) ,
    "airplane":  (67, 143) ,
    "pizza":  (139, 285) ,
    "backpack":  (156, 371) ,
    "kite":  (82, 336) ,
    "sheep":  (70, 361) ,
    "umbrella":  (100, 413) ,
    "stop sign":  (97, 75) ,
    "truck":  (274, 415) ,
    "skis":  (175, 241) ,
    "sandwich":  (176, 177) ,
    "broccoli":  (74, 316) ,
    "wine glass":  (80, 343) ,
    "surfboard":  (198, 269) ,
    "sports ball":  (76, 263) ,
    "cell phone":  (126, 262) ,
    "dog":  (30, 218) ,
    "bed":  (124, 163) ,
    "toilet":  (115, 179) ,
    "fire hydrant":  (63, 101) ,
    "oven":  (136, 143) ,
    "zebra":  (78, 268) ,
    "tv":  (116, 288) ,
    "potted plant":  (160, 343) ,
    "parking meter":  (42, 60) ,
    "spoon":  (102, 253) ,
    "bus":  (74, 285) ,
    "laptop":  (118, 231) ,
    "cup":  (440, 899) ,
    "bird":  (122, 440) ,
    "sink":  (193, 225) ,
    "remote":  (202, 283) ,
    "bicycle":  (93, 316) ,
    "tennis racket":  (160, 225) ,
    "baseball bat":  (26, 146) ,
    "cat":  (60, 202) ,
    "fork":  (82, 215) ,
    "suitcase":  (105, 303) ,
    "snowboard":  (51, 69) ,
    "clock":  (157, 267) ,
    "apple":  (100, 239) ,
    "mouse":  (90, 106) ,
    "bottle":  (308, 1025) ,
    "frisbee":  (95, 115) ,
    "carrot":  (161, 371) ,
    "bear":  (54, 71) ,
    "hot dog":  (93, 127) ,
    "teddy bear":  (98, 191) ,
    "knife":  (133, 326) ,
    "train":  (73, 190) ,
    "vase":  (132, 277) ,
    "banana":  (105, 379) ,
    "motorcycle":  (92, 371) ,
    "orange":  (104, 287) ,
    "total":  (5000, 36781)}

SPLITS_TO_SIZES = {
    'train2017': 118287,
    'val2017': 5000,
}

SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
    'val': VAL_STATISTICS,
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
    return dataset_common.get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      image_preprocessing_fn,
                                      dataset_name,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,
                                      NUM_CLASSES,
                                      **kwargs)
