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
import os

import tensorflow as tf

from tensorflow.python.framework import sparse_tensor

from . import dataset_utils

slim = tf.contrib.slim

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

COCO_LABELS = {
    "bench":  (14, 'outdoor') ,
    "skateboard":  (37, 'sports') ,
    "toothbrush":  (80, 'indoor') ,
    "person":  (1, 'person') ,
    "donut":  (55, 'food') ,
    "none":  (0, 'background') ,
    "refrigerator":  (73, 'appliance') ,
    "horse":  (18, 'animal') ,
    "elephant":  (21, 'animal') ,
    "book":  (74, 'indoor') ,
    "car":  (3, 'vehicle') ,
    "keyboard":  (67, 'electronic') ,
    "cow":  (20, 'animal') ,
    "microwave":  (69, 'appliance') ,
    "traffic light":  (10, 'outdoor') ,
    "tie":  (28, 'accessory') ,
    "dining table":  (61, 'furniture') ,
    "toaster":  (71, 'appliance') ,
    "baseball glove":  (36, 'sports') ,
    "giraffe":  (24, 'animal') ,
    "cake":  (56, 'food') ,
    "handbag":  (27, 'accessory') ,
    "scissors":  (77, 'indoor') ,
    "bowl":  (46, 'kitchen') ,
    "couch":  (58, 'furniture') ,
    "chair":  (57, 'furniture') ,
    "boat":  (9, 'vehicle') ,
    "hair drier":  (79, 'indoor') ,
    "airplane":  (5, 'vehicle') ,
    "pizza":  (54, 'food') ,
    "backpack":  (25, 'accessory') ,
    "kite":  (34, 'sports') ,
    "sheep":  (19, 'animal') ,
    "umbrella":  (26, 'accessory') ,
    "stop sign":  (12, 'outdoor') ,
    "truck":  (8, 'vehicle') ,
    "skis":  (31, 'sports') ,
    "sandwich":  (49, 'food') ,
    "broccoli":  (51, 'food') ,
    "wine glass":  (41, 'kitchen') ,
    "surfboard":  (38, 'sports') ,
    "sports ball":  (33, 'sports') ,
    "cell phone":  (68, 'electronic') ,
    "dog":  (17, 'animal') ,
    "bed":  (60, 'furniture') ,
    "toilet":  (62, 'furniture') ,
    "fire hydrant":  (11, 'outdoor') ,
    "oven":  (70, 'appliance') ,
    "zebra":  (23, 'animal') ,
    "tv":  (63, 'electronic') ,
    "potted plant":  (59, 'furniture') ,
    "parking meter":  (13, 'outdoor') ,
    "spoon":  (45, 'kitchen') ,
    "bus":  (6, 'vehicle') ,
    "laptop":  (64, 'electronic') ,
    "cup":  (42, 'kitchen') ,
    "bird":  (15, 'animal') ,
    "sink":  (72, 'appliance') ,
    "remote":  (66, 'electronic') ,
    "bicycle":  (2, 'vehicle') ,
    "tennis racket":  (39, 'sports') ,
    "baseball bat":  (35, 'sports') ,
    "cat":  (16, 'animal') ,
    "fork":  (43, 'kitchen') ,
    "suitcase":  (29, 'accessory') ,
    "snowboard":  (32, 'sports') ,
    "clock":  (75, 'indoor') ,
    "apple":  (48, 'food') ,
    "mouse":  (65, 'electronic') ,
    "bottle":  (40, 'kitchen') ,
    "frisbee":  (30, 'sports') ,
    "carrot":  (52, 'food') ,
    "bear":  (22, 'animal') ,
    "hot dog":  (53, 'food') ,
    "teddy bear":  (78, 'indoor') ,
    "knife":  (44, 'kitchen') ,
    "train":  (7, 'vehicle') ,
    "vase":  (76, 'indoor') ,
    "banana":  (47, 'food') ,
    "motorcycle":  (4, 'vehicle') ,
    "orange":  (50, 'food')}


#def slim_get_split(split_name, dataset_dir, file_pattern, reader, image_preprocessing_fn,
#              split_to_sizes, items_to_descriptions, num_classes, **kwargs):
def slim_get_split(split_name, dataset_dir, file_pattern, reader, image_preprocessing_fn, dataset_name, split_to_sizes, items_to_descriptions, num_classes, **kwargs):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

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
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    if 'coco' in dataset_name:
        # Features in CoCo TFRecords.
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/channels': tf.FixedLenFeature([1], tf.int64),
            'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/iscrowd': tf.VarLenFeature(dtype=tf.int64),
        }
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
            'shape': slim.tfexample_decoder.Tensor('image/shape'),
            'object/bbox': slim.tfexample_decoder.BoundingBox(
                    ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
            'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
            'object/iscrowd': slim.tfexample_decoder.Tensor('image/object/bbox/iscrowd'),
        }
    else:
        # Features in Pascal VOC TFRecords.
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/channels': tf.FixedLenFeature([1], tf.int64),
            'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
        }
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
            'shape': slim.tfexample_decoder.Tensor('image/shape'),
            'object/bbox': slim.tfexample_decoder.BoundingBox(
                    ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
            'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
            'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
            'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
        }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = {}
    if 'coco' in dataset_name:
        for name, pair in COCO_LABELS.items():
            labels_to_names[pair[0]] = name
    else:
        for name, pair in VOC_LABELS.items():
            labels_to_names[pair[0]] = name

    dataset = slim.dataset.Dataset(
                data_sources=file_pattern,
                reader=reader,
                decoder=decoder,
                num_samples=split_to_sizes[split_name],
                items_to_descriptions=items_to_descriptions,
                num_classes=num_classes,
                labels_to_names=labels_to_names)
    is_training = True
    # check additional arguments
    if 'method' in kwargs:
        if kwargs['method'] == 'eval':
            is_training = False
    if 'batch_size' not in kwargs:
        raise ValueError('Must provide "batch_size" for slim DatasetDataProvider.')
    if 'num_readers' not in kwargs:
        raise ValueError('Must provide "num_readers" for slim DatasetDataProvider.')
    if 'num_preprocessing_threads' not in kwargs:
        raise ValueError('Must provide "num_preprocessing_threads" for slim DatasetDataProvider.')
    if 'anchor_encoder' not in kwargs:
        raise ValueError('Must provide "anchor_encoder" for slim DatasetDataProvider.')
    if 'num_epochs' not in kwargs:
        num_epochs = None
    else:
        num_epochs = kwargs['num_epochs']

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=kwargs['num_readers'],
            common_queue_capacity=32 * kwargs['batch_size'],
            common_queue_min=8 * kwargs['batch_size'],
            shuffle=True,
            num_epochs = num_epochs)

    [org_image, shape, glabels_raw, gbboxes_raw, isdifficult] = provider.get(['image', 'shape',
                                                     'object/label',
                                                     'object/bbox',
                                                     'object/difficult'])
    # if is_training:
    #     glabels_raw = tf.cast(isdifficult < tf.ones_like(isdifficult), glabels_raw.dtype) * glabels_raw
    if is_training:
        # isdifficult = tf.ones_like(isdifficult)
        # isdifficult= tf.Print(isdifficult,[isdifficult])
        # if all is difficult, then keep the first one
        isdifficult_mask =tf.cond(tf.reduce_sum(tf.cast(tf.logical_not(tf.equal(tf.ones_like(isdifficult), isdifficult)), tf.float32)) < 1., lambda : tf.one_hot(0, tf.shape(isdifficult)[0], on_value=True, off_value=False, dtype=tf.bool), lambda : isdifficult < tf.ones_like(isdifficult))

        glabels_raw = tf.boolean_mask(glabels_raw, isdifficult_mask)
        gbboxes_raw = tf.boolean_mask(gbboxes_raw, isdifficult_mask)
        #glabels_raw = tf.cast(isdifficult < tf.ones_like(isdifficult), glabels_raw.dtype) * glabels_raw

    # Pre-processing image, labels and bboxes.
    if is_training:
        image, glabels_, gbboxes_ = image_preprocessing_fn(org_image, shape, glabels_raw, gbboxes_raw)
    else:
        image, glabels_, gbboxes_, bbox_img = image_preprocessing_fn(org_image, shape, glabels_raw, gbboxes_raw)

    glabels_raw = tf.cast(glabels_raw, tf.int32)

    glabels, gtargets, gscores, _ = kwargs['anchor_encoder'](glabels_, gbboxes_)

    list_for_batch = []
    for glabel in glabels:
        list_for_batch.append(glabel)
    for gtarget in gtargets:
        list_for_batch.append(gtarget)
    for gscore in gscores:
        list_for_batch.append(gscore)

    if not is_training:
        list_for_batch.append(glabels_raw)
        list_for_batch.append(gbboxes_raw)
    else:
        list_for_batch.append(glabels_)
        list_for_batch.append(gbboxes_)
    if not is_training:
        list_for_batch.append(bbox_img)
        list_for_batch.append(isdifficult)
        list_for_batch.append(org_image)

    list_for_batch.append(shape)
    list_for_batch.append(image)

    return tf.train.batch(list_for_batch,
                dynamic_pad=True,#(not is_training),
                batch_size = kwargs['batch_size'],
                allow_smaller_final_batch=(not is_training),
                num_threads = kwargs['num_preprocessing_threads'],
                capacity = 64 * kwargs['batch_size']), None


def get_split(split_name, dataset_dir, file_pattern, reader, image_preprocessing_fn,
              dataset_name, split_to_sizes, items_to_descriptions, num_classes, **kwargs):
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    input_file_list = tf.gfile.Glob(os.path.join(dataset_dir, file_pattern % split_name))

    is_training = True
    # check additional arguments
    if 'method' in kwargs:
        if kwargs['method'] == 'eval':
            is_training = False
    if 'batch_size' not in kwargs:
        raise ValueError('Must provide "batch_size" for Dataset.')
    if 'num_readers' not in kwargs:
        raise ValueError('Must provide "num_readers" for Dataset.')
    if 'num_preprocessing_threads' not in kwargs:
        raise ValueError('Must provide "num_preprocessing_threads" for Dataset.')
    if 'anchor_encoder' not in kwargs:
        raise ValueError('Must provide "anchor_encoder" for Dataset.')
    if 'num_epochs' not in kwargs:
        num_epochs = None
    else:
        num_epochs = kwargs['num_epochs']

    def _parse_function(example_proto):
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height': tf.FixedLenFeature([1], tf.int64),
            'image/width': tf.FixedLenFeature([1], tf.int64),
            'image/channels': tf.FixedLenFeature([1], tf.int64),
            'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
        }

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        org_image = tf.image.decode_image(parsed_features["image/encoded"])
        org_image.set_shape([None, None, 3])

        shape = parsed_features["image/shape"]
        glabels_raw = parsed_features["image/object/bbox/label"].values
        gbboxes_xmin = parsed_features["image/object/bbox/xmin"].values
        gbboxes_ymin = parsed_features["image/object/bbox/ymin"].values
        gbboxes_xmax = parsed_features["image/object/bbox/xmax"].values
        gbboxes_ymax = parsed_features["image/object/bbox/ymax"].values

        isdifficult = parsed_features['image/object/bbox/difficult'].values

        gbboxes_raw = tf.stack([gbboxes_ymin, gbboxes_xmin, gbboxes_ymax, gbboxes_xmax], axis=-1)

        # if is_training:
        #     glabels_raw = tf.cast(isdifficult < tf.ones_like(isdifficult), glabels_raw.dtype) * glabels_raw
        if is_training:
            # isdifficult = tf.ones_like(isdifficult)
            # isdifficult= tf.Print(isdifficult,[isdifficult])
            # if all is difficult, then keep the first one
            isdifficult_mask =tf.cond(tf.reduce_sum(tf.cast(tf.logical_not(tf.equal(tf.ones_like(isdifficult), isdifficult)), tf.float32)) < 1., lambda : tf.one_hot(0, tf.shape(isdifficult)[0], on_value=True, off_value=False, dtype=tf.bool), lambda : isdifficult < tf.ones_like(isdifficult))

            glabels_raw = tf.boolean_mask(glabels_raw, isdifficult_mask)
            gbboxes_raw = tf.boolean_mask(gbboxes_raw, isdifficult_mask)
            #glabels_raw = tf.cast(isdifficult < tf.ones_like(isdifficult), glabels_raw.dtype) * glabels_raw

        if is_training:
            image, glabels_, gbboxes_ = image_preprocessing_fn(org_image, shape, glabels_raw, gbboxes_raw)
        else:
            image, glabels_, gbboxes_, bbox_img = image_preprocessing_fn(org_image, shape, glabels_raw, gbboxes_raw)

        glabels_raw = tf.cast(glabels_raw, tf.int32)

        #return image, glabels

        glabels, gtargets, gscores, _ = kwargs['anchor_encoder'](glabels_, gbboxes_)

        list_for_batch = []
        for glabel in glabels:
            list_for_batch.append(glabel)
        for gtarget in gtargets:
            list_for_batch.append(gtarget)
        for gscore in gscores:
            list_for_batch.append(gscore)

        if not is_training:
            list_for_batch.append(glabels_raw)
            list_for_batch.append(gbboxes_raw)
        else:
            list_for_batch.append(glabels_)
            list_for_batch.append(gbboxes_)
        if not is_training:
            list_for_batch.append(bbox_img)
            list_for_batch.append(isdifficult)
            list_for_batch.append(org_image)

        list_for_batch.append(shape)
        list_for_batch.append(image)

        return list_for_batch

    # dataset = tf.data.TFRecordDataset(input_file_list, compression_type=None, buffer_size = int(128 * 1024 * 1024))
    # dataset = dataset.map(_parse_function, num_parallel_calls=kwargs['num_preprocessing_threads'])
    # dataset = dataset.prefetch(kwargs['batch_size'] * 20)
    # # When choosing shuffle buffer sizes, larger sizes result in better
    # # randomness, while smaller sizes have better performance.
    # dataset = dataset.shuffle(buffer_size = 32 * kwargs['batch_size'])
    # dataset = dataset.batch(kwargs['batch_size'])
    # dataset = dataset.cache()
    # dataset = dataset.repeat(count = num_epochs)
    # dataset_iterator = dataset.make_one_shot_iterator()
    # #dataset_iterator = dataset.make_initializable_iterator()

    # return dataset_iterator.get_next(), None#dataset_iterator.initializer

    dataset = tf.data.TFRecordDataset(input_file_list, compression_type=None, buffer_size = int(128 * 1024 * 1024))
    dataset = dataset.map(_parse_function, num_parallel_calls=kwargs['num_preprocessing_threads'])
    dataset = dataset.prefetch(kwargs['batch_size'] * 20)
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    dataset = dataset.shuffle(buffer_size = 32 * kwargs['batch_size'])
    # remove batch op here, just use tf.data for parallelism in C++, then batch data use python queue for lower latency
    dataset = dataset.batch(kwargs['batch_size'])
    dataset = dataset.cache()
    dataset = dataset.repeat(count = num_epochs)
    dataset_iterator = dataset.make_one_shot_iterator()
    #dataset_iterator = dataset.make_initializable_iterator()

    return dataset_iterator.get_next(), None#dataset_iterator.initializer

    # return tf.train.batch(dataset_iterator.get_next(),
    #             batch_size = kwargs['batch_size'],
    #             num_threads = kwargs['num_readers'],
    #             capacity = 64 * kwargs['batch_size']), None
