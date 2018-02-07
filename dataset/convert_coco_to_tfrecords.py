from pycocotools.coco import COCO

import os
import sys
import random

import numpy as np
import skimage.io as io
import scipy
import tensorflow as tf

from dataset_utils import int64_feature, float_feature, bytes_feature

# TFRecords convertion parameters.
SAMPLES_PER_FILES = 5000

class CoCoDataset(object):
    def __init__(self, dataset_dir, image_set='val2017'):
        super(CoCoDataset, self).__init__()
        self._image_set = image_set
        self._ann_file = self.get_ann_file(dataset_dir, self._image_set)
        self._filename_pattern = self.get_image_file_pattern(dataset_dir, self._image_set) + '{}'
        self._coco = COCO(self._ann_file)
        self._cats = self._coco.loadCats(self._coco.getCatIds())
        self._classes = tuple(['none'] + [c['name'] for c in self._cats])
        self._num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, list(range(self._num_classes))))
        self._ind_to_class = dict(zip(list(range(self._num_classes)), self._classes))
        self._super_classes = tuple(['background'] + [c['supercategory'] for c in self._cats])
        self._class_to_coco_cat_id = dict(zip([c['name'] for c in self._cats], self._coco.getCatIds()))
        self._labels = {'none': (0, 'background'),}
        for ind, cls in enumerate(self._classes[1:]):
            self._labels[cls] = (self._class_to_ind[cls], self._super_classes[ind + 1])
        self._image_index = self._coco.getImgIds()
        self._num_examples = len(self._image_index)

    def get_ann_file(self, dataset_dir, data_type):
        return '{}/annotations/instances_{}.json'.format(dataset_dir, data_type)
    def get_image_file_pattern(self, dataset_dir, data_type):
        return '{}/{}/'.format(dataset_dir, data_type)

    def validate_boxes(self, boxes, width=0, height=0):
        """Check that a set of boxes are valid."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        assert (x1 >= 0).all()
        assert (y1 >= 0).all()
        assert (x2 >= x1).all()
        assert (y2 >= y1).all()
        assert (x2 < width).all()
        assert (y2 < height).all()
    def _load_coco_annotation(self, index):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = self._coco.loadImgs(index)[0]
        filaname = im_ann['file_name']
        width = im_ann['width']
        height = im_ann['height']

        annIds = self._coco.getAnnIds(imgIds=index, iscrowd=None)
        objs = self._coco.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['iscrowd'] and (self._image_set == 'train2017' or self._image_set == 'val2017'):
                continue
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                #obj['clean_bbox'] = [x1, y1, x2, y2]
                obj['clean_bbox'] = [y1/height, x1/width, y2/height, x2/width]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)
        has_boxes = 1
        if num_objs == 0:
            has_boxes = 0

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        gt_iscrowd = np.zeros((num_objs), dtype=np.int32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Lookup table to map from COCO category ids to our internal class
        # indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                          self._class_to_ind[cls])
                                         for cls in self._classes[1:]])

        for ix, obj in enumerate(objs):
            cls = coco_cat_id_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            if obj['iscrowd']:
                gt_iscrowd[ix] = 1

        self.validate_boxes(boxes, width=width, height=height)
        return {'filaname' : filaname,
                'boxes' : boxes,
                'shape' : (height, width),
                'gt_classes': gt_classes,
                'gt_iscrowd' : gt_iscrowd,
                'has_boxes': has_boxes}

    def _get_statistic(self):
        class_name_list = ['none', 'total']
        class_name_list.extend([_ for _ in self._classes[1:]])
        stat_by_obj = dict(zip(class_name_list, [0]*len(class_name_list)))
        stat_by_image = dict(zip(class_name_list, [0]*len(class_name_list)))

        for index in self._image_index:
            im_ann = self._coco.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']

            annIds = self._coco.getAnnIds(imgIds=index, iscrowd=None)
            objs = self._coco.loadAnns(annIds)
            # Sanitize bboxes -- some are invalid
            valid_objs = []
            for obj in objs:
                x1 = np.max((0, obj['bbox'][0]))
                y1 = np.max((0, obj['bbox'][1]))
                x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
                if obj['iscrowd'] and (self._image_set == 'train' or self._image_set == 'trainval'):
                    continue
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    valid_objs.append(obj)
            objs = valid_objs
            num_objs = len(objs)

            coco_cat_id_to_name = dict(zip(self._coco.getCatIds(), [c['name'] for c in self._cats]))
            cls_in_image_list = {}
            for ix, obj in enumerate(objs):
                cls = coco_cat_id_to_name[obj['category_id']]
                stat_by_obj[cls] += 1
                stat_by_obj['total'] += 1
                cls_in_image_list[cls] = 0
            for key in cls_in_image_list.keys():
                stat_by_image[cls] += 1
            stat_by_image['total'] += 1

        statistics = dict(zip(class_name_list, [(stat_by_image[cls_name], stat_by_obj[cls_name]) for cls_name in class_name_list]))
        return statistics

# d = CoCoDataset(dataDir, dataType)
# STS = d._get_statistic()
# for k, v in STS.items():
#     print('"%s": '%k, v, ',')

# print('ok')
# for k, v in d._labels.items():
#     print('"%s": '%k, v, ',')

#print(len(d._image_index))
#print([d._load_coco_annotation(index) for index in d._image_index])


# {'filaname' : filaname,
#                 'boxes' : boxes,
#                 'shape' : (height, width),
#                 'gt_classes': gt_classes,
#                 'gt_iscrowd' : gt_iscrowd,
#                 'has_boxes': has_boxes}
# boxes = np.zeros((num_objs, 4), dtype=np.float32)
#         gt_classes = np.zeros((num_objs), dtype=np.int32)
#         gt_iscrowd = np.zeros((num_objs), dtype=np.int32)
#         seg_areas = np.zeros((num_objs), dtype=np.float32)

def _process_image(filename_pattern, ann_dict):
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    filename = filename_pattern.format(ann_dict['filaname'])
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Find annotations.
    bboxes = []
    labels = []
    iscrowd = []
    for index in range(ann_dict['boxes'].shape[0]):
        labels.append(int(ann_dict['gt_classes'][index]))
        iscrowd.append(int(ann_dict['gt_iscrowd'][index]))

        bboxes.append((ann_dict['boxes'][index, 0], ann_dict['boxes'][index, 1], ann_dict['boxes'][index, 2], ann_dict['boxes'][index, 3]
                       ))
    return image_data, ann_dict['shape'], bboxes, labels, iscrowd


def _convert_to_example(image_data, labels, bboxes, shape, iscrowd):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        #          [(ymin_0, xmin_0, ymax_0, xmax_0), (ymin_1, xmin_1, ymax_1, xmax_1), ....]
        #                                            |
        # [ymin_0, ymin_1, ...], [xmin_0, xmin_1, ...], [ymax_0, ymax_1, ...], [xmax_0, xmax_1, ...]
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(3),
            'image/shape': int64_feature([shape[0], shape[1], 3]),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/iscrowd': int64_feature(iscrowd),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example

def _add_to_tfrecord(filename_pattern, ann_dict, tfrecord_writer):
    image_data, shape, bboxes, labels, iscrowd = _process_image(filename_pattern, ann_dict)
    example = _convert_to_example(image_data, labels, bboxes, shape, iscrowd)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return os.path.join(output_dir, '%s_%03d.tfrecord' % (name, idx))


def run(dataset_dir, output_dir, output_name, name='train2017'):
    coco_dataset = CoCoDataset(dataset_dir, name)
    num_examples = coco_dataset._num_examples

    # Process dataset files.
    i = 0
    fidx = 0
    while True:
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, output_name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < num_examples and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, num_examples))
                sys.stdout.flush()

                ann_dict = coco_dataset._load_coco_annotation(coco_dataset._image_index[i])

                _add_to_tfrecord(coco_dataset._filename_pattern, ann_dict, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1
        if not i < num_examples:
            break

    print('\nFinished converting the CoCo dataset!')

if __name__ == '__main__':
    split_name = 'train2017' # 'train2017' or 'val2017'
    output_name = 'coco_{}'.format(split_name)
    dataset_dir = '/media/rs/7A0EE8880EE83EAF/Detections/CoCo'
    output_dir = '../CoCo/tfrecords/{}/'.format(split_name)

    run(dataset_dir, output_dir, output_name, split_name)


    split_name = 'val2017' # 'train2017' or 'val2017'
    output_name = 'coco_{}'.format(split_name)
    dataset_dir = '/media/rs/7A0EE8880EE83EAF/Detections/CoCo'
    output_dir = '../CoCo/tfrecords/{}/'.format(split_name)

    run(dataset_dir, output_dir, output_name, split_name)



