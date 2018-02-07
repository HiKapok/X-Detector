"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function

from . import dataset_common

import sys
import os
import time
import numpy as np
import pickle
import cv2

import tensorflow as tf

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class DetectorEvalPascal(object):
    def __init__(self, voc_root, devkit_root, set_type = 'test', output_dir='output_{}'):
        super(DetectorEvalPascal, self).__init__()
        self._set_type = set_type
        output_dir=output_dir.format(set_type)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        self._voc_root = voc_root
        self._output_dir = output_dir
        self._annopath = os.path.join(self._voc_root, 'VOC2007', 'Annotations', '%s.xml')
        self._imgpath = os.path.join(self._voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(self._voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
        self._devkit_path = os.path.join(devkit_root, 'VOC2007')
        self._image_ids = list()
        rootpath = os.path.join(self._voc_root, 'VOC2007')
        for line in open(os.path.join(rootpath, 'ImageSets', 'Main', self._set_type + '.txt')):
            self._image_ids.append((rootpath, line.strip()))

    @property
    def image_ids(self):
        return self._image_ids
    @property
    def output_dir(self):
        return self._output_dir
    def evaluate_detections(self, box_list):
        self.write_voc_results_file(box_list)
        self.do_python_eval()

    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                                  int(bbox.find('ymin').text) - 1,
                                  int(bbox.find('xmax').text) - 1,
                                  int(bbox.find('ymax').text) - 1]
            objects.append(obj_struct)

        return objects


    def get_voc_results_file_template(self, cls):
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = 'det_' + self._set_type + '_%s.txt' % (cls)
        filedir = os.path.join(self._devkit_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path


    def write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(pascalvoc_common.VOC_CLASSES):
            print('Writing {:s} VOC results file'.format(cls))
            filename = self.get_voc_results_file_template(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_ids):
                    dets = all_boxes[cls_ind+1][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))


    def do_python_eval(self, use_07=True):
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        for i, cls in enumerate(pascalvoc_common.VOC_CLASSES):
            filename = self.get_voc_results_file_template(cls)
            rec, prec, ap = self.voc_eval(filename, cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(self._output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('--------------------------------------------------------------')


    def voc_ap(self, rec, prec, use_07_metric=True):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


    def voc_eval(self, detpath,
                 classname,
                 cachedir,
                 ovthresh=0.5,
                 use_07_metric=True):
        """rec, prec, ap = voc_eval(detpath,
                               annopath,
                               imagesetfile,
                               classname,
                               [ovthresh],
                               [use_07_metric])
        Top level function that does the PASCAL VOC evaluation.
        detpath: Path to detections
           detpath.format(classname) should produce the detection results file.
        annopath: Path to annotations
           annopath.format(imagename) should be the xml annotations file.
        imagesetfile: Text file containing the list of images, one image per line.
        classname: Category name (duh)
        cachedir: Directory for caching the annotations
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
           (default False)
        """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name
        # cachedir caches the annotations in a pickle file
        # first load gt
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')
        # read list of images
        with open(self._imgsetpath.format(self._set_type), 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_rec(self._annopath % (imagename))
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(
                       i + 1, len(imagenames)))
            # save
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

        # read dets
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                           (BBGT[:, 2] - BBGT[:, 0]) *
                           (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = -1.

        return rec, prec, ap

