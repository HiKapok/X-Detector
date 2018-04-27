# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python import debug as tf_debug

from scipy.misc import imread, imsave, imshow, imresize
import numpy as np

from net import xception_body
from utility import train_helper
from utility import eval_helper
from utility import metrics
from utility import draw_toolbox

from dataset import dataset_factory
from preprocessing import preprocessing_factory
from preprocessing import anchor_manipulator
from preprocessing import common_preprocessing

# scaffold related configuration
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'debug_dir', './debug/',
    'The directory where the debug files will be stored.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', 480,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'resnet_size', 50,
    'The size of the ResNet model to use.')
tf.app.flags.DEFINE_integer(
    'roi_one_image', 64,
    'Batch size of RoIs for training in the second stage.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_last', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'select_threshold', 0.5, 'Class-specific confidence score threshold for selecting a box.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.3, 'nms threshold.')
tf.app.flags.DEFINE_integer(
    'nms_topk_percls', 20, 'Number of object for each class to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'nms_topk', 20, 'Number of total object to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'rpn_anchors_per_image', 256, 'total rpn anchors to calculate loss and backprop.')
tf.app.flags.DEFINE_integer(
    'rpn_pre_nms_top_n', 5000, 'selected numbers of proposals to nms.')
tf.app.flags.DEFINE_integer(
    'rpn_post_nms_top_n', 1000, 'keep numbers of proposals after nms.')
tf.app.flags.DEFINE_float(
    'rpn_min_size', 16*1./480, 'minsize threshold of proposals to be filtered for rpn.')
tf.app.flags.DEFINE_float(
    'rpn_nms_thres', 0.7, 'nms threshold for rpn.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', '/media/rs/7A0EE8880EE83EAF/Detections/DetInDet_Tensorflow/logs_light/model.ckpt-122320',#None,
    'The path of the checkpoint used to test new images.')
tf.app.flags.DEFINE_string(
    'model_scope', 'xception_lighthead',
    'Model scope name used to replace the name_scope in checkpoint.')
#CUDA_VISIBLE_DEVICES
FLAGS = tf.app.flags.FLAGS

LIB_NAME = 'ps_roi_align'

op_module = tf.load_op_library('./' + 'lib{0}.so'.format(LIB_NAME))
ps_roi_align = op_module.ps_roi_align
pool_method = 'max'

@ops.RegisterGradient("PsRoiAlign")
def _ps_roi_align_grad(op, grad, _):
  '''The gradients for `PsRoiAlign`.
  '''
  inputs_features = op.inputs[0]
  rois = op.inputs[1]
  pooled_features_grad = op.outputs[0]
  pooled_index = op.outputs[1]
  grid_dim_width = op.get_attr('grid_dim_width')
  grid_dim_height = op.get_attr('grid_dim_height')

  #return [tf.ones_like(inputs_features), None]
  return [op_module.ps_roi_align_grad(inputs_features, rois, grad, pooled_index, grid_dim_width, grid_dim_height, pool_method), None]

def main(_):
    with tf.Graph().as_default():
        image_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        shape_input = tf.placeholder(tf.int32, shape=(2,))

        features = common_preprocessing.light_head_preprocess_for_test(image_input, [FLAGS.train_image_size] * 2, data_format=('NCHW' if FLAGS.data_format=='channels_first' else 'NHWC'))

        features = tf.expand_dims(features, axis=0)

        anchor_creator = anchor_manipulator.AnchorCreator([FLAGS.train_image_size] * 2,
                                                        layers_shapes = [(30, 30)],
                                                        anchor_scales = [[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
                                                        extra_anchor_scales = [[0.1]],
                                                        anchor_ratios = [[1., 2., .5]],
                                                        layer_steps = [16])

        all_anchors, num_anchors_list = anchor_creator.get_all_anchors()

        anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(all_anchors,
                                        num_classes = FLAGS.num_classes,
                                        allowed_borders = None,
                                        positive_threshold = None,
                                        ignore_threshold = None,
                                        prior_scaling=[1., 1., 1., 1.])

        with tf.variable_scope(FLAGS.model_scope, default_name = None, values = [features], reuse=tf.AUTO_REUSE):
            rpn_feat_map, backbone_feat = xception_body.XceptionBody(features, FLAGS.num_classes, is_training=False, data_format=FLAGS.data_format)
            #rpn_feat_map = tf.Print(rpn_feat_map,[tf.shape(rpn_feat_map), rpn_feat_map,backbone_feat])
            rpn_cls_score, rpn_bbox_pred = xception_body.get_rpn(rpn_feat_map, num_anchors_list[0], False, FLAGS.data_format, 'rpn_head')

            large_sep_feature = xception_body.large_sep_kernel(backbone_feat, 256, 10 * 7 * 7, False, FLAGS.data_format, 'large_sep_feature')

            if FLAGS.data_format == 'channels_first':
                rpn_cls_score = tf.transpose(rpn_cls_score, [0, 2, 3, 1])
                rpn_bbox_pred = tf.transpose(rpn_bbox_pred, [0, 2, 3, 1])

            rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
            rpn_object_score = tf.nn.softmax(rpn_cls_score)[:, -1]

            rpn_object_score = tf.reshape(rpn_object_score, [1, -1])
            rpn_location_pred = tf.reshape(rpn_bbox_pred, [1, -1, 4])

            rpn_bboxes_pred = anchor_encoder_decoder.decode_all_anchors([rpn_location_pred], squeeze_inner=True)[0]

            proposals_bboxes = xception_body.get_proposals(rpn_object_score, rpn_bboxes_pred, None, FLAGS.rpn_pre_nms_top_n, FLAGS.rpn_post_nms_top_n, FLAGS.rpn_nms_thres, FLAGS.rpn_min_size, False, FLAGS.data_format)

            cls_score, bboxes_reg = xception_body.get_head(large_sep_feature, lambda input_, bboxes_, grid_width_, grid_height_ : ps_roi_align(input_, bboxes_, grid_width_, grid_height_, pool_method), 7, 7, None, proposals_bboxes, FLAGS.num_classes, False, False, 0, FLAGS.data_format, 'final_head')

            head_bboxes_pred = anchor_encoder_decoder.ext_decode_rois(proposals_bboxes, bboxes_reg, head_prior_scaling=[1., 1., 1., 1.])

            head_cls_score = tf.reshape(cls_score, [-1, FLAGS.num_classes])
            head_cls_score = tf.nn.softmax(head_cls_score)
            head_bboxes_pred = tf.reshape(head_bboxes_pred, [-1, 4])

            with tf.device('/device:CPU:0'):
                selected_scores, selected_bboxes = eval_helper.tf_bboxes_select([head_cls_score], [head_bboxes_pred], FLAGS.select_threshold, FLAGS.num_classes, scope='xdet_v2_select')

                selected_bboxes = eval_helper.bboxes_clip(tf.constant([0., 0., 1., 1.]), selected_bboxes)
                selected_scores, selected_bboxes = eval_helper.filter_boxes(selected_scores, selected_bboxes, 0.03, shape_input, [FLAGS.train_image_size] * 2, keep_top_k = FLAGS.nms_topk * 2)

                # Resize bboxes to original image shape.
                selected_bboxes = eval_helper.bboxes_resize(tf.constant([0., 0., 1., 1.]), selected_bboxes)

                selected_scores, selected_bboxes = eval_helper.bboxes_sort(selected_scores, selected_bboxes, top_k=FLAGS.nms_topk * 2)

                # Apply NMS algorithm.
                selected_scores, selected_bboxes = eval_helper.bboxes_nms_batch(selected_scores, selected_bboxes,
                                         nms_threshold=FLAGS.nms_threshold,
                                         keep_top_k=FLAGS.nms_topk)

                labels_list = []
                for k, v in selected_scores.items():
                    labels_list.append(tf.ones_like(v, tf.int32) * k)
                all_labels = tf.concat(labels_list, axis=0)
                all_scores = tf.concat(list(selected_scores.values()), axis=0)
                all_bboxes = tf.concat(list(selected_bboxes.values()), axis=0)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver.restore(sess, FLAGS.checkpoint_path)

            np_image = imread('./demo/test.jpg')
            labels_, scores_, bboxes_ = sess.run([all_labels, all_scores, all_bboxes], feed_dict = {image_input : np_image, shape_input : np_image.shape[:-1]})

            img_to_draw = draw_toolbox.bboxes_draw_on_img(np_image, labels_, scores_, bboxes_, thickness=2)
            imsave(os.path.join(FLAGS.debug_dir, 'test_out.jpg'), img_to_draw)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
