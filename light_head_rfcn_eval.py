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


#from scipy.misc import imread, imsave, imshow, imresize
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python import debug as tf_debug

import numpy as np

from net import xception_body
from utility import train_helper
from utility import eval_helper
from utility import metrics

from dataset import dataset_factory
from preprocessing import preprocessing_factory
from preprocessing import anchor_manipulator
from preprocessing import common_preprocessing

# hardware related configuration
tf.app.flags.DEFINE_integer(
    'num_readers', 16,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 48,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')
# scaffold related configuration
tf.app.flags.DEFINE_string(
    'data_dir', '../PASCAL/VOC_TF/VOC0712TF/',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc_0712', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'model_dir', './logs_light/',
    'The directory where the model will be stored.')
tf.app.flags.DEFINE_string(
    'debug_dir', './Debug_light/',
    'The directory where the debug files will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 500,
    'The frequency with which summaries are saved, in seconds.')
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
    'data_format', 'channels_first', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'select_threshold', 0.01, 'Class-specific confidence score threshold for selecting a box.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.3, 'nms threshold.')
tf.app.flags.DEFINE_integer(
    'nms_topk_percls', 200, 'Number of object for each class to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'nms_topk', 200, 'Number of total object to keep after NMS.')
tf.app.flags.DEFINE_float(
    'fg_ratio', 0.25, 'fore-ground ratio in the total proposals.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.55, 'Matching threshold in the loss function for proposals.')
tf.app.flags.DEFINE_float(
    'neg_threshold_high', 0.5, 'Matching threshold for the negtive examples in the loss function for proposals.')
tf.app.flags.DEFINE_float(
    'neg_threshold_low', 0., 'Matching threshold for the negtive examples in the loss function for proposals.')
tf.app.flags.DEFINE_integer(
    'rpn_anchors_per_image', 256, 'total rpn anchors to calculate loss and backprop.')
tf.app.flags.DEFINE_integer(
    'rpn_pre_nms_top_n', 4000, 'selected numbers of proposals to nms.')
tf.app.flags.DEFINE_integer(
    'rpn_post_nms_top_n', 300, 'keep numbers of proposals after nms.')
tf.app.flags.DEFINE_float(
    'rpn_min_size', 16*1./480, 'minsize threshold of proposals to be filtered for rpn.')
tf.app.flags.DEFINE_float(
    'rpn_nms_thres', 0.7, 'nms threshold for rpn.')
tf.app.flags.DEFINE_float(
    'rpn_fg_ratio', 0.5, 'fore-ground ratio in the total samples for rpn.')
tf.app.flags.DEFINE_float(
    'rpn_match_threshold', 0.7, 'Matching threshold in the loss function for rpn.')
tf.app.flags.DEFINE_float(
    'rpn_neg_threshold', 0.3, 'Matching threshold for the negtive examples in the loss function for rpn.')
# optimizer related configuration
tf.app.flags.DEFINE_float(
    'weight_decay', 0.0003, 'The weight decay on the model weights.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model/xception',#None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'model_scope', 'xception_lighthead',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_boolean(
    'run_on_cloud', True,
    'Wether we will train on cloud (pre-trained model will be placed in the "data_dir/cloud_checkpoint_path").')
tf.app.flags.DEFINE_string(
    'cloud_checkpoint_path', 'xception_model/xception_model.ckpt',
    'The path to a checkpoint from which to fine-tune.')
#CUDA_VISIBLE_DEVICES
FLAGS = tf.app.flags.FLAGS

LIB_NAME = 'ps_roi_align'

def load_op_module(lib_name):
  """
  Load TensorFlow operator library.
  """
  # use absolute path so that ops.py can be called from other directory
  if FLAGS.run_on_cloud:
      lib_path = os.path.join(FLAGS.data_dir, 'lib{0}.so'.format(lib_name))
      tf.gfile.Copy(lib_path, './' + 'lib{0}.so'.format(lib_name), overwrite=True)
  return tf.load_op_library('./' + 'lib{0}.so'.format(lib_name))

op_module = load_op_module(LIB_NAME)
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

from dataset import dataset_common
def gain_translate_table():
    label2name_table = {}
    for class_name, labels_pair in dataset_common.VOC_LABELS.items():
        label2name_table[labels_pair[0]] = class_name
    return label2name_table
label2name_table = gain_translate_table()

def input_pipeline():
    image_preprocessing_fn = lambda image_, shape_, glabels_, gbboxes_ : preprocessing_factory.get_preprocessing(
        'xception_lighthead', is_training=False)(image_, glabels_, gbboxes_, out_shape=[FLAGS.train_image_size] * 2, data_format=('NCHW' if FLAGS.data_format=='channels_first' else 'NHWC'))

    anchor_creator = anchor_manipulator.AnchorCreator([FLAGS.train_image_size] * 2,
                                                    layers_shapes = [(30, 30)],
                                                    anchor_scales = [[0.05, 0.1, 0.25, 0.45, 0.65, 0.85]],
                                                    extra_anchor_scales = [[]],
                                                    anchor_ratios = [[1., 2., .5]],
                                                    layer_steps = [16])

    def input_fn():
        all_anchors, num_anchors_list = anchor_creator.get_all_anchors()

        anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(all_anchors,
                                        num_classes = FLAGS.num_classes,
                                        allowed_borders = [0.],
                                        positive_threshold = FLAGS.rpn_match_threshold,
                                        ignore_threshold = FLAGS.rpn_neg_threshold,
                                        prior_scaling=[1., 1., 1., 1.],#[0.1, 0.1, 0.2, 0.2],
                                        rpn_fg_thres = FLAGS.match_threshold,
                                        rpn_bg_high_thres = FLAGS.neg_threshold_high,
                                        rpn_bg_low_thres = FLAGS.neg_threshold_low)

        num_readers_to_use = FLAGS.num_readers if FLAGS.run_on_cloud else 2
        num_preprocessing_threads_to_use = FLAGS.num_preprocessing_threads if FLAGS.run_on_cloud else 2

        list_from_batch, _ = dataset_factory.get_dataset(FLAGS.dataset_name,
                                                FLAGS.dataset_split_name,
                                                FLAGS.data_dir,
                                                image_preprocessing_fn,
                                                file_pattern = None,
                                                reader = None,
                                                batch_size = 1,
                                                num_readers = num_readers_to_use,
                                                num_preprocessing_threads = num_preprocessing_threads_to_use,
                                                num_epochs = 1,
                                                method = 'eval',
                                                anchor_encoder = anchor_encoder_decoder.encode_all_anchors)
        #print(list_from_batch[-4], list_from_batch[-3])
        return list_from_batch[-1], {'targets': list_from_batch[:-1],
                                    'rpn_decode_fn': lambda pred : anchor_encoder_decoder.decode_all_anchors([pred], squeeze_inner=True)[0],
                                    'head_decode_fn': lambda rois, pred : anchor_encoder_decoder.ext_decode_rois(rois, pred, head_prior_scaling=[1., 1., 1., 1.]),
                                    'num_anchors_list': num_anchors_list}
    return input_fn

def modified_smooth_l1(bbox_pred, bbox_targets, bbox_inside_weights = 1., bbox_outside_weights = 1., sigma = 1.):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    """
    sigma2 = sigma * sigma

    inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                              tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

    outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

    return outside_mul

if not FLAGS.run_on_cloud:
    from scipy.misc import imread, imsave, imshow, imresize
    from utility import draw_toolbox

def save_image_with_bbox(image, labels_, scores_, bboxes_):
    if not hasattr(save_image_with_bbox, "counter"):
        save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
    save_image_with_bbox.counter += 1

    img_to_draw = np.copy(image)#common_preprocessing.np_image_unwhitened(image))
    if not FLAGS.run_on_cloud:
        img_to_draw = draw_toolbox.bboxes_draw_on_img(img_to_draw, labels_, scores_, bboxes_, thickness=2)
        imsave(os.path.join(FLAGS.debug_dir, '{}.jpg').format(save_image_with_bbox.counter), img_to_draw)
    return save_image_with_bbox.counter#np.array([save_image_with_bbox.counter])


#[feature_h, feature_w, num_anchors, 4]
# only support batch_size 1
def bboxes_eval(org_image, image_shape, bbox_img, cls_pred_logits, bboxes_pred, glabels_raw, gbboxes_raw, isdifficult, num_classes):
    # Performing post-processing on CPU: loop-intensive, usually more efficient.
    cls_pred_prob = tf.nn.softmax(tf.reshape(cls_pred_logits, [-1, num_classes]))
    bboxes_pred = tf.reshape(bboxes_pred, [-1, 4])
    glabels_raw = tf.reshape(glabels_raw, [-1])
    gbboxes_raw = tf.reshape(gbboxes_raw, [-1, 4])
    gbboxes_raw = tf.boolean_mask(gbboxes_raw, glabels_raw > 0)
    glabels_raw = tf.boolean_mask(glabels_raw, glabels_raw > 0)
    isdifficult = tf.reshape(isdifficult, [-1])

    with tf.device('/device:CPU:0'):
        selected_scores, selected_bboxes = eval_helper.tf_bboxes_select([cls_pred_prob], [bboxes_pred], FLAGS.select_threshold, num_classes, scope='xdet_v2_select')

        selected_bboxes = eval_helper.bboxes_clip(bbox_img, selected_bboxes)
        selected_scores, selected_bboxes = eval_helper.filter_boxes(selected_scores, selected_bboxes, 0.03, image_shape, [FLAGS.train_image_size] * 2, keep_top_k = FLAGS.nms_topk * 2)

        # Resize bboxes to original image shape.
        selected_bboxes = eval_helper.bboxes_resize(bbox_img, selected_bboxes)

        selected_scores, selected_bboxes = eval_helper.bboxes_sort(selected_scores, selected_bboxes, top_k=FLAGS.nms_topk * 2)

        # Apply NMS algorithm.
        selected_scores, selected_bboxes = eval_helper.bboxes_nms_batch(selected_scores, selected_bboxes,
                                 nms_threshold=FLAGS.nms_threshold,
                                 keep_top_k=FLAGS.nms_topk)

        # label_scores, pred_labels, bboxes_pred = eval_helper.xdet_predict(bbox_img, cls_pred_prob, bboxes_pred, image_shape, FLAGS.train_image_size, FLAGS.nms_threshold, FLAGS.select_threshold, FLAGS.nms_topk, num_classes, nms_mode='union')

        dict_metrics = {}
        # Compute TP and FP statistics.
        num_gbboxes, tp, fp = eval_helper.bboxes_matching_batch(selected_scores.keys(), selected_scores, selected_bboxes, glabels_raw, gbboxes_raw, isdifficult)

        # FP and TP metrics.
        tp_fp_metric = metrics.streaming_tp_fp_arrays(num_gbboxes, tp, fp, selected_scores)
        metrics_name = ('nobjects', 'ndetections', 'tp', 'fp', 'scores')
        for c in tp_fp_metric[0].keys():
            for _ in range(len(tp_fp_metric[0][c])):
                dict_metrics['tp_fp_%s_%s' % (label2name_table[c], metrics_name[_])] = (tp_fp_metric[0][c][_],
                                                tp_fp_metric[1][c][_])

        # Add to summaries precision/recall values.
        aps_voc07 = {}
        aps_voc12 = {}
        for c in tp_fp_metric[0].keys():
            # Precison and recall values.
            prec, rec = metrics.precision_recall(*tp_fp_metric[0][c])

            # Average precision VOC07.
            v = metrics.average_precision_voc07(prec, rec)
            summary_name = 'AP_VOC07/%s' % c
            op = tf.summary.scalar(summary_name, v)
            # op = tf.Print(op, [v], summary_name)
            #tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            aps_voc07[c] = v

            # Average precision VOC12.
            v = metrics.average_precision_voc12(prec, rec)
            summary_name = 'AP_VOC12/%s' % c
            op = tf.summary.scalar(summary_name, v)
            # op = tf.Print(op, [v], summary_name)
            #tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            aps_voc12[c] = v

        # Mean average precision VOC07.
        summary_name = 'AP_VOC07/mAP'
        mAP = tf.add_n(list(aps_voc07.values())) / len(aps_voc07)
        mAP = tf.Print(mAP, [mAP], summary_name)
        op = tf.summary.scalar(summary_name, mAP)
        #tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # Mean average precision VOC12.
        summary_name = 'AP_VOC12/mAP'
        mAP = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
        mAP = tf.Print(mAP, [mAP], summary_name)
        op = tf.summary.scalar(summary_name, mAP)
        #tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        labels_list = []
        for k, v in selected_scores.items():
            labels_list.append(tf.ones_like(v, tf.int32) * k)
        save_image_op = tf.py_func(save_image_with_bbox,
                                    [org_image,
                                    tf.concat(labels_list, axis=0),
                                    #tf.convert_to_tensor(list(selected_scores.keys()), dtype=tf.int64),
                                    tf.concat(list(selected_scores.values()), axis=0),
                                    tf.concat(list(selected_bboxes.values()), axis=0)],
                                    tf.int64, stateful=True)

        #dict_metrics['save_image_with_bboxes'] = save_image_count#tf.tuple([save_image_count, save_image_count_update_op])
    # for i, v in enumerate(l_precisions):
    #     summary_name = 'eval/precision_at_recall_%.2f' % LIST_RECALLS[i]
    #     op = tf.summary.scalar(summary_name, v, collections=[])
    #     op = tf.Print(op, [v], summary_name)
    #     tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    # print(dict_metrics)
    # metric_names = dict_metrics.keys()
    # value_ops, update_ops = zip(*dict_metrics.values())
    # return dict(zip(metric_names, update_ops)), save_image_op

    return dict_metrics, save_image_op

def lighr_head_model_fn(features, labels, mode, params):
    """Our model_fn for ResNet to be used with our Estimator."""
    num_anchors_list = labels['num_anchors_list']
    num_feature_layers = len(num_anchors_list)

    shape = labels['targets'][-1]
    if mode != tf.estimator.ModeKeys.TRAIN:
        org_image = labels['targets'][-2]
        isdifficult = labels['targets'][-3]
        bbox_img = labels['targets'][-4]
        gbboxes_raw = labels['targets'][-5]
        glabels_raw = labels['targets'][-6]

    glabels = labels['targets'][:num_feature_layers][0]
    gtargets = labels['targets'][num_feature_layers : 2 * num_feature_layers][0]
    gscores = labels['targets'][2 * num_feature_layers : 3 * num_feature_layers][0]

    #features = tf.ones([4,480,480,3]) * 0.5
    with tf.variable_scope(params['model_scope'], default_name = None, values = [features], reuse=tf.AUTO_REUSE):
        rpn_feat_map, backbone_feat = xception_body.XceptionBody(features, params['num_classes'], is_training=(mode == tf.estimator.ModeKeys.TRAIN), data_format=params['data_format'])
        #rpn_feat_map = tf.Print(rpn_feat_map,[tf.shape(rpn_feat_map), rpn_feat_map,backbone_feat])
        rpn_cls_score, rpn_bbox_pred = xception_body.get_rpn(rpn_feat_map, num_anchors_list[0], (mode == tf.estimator.ModeKeys.TRAIN), params['data_format'], 'rpn_head')

        large_sep_feature = xception_body.large_sep_kernel(backbone_feat, 256, 10 * 7 * 7, (mode == tf.estimator.ModeKeys.TRAIN), params['data_format'], 'large_sep_feature')

        if params['data_format'] == 'channels_first':
            rpn_cls_score = tf.transpose(rpn_cls_score, [0, 2, 3, 1])
            rpn_bbox_pred = tf.transpose(rpn_bbox_pred, [0, 2, 3, 1])

        rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
        rpn_object_score = tf.nn.softmax(rpn_cls_score)[:, -1]

        rpn_object_score = tf.reshape(rpn_object_score, [1, -1])
        rpn_location_pred = tf.reshape(rpn_bbox_pred, [1, -1, 4])

        rpn_bboxes_pred = labels['rpn_decode_fn'](rpn_location_pred)

        proposals_bboxes = xception_body.get_proposals(rpn_object_score, rpn_bboxes_pred, None, params['rpn_pre_nms_top_n'], params['rpn_post_nms_top_n'], params['nms_threshold'], params['rpn_min_size'], (mode == tf.estimator.ModeKeys.TRAIN), params['data_format'])
        #proposals_targets = tf.Print(proposals_targets, [proposals_targets], message='proposals_targets0:')

        cls_score, bboxes_reg = xception_body.get_head(large_sep_feature, lambda input_, bboxes_, grid_width_, grid_height_ : ps_roi_align(input_, bboxes_, grid_width_, grid_height_, pool_method), 7, 7, None, proposals_bboxes, params['num_classes'], (mode == tf.estimator.ModeKeys.TRAIN), False, 0, params['data_format'], 'final_head')

        head_bboxes_pred = labels['head_decode_fn'](proposals_bboxes, bboxes_reg)

        head_cls_score = tf.reshape(cls_score, [-1, params['num_classes']])
        head_cls_score = tf.nn.softmax(head_cls_score)
        head_bboxes_pred = tf.reshape(head_bboxes_pred, [-1, 4])

        shape = tf.squeeze(shape, axis = 0)
        glabels = tf.squeeze(glabels, axis = 0)
        gtargets = tf.squeeze(gtargets, axis = 0)
        gscores = tf.squeeze(gscores, axis = 0)
        if mode != tf.estimator.ModeKeys.TRAIN:
            org_image = tf.squeeze(org_image, axis = 0)
            isdifficult = tf.squeeze(isdifficult, axis = 0)
            gbboxes_raw = tf.squeeze(gbboxes_raw, axis = 0)
            glabels_raw = tf.squeeze(glabels_raw, axis = 0)
            bbox_img = tf.squeeze(bbox_img, axis = 0)

        eval_ops, save_image_op = bboxes_eval(org_image, shape, bbox_img, cls_score, head_bboxes_pred, glabels_raw, gbboxes_raw, isdifficult, params['num_classes'])
        _ = tf.identity(save_image_op, name='save_image_with_bboxes_op')

    with tf.control_dependencies([save_image_op]):
        weight_decay_loss = params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'batch_normalization' not in v.name])

    predictions = {
        'classes': tf.argmax(head_cls_score, axis=-1),
        'probabilities': tf.reduce_max(head_cls_score, axis=-1),
        'bboxes_predict': head_bboxes_pred,
        'saved_image_index': save_image_op }

    summary_hook = tf.train.SummarySaverHook(
                        save_secs=FLAGS.save_summary_steps,
                        output_dir=FLAGS.model_dir,
                        summary_op=tf.summary.merge_all())

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                        evaluation_hooks = [summary_hook],
                        loss=weight_decay_loss, eval_metric_ops=eval_ops)
    else:
        raise ValueError('This script only support predict mode!')

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False, intra_op_parallelism_threads = FLAGS.num_cpu_threads, inter_op_parallelism_threads = FLAGS.num_cpu_threads, gpu_options = gpu_options)

    # Set up RunConfig
    run_config = tf.estimator.RunConfig().replace(
                                        save_checkpoints_secs=None).replace(
                                        save_checkpoints_steps=None).replace(
                                        save_summary_steps=FLAGS.save_summary_steps).replace(
                                        keep_checkpoint_max=5).replace(
                                        log_step_count_steps=FLAGS.log_every_n_steps).replace(
                                        session_config=config)

    light_head_detector = tf.estimator.Estimator(
        model_fn=lighr_head_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'resnet_size': FLAGS.resnet_size,
            'data_format': FLAGS.data_format,
            'model_scope': FLAGS.model_scope,
            'num_classes': FLAGS.num_classes,
            'roi_one_image': FLAGS.roi_one_image,
            'fg_ratio': FLAGS.fg_ratio,
            'match_threshold': FLAGS.match_threshold,
            'neg_threshold_high': FLAGS.neg_threshold_high,
            'neg_threshold_low': FLAGS.neg_threshold_low,
            'rpn_anchors_per_image': FLAGS.rpn_anchors_per_image,
            'rpn_pre_nms_top_n': FLAGS.rpn_pre_nms_top_n,
            'rpn_post_nms_top_n': FLAGS.rpn_post_nms_top_n,
            'nms_threshold': FLAGS.nms_threshold,
            'rpn_min_size': FLAGS.rpn_min_size,
            'rpn_nms_thres': FLAGS.rpn_nms_thres,
            'rpn_fg_ratio': FLAGS.rpn_fg_ratio,
            'rpn_match_threshold': FLAGS.rpn_match_threshold,
            'rpn_neg_threshold': FLAGS.rpn_neg_threshold,
            'weight_decay': FLAGS.weight_decay,
        })


    tensors_to_log = {
        'saved_image_index':'xception_lighthead/save_image_with_bboxes_op'
    }

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps)

    print('Starting evaluate cycle.')

    light_head_detector.evaluate(input_fn=input_pipeline(), hooks=[logging_hook], checkpoint_path=train_helper.get_latest_checkpoint_for_evaluate(FLAGS))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
