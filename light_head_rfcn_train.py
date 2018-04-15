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

from net import xception_body
from utility import train_helper

from dataset import dataset_factory
from preprocessing import preprocessing_factory
from preprocessing import anchor_manipulator

#--run_on_cloud=False --data_format=channels_last --batch_size=1 --log_every_n_steps=1
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
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 500,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_secs', 7200,
    'The frequency with which the model is saved, in seconds.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', 480,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'resnet_size', 50,
    'The size of the ResNet model to use.')
tf.app.flags.DEFINE_integer(
    'train_epochs', None,
    'The number of epochs to use for training.')
tf.app.flags.DEFINE_integer(
    'batch_size', 8,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_boolean(
    'using_ohem', True, 'Wether to use OHEM.')
tf.app.flags.DEFINE_integer(
    'ohem_roi_one_image', 32,
    'Batch size of RoIs for training in the second stage after OHEM.')
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
    'nms_threshold', 0.3, 'nms threshold.')
tf.app.flags.DEFINE_float(
    'fg_ratio', 0.25, 'fore-ground ratio in the total proposals.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.53, 'Matching threshold in the loss function for proposals.')
tf.app.flags.DEFINE_float(
    'neg_threshold_high', 0.5, 'Matching threshold for the negtive examples in the loss function for proposals.')
tf.app.flags.DEFINE_float(
    'neg_threshold_low', 0., 'Matching threshold for the negtive examples in the loss function for proposals.')
tf.app.flags.DEFINE_integer(
    'rpn_anchors_per_image', 256, 'total rpn anchors to calculate loss and backprop.')
tf.app.flags.DEFINE_integer(
    'rpn_pre_nms_top_n', 10000, 'selected numbers of proposals to nms.')
tf.app.flags.DEFINE_integer(
    'rpn_post_nms_top_n', 1800, 'keep numbers of proposals after nms.')
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
    'weight_decay', 0.00002, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
# for learning rate exponential_decay
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.96, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'decay_steps', 1000,
    'Number of epochs after which learning rate decays.')
# for learning rate piecewise_constant decay
tf.app.flags.DEFINE_string(
    'decay_boundaries', '60000, 80000',
    'Learning rate decay boundaries by global_step (comma-separated list).')
tf.app.flags.DEFINE_string(
    'lr_decay_factors', '1, 0.8, 0.1',
    'The values of learning_rate decay factor for each segment between boundaries (comma-separated list).')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model/xception',#None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', '',
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'model_scope', 'xception_lighthead',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'xception_lighthead/rpn_head, xception_lighthead/large_sep_feature, xception_lighthead/final_head',#None
    'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_boolean(
    'run_on_cloud', True,
    'Wether we will train on cloud (pre-trained model will be placed in the "data_dir/cloud_checkpoint_path").')
tf.app.flags.DEFINE_string(
    'cloud_checkpoint_path', 'xception_model/xception_model.ckpt',
    'The path to a checkpoint from which to fine-tune.')
#CUDA_VISIBLE_DEVICES
FLAGS = tf.app.flags.FLAGS

LIB_NAME = 'ps_roi_align'

if FLAGS.run_on_cloud:
    # when run on cloud we have no access to /tmp directory, so we change TMPDIR first
    import subprocess
    os.environ["TMPDIR"] = os.getcwd()
    cmake_process = subprocess.Popen(str("cmake " + os.path.join(os.getcwd(), 'cpp/PSROIPooling/') + " ").split(), stdout=subprocess.PIPE, cwd=os.path.join(os.getcwd(), 'cpp/PSROIPooling/build'))
    output, _ = cmake_process.communicate()
    print(output)
    make_process = subprocess.Popen(str("make").split(), stdout=subprocess.PIPE, cwd=os.path.join(os.getcwd(), 'cpp/PSROIPooling/build'))
    output, _ = make_process.communicate()
    print(output)
    print(os.getcwd())
    tf.gfile.Copy(os.path.join(os.getcwd(), 'cpp/PSROIPooling/build/libps_roi_align.so'), os.path.join(FLAGS.data_dir, 'libps_roi_align.so'), overwrite=True)

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

def input_pipeline():
    image_preprocessing_fn = lambda image_, shape_, glabels_, gbboxes_ : preprocessing_factory.get_preprocessing(
        'xception_lighthead', is_training=True)(image_, glabels_, gbboxes_, out_shape=[FLAGS.train_image_size] * 2, data_format=('NCHW' if FLAGS.data_format=='channels_first' else 'NHWC'))

    anchor_creator = anchor_manipulator.AnchorCreator([FLAGS.train_image_size] * 2,
                                                    layers_shapes = [(30, 30)],
                                                    anchor_scales = [[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
                                                    extra_anchor_scales = [[0.1]],
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
        list_from_batch, _ = dataset_factory.get_dataset(FLAGS.dataset_name,
                                                FLAGS.dataset_split_name,
                                                FLAGS.data_dir,
                                                image_preprocessing_fn,
                                                file_pattern = None,
                                                reader = None,
                                                batch_size = FLAGS.batch_size,
                                                num_readers = FLAGS.num_readers,
                                                num_preprocessing_threads = FLAGS.num_preprocessing_threads,
                                                num_epochs = FLAGS.train_epochs,
                                                anchor_encoder = anchor_encoder_decoder.encode_all_anchors)
        #print(list_from_batch[-4], list_from_batch[-3])
        return list_from_batch[-1], {'targets': list_from_batch[:-1],
                                    'rpn_decode_fn': lambda pred : anchor_encoder_decoder.decode_all_anchors([pred], squeeze_inner=True)[0],
                                    'head_decode_fn': lambda rois, pred : anchor_encoder_decoder.ext_decode_rois(rois, pred, head_prior_scaling=[1., 1., 1., 1.]),
                                    'rpn_encode_fn': lambda rois : anchor_encoder_decoder.ext_encode_rois(rois, list_from_batch[-4], list_from_batch[-3], FLAGS.roi_one_image, FLAGS.fg_ratio, 0.1, head_prior_scaling=[1., 1., 1., 1.]),
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

def lighr_head_model_fn(features, labels, mode, params):
    """Our model_fn for ResNet to be used with our Estimator."""
    num_anchors_list = labels['num_anchors_list']
    num_feature_layers = len(num_anchors_list)

    shape = labels['targets'][-1]
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


        #with tf.device('/cpu:0'):
        rpn_object_score = tf.reshape(rpn_object_score, [params['batch_size'], -1])
        rpn_location_pred = tf.reshape(rpn_bbox_pred, [params['batch_size'], -1, 4])

        #rpn_location_pred = tf.Print(rpn_location_pred,[tf.shape(rpn_location_pred), rpn_location_pred])

        rpn_bboxes_pred = labels['rpn_decode_fn'](rpn_location_pred)

        #rpn_bboxes_pred = tf.Print(rpn_bboxes_pred,[tf.shape(rpn_bboxes_pred), rpn_bboxes_pred])
        # rpn loss here
        cls_pred = tf.reshape(rpn_cls_score, [-1, 2])
        location_pred = tf.reshape(rpn_bbox_pred, [-1, 4])
        glabels = tf.reshape(glabels, [-1])
        gscores = tf.reshape(gscores, [-1])
        gtargets = tf.reshape(gtargets, [-1, 4])

        expected_num_fg_rois = tf.cast(tf.round(tf.cast(params['batch_size'] * params['rpn_anchors_per_image'], tf.float32) * params['rpn_fg_ratio']), tf.int32)

        def select_samples(cls_pred, location_pred, glabels, gscores, gtargets):
            def upsampel_impl(now_count, need_count):
                # sample with replacement
                left_count = need_count - now_count
                select_indices = tf.random_shuffle(tf.range(now_count))[:tf.floormod(left_count, now_count)]
                select_indices = tf.concat([tf.tile(tf.range(now_count), [tf.floor_div(left_count, now_count) + 1]), select_indices], axis = 0)

                return select_indices
            def downsample_impl(now_count, need_count):
                # downsample with replacement
                select_indices = tf.random_shuffle(tf.range(now_count))[:need_count]
                return select_indices

            positive_mask = glabels > 0
            positive_indices = tf.squeeze(tf.where(positive_mask), axis = -1)
            n_positives = tf.shape(positive_indices)[0]
            # either downsample or take all
            fg_select_indices = tf.cond(n_positives < expected_num_fg_rois, lambda : positive_indices, lambda : tf.gather(positive_indices, downsample_impl(n_positives, expected_num_fg_rois)))
            # now the all rois taken as positive is min(n_positives, expected_num_fg_rois)

            #negtive_mask = tf.logical_and(tf.logical_and(tf.logical_not(tf.logical_or(positive_mask, glabels < 0)), gscores < params['rpn_neg_threshold']), gscores > 0.)
            negtive_mask = tf.logical_and(tf.equal(glabels, 0), gscores > 0.)
            negtive_indices = tf.squeeze(tf.where(negtive_mask), axis = -1)
            n_negtives = tf.shape(negtive_indices)[0]

            expected_num_bg_rois = params['batch_size'] * params['rpn_anchors_per_image'] - tf.minimum(n_positives, expected_num_fg_rois)
            # either downsample or take all
            bg_select_indices = tf.cond(n_negtives < expected_num_bg_rois, lambda : negtive_indices, lambda : tf.gather(negtive_indices, downsample_impl(n_negtives, expected_num_bg_rois)))
            # now the all rois taken as positive is min(n_negtives, expected_num_bg_rois)

            keep_indices = tf.concat([fg_select_indices, bg_select_indices], axis = 0)
            n_keeps = tf.shape(keep_indices)[0]
            # now n_keeps must be equal or less than rpn_anchors_per_image
            final_keep_indices = tf.cond(n_keeps < params['batch_size'] * params['rpn_anchors_per_image'], lambda : tf.gather(keep_indices, upsampel_impl(n_keeps, params['batch_size'] * params['rpn_anchors_per_image'])), lambda : keep_indices)

            return tf.gather(cls_pred, final_keep_indices), tf.gather(location_pred, final_keep_indices), tf.cast(tf.gather(tf.clip_by_value(glabels, 0, params['num_classes']), final_keep_indices) > 0, tf.int64), tf.gather(gscores, final_keep_indices), tf.gather(gtargets, final_keep_indices)

        cls_pred, location_pred, glabels, gscores, gtargets = select_samples(cls_pred, location_pred, glabels, gscores, gtargets)

        # Calculate loss, which includes softmax cross entropy and L2 regularization.
        rpn_cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=glabels, logits=cls_pred)

        # Create a tensor named cross_entropy for logging purposes.
        rpn_cross_entropy = tf.identity(rpn_cross_entropy, name='rpn_cross_entropy_loss')
        tf.summary.scalar('rpn_cross_entropy_loss', rpn_cross_entropy)

        total_positive_mask = (glabels > 0)
        gtargets = tf.boolean_mask(gtargets, tf.stop_gradient(total_positive_mask))
        location_pred = tf.boolean_mask(location_pred, tf.stop_gradient(total_positive_mask))
        #gtargets = tf.Print(gtargets, [gtargets], message='gtargets:', summarize=100)

        rpn_l1_distance = modified_smooth_l1(location_pred, gtargets, sigma=1.)
        rpn_loc_loss = tf.reduce_mean(tf.reduce_sum(rpn_l1_distance, axis=-1)) * params['rpn_fg_ratio']
        rpn_loc_loss = tf.identity(rpn_loc_loss, name='rpn_location_loss')
        tf.summary.scalar('rpn_location_loss', rpn_loc_loss)
        tf.losses.add_loss(rpn_loc_loss)

        rpn_loss = tf.identity(rpn_loc_loss + rpn_cross_entropy, name='rpn_loss')
        tf.summary.scalar('rpn_loss', rpn_loss)
        #print(rpn_loc_loss)

        proposals_bboxes, proposals_targets, proposals_labels, proposals_scores = xception_body.get_proposals(rpn_object_score, rpn_bboxes_pred, labels['rpn_encode_fn'], params['rpn_pre_nms_top_n'], params['rpn_post_nms_top_n'], params['rpn_nms_thres'], params['rpn_min_size'], (mode == tf.estimator.ModeKeys.TRAIN), params['data_format'])
        #proposals_targets = tf.Print(proposals_targets, [proposals_targets], message='proposals_targets0:')
        def head_loss_func(cls_score, bboxes_reg, select_indices, proposals_targets, proposals_labels):
            if select_indices is not None:
                proposals_targets = tf.gather(proposals_targets, select_indices, axis=1)
                proposals_labels = tf.gather(proposals_labels, select_indices, axis=1)
            # Calculate loss, which includes softmax cross entropy and L2 regularization.
            head_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=proposals_labels, logits=cls_score)

            total_positive_mask = tf.cast((proposals_labels > 0), tf.float32)
            # proposals_targets = tf.boolean_mask(proposals_targets, tf.stop_gradient(total_positive_mask))
            # bboxes_reg = tf.boolean_mask(bboxes_reg, tf.stop_gradient(total_positive_mask))
            head_loc_loss = modified_smooth_l1(bboxes_reg, proposals_targets, sigma=1.)
            head_loc_loss = tf.reduce_sum(head_loc_loss, axis=-1) * total_positive_mask
            if (params['using_ohem'] and (select_indices is not None)) or (not params['using_ohem']):
                head_cross_entropy_loss = tf.reduce_mean(head_cross_entropy)
                head_cross_entropy_loss = tf.identity(head_cross_entropy_loss, name='head_cross_entropy_loss')
                tf.summary.scalar('head_cross_entropy_loss', head_cross_entropy_loss)

                head_location_loss = tf.reduce_mean(head_loc_loss)#/params['fg_ratio']
                head_location_loss = tf.identity(head_location_loss, name='head_location_loss')
                tf.summary.scalar('head_location_loss', head_location_loss)

            return head_cross_entropy + head_loc_loss#/params['fg_ratio']

        head_loss = xception_body.get_head(large_sep_feature, lambda input_, bboxes_, grid_width_, grid_height_ : ps_roi_align(input_, bboxes_, grid_width_, grid_height_, pool_method), 7, 7, lambda cls, bbox, indices : head_loss_func(cls, bbox, indices, proposals_targets, proposals_labels), proposals_bboxes, params['num_classes'], (mode == tf.estimator.ModeKeys.TRAIN), params['using_ohem'], params['ohem_roi_one_image'], params['data_format'], 'final_head')

        # Create a tensor named cross_entropy for logging purposes.
        head_loss = tf.identity(head_loss, name='head_loss')
        tf.summary.scalar('head_loss', head_loss)

        tf.losses.add_loss(head_loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=None)

    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = rpn_cross_entropy + rpn_loc_loss + head_loss + params['weight_decay'] * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if (('batch_normalization' not in v.name) and ('_bn' not in v.name))])#_bn
    total_loss = tf.identity(loss, name='total_loss')

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        lr_values = [params['learning_rate'] * decay for decay in params['lr_decay_factors']]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
                                                    [int(_) for _ in params['decay_boundaries']],
                                                    lr_values)
        truncated_learning_rate = tf.maximum(learning_rate, tf.constant(params['end_learning_rate'], dtype=learning_rate.dtype))
        # Create a tensor named learning_rate for logging purposes.
        tf.identity(truncated_learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', truncated_learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=truncated_learning_rate,
                                                momentum=params['momentum'])

        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=None,
          loss=loss,
          train_op=train_op,
          eval_metric_ops=None,
          scaffold = tf.train.Scaffold(init_fn=train_helper.get_init_fn_for_scaffold(FLAGS)))

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False, intra_op_parallelism_threads = FLAGS.num_cpu_threads, inter_op_parallelism_threads = FLAGS.num_cpu_threads, gpu_options = gpu_options)

    #trace_level=tf.RunOptions.FULL_TRACE
    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(
                                        save_checkpoints_secs=FLAGS.save_checkpoints_secs).replace(
                                        save_checkpoints_steps=None).replace(
                                        save_summary_steps=FLAGS.save_summary_steps).replace(
                                        keep_checkpoint_max=5).replace(
                                        log_step_count_steps=FLAGS.log_every_n_steps).replace(
                                        session_config=config)

    xdetector = tf.estimator.Estimator(
        model_fn=lighr_head_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'resnet_size': FLAGS.resnet_size,
            'data_format': FLAGS.data_format,
            'model_scope': FLAGS.model_scope,
            'batch_size': FLAGS.batch_size,
            'num_classes': FLAGS.num_classes,
            'ohem_roi_one_image': FLAGS.ohem_roi_one_image,
            'using_ohem': FLAGS.using_ohem,
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
            'momentum': FLAGS.momentum,
            'learning_rate': FLAGS.learning_rate,
            'end_learning_rate': FLAGS.end_learning_rate,
            'learning_rate_decay_factor': FLAGS.learning_rate_decay_factor,
            'decay_steps': FLAGS.decay_steps,
            'decay_boundaries': parse_comma_list(FLAGS.decay_boundaries),
            'lr_decay_factors': parse_comma_list(FLAGS.lr_decay_factors),
        })


    tensors_to_log = {
        'lr': 'learning_rate',
        'rpn_ce_loss': 'xception_lighthead/rpn_cross_entropy_loss',
        'rpn_loc_loss': 'xception_lighthead/rpn_location_loss',
        'rpn_loss': 'xception_lighthead/rpn_loss',
        'head_loss': 'xception_lighthead/head_loss',
        'head_ce_loss': 'xception_lighthead/final_head/head_cross_entropy_loss',
        'head_loc_loss': 'xception_lighthead/final_head/head_location_loss',
        'total_loss': 'total_loss',
    }

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps)

    print('Starting a training cycle.')
    #hook = tf.train.ProfilerHook(save_steps=50, output_dir='.')
    # debug_hook = tf_debug.LocalCLIDebugHook(thread_name_filter="MainThread$")
    # debug_hook.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    # xdetector.train(input_fn=input_pipeline(), hooks=[debug_hook])
    xdetector.train(input_fn=input_pipeline(), hooks=[logging_hook])

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

# Epoch[0] Batch [100]    Speed: 3.37 samples/sec Train-RPNAcc=0.896658,  RPNLogLoss=0.335296, RPNL1Loss=0.064354,    RCNNAcc=0.387995,   RCNNLogLoss=1.381760,   RCNNL1Loss=0.195688,
# Epoch[0] Batch [200]    Speed: 3.32 samples/sec Train-RPNAcc=0.926617,  RPNLogLoss=0.247772, RPNL1Loss=0.059578,    RCNNAcc=0.637477,   RCNNLogLoss=1.363275,   RCNNL1Loss=0.218161,
# Epoch[0] Batch [300]    Speed: 3.36 samples/sec Train-RPNAcc=0.936566,  RPNLogLoss=0.212495, RPNL1Loss=0.059787,    RCNNAcc=0.722254,   RCNNLogLoss=1.340625,   RCNNL1Loss=0.230750,
# Epoch[0] Batch [400]    Speed: 3.33 samples/sec Train-RPNAcc=0.943988,  RPNLogLoss=0.188992, RPNL1Loss=0.059758,    RCNNAcc=0.765644,   RCNNLogLoss=1.286642,   RCNNL1Loss=0.233597,
# Epoch[0] Batch [500]    Speed: 3.30 samples/sec Train-RPNAcc=0.947893,  RPNLogLoss=0.176932, RPNL1Loss=0.063037,    RCNNAcc=0.790357,   RCNNLogLoss=1.141996,   RCNNL1Loss=0.237377,
# Epoch[0] Batch [600]    Speed: 3.38 samples/sec Train-RPNAcc=0.949804,  RPNLogLoss=0.166182, RPNL1Loss=0.064645,    RCNNAcc=0.808197,   RCNNLogLoss=1.025559,   RCNNL1Loss=0.235833,
# Epoch[0] Batch [700]    Speed: 3.29 samples/sec Train-RPNAcc=0.951949,  RPNLogLoss=0.158510, RPNL1Loss=0.066234,    RCNNAcc=0.820647,   RCNNLogLoss=0.939149,   RCNNL1Loss=0.234840,
# Epoch[0] Batch [800]    Speed: 3.31 samples/sec Train-RPNAcc=0.952047,  RPNLogLoss=0.154384, RPNL1Loss=0.066083,    RCNNAcc=0.832026,   RCNNLogLoss=0.865931,   RCNNL1Loss=0.228402,
# Epoch[0] Batch [900]    Speed: 3.26 samples/sec Train-RPNAcc=0.953667,  RPNLogLoss=0.147420, RPNL1Loss=0.064395,    RCNNAcc=0.839128,   RCNNLogLoss=0.812424,   RCNNL1Loss=0.226792,

# In Epoch[0] RPNL1Loss = 0.403792. Then always nan

# Epoch[0] Batch [300] Speed: 1.15 samples/sec Train-RPNAcc=0.812539, RPNLogLoss=0.570043, RPNL1Loss=nan, RCNNAcc=0.767390, RCNNLogLoss=3.596807, RCNNL1Loss=0.010698,
# Epoch[0] Batch [400] Speed: 1.16 samples/sec Train-RPNAcc=0.821910, RPNLogLoss=0.599778, RPNL1Loss=nan, RCNNAcc=0.818267, RCNNLogLoss=3.577709, RCNNL1Loss=0.008031,
# Epoch[0] Batch [500] Speed: 1.14 samples/sec Train-RPNAcc=0.828243, RPNLogLoss=0.617132, RPNL1Loss=nan, RCNNAcc=0.848381, RCNNLogLoss=3.396140, RCNNL1Loss=0.006434,
# Epoch[0] Batch [600] Speed: 1.15 samples/sec Train-RPNAcc=0.832391, RPNLogLoss=0.628403, RPNL1Loss=nan, RCNNAcc=0.869345, RCNNLogLoss=2.870057, RCNNL1Loss=0.005387,
# Epoch[0] Batch [700] Speed: 1.16 samples/sec Train-RPNAcc=0.834384, RPNLogLoss=0.636100, RPNL1Loss=nan, RCNNAcc=0.883437, RCNNLogLoss=2.499374, RCNNL1Loss=0.004622,
# Epoch[0] Batch [800] Speed: 1.16 samples/sec Train-RPNAcc=0.836050, RPNLogLoss=0.641726, RPNL1Loss=nan, RCNNAcc=0.894673, RCNNLogLoss=2.215910, RCNNL1Loss=0.004046,
# Epoch[0] Batch [900] Speed: 1.14 samples/sec Train-RPNAcc=0.837489, RPNLogLoss=0.645880, RPNL1Loss=nan, RCNNAcc=0.903519, RCNNLogLoss=1.994231, RCNNL1Loss=0.003597,
# Epoch[0] Batch [1000] Speed: 1.16 samples/sec Train-RPNAcc=0.838438, RPNLogLoss=0.649027, RPNL1Loss=nan, RCNNAcc=0.910550, RCNNLogLoss=1.816863, RCNNL1Loss=0.003238,
# Epoch[0] Batch [1100] Speed: 1.16 samples/sec Train-RPNAcc=0.839179, RPNLogLoss=0.650772, RPNL1Loss=nan, RCNNAcc=0.915993, RCNNLogLoss=1.673953, RCNNL1Loss=0.003987,
# Epoch[0] Batch [1200] Speed: 1.15 samples/sec Train-RPNAcc=0.839684, RPNLogLoss=0.650882, RPNL1Loss=nan, RCNNAcc=0.920535, RCNNLogLoss=1.553478, RCNNL1Loss=0.003658,
# Epoch[0] Batch [1300] Speed: 1.15 samples/sec Train-RPNAcc=0.840592, RPNLogLoss=0.649718, RPNL1Loss=nan, RCNNAcc=0.924115, RCNNLogLoss=1.452725, RCNNL1Loss=0.003903,
# Epoch[0] Batch [1400] Speed: 1.14 samples/sec Train-RPNAcc=0.841459, RPNLogLoss=0.647637, RPNL1Loss=nan, RCNNAcc=0.927468, RCNNLogLoss=1.363495, RCNNL1Loss=0.003740,
# Epoch[0] Batch [1500] Speed: 1.16 samples/sec Train-RPNAcc=0.841013, RPNLogLoss=0.645282, RPNL1Loss=nan, RCNNAcc=0.930463, RCNNLogLoss=1.284670, RCNNL1Loss=0.003492,
# Epoch[0] Batch [1600] Speed: 1.16 samples/sec Train-RPNAcc=0.841707, RPNLogLoss=0.642184, RPNL1Loss=nan, RCNNAcc=0.932781, RCNNLogLoss=1.216500, RCNNL1Loss=0.003344,
# Epoch[0] Batch [1700] Speed: 1.16 samples/sec Train-RPNAcc=0.841856, RPNLogLoss=0.638847, RPNL1Loss=nan, RCNNAcc=0.935341, RCNNLogLoss=1.151333, RCNNL1Loss=0.003149,
# Epoch[0] Batch [1800] Speed: 1.16 samples/sec Train-RPNAcc=0.842007, RPNLogLoss=0.635248, RPNL1Loss=nan, RCNNAcc=0.937574, RCNNLogLoss=1.095160, RCNNL1Loss=0.004580,
# Epoch[0] Batch [1900] Speed: 1.17 samples/sec Train-RPNAcc=0.842244, RPNLogLoss=0.631594, RPNL1Loss=nan, RCNNAcc=0.939325, RCNNLogLoss=1.043886, RCNNL1Loss=0.004343,
# Epoch[0] Batch [2000] Speed: 1.17 samples/sec Train-RPNAcc=0.842616, RPNLogLoss=0.627715, RPNL1Loss=nan, RCNNAcc=0.941225, RCNNLogLoss=0.996324, RCNNL1Loss=0.004129,
# Epoch[0] Batch [2100] Speed: 1.18 samples/sec Train-RPNAcc=0.843182, RPNLogLoss=0.623727, RPNL1Loss=nan, RCNNAcc=0.942862, RCNNLogLoss=0.953685, RCNNL1Loss=0.003934,
# Epoch[0] Batch [2200] Speed: 1.18 samples/sec Train-RPNAcc=0.843663, RPNLogLoss=0.619788, RPNL1Loss=nan, RCNNAcc=0.944095, RCNNLogLoss=0.915521, RCNNL1Loss=0.003757,
# Epoch[0] Batch [2300] Speed: 1.17 samples/sec Train-RPNAcc=0.844234, RPNLogLoss=0.615760, RPNL1Loss=nan, RCNNAcc=0.945496, RCNNLogLoss=0.879742, RCNNL1Loss=0.003595,
# Epoch[0] Batch [2400] Speed: 1.18 samples/sec Train-RPNAcc=0.844505, RPNLogLoss=0.611821, RPNL1Loss=nan, RCNNAcc=0.947011, RCNNLogLoss=0.846207, RCNNL1Loss=0.003446,
# Epoch[0] Batch [2500] Speed: 1.16 samples/sec Train-RPNAcc=0.844367, RPNLogLoss=0.608176, RPNL1Loss=nan, RCNNAcc=0.947858, RCNNLogLoss=0.819818, RCNNL1Loss=0.004606,
# Epoch[0] Batch [2600] Speed: 1.18 samples/sec Train-RPNAcc=0.844443, RPNLogLoss=0.604457, RPNL1Loss=nan, RCNNAcc=0.948941, RCNNLogLoss=0.791787, RCNNL1Loss=0.004434,
