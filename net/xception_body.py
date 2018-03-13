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
import tensorflow as tf

from . import resnet_v2
from utility import eval_helper

USE_FUSED_BN = True
BN_EPSILON = 0.001
BN_MOMENTUM = 0.99

initializer_to_use = lambda : tf.truncated_normal_initializer(mean=0.0, stddev=0.005)

def get_shape(x, rank=None):
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]

def _pad_axis(x, offset, size, axis=0, name=None):
    with tf.name_scope(name, 'rpn_pad_axis'):
        shape = get_shape(x)
        rank = len(shape)
        # Padding description.
        new_size = tf.maximum(size-offset-shape[axis], 0)
        pad1 = tf.stack([0]*axis + [offset] + [0]*(rank-axis-1))
        pad2 = tf.stack([0]*axis + [new_size] + [0]*(rank-axis-1))
        paddings = tf.stack([pad1, pad2], axis=1)
        x = tf.pad(x, paddings, mode='CONSTANT')
        # Reshape, to get fully defined shape if possible.
        # TODO: fix with tf.slice
        shape[axis] = tf.maximum(size, shape[axis])
        x = tf.reshape(x, tf.stack(shape))
        return x

def _bboxes_nms(scores, bboxes, nms_threshold = 0.5, keep_top_k = 200, mode = 'union', scope=None):
    with tf.name_scope(scope, 'rpn_bboxes_nms', [scores, bboxes]):
        num_bboxes = tf.shape(scores)[0]
        def nms_proc(scores, bboxes):
            # sort all the bboxes
            scores, idxes = tf.nn.top_k(scores, k = num_bboxes, sorted = True)
            bboxes = tf.gather(bboxes, idxes)

            ymin = bboxes[:, 0]
            xmin = bboxes[:, 1]
            ymax = bboxes[:, 2]
            xmax = bboxes[:, 3]

            vol_anchors = (xmax - xmin) * (ymax - ymin)

            nms_mask = tf.cast(tf.ones_like(scores, dtype=tf.int8), tf.bool)
            keep_mask = tf.cast(tf.zeros_like(scores, dtype=tf.int8), tf.bool)

            def safe_divide(numerator, denominator):
                return tf.where(tf.greater(denominator, 0), tf.divide(numerator, denominator), tf.zeros_like(denominator))

            def get_scores(bbox, nms_mask):
                # the inner square
                inner_ymin = tf.maximum(ymin, bbox[0])
                inner_xmin = tf.maximum(xmin, bbox[1])
                inner_ymax = tf.minimum(ymax, bbox[2])
                inner_xmax = tf.minimum(xmax, bbox[3])
                h = tf.maximum(inner_ymax - inner_ymin, 0.)
                w = tf.maximum(inner_xmax - inner_xmin, 0.)
                inner_vol = h * w
                this_vol = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if mode == 'union':
                    union_vol = vol_anchors - inner_vol  + this_vol
                elif mode == 'min':
                    union_vol = tf.minimum(vol_anchors, this_vol)
                else:
                    raise ValueError('unknown mode to use for nms.')
                return safe_divide(inner_vol, union_vol) * tf.cast(nms_mask, tf.float32)

            def condition(index, nms_mask, keep_mask):
                return tf.logical_and(tf.reduce_sum(tf.cast(nms_mask, tf.int32)) > 0, tf.less(index, keep_top_k))

            def body(index, nms_mask, keep_mask):
                # at least one True in nms_mask
                indices = tf.where(nms_mask)[0][0]
                bbox = bboxes[indices]
                this_mask = tf.one_hot(indices, num_bboxes, on_value=False, off_value=True, dtype=tf.bool)
                keep_mask = tf.logical_or(keep_mask, tf.logical_not(this_mask))
                nms_mask = tf.logical_and(nms_mask, this_mask)

                nms_scores = get_scores(bbox, nms_mask)

                nms_mask = tf.logical_and(nms_mask, nms_scores < nms_threshold)
                return [index+1, nms_mask, keep_mask]

            index = 0
            [index, nms_mask, keep_mask] = tf.while_loop(condition, body, [index, nms_mask, keep_mask])

            return [_pad_axis(tf.boolean_mask(scores, keep_mask), 0, keep_top_k), _pad_axis(tf.boolean_mask(bboxes, keep_mask), 0, keep_top_k)]

        return tf.cond(tf.less(num_bboxes, 1), lambda: [scores, bboxes], lambda: nms_proc(scores, bboxes))

def _filter_boxes(scores, bboxes, min_size, scope=None):#, keep_top_k = 100
    #scores = tf.Print(scores,[tf.shape(scores)])
    # scores = tf.Print(scores,[tf.shape(scores), min_size])
    # bboxes = tf.Print(bboxes,[tf.shape(bboxes), bboxes])
    with tf.name_scope(scope, 'rpn_filter_boxes', [scores, bboxes]):
        ymin = bboxes[:, 0]
        #ymin = tf.Print(ymin, [tf.shape(ymin)])
        xmin = bboxes[:, 1]
        ymax = bboxes[:, 2]
        xmax = bboxes[:, 3]

        ws = xmax - xmin
        hs = ymax - ymin
        x_ctr = xmin + ws / 2.
        y_ctr = ymin + hs / 2.

        keep_mask = tf.logical_and(tf.greater(ws, min_size), tf.greater(hs, min_size))
        keep_mask = tf.logical_and(keep_mask, tf.greater(x_ctr, 0.))
        keep_mask = tf.logical_and(keep_mask, tf.greater(y_ctr, 0.))
        keep_mask = tf.logical_and(keep_mask, tf.less(x_ctr, 1.))
        keep_mask = tf.logical_and(keep_mask, tf.less(y_ctr, 1.))

        # scores = _pad_axis(tf.boolean_mask(scores, keep_mask), 0, keep_top_k, axis=0)
        # bboxes = _pad_axis(tf.boolean_mask(bboxes, keep_mask), 0, keep_top_k, axis=0)
        return [tf.boolean_mask(scores, keep_mask), tf.boolean_mask(bboxes, keep_mask)]

def _bboxes_sort(scores, bboxes, top_k=100, scope=None):
    #scores = tf.Print(scores,[tf.shape(scores)])
    with tf.name_scope(scope, 'rpn_bboxes_sort', [scores, bboxes]):
        # Sort scores...
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)
        #return scores, tf.gather(bboxes, idxes)
        return [_pad_axis(scores, 0, top_k, axis=0), _pad_axis(tf.gather(bboxes, idxes), 0, top_k, axis=0)]


def _bboxes_clip(bbox_ref, bboxes, scope=None):
    #bboxes = tf.Print(bboxes,[tf.shape(bboxes)])
    with tf.name_scope(scope, 'rpn_bboxes_clip'):
        # Easier with transposed bboxes. Especially for broadcasting.
        bbox_ref = tf.transpose(bbox_ref)
        bboxes = tf.transpose(bboxes)
        # Intersection bboxes and reference bbox.
        ymin = tf.maximum(bboxes[0], bbox_ref[0])
        xmin = tf.maximum(bboxes[1], bbox_ref[1])
        ymax = tf.minimum(bboxes[2], bbox_ref[2])
        xmax = tf.minimum(bboxes[3], bbox_ref[3])
        # Double check! Empty boxes when no-intersection.
        #      _____
        #     |     |
        #     |_____|
        #                ______
        #               |      |
        #               |______|
        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)
        bboxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax], axis=0))
        return bboxes

def _upsample_rois(scores, bboxes, keep_top_k):
    # upsample with replacement
    bboxes = tf.boolean_mask(bboxes, scores > 0.)
    scores = tf.boolean_mask(scores, scores > 0.)
    def upsampel_impl():
        num_bboxes = tf.shape(scores)[0]
        left_count = keep_top_k - num_bboxes

        select_indices = tf.random_shuffle(tf.range(num_bboxes))[:tf.floormod(left_count, num_bboxes)]
        select_indices = tf.concat([tf.tile(tf.range(num_bboxes), [tf.floor_div(left_count, num_bboxes) + 1]), select_indices], axis = 0)

        return [tf.gather(scores, select_indices), tf.gather(bboxes, select_indices)]
    return tf.cond(tf.shape(scores)[0] < keep_top_k, lambda : upsampel_impl(), lambda : [scores, bboxes])

def _point2center(proposals_bboxes):
    ymin, xmin, ymax, xmax = proposals_bboxes[:, :, 0], proposals_bboxes[:, :, 1], proposals_bboxes[:, :, 2], proposals_bboxes[:, :, 3]
    height, width = (ymax - ymin), (xmax - xmin)
    return tf.stack([ymin + height / 2., xmin + width / 2., height, width], axis=-1)

def relu_separable_bn_block(inputs, filters, name_prefix, is_training, data_format):
    bn_axis = -1 if data_format == 'channels_last' else 1

    inputs = tf.nn.relu(inputs, name=name_prefix + '_act')
    inputs = tf.layers.separable_conv2d(inputs, filters, (3, 3),
                        strides=(1, 1), padding='same',
                        data_format=data_format,
                        activation=None, use_bias=False,
                        depthwise_initializer=initializer_to_use(),
                        pointwise_initializer=initializer_to_use(),
                        bias_initializer=tf.zeros_initializer(),
                        name=name_prefix, reuse=None)
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name=name_prefix + '_bn', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    return inputs

def XceptionBody(input_image, num_classes, is_training = False, data_format='channels_last'):
    # one question: is there problems with the unaligned 'valid-conv' when mapping ROI from input image to last feature maps

    # modify the input size to 481
    bn_axis = -1 if data_format == 'channels_last' else 1

    # (481-3+0*2)/2 + 1 = 240
    inputs = tf.layers.conv2d(input_image, 32, (3, 3), use_bias=False, name='block1_conv1', strides=(2, 2),
                padding='valid', data_format=data_format, activation=None,
                kernel_initializer=initializer_to_use(),
                bias_initializer=tf.zeros_initializer())
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block1_conv1_bn', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name='block1_conv1_act')

    # (240-3+0*2)/1 + 1 = 238
    inputs = tf.layers.conv2d(inputs, 64, (3, 3), use_bias=False, name='block1_conv2', strides=(1, 1),
                padding='valid', data_format=data_format, activation=None,
                kernel_initializer=initializer_to_use(),
                bias_initializer=tf.zeros_initializer())
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block1_conv2_bn', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name='block1_conv2_act')

    # (238-1+0*2)/2 + 1 = 119
    residual = tf.layers.conv2d(inputs, 128, (1, 1), use_bias=False, name='conv2d_1', strides=(2, 2),
                padding='same', data_format=data_format, activation=None,
                kernel_initializer=initializer_to_use(),
                bias_initializer=tf.zeros_initializer())
    residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_1', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = tf.layers.separable_conv2d(inputs, 128, (3, 3),
                        strides=(1, 1), padding='same',
                        data_format=data_format,
                        activation=None, use_bias=False,
                        depthwise_initializer=initializer_to_use(),
                        pointwise_initializer=initializer_to_use(),
                        bias_initializer=tf.zeros_initializer(),
                        name='block2_sepconv1', reuse=None)
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block2_sepconv1_bn', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 128, 'block2_sepconv2', is_training, data_format)
    # (238-3+1*2)/2 + 1 = 119
    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                    padding='same', data_format=data_format,
                                    name='block2_pool')
    # 119
    inputs = inputs + residual

    #inputs = tf.Print(inputs,[tf.shape(inputs), inputs,residual])

    # (119-1+0*2)/2 + 1 = 60
    residual = tf.layers.conv2d(inputs, 256, (1, 1), use_bias=False, name='conv2d_2', strides=(2, 2),
                padding='same', data_format=data_format, activation=None,
                kernel_initializer=initializer_to_use(),
                bias_initializer=tf.zeros_initializer())
    residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_2', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 256, 'block3_sepconv1', is_training, data_format)
    inputs = relu_separable_bn_block(inputs, 256, 'block3_sepconv2', is_training, data_format)

    # (119-3+1*2)/2 + 1 = 60
    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                    padding='same', data_format=data_format,
                                    name='block3_pool')
    # 60
    inputs = inputs + residual

    # (119-1+0*2)/2 + 1 = 30
    residual = tf.layers.conv2d(inputs, 728, (1, 1), use_bias=False, name='conv2d_3', strides=(2, 2),
                padding='same', data_format=data_format, activation=None,
                kernel_initializer=initializer_to_use(),
                bias_initializer=tf.zeros_initializer())
    residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_3', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 728, 'block4_sepconv1', is_training, data_format)
    inputs = relu_separable_bn_block(inputs, 728, 'block4_sepconv2', is_training, data_format)

    # (119-3+1*2)/2 + 1 = 30
    inputs = tf.layers.max_pooling2d(inputs, pool_size=(3, 3), strides=(2, 2),
                                    padding='same', data_format=data_format,
                                    name='block4_pool')
    # 30
    inputs = inputs + residual

    for index in range(8):
        residual = inputs
        prefix = 'block' + str(index + 5)

        inputs = relu_separable_bn_block(inputs, 728, prefix + '_sepconv1', is_training, data_format)
        inputs = relu_separable_bn_block(inputs, 728, prefix + '_sepconv2', is_training, data_format)
        inputs = relu_separable_bn_block(inputs, 728, prefix + '_sepconv3', is_training, data_format)

        inputs = inputs + residual

    mid_outputs = tf.nn.relu(inputs, name='before_block13_act')
    # remove stride 2 for the residual connection
    residual = tf.layers.conv2d(inputs, 1024, (1, 1), use_bias=False, name='conv2d_4', strides=(1, 1),
                padding='same', data_format=data_format, activation=None,
                kernel_initializer=initializer_to_use(),
                bias_initializer=tf.zeros_initializer())
    residual = tf.layers.batch_normalization(residual, momentum=BN_MOMENTUM, name='batch_normalization_4', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)

    inputs = relu_separable_bn_block(inputs, 728, 'block13_sepconv1', is_training, data_format)
    inputs = relu_separable_bn_block(inputs, 1024, 'block13_sepconv2', is_training, data_format)

    inputs = inputs + residual
    # use atrous algorithm at last two conv
    inputs = tf.layers.separable_conv2d(inputs, 1536, (3, 3),
                        strides=(1, 1), dilation_rate=(2, 2), padding='same',
                        data_format=data_format,
                        activation=None, use_bias=False,
                        depthwise_initializer=initializer_to_use(),
                        pointwise_initializer=initializer_to_use(),
                        bias_initializer=tf.zeros_initializer(),
                        name='block14_sepconv1', reuse=None)
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block14_sepconv1_bn', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    inputs = tf.nn.relu(inputs, name='block14_sepconv1_act')

    inputs = tf.layers.separable_conv2d(inputs, 2048, (3, 3),
                        strides=(1, 1), dilation_rate=(2, 2), padding='same',
                        data_format=data_format,
                        activation=None, use_bias=False,
                        depthwise_initializer=initializer_to_use(),
                        pointwise_initializer=initializer_to_use(),
                        bias_initializer=tf.zeros_initializer(),
                        name='block14_sepconv2', reuse=None)
    inputs = tf.layers.batch_normalization(inputs, momentum=BN_MOMENTUM, name='block14_sepconv2_bn', axis=bn_axis,
                            epsilon=BN_EPSILON, training=is_training, reuse=None, fused=USE_FUSED_BN)
    outputs = tf.nn.relu(inputs, name='block14_sepconv2_act')
    # the output size here is 30 x 30


    return mid_outputs, outputs

def get_rpn(net_input, num_anchors, is_training, data_format, var_scope):
    with tf.variable_scope(var_scope):
        rpn_relu = tf.layers.conv2d(inputs=net_input, filters=512, kernel_size=(3, 3), strides=1,
                                  padding='SAME', use_bias=True, activation=tf.nn.relu,
                                  kernel_initializer=initializer_to_use(),
                                  bias_initializer=tf.zeros_initializer(),
                                  data_format=data_format)
        rpn_cls_score = tf.layers.conv2d(inputs=rpn_relu, filters=2 * num_anchors, kernel_size=(1, 1), strides=1,
                                      padding='SAME', use_bias=True, activation=None,
                                      kernel_initializer=initializer_to_use(),
                                      bias_initializer=tf.zeros_initializer(),
                                      data_format=data_format)
        rpn_bbox_pred = tf.layers.conv2d(inputs=rpn_relu, filters=4 * num_anchors, kernel_size=(1, 1), strides=1,
                                      padding='SAME', use_bias=True, activation=None,
                                      kernel_initializer=initializer_to_use(),
                                      bias_initializer=tf.zeros_initializer(),
                                      data_format=data_format)

        #rpn_bbox_pred = tf.Print(rpn_bbox_pred,[tf.shape(rpn_bbox_pred), net_input, rpn_bbox_pred])
        return rpn_cls_score, rpn_bbox_pred

def get_proposals(object_score, bboxes_pred, encode_fn, rpn_pre_nms_top_n, rpn_post_nms_top_n, nms_threshold, rpn_min_size, data_format):
    '''About the input:
    object_score: N x num_bboxes
    bboxes_pred: N x num_bboxes x 4
    decode_fn: accept location_pred and return the decoded bboxes in format 'batch_size, feature_h, feature_w, num_anchors, 4'
    rpn_min_size: the absolute pixels a bbox must have in both side
    data_format: specify the input format

    this function do the following process:
    1. for each (H, W) location i
      generate A anchor boxes centered on cell i
      apply predicted bbox deltas at cell i to each of the A anchors (this is done before this function)
    2. clip predicted boxes to image
    3. remove predicted boxes with either height or width < rpn_min_size
    4. sort all (proposal, score) pairs by score from highest to lowest
    5. take top pre_nms_topN rois before NMS
    6. apply NMS with threshold 0.7 to remaining rois
    7. take after_nms_topN rois after NMS (if number of bboxes if less than after_nms_topN, the upsample them)
    8. take both the top rois and all the ground truth bboxes as all_rois
    9. rematch all_rois to get regress and classification target
    10.sample all_rois as proposals
    '''
    #bboxes_pred = decode_fn(location_pred)
    bboxes_pred = tf.map_fn(lambda _bboxes : _bboxes_clip([0., 0., 1., 1.], _bboxes), bboxes_pred)
    object_score, bboxes_pred = tf.map_fn(lambda _score_bboxes : _filter_boxes(_score_bboxes[0], _score_bboxes[1], rpn_min_size), [object_score, bboxes_pred])
    object_score, bboxes_pred = tf.map_fn(lambda _score_bboxes : _bboxes_sort(_score_bboxes[0], _score_bboxes[1], top_k=tf.minimum(tf.shape(_score_bboxes[0])[0], rpn_pre_nms_top_n)), [object_score, bboxes_pred])
    object_score, bboxes_pred = tf.map_fn(lambda _score_bboxes : _bboxes_nms(_score_bboxes[0], _score_bboxes[1], nms_threshold = nms_threshold, keep_top_k=tf.minimum(tf.shape(_score_bboxes[0])[0], rpn_post_nms_top_n), mode = 'union'), [object_score, bboxes_pred], dtype=[tf.float32, tf.float32], infer_shape=True)
    # padding to fix the size of rois
    # the object_score is not in descending order when the upsample padding happened
    object_score, bboxes_pred = tf.map_fn(lambda _score_bboxes : _upsample_rois(_score_bboxes[0], _score_bboxes[1], keep_top_k= rpn_post_nms_top_n), [object_score, bboxes_pred])
    # match and sample to get proposals and targets
    #print(encode_fn(bboxes_pred))
    proposals_bboxes, proposals_targets, proposals_labels, proposals_scores = encode_fn(bboxes_pred)

    return proposals_bboxes, proposals_targets, proposals_labels, proposals_scores

def large_sep_kernel(net_input, depth_mid, depth_output, is_training, data_format, var_scope):
  with tf.variable_scope(var_scope):
    with tf.variable_scope('Branch_0'):
      branch_0a = tf.layers.conv2d(inputs=net_input, filters=depth_mid, kernel_size=(15, 1), strides=1,
                                  padding='SAME', use_bias=True, activation=None,
                                  kernel_initializer=initializer_to_use(),
                                  bias_initializer=tf.zeros_initializer(),
                                  data_format=data_format)
      branch_0b = tf.layers.conv2d(inputs=branch_0a, filters=depth_output, kernel_size=(1, 15), strides=1,
                                  padding='SAME', use_bias=True, activation=None,
                                  kernel_initializer=initializer_to_use(),
                                  bias_initializer=tf.zeros_initializer(),
                                  data_format=data_format)
    with tf.variable_scope('Branch_1'):
      branch_1a = tf.layers.conv2d(inputs=net_input, filters=depth_mid, kernel_size=(15, 1), strides=1,
                                  padding='SAME', use_bias=True, activation=None,
                                  kernel_initializer=initializer_to_use(),
                                  bias_initializer=tf.zeros_initializer(),
                                  data_format=data_format)
      branch_1b = tf.layers.conv2d(inputs=branch_1a, filters=depth_output, kernel_size=(1, 15), strides=1,
                                  padding='SAME', use_bias=True, activation=None,
                                  kernel_initializer=initializer_to_use(),
                                  bias_initializer=tf.zeros_initializer(),
                                  data_format=data_format)

    return resnet_v2.batch_norm_relu(branch_0b + branch_1b, is_training, data_format)

def get_head(net_input, pooling_op, grid_width, grid_height, loss_func, proposals_bboxes, proposals_targets, proposals_labels, proposals_scores, num_classes, is_training, using_ohem, ohem_roi_one_image, data_format, var_scope):
    with tf.variable_scope(var_scope):
        # two pooling op here in original r-fcn
        # rfcn_cls = tf.layers.conv2d(inputs=net_input, filters=10 * grid_width * grid_height, kernel_size=(1, 1), strides=1,
        #                           padding='SAME', use_bias=True, activation=tf.nn.relu,
        #                           kernel_initializer=initializer_to_use(),
        #                           bias_initializer=tf.zeros_initializer(),
        #                           data_format=data_format)
        # rfcn_bbox = tf.layers.conv2d(inputs=net_input, filters=10 * grid_width * grid_height, kernel_size=(1, 1), strides=1,
        #                           padding='SAME', use_bias=True, activation=tf.nn.relu,
        #                           kernel_initializer=initializer_to_use(),
        #                           bias_initializer=tf.zeros_initializer(),
        #                           data_format=data_format)
        # {num_per_batch, num_rois, grid_size, bank_size}

        yxhw_bboxes = _point2center(proposals_bboxes)
        if data_format == 'channels_last':
            net_input = tf.transpose(net_input, [0, 3, 1, 2])

        psroipooled_rois, _ = pooling_op(net_input, yxhw_bboxes, grid_width, grid_height, 'max')

        psroipooled_rois = tf.map_fn(lambda pooled_feat: tf.reshape(pooled_feat, [-1, 10 * grid_width * grid_height]), psroipooled_rois)
        #psroipooled_rois = tf.reshape(psroipooled_rois, [-1, proposals_labels.get_shape().as_list()[-1], 10 * grid_width * grid_height])

        select_indices = None

        if using_ohem:
            subnet_fc_feature = tf.layers.dense(psroipooled_rois, 2048,
                                    activation=tf.nn.relu,
                                    use_bias=True,
                                    kernel_initializer=initializer_to_use(),
                                    bias_initializer=tf.zeros_initializer(),
                                    name='subnet_fc', reuse=False)
            cls_score = tf.layers.dense(subnet_fc_feature, num_classes,
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer=initializer_to_use(),
                                        bias_initializer=tf.zeros_initializer(),
                                        name='fc_cls', reuse=False)
            bboxes_reg = tf.layers.dense(subnet_fc_feature, 4,
                                        activation=None,
                                        use_bias=True,
                                        kernel_initializer=initializer_to_use(),
                                        bias_initializer=tf.zeros_initializer(),
                                        name='fc_loc', reuse=False)
            # the input of loss_func is (batch, num_rois, num_classes), (batch, num_rois, 4)
            # the output should be (batch, num_rois)
            ohem_loss = loss_func(tf.stop_gradient(cls_score), tf.stop_gradient(bboxes_reg), None)

            ohem_select_num = tf.minimum(ohem_roi_one_image, tf.shape(ohem_loss)[1])

            _, select_indices = tf.nn.top_k(ohem_loss, k=ohem_select_num)

            select_indices = tf.stop_gradient(select_indices)

            psroipooled_rois = tf.gather(psroipooled_rois, select_indices, axis=1)

            # proposals_bboxes = tf.gather(proposals_bboxes, select_indices, axis=1)
            # proposals_targets = tf.gather(proposals_targets, select_indices, axis=1)
            # proposals_labels = tf.gather(proposals_labels, select_indices, axis=1)
            # proposals_scores = tf.gather(proposals_scores, select_indices, axis=1)

        subnet_fc_feature = tf.layers.dense(psroipooled_rois, 2048,
                                    activation=tf.nn.relu,
                                    use_bias=True,
                                    kernel_initializer=initializer_to_use(),
                                    bias_initializer=tf.zeros_initializer(),
                                    name='subnet_fc', reuse=using_ohem)

        cls_score = tf.layers.dense(subnet_fc_feature, num_classes,
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=initializer_to_use(),
                                    bias_initializer=tf.zeros_initializer(),
                                    name='fc_cls', reuse=using_ohem)
        bboxes_reg = tf.layers.dense(subnet_fc_feature, 4,
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=initializer_to_use(),
                                    bias_initializer=tf.zeros_initializer(),
                                    name='fc_loc', reuse=using_ohem)

        return tf.reduce_mean(loss_func(cls_score, bboxes_reg, select_indices))



