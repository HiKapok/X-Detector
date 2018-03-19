
import tensorflow as tf

import numpy as np
sess = tf.Session()


t1 = tf.constant([113,114,13,214,111,16])
t2 = tf.constant([111,112,113,114,115])
a=tf.expand_dims(t1,0)+tf.expand_dims(t2,1)

b = t1==113
print(sess.run(b))


bboxes = tf.constant([[0.1,0.2,0.3,0.4], [0.1,0.25,0.3,0.4], [0.15,0.2,0.3,0.4], [0.15,0.28,0.32,0.49]])
gt_bboxes = tf.constant([[0.12,0.22,0.33,0.44], [0.14,0.2,0.35,0.4], [0.1,0.32,0.3,0.54]])


from tensorflow.contrib.image.python.ops import image_ops

def areas(bboxes):
    ymin, xmin, ymax, xmax = tf.split(bboxes, 4, axis=1)
    return (xmax - xmin) * (ymax - ymin)
def intersection(bboxes, gt_bboxes):
    ymin, xmin, ymax, xmax = tf.split(bboxes, 4, axis=1)
    gt_ymin, gt_xmin, gt_ymax, gt_xmax = [tf.transpose(b, perm=[1, 0]) for b in tf.split(gt_bboxes, 4, axis=1)]

    int_ymin = tf.maximum(ymin, gt_ymin)
    int_xmin = tf.maximum(xmin, gt_xmin)
    int_ymax = tf.minimum(ymax, gt_ymax)
    int_xmax = tf.minimum(xmax, gt_xmax)
    h = tf.maximum(int_ymax - int_ymin, 0.)
    w = tf.maximum(int_xmax - int_xmin, 0.)

    return h * w
def iou_matrix(bboxes, gt_bboxes):
    inter_vol = intersection(bboxes, gt_bboxes)
    union_vol = areas(bboxes) + tf.transpose(areas(gt_bboxes), perm=[1, 0]) - inter_vol

    return tf.where(tf.equal(inter_vol, 0.0),
                    tf.zeros_like(inter_vol), tf.truediv(inter_vol, union_vol))

def do_dual_max_match(overlap_matrix, high_thres, low_thres, ignore_between = True):
    '''
    overlap_matrix: num_gt * num_anchors
    '''
    anchors_to_gt = tf.argmax(overlap_matrix, axis=0)
    match_values = tf.reduce_max(overlap_matrix, axis=0)

    positive_mask = tf.greater_equal(match_values, high_thres)
    less_mask = tf.less(match_values, low_thres)
    between_mask = tf.logical_and(tf.less(match_values, high_thres), tf.greater_equal(match_values, low_thres))
    negative_mask = less_mask if ignore_between else between_mask
    ignore_mask = between_mask if ignore_between else less_mask

    match_indices = tf.where(negative_mask, -1 * tf.ones_like(anchors_to_gt), anchors_to_gt)
    match_indices = tf.where(ignore_mask, -2 * tf.ones_like(match_indices), match_indices)

    anchors_to_gt_mask = tf.one_hot(tf.clip_by_value(match_indices, -1, tf.cast(tf.shape(overlap_matrix)[0], tf.int64)), tf.shape(overlap_matrix)[0], on_value=1, off_value=0, axis=0, dtype=tf.int32)

    gt_to_anchors = tf.argmax(overlap_matrix, axis=1)

    left_gt_to_anchors_mask = tf.cast(tf.logical_and(tf.reduce_max(anchors_to_gt_mask, axis=1, keep_dims=True) < 1, tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[1], on_value=True, off_value=False, axis=1, dtype=tf.bool)), tf.int64)

    selected_scores = tf.gather_nd(overlap_matrix, tf.stack([tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0, tf.argmax(left_gt_to_anchors_mask, axis=0), anchors_to_gt), tf.range(tf.cast(tf.shape(overlap_matrix)[1], tf.int64))], axis=1))
    return tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0, tf.argmax(left_gt_to_anchors_mask, axis=0), match_indices)

print(sess.run([tf.transpose(areas(gt_bboxes), perm=[1, 0])]))
print(sess.run([tf.maximum(tf.expand_dims(t1,0),tf.expand_dims(t2,1))]))
print(sess.run([do_dual_max_match(iou_matrix(gt_bboxes, bboxes), 0.65, 0.6)]))
print(sess.run([image_ops.bipartite_match(iou_matrix(gt_bboxes, bboxes), -1)]))


gt_bboxes1 = tf.constant([[0.12,0.22,0.33,0.44], [0.14,0.2,0.35,0.4], [0.1,0.32,0.3,0.54]])
gt_bboxes2 = tf.constant([[2,1,0,3.]])
print(sess.run(gt_bboxes1 * tf.expand_dims(gt_bboxes2,0)))

# elems = np.array([[1,2], [2,3], [3,4]])
# alternates = tf.map_fn(lambda x: [x[0][0],x[1][1]], [elems,elems])
# print(sess.run(alternates))
# scores = tf.constant([11,12,13,14,15])
# labels = tf.constant([[111,12,13,14,15], [211,12,13,14,15], [311,12,13,14,15], [411,12,13,14,15],[511,12,13,14,15]])

# num_bboxes = tf.shape(scores)[0]

# left_count = 8 - num_bboxes


# print(sess.run([tf.gather(labels, [1,2,3], axis=1)]))

# select_indices = tf.random_shuffle(tf.range(num_bboxes))[:tf.floormod(left_count, num_bboxes)]

# select_indices = tf.concat([tf.tile(tf.range(num_bboxes), [tf.floor_div(left_count, num_bboxes) + 1]), select_indices], axis = 0)
# print(sess.run([tf.gather(scores, select_indices), tf.gather(labels, select_indices)]))

# print(sess.run(tf.where([[True],[False],[True],[True],[False],[False]])))

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import os
# import sys

# from scipy.misc import imread, imsave, imshow, imresize
# import tensorflow as tf

# name_map = {'beta': 'bias',
#             'gamma': 'scale',
#             'moving_mean': 'mean',
#             'moving_variance': 'var',
#             'kernel':'weight',
#             'depthwise_kernel': 'df',
#             'pointwise_kernel': 'pf'}

# test_var_name_list = ['block1_conv1/kernel',
#                     'block1_conv1_bn/gamma',
#                     'block1_conv1_bn/beta',
#                     'block1_conv1_bn/moving_mean',
#                     'block1_conv1_bn/moving_variance',
#                     'block1_conv2/kernel',
#                     'block1_conv2_bn/gamma',
#                     'block1_conv2_bn/beta',
#                     'block1_conv2_bn/moving_mean',
#                     'block1_conv2_bn/moving_variance',
#                     'conv2d_1/kernel',
#                     'batch_normalization_1/gamma',
#                     'batch_normalization_1/beta',
#                     'batch_normalization_1/moving_mean',
#                     'batch_normalization_1/moving_variance',
#                     'block2_sepconv1/depthwise_kernel',
#                     'block2_sepconv1/pointwise_kernel']

# for var_name in test_var_name_list:
#     prefix = var_name[:var_name.rfind('/')]
#     suffix = var_name[var_name.rfind('/') + 1 :]
#     print(prefix + '_' + name_map[suffix])

# # hardware related configuration
# tf.app.flags.DEFINE_string(
#     'boundaries', '60000, 90000, 120000',
#     'Learning rate decay boundaries.')
# tf.app.flags.DEFINE_string(
#     'lr_decay_factors', '0.1, 0.01, 0.001',
#     'The values of learning rate for each segment between boundaries.')

# FLAGS = tf.app.flags.FLAGS

# def parse_comma_list(args):
#     return [float(s.strip()) for s in args.split(',')]

# def main(_):
#     print(parse_comma_list(FLAGS.boundaries))

# if __name__ == '__main__':
#   tf.logging.set_verbosity(tf.logging.INFO)
#   tf.app.run()




# # from tensorflow.python.framework import constant_op
# # from tensorflow.python.framework import dtypes
# # from tensorflow.python.framework import ops
# # from tensorflow.python.framework import tensor_shape
# # from tensorflow.python.framework import tensor_util
# # from tensorflow.python.ops import array_ops
# # from tensorflow.python.ops import check_ops
# # from tensorflow.python.ops import clip_ops
# # from tensorflow.python.ops import control_flow_ops
# # from tensorflow.python.ops import gen_image_ops
# # from tensorflow.python.ops import gen_nn_ops
# # from tensorflow.python.ops import string_ops
# # from tensorflow.python.ops import math_ops
# # from tensorflow.python.ops import random_ops
# # from tensorflow.python.ops import variables

# # min_iou_list = tf.convert_to_tensor([0.1, 0.3, 0.5, 0.7, 0.9, 1.])
# # samples_min_iou = tf.multinomial(tf.log([[1./6., 1./6., 1./6., 1./6., 1./6., 1./6.]]), 1) # note log-prob

# # sampled_min_iou = min_iou_list[tf.cast(samples_min_iou[0][0], tf.int32)]

# # result = control_flow_ops.cond(math_ops.less(sampled_min_iou, 1.), lambda: 1, lambda: 2)

# # sess = tf.Session()


# # print(sess.run(result))
# # print(sess.run(result))
# # print(sess.run(result))
# # print(sess.run(result))
# # print(sess.run(result))
# # print(sess.run(result))
# import numpy as np


# import tensorflow as tf
# from tensorflow.contrib.framework.python.ops import add_arg_scope
# from tensorflow.contrib.framework.python.ops import arg_scope
# b = 8
# a = 2 if b else 3
# print(a)
# aaa=[12,4,5]
# print(aaa[1:2])
# print(aaa[0:-2] + [-1,3])


# dsf





# @add_arg_scope
# def ffff(net, a = 1):
#     print(a)

# with arg_scope(
#       [ffff],
#       a=2):
#     print(ffff(1))
# sess = tf.Session()
















# print(sess.run(tf.constant([[1,2,3,4,5],[5,6,7,8,9]])[:, -2:]))

# print(np.mgrid[0 : 5, 0 : 5])
# print(sess.run(tf.meshgrid(tf.range(5), tf.range(6))))

# def sample_width_height(width, height):
#     index = 0
#     max_attempt = 100
#     sampled_width, sampled_height = width, height

#     def condition(index, sampled_width, sampled_height, width, height):
#         return tf.logical_or(tf.logical_and(tf.logical_or(tf.greater(sampled_width, sampled_height * 2), tf.greater(sampled_height, sampled_width * 2)), tf.less(index, max_attempt)), tf.less(index, 1))

#     def body(index, sampled_width, sampled_height, width, height):
#         sampled_width = tf.random_uniform([1], minval=0.1, maxval=1., dtype=tf.float32)[0] * width
#         sampled_height = tf.random_uniform([1], minval=0.1, maxval=1., dtype=tf.float32)[0] *height

#         return index+1, sampled_width, sampled_height, width, height

#     [index, sampled_width, sampled_height, _, _] = tf.while_loop(condition, body,
#                                        [index, sampled_width, sampled_height, width, height])


#     return tf.cast(sampled_width, tf.int32), tf.cast(sampled_height, tf.int32)

# width = tf.constant(5, dtype=tf.int32)
# height = tf.constant(6, dtype=tf.int32)

# sampled_width,  sampled_height= sample_width_height(tf.cast(width, tf.float32), tf.cast(height, tf.float32))

# x = tf.random_uniform([1], minval=0, maxval=width - sampled_width, dtype=tf.int32)[0]
# y = tf.random_uniform([1], minval=0, maxval=height - sampled_height, dtype=tf.int32)[0]

# print(sess.run(tf.cast(tf.zeros_like(x, dtype=tf.uint8), tf.bool)))
# print(sess.run(tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32)[0]))

# roi = [y/height, x, (y + sampled_height)/height, x + sampled_width]

# x = tf.convert_to_tensor([[[1.,2.,29.], [31.,32.,9.]], [[11.,12.,19.], [11.,22.,91.]], [[11.,12.,19.], [11.,22.,91.]]])


# def ssd_random_expand(image, ratio = 4, name=None):
#     with tf.name_scope('ssd_random_expand'):
#         height, width, depth = 3, 2, 3

#         canvas_width, canvas_height = width * ratio, height * ratio

#         mean_color_of_image = tf.reduce_mean(tf.reshape(image, [-1, 3]), 0)

#         x = tf.random_uniform([1], minval=0, maxval=canvas_width - width, dtype=tf.int32)[0]
#         y = tf.random_uniform([1], minval=0, maxval=canvas_height - height, dtype=tf.int32)[0]

#         paddings = tf.convert_to_tensor([[y, x], [canvas_height - height - y, canvas_width - width - x]])

#         big_canvas = tf.stack([tf.pad(image[:, :, 0], paddings, "CONSTANT", constant_values = mean_color_of_image[0]), tf.pad(image[:, :, 1], paddings, "CONSTANT", constant_values = mean_color_of_image[1]), tf.pad(image[:, :, 2], paddings, "CONSTANT", constant_values = mean_color_of_image[2])], axis=-1)
#         return big_canvas

# nms_mask = tf.cast(tf.ones_like(tf.constant([0,2,34,5,6]), dtype=tf.int8), tf.bool)
# indices = tf.where(nms_mask)
# print(sess.run(tf.shape(indices)[0]))
# print(sess.run(indices[-1][0]))

# print(not 1 < 2)

# tower_grads = [[(1,2),(3,4),(13,4)], [(1,2),(3,4),(13,4)], [(1,2),(3,4),(13,4)]]
# print([f for f in zip(*tower_grads)])

# [((1, 2), (1, 2)), ((3, 4), (3, 4))]

# print(sess.run(tf.one_hot(1, 5, on_value=False, off_value=True, dtype=tf.bool)))




# a = np.array([1,2,3], dtype=np.float32)
# b = np.array([[5,6], [15,16]], dtype=np.float32)

# print(np.expand_dims(b, axis=-1) - a)
# #print(tf.gfile.Glob('F:/*.jpg'))

# print(sess.run(ssd_random_expand(x)))
# mean_image = tf.reduce_mean(tf.reshape(x, [-1, 3]), 0)
# print(sess.run(tf.stack([tf.fill([5,6], mean_image[0]), tf.fill([5,6], mean_image[1]), tf.fill([5,6], mean_image[2])], axis=-1)))
# print(sess.run(tf.cast([1.2,3.4],tf.int32)[0]))
# print(int(0.999999))
# print(sess.run(roi))
# print(sess.run([sampled_width, sampled_height, x,y]))
# print(sess.run([sampled_width, sampled_height, x,y]))
# print(sess.run([sampled_width, sampled_height, x,y]))
# print(sess.run([sampled_width, sampled_height, x,y]))
# print(sess.run([sampled_width, sampled_height, x,y]))
# print(sess.run([sampled_width, sampled_height, x,y]))
# print(sess.run([sampled_width, sampled_height, x,y]))
# print(sess.run([sampled_width, sampled_height, x,y]))
# print(sess.run([sampled_width, sampled_height, x,y]))
# print(sess.run([sampled_width, sampled_height, x,y]))
# print(sess.run([sampled_width, sampled_height, x,y]))
# print(sess.run([sampled_width, sampled_height, x,y]))
# print(sess.run(tf.random_uniform([1], minval=0, maxval=0, dtype=tf.int32)[0]))
