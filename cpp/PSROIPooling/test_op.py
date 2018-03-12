# MIT License

# Copyright (c) 2018 Changan Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import shutil
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import math

LIB_NAME = 'ps_roi_align'

def load_op_module(lib_name):
  """
  Load TensorFlow operator library.
  """
  # use absolute path so that ops.py can be called from other directory
  lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'build/lib{0}.so'.format(lib_name))
  # duplicate library with a random new name so that
  # a running program will not be interrupted when the original library is updated
  lib_copy_path = '/tmp/lib{0}_{1}.so'.format(str(uuid.uuid4())[:8], LIB_NAME)
  shutil.copyfile(lib_path, lib_copy_path)
  oplib = tf.load_op_library(lib_copy_path)
  #print(_)
  return oplib

op_module = load_op_module(LIB_NAME)
#print("----",op_module.OP_LIST)

# map_to_pool = [[[[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]], [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]], [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]], [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]]]]

map_to_pool = [[
              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],
              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],
              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],
              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],

              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],
              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],
              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],
              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],

              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],
              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],
              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],
              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],

              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],
              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],
              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]],
              [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.], [11., 12., 13., 14., 15.], [16., 17., 18., 19., 20.], [21., 22., 23., 24., 25.]]
            ]]

pool_method = 'mean'

class PSROIAlignTest(tf.test.TestCase):
  def testPSROIAlign(self):
    with tf.device('/gpu:1'):
      # map C++ operators to python objects
      ps_roi_align = op_module.ps_roi_align
      result = ps_roi_align(map_to_pool, [[[0.2, 0.2, 0.7, 0.7], [0.5, 0.5, 0.9, 0.9], [0.9, 0.9, 1., 1.]]], 2, 2, pool_method)
      with self.test_session() as sess:
        print('ps_roi_align in gpu:', sess.run(result))
    with tf.device('/cpu:0'):
      # map C++ operators to python objects
      ps_roi_align = op_module.ps_roi_align
      result = ps_roi_align(map_to_pool, [[[0.2, 0.2, 0.7, 0.7], [0.5, 0.5, 0.9, 0.9], [0.9, 0.9, 1., 1.]]], 2, 2, pool_method)
      with self.test_session() as sess:
        print('ps_roi_align in cpu:', sess.run(result))
        # expect [3.18034267  0.39960092  0.00709875  2.96500921]
        #self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

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

  return [op_module.ps_roi_align_grad(inputs_features, rois, grad, pooled_index, grid_dim_width, grid_dim_height, pool_method), None]

class PSROIAlignGradTest(tf.test.TestCase):
  def testPSROIAlignGrad(self):
    with tf.device('/cpu:0'):
      ps_roi_align = op_module.ps_roi_align
      inputs_features = tf.constant(map_to_pool, dtype=tf.float32)
      pool_result = ps_roi_align(inputs_features, [[[0.2, 0.2, 0.7, 0.7], [0.5, 0.5, 0.9, 0.9], [0.9, 0.9, 1., 1.]]], 2, 2, pool_method)
      with tf.Session() as sess:
        #print(sess.run(tf.gradients(pool_result[0], [inputs_features])))
        print(tf.test.compute_gradient_error(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool)))
        # _, jaccobian = tf.test.compute_gradient(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool))
        # y = sess.run(pool_result[0])
        # print(jaccobian.shape)
        # print(np.reshape(np.matmul(jaccobian, np.ones_like(y.flatten())), np.array(map_to_pool).shape))
        print(tf.test.compute_gradient(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool)))
    with tf.device('/gpu:0'):
      ps_roi_align = op_module.ps_roi_align
      inputs_features = tf.constant(map_to_pool, dtype=tf.float32)
      pool_result = ps_roi_align(inputs_features, [[[0.2, 0.2, 0.7, 0.7], [0.5, 0.5, 0.9, 0.9], [0.9, 0.9, 1., 1.]]], 2, 2, pool_method)
      with tf.Session() as sess:
        #print(sess.run(tf.gradients(pool_result[0], [inputs_features])))
        print(tf.test.compute_gradient_error(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool)))
        # _, jaccobian = tf.test.compute_gradient(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool))
        # y = sess.run(pool_result[0])
        # print(jaccobian.shape)
        # print(np.reshape(np.matmul(jaccobian, np.ones_like(y.flatten())), np.array(map_to_pool).shape))
        print(tf.test.compute_gradient(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool)))

class RotatedPSROIAlignTest(tf.test.TestCase):
  def testRotatedPSROIAlign(self):
    with tf.device('/gpu:1'):
      # map C++ operators to python objects
      rotated_ps_roi_align = op_module.rotated_ps_roi_align
      result = rotated_ps_roi_align(map_to_pool, [[[0.1, 0.1, 0.2, 0.3, 0.5, 0.5, 0.3, 0.2], [0.5, 0.5, 0.6, 0.7, 0.9, 0.9, 0.7, 0.6], [0.6, 0.7, 0.9, 0.9, 0.7, 0.6, 0.2, 0.2]]], [[1, -1, 0]], 2, 2, pool_method)
      with self.test_session() as sess:
        print('rotated_ps_roi_align in gpu:', sess.run(result))
    with tf.device('/cpu:0'):
      # map C++ operators to python objects
      rotated_ps_roi_align = op_module.rotated_ps_roi_align
      result = rotated_ps_roi_align(map_to_pool, [[[0.1, 0.1, 0.2, 0.3, 0.5, 0.5, 0.3, 0.2], [0.5, 0.5, 0.6, 0.7, 0.9, 0.9, 0.7, 0.6], [0.6, 0.7, 0.9, 0.9, 0.7, 0.6, 0.2, 0.2]]], [[1, -1, 0]], 2, 2, pool_method)
      with self.test_session() as sess:
        print('rotated_ps_roi_align in cpu:', sess.run(result))
        # expect [3.18034267  0.39960092  0.00709875  2.96500921]
        #self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

@ops.RegisterGradient("RotatedPsRoiAlign")
def _rotated_ps_roi_align_grad(op, grad, _):
  '''The gradients for `RotatedPsRoiAlign`.
  '''
  inputs_features = op.inputs[0]
  rois = op.inputs[1]
  orders = op.inputs[2]
  pooled_features_grad = op.outputs[0]
  pooled_index = op.outputs[1]
  grid_dim_width = op.get_attr('grid_dim_width')
  grid_dim_height = op.get_attr('grid_dim_height')

  return [op_module.rotated_ps_roi_align_grad(inputs_features, rois, orders, grad, pooled_index, grid_dim_width, grid_dim_height, pool_method), None, None]

class RotatedPSROIAlignGradTest(tf.test.TestCase):
  def testRotatedPSROIAlignGrad(self):
    with tf.device('/cpu:0'):
      rotated_ps_roi_align = op_module.rotated_ps_roi_align
      inputs_features = tf.constant(map_to_pool, dtype=tf.float32)
      pool_result = rotated_ps_roi_align(inputs_features, [[[0.1, 0.1, 0.2, 0.3, 0.5, 0.5, 0.3, 0.2], [0.5, 0.5, 0.6, 0.7, 0.9, 0.9, 0.7, 0.6], [0.6, 0.7, 0.9, 0.9, 0.7, 0.6, 0.2, 0.2]]], [[1, -1, 0]], 2, 2, pool_method)
      with tf.Session() as sess:
        #print(sess.run(tf.gradients(pool_result[0], [inputs_features])))
        print(tf.test.compute_gradient_error(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool)))
        # _, jaccobian = tf.test.compute_gradient(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool))
        # y = sess.run(pool_result[0])
        # print(jaccobian.shape)
        # print(np.reshape(np.matmul(jaccobian, np.ones_like(y.flatten())), np.array(map_to_pool).shape))
        print(tf.test.compute_gradient(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool)))
    with tf.device('/gpu:0'):
      rotated_ps_roi_align = op_module.rotated_ps_roi_align
      inputs_features = tf.constant(map_to_pool, dtype=tf.float32)
      pool_result = rotated_ps_roi_align(inputs_features, [[[0.1, 0.1, 0.2, 0.3, 0.5, 0.5, 0.3, 0.2], [0.5, 0.5, 0.6, 0.7, 0.9, 0.9, 0.7, 0.6], [0.6, 0.7, 0.9, 0.9, 0.7, 0.6, 0.2, 0.2]]], [[1, -1, 0]], 2, 2, pool_method)
      with tf.Session() as sess:
        #print(sess.run(tf.gradients(pool_result[0], [inputs_features])))
        print(tf.test.compute_gradient_error(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool)))
        # _, jaccobian = tf.test.compute_gradient(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool))
        # y = sess.run(pool_result[0])
        # print(jaccobian.shape)
        # print(np.reshape(np.matmul(jaccobian, np.ones_like(y.flatten())), np.array(map_to_pool).shape))
        print(tf.test.compute_gradient(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool)))

if __name__ == "__main__":
  tf.test.main()
