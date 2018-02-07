import os
import shutil
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import math

LIB_NAME = 'ps_roi_pooling'

def load_op_module(lib_name):
  """
  Load TensorFlow operator library.
  """
  # use absolute path so that ops.py can be called from other directory
  lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cpp/build/lib{0}.so'.format(lib_name))
  # duplicate library with a random new name so that
  # a running program will not be interrupted when the original library is updated
  lib_copy_path = '/tmp/lib{0}_{1}.so'.format(str(uuid.uuid4())[:8], LIB_NAME)
  shutil.copyfile(lib_path, lib_copy_path)
  oplib = tf.load_op_library(lib_copy_path)
  return oplib

op_module = load_op_module(LIB_NAME)

class PSROIPoolingTest(tf.test.TestCase):
  def testPSROIPooling(self):
    with tf.device('/gpu:0'):
      # map C++ operators to python objects
      ps_roi_pooling = op_module.ps_roi_pooling
      with self.test_session():
        result = ps_roi_pooling([[1., 2., 3., 0, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0]], [[1., 0, 0, 0, 0],[0, 1., 0, 0, 0],[0, 0, 1., 0, 0],[0, 0, 0, 1., 0]], [1, 1., 1, 1], 5.)
        print('ps_roi_pooling in gpu:', result.eval())
    with tf.device('/cpu:0'):
      # map C++ operators to python objects
      ps_roi_pooling = op_module.ps_roi_pooling
      with self.test_session():
        result = ps_roi_pooling([[1., 2., 3., 0, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0]], [[1., 0, 0, 0, 0],[0, 1., 0, 0, 0],[0, 0, 1., 0, 0],[0, 0, 0, 1., 0]], [1, 1., 1, 1], 5.)
        print('ps_roi_pooling in cpu:', result.eval())
        # expect [3.18034267  0.39960092  0.00709875  2.96500921]
        #self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

# @ops.RegisterGradient("PSROIPooling")
# def _ps_roi_pooling_grad(op, grad):
#     """The gradients for `ps_roi_pooling`.

#     Args:
#     op: The `ps_roi_pooling` `Operation` that we are differentiating, which we can use
#       to find the inputs and outputs of the original op.
#     grad: Gradient with respect to the output of the `ps_roi_pooling` op.

#     Returns:
#     Gradients with respect to the input of `ps_roi_pooling`.
#     """
#     logits = op.inputs[0]
#     one_hot_labels = op.inputs[1]
#     batch_weight = op.inputs[2]
#     ps_roi_pooling_gamma = op.inputs[3]
#     ps_roi_pooling = op.outputs[0]
#     ligits_shape = array_ops.shape(logits)

#     probs = tf.nn.softmax(logits)
#     # in fact, tf.shape(probs)[0] is also a tensor
#     # tf.get_shape().as_list() for known shape
#     indices = tf.stack((tf.range(tf.cast(tf.shape(probs)[0], tf.int64), dtype=tf.int64), tf.argmax(one_hot_labels, axis=1)), axis=1)
#     #indices = tf.stack((tf.range(probs.get_shape()[0], dtype=tf.int64), tf.argmax(one_hot_labels, axis=1)), axis=1)
#     prob_foreach = tf.gather_nd( probs, indices )
#     prob_foreach_subbyone = 1 - prob_foreach

#     grad_true = 0. - tf.add(prob_foreach*ps_roi_pooling*focal_loss_gamma, batch_weight * tf.pow(prob_foreach_subbyone, focal_loss_gamma + 1))
#     scatter_mask = 1. - tf.scatter_nd(tf.cast(indices, tf.int32), tf.ones_like(prob_foreach), ligits_shape)
#     grad_false = tf.expand_dims(tf.div(prob_foreach * ps_roi_pooling * focal_loss_gamma, prob_foreach_subbyone) + batch_weight*tf.pow(prob_foreach_subbyone, focal_loss_gamma), axis=1) * probs
#     scatter_grad_true = tf.scatter_nd(tf.cast(indices, tf.int32), grad_true, ligits_shape)
#     #grad_false * scatter_mask + scatter_grad_true
#     return [tf.expand_dims(grad, 1)  * (grad_false * scatter_mask + scatter_grad_true), None, None, None]  # List of one Tensor, use None for no well-defined gradient of some input,

@ops.RegisterGradient("PSROIPooling")
def _ps_roi_pooling_grad(op, grad):
    """The gradients for `focal_loss`.

    Args:
    op: The `focal_loss` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `focal_loss` op.

    Returns:
    Gradients with respect to the input of `focal_loss`.
    """
    ps_roi_pooling_grad = op_module.ps_roi_pooling_grad
    logits = op.inputs[0]
    one_hot_labels = op.inputs[1]
    batch_weight = op.inputs[2]
    focal_loss_gamma = op.inputs[3]
    focal_loss = op.outputs[0]

    return [tf.expand_dims(grad, 1) * ps_roi_pooling_grad(logits, tf.cast(tf.argmax(one_hot_labels, 1), tf.float32), batch_weight, focal_loss_gamma, focal_loss), None, None, None]

def ps_roi_pooling_tf(logits, one_hot, weight, gamma):
  # prob = tf.nn.softmax(logits)
  # percls_loss = tf.subtract(tf.zeros_like(prob), tf.multiply(tf.pow(tf.subtract(tf.ones_like(prob), prob), gamma), tf.log(prob)))
  # losses = tf.multiply(tf.reduce_sum(tf.multiply(one_hot, percls_loss), 1), weight)
  # return losses
  prob = tf.nn.softmax(logits)
  return tf.reduce_sum(one_hot * (0. - tf.pow(1 - prob, gamma) * tf.nn.log_softmax(logits)), 1) * weight#tf.reduce_mean(tf.reduce_sum(one_hot * (0. - tf.pow(1 - prob, gamma) * tf.nn.log_softmax(logits)), 1) * weight)

class PSROIPoolingTFTest(tf.test.TestCase):
  def testPSROIPoolingTF(self):
    with self.test_session():
      result = ps_roi_pooling_tf([[1., 2., 3., 0, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0]], [[1., 0, 0, 0, 0],[0, 1., 0, 0, 0],[0, 0, 1., 0, 0],[0, 0, 0, 1., 0]], [1, 1., 1, 1], 5.)
      print(result.eval())


class PSROIPoolingGradTest(tf.test.TestCase):
  def testPSROIPoolingGrad(self):
    ps_roi_pooling = op_module.ps_roi_pooling

    logits = tf.constant([[1.0, 2., 3., 30, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0]], dtype=tf.float32)
    rand_logits = tf.placeholder(tf.float32, shape=(4, 5))
    one_hot = tf.constant([[1., 0, 0, 0, 0],[0, 1., 0., 0, 0],[0, 0, 1., 0, 0],[0, 0, 0, 1., 0]], dtype=tf.float32)
    weight = tf.constant([2, 1., 1, 1], dtype=tf.float32)
    loss = ps_roi_pooling(logits, one_hot, weight, tf.constant(5., dtype=tf.float32))
    loss_tf = ps_roi_pooling_tf(logits, one_hot, weight, tf.constant(5., dtype=tf.float32))

    with tf.Session() as sess:
      print(tf.test.compute_gradient_error(logits, [4, 5], loss, [4], delta=0.0001, x_init_value=np.array([[1.0, 2., 3., 30, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0]])))
      print(tf.test.compute_gradient_error(logits, [4, 5], loss_tf, [4], delta=0.0001, x_init_value=np.array([[1.0, 2., 3., 30, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0]])))
      print(tf.test.compute_gradient(logits, [4, 5], loss, [4], x_init_value=np.array([[1.0, 2., 3., 30, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0]])))
      print(tf.test.compute_gradient(logits, [4, 5], loss_tf, [4], x_init_value=np.array([[1.0, 2., 3., 30, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0]])))


class PSROIPoolingBackTest(tf.test.TestCase):
  def testPSROIPoolingBack(self):
    # map C++ operators to python objects
    ps_roi_pooling_grad = op_module.ps_roi_pooling_grad
    with self.test_session():
      pass
      # result = ps_roi_pooling_grad([[1., 2., 3., 0, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0],[1., 2., 3., 0, 0]], [1, 2, 3, 4], [2, 1., 2, 1, 1], 5., [3.18034267, 0.39960092, 0.00709875, 2.96500921])
      #print(result.eval())

if __name__ == "__main__":
  tf.test.main()
