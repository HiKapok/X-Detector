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

import tensorflow as tf

from . import resnet_v2
from . import depth_conv2d

#initializer_to_use = tf.glorot_uniform_initializer
initializer_to_use = tf.glorot_normal_initializer
conv_bn_initializer_to_use = tf.glorot_normal_initializer#lambda : tf.truncated_normal_initializer(mean=0.0, stddev=0.005)

def dilate_conv2d(inputs, filters, kernel_size, dilation_rate, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=1, dilation_rate = dilation_rate,
      padding='SAME', use_bias=False,#True,
      kernel_initializer=conv_bn_initializer_to_use(),
      bias_initializer=None,#tf.zeros_initializer(),
      data_format=data_format)

def xdet_bottleneck_block(inputs, filters, is_training, projection_shortcut,
                     dilation_rate, data_format):
  """Bottleneck block variant for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    dilation_rate: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = resnet_v2.batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = resnet_v2.conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format, kernel_initializer=conv_bn_initializer_to_use)

  inputs = resnet_v2.batch_norm_relu(inputs, is_training, data_format)
  inputs = dilate_conv2d(
      inputs=inputs, filters=filters, kernel_size=3, dilation_rate=dilation_rate,
      data_format=data_format)

  inputs = resnet_v2.batch_norm_relu(inputs, is_training, data_format)
  # default activation is None
  inputs = resnet_v2.conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format, kernel_initializer=conv_bn_initializer_to_use)

  return inputs + shortcut


def xdet_block_layer(inputs, filters, block_fn, blocks, dilation_rate, is_training, name,
                data_format):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    dilation_rate: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is xdet_bottleneck_block else filters

  def projection_shortcut(inputs):
    return tf.layers.conv2d(
                            inputs=inputs, filters=filters_out, kernel_size=1, strides=1,
                            padding='SAME', use_bias=False,
                            kernel_initializer=initializer_to_use(),
                            data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and dilation_rate
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, dilation_rate, data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, dilation_rate, data_format)

  return tf.identity(inputs, name)

def SEBlock(inputs, filters, data_format, is_training, rate = 8):
    # new_shape = x.get_shape().as_list()
    # new_shape[1] *= 2
    # new_shape[2] *= 2
    # new_shape[3] = to_channel_num

    # upsampled_x = tf.image.resize_images(x, new_shape[1:3])
    # ( input_iamge_size / 16 - (5 - 1) ) / 3, use suitable input_iamge_size to produce integer here
    #inputs = resnet_v2.batch_norm_relu(inputs, is_training, data_format)
    inputs = tf.layers.separable_conv2d(inputs, filters/rate, 5, strides = 3, padding='VALID', data_format=data_format,
                              activation = None, use_bias = True,
                              depthwise_initializer = initializer_to_use(),
                              pointwise_initializer = initializer_to_use(),
                              bias_initializer = tf.zeros_initializer())
    inputs = tf.nn.relu(inputs)
    inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size = 1, strides = 1, dilation_rate = 1,
                  padding='SAME', use_bias=True, activation = tf.sigmoid,
                  kernel_initializer=initializer_to_use(),
                  bias_initializer=tf.zeros_initializer(),
                  data_format=data_format)

    if data_format == 'channels_first':
      inputs = tf.transpose(inputs, [0, 2, 3, 1])

    input_shape = tf.shape(inputs)[-3:-1]
    # use default ResizeMethod.BILINEAR to zoom-in the input size
    outputs = tf.image.resize_images(inputs, input_shape * 3 + (5 - 1))
    if data_format == 'channels_first':
      outputs = tf.transpose(outputs, [0, 3, 1, 2])
    return outputs

def xdet_resnet_v3_generator(block_fn, layers, data_format=None):
  """Generator for X-Det ResNet v2 models.

  Args:
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    layers: A length-4 array denoting the number of blocks to include in each
      layer. Each layer consists of blocks that take inputs of the same size.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.

  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the ResNet model.
  """
  if data_format is None:
    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""
    # we do this in preprocess func
    #if data_format == 'channels_first':
      # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
      # This provides a large performance boost on GPU. See
      # https://www.tensorflow.org/performance/performance_guide#data_formats
      #inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = resnet_v2.conv2d_fixed_padding(
        inputs=inputs, filters=64, kernel_size=7, strides=2,
        data_format=data_format, kernel_initializer=conv_bn_initializer_to_use)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    inputs = resnet_v2.block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = resnet_v2.block_layer(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = resnet_v2.block_layer(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    output_conv5 = xdet_block_layer(
        inputs=inputs, filters=512, block_fn=xdet_bottleneck_block, blocks=layers[3],
        dilation_rate=2, is_training=is_training, name='block_layer4',
        data_format=data_format)
    with tf.variable_scope('xdet_additional_conv', default_name = None, values = [output_conv5], reuse=tf.AUTO_REUSE):
      output_conv6 = xdet_block_layer(
          inputs=output_conv5, filters=512, block_fn=xdet_bottleneck_block, blocks=layers[4],
          dilation_rate=4, is_training=is_training, name='block_layer5',
          data_format=data_format)
      output_conv7 = xdet_block_layer(
          inputs=output_conv6, filters=512, block_fn=xdet_bottleneck_block, blocks=layers[4],
          dilation_rate=8, is_training=is_training, name='block_layer6',
          data_format=data_format)
    with tf.variable_scope('xdet_multi_path', default_name = None, values = [output_conv5, output_conv6, output_conv7], reuse=tf.AUTO_REUSE):
      output_conv5 = resnet_v2.batch_norm_relu(output_conv5, is_training, data_format)
      output_conv5 = tf.layers.conv2d(inputs=output_conv5, filters=256, kernel_size=1, strides=1,
                                  padding='SAME', use_bias=False, activation=None,
                                  kernel_initializer=conv_bn_initializer_to_use(),
                                  bias_initializer=None,#tf.zeros_initializer(),
                                  data_format=data_format)
      output_conv6 = resnet_v2.batch_norm_relu(output_conv6, is_training, data_format)
      output_conv6 = tf.layers.conv2d(inputs=output_conv6, filters=256, kernel_size=1, strides=1,
                                  padding='SAME', use_bias=False, activation=None,
                                  kernel_initializer=conv_bn_initializer_to_use(),
                                  bias_initializer=None,#tf.zeros_initializer(),
                                  data_format=data_format)
      output_conv7 = resnet_v2.batch_norm_relu(output_conv7, is_training, data_format)
      output_conv7 = tf.layers.conv2d(inputs=output_conv7, filters=256, kernel_size=1, strides=1,
                                  padding='SAME', use_bias=False, activation=None,
                                  kernel_initializer=conv_bn_initializer_to_use(),
                                  bias_initializer=None,#tf.zeros_initializer(),
                                  data_format=data_format)
      if data_format == 'channels_first':
        concat_output = tf.concat([output_conv5, output_conv6, output_conv7], axis = 1)
      else:
        concat_output = tf.concat([output_conv5, output_conv6, output_conv7], axis = -1)

      concat_output = resnet_v2.batch_norm_only(concat_output, is_training, data_format)

      with tf.variable_scope('xdet_cls_seblock', default_name = None, values = [concat_output], reuse=tf.AUTO_REUSE):
        cls_seblock_output = SEBlock(concat_output, 256 * 3, data_format, is_training)
      with tf.variable_scope('xdet_regress_seblock', default_name = None, values = [concat_output], reuse=tf.AUTO_REUSE):
        regress_seblock_output = SEBlock(concat_output, 256 * 3, data_format, is_training)

      return tf.nn.relu(concat_output * cls_seblock_output), tf.nn.relu(concat_output * regress_seblock_output)

  return model


def pred_inception_module(net_input, depth_output, is_training, data_format, var_scope):
  with tf.variable_scope(var_scope):
    with tf.variable_scope('Branch_0'):
      branch_0 = tf.layers.conv2d(inputs=net_input, filters=depth_output, kernel_size=3, strides=1,
                                  padding='SAME', use_bias=True, activation=None,
                                  kernel_initializer=conv_bn_initializer_to_use(),
                                  bias_initializer=tf.zeros_initializer(),
                                  data_format=data_format)
    with tf.variable_scope('Branch_1'):
      branch_1 = tf.layers.conv2d(inputs=net_input, filters=depth_output, kernel_size=1, strides=1,
                                  padding='SAME', use_bias=True, activation=None,
                                  kernel_initializer=conv_bn_initializer_to_use(),
                                  bias_initializer=tf.zeros_initializer(),
                                  data_format=data_format)

    if data_format == 'channels_first':
      net_input = tf.concat([branch_0, branch_1], axis = 1)
    else:
      net_input = tf.concat([branch_0, branch_1], axis = -1)

    return resnet_v2.batch_norm_relu(net_input, is_training, data_format)


def xdet_head(body_cls_input, body_regress_input, num_classes, num_anchors, is_training, data_format=None):
  with tf.variable_scope('xdet_head', default_name = None, values = [body_cls_input, body_regress_input], reuse=tf.AUTO_REUSE):
    if data_format is None:
      data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    def pred_submodule(input_feature, output_channals):
      inputs = resnet_v2.conv2d_fixed_padding(inputs=input_feature, filters=output_channals, kernel_size=3, strides=1, data_format=data_format, kernel_initializer=conv_bn_initializer_to_use)
      return resnet_v2.batch_norm_relu(inputs, is_training, data_format)

    # never cause information bottleneck
    # classification module
    # cls_inputs = body_cls_input
    # cls_depth_list = [512, 256]
    # for depth_ in cls_depth_list:
    #   if num_anchors * num_classes < 0.8 * depth_:
    #     cls_inputs = pred_submodule(cls_inputs, depth_)

    cls_inputs = pred_inception_module(body_cls_input, 256, is_training, data_format, 'inception_1')
    cls_inputs = pred_inception_module(cls_inputs, 256, is_training, data_format, 'inception_2')

    cls_outputs = tf.layers.conv2d(
                                inputs=cls_inputs, filters=num_anchors * num_classes, kernel_size=3, strides=1,
                                padding='SAME', use_bias=True, activation=None,
                                kernel_initializer=initializer_to_use(),
                                data_format=data_format)

    # regress module
    regress_inputs = body_regress_input
    regress_depth_list = [512, 256]
    for depth_ in regress_depth_list:
      if num_anchors < depth_/(4 + 1):
        regress_inputs = pred_submodule(regress_inputs, depth_)

    regress_outputs = tf.layers.conv2d(
                                inputs=regress_inputs, filters=num_anchors * 4, kernel_size=3, strides=1,
                                padding='SAME', use_bias=True, activation=None,
                                kernel_initializer=initializer_to_use(), bias_initializer=tf.zeros_initializer(),
                                data_format=data_format)

    return tf.identity(cls_outputs, 'class_module'), tf.identity(regress_outputs, 'location_module')

def xdet_resnet_v3(resnet_size, data_format=None):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': resnet_v2.building_block, 'layers': [2, 2, 2, 2, 2, 2]},
      34: {'block': resnet_v2.building_block, 'layers': [3, 4, 6, 3, 3, 3]},
      50: {'block': resnet_v2.bottleneck_block, 'layers': [3, 4, 6, 3, 3, 3]},
      101: {'block': resnet_v2.bottleneck_block, 'layers': [3, 4, 23, 3, 3, 3]},
      152: {'block': resnet_v2.bottleneck_block, 'layers': [3, 8, 36, 3, 3, 3]},
      200: {'block': resnet_v2.bottleneck_block, 'layers': [3, 24, 36, 3, 3, 3]}
  }

  if resnet_size not in model_params:
    raise ValueError('Not a valid resnet_size:', resnet_size)

  params = model_params[resnet_size]
  return xdet_resnet_v3_generator(
      params['block'], params['layers'], data_format)
