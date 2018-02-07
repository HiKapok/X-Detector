# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""Xception V1 model for Keras.

On ImageNet, this model gets to a top-1 validation accuracy of 0.790
and a top-5 validation accuracy of 0.945.

Do note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function
is also different (same as Inception V3).

Also do note that this model is only available for the TensorFlow backend,
due to its reliance on `SeparableConvolution` layers.

# Reference

- [Xception: Deep Learning with Depthwise Separable
Convolutions](https://arxiv.org/abs/1610.02357)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import layers
from keras.preprocessing import image
from tensorflow.python.keras._impl.keras.applications import imagenet_utils
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import decode_predictions  # pylint: disable=unused-import
from tensorflow.python.keras._impl.keras.engine.topology import get_source_inputs
from tensorflow.python.keras._impl.keras.layers import Activation
from tensorflow.python.keras._impl.keras.layers import BatchNormalization
from tensorflow.python.keras._impl.keras.layers import Conv2D
from tensorflow.python.keras._impl.keras.layers import Dense
from tensorflow.python.keras._impl.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras._impl.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras._impl.keras.layers import Input
from tensorflow.python.keras._impl.keras.layers import MaxPooling2D
from tensorflow.python.keras._impl.keras.layers import SeparableConv2D
from tensorflow.python.keras._impl.keras.models import Model
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file
from tensorflow.python.platform import tf_logging as logging


TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


def Xception(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):
  """Instantiates the Xception architecture.

  Optionally loads weights pre-trained
  on ImageNet. This model is available for TensorFlow only,
  and can only be used with inputs following the TensorFlow
  data format `(width, height, channels)`.
  You should set `image_data_format="channels_last"` in your Keras config
  located at ~/.keras/keras.json.

  Note that the default input image size for this model is 299x299.

  Arguments:
      include_top: whether to include the fully-connected
          layer at the top of the network.
      weights: one of `None` (random initialization),
          'imagenet' (pre-training on ImageNet),
          or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(299, 299, 3)`.
          It should have exactly 3 input channels,
          and width and height should be no smaller than 71.
          E.g. `(150, 150, 3)` would be one valid value.
      pooling: Optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.

  Returns:
      A Keras model instance.

  Raises:
      ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
      RuntimeError: If attempting to run this model with a
          backend that does not support separable convolutions.
  """
  if not (weights in {'imagenet', None} or os.path.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as imagenet with `include_top`'
                     ' as true, `classes` should be 1000')

  if K.image_data_format() != 'channels_last':
    logging.warning(
        'The Xception model is only available for the '
        'input data format "channels_last" '
        '(width, height, channels). '
        'However your settings specify the default '
        'data format "channels_first" (channels, width, height). '
        'You should set `image_data_format="channels_last"` in your Keras '
        'config located at ~/.keras/keras.json. '
        'The model being returned right now will expect inputs '
        'to follow the "channels_last" data format.')
    K.set_image_data_format('channels_last')
    old_data_format = 'channels_first'
  else:
    old_data_format = None

  # Determine proper input shape
  input_shape = _obtain_input_shape(
      input_shape,
      default_size=299,
      min_size=71,
      data_format=K.image_data_format(),
      require_flatten=False,
      weights=weights)

  if input_tensor is None:
    img_input = Input(shape=input_shape)
  else:
    img_input = Input(tensor=input_tensor, shape=input_shape)

  x = Conv2D(
      32, (3, 3), strides=(2, 2), use_bias=False,
      name='block1_conv1')(img_input)
  x = BatchNormalization(name='block1_conv1_bn')(x)
  x = Activation('relu', name='block1_conv1_act')(x)
  x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
  x = BatchNormalization(name='block1_conv2_bn')(x)
  x = Activation('relu', name='block1_conv2_act')(x)

  residual = Conv2D(
      128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
  residual = BatchNormalization()(residual)

  x = SeparableConv2D(
      128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
  x = BatchNormalization(name='block2_sepconv1_bn')(x)
  x = Activation('relu', name='block2_sepconv2_act')(x)
  x = SeparableConv2D(
      128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
  x = BatchNormalization(name='block2_sepconv2_bn')(x)

  x = MaxPooling2D(
      (3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
  x = layers.add([x, residual])

  residual = Conv2D(
      256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
  residual = BatchNormalization()(residual)

  x = Activation('relu', name='block3_sepconv1_act')(x)
  x = SeparableConv2D(
      256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
  x = BatchNormalization(name='block3_sepconv1_bn')(x)
  x = Activation('relu', name='block3_sepconv2_act')(x)
  x = SeparableConv2D(
      256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
  x = BatchNormalization(name='block3_sepconv2_bn')(x)

  x = MaxPooling2D(
      (3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
  x = layers.add([x, residual])

  residual = Conv2D(
      728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
  residual = BatchNormalization()(residual)

  x = Activation('relu', name='block4_sepconv1_act')(x)
  x = SeparableConv2D(
      728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
  x = BatchNormalization(name='block4_sepconv1_bn')(x)
  x = Activation('relu', name='block4_sepconv2_act')(x)
  x = SeparableConv2D(
      728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
  x = BatchNormalization(name='block4_sepconv2_bn')(x)

  x = MaxPooling2D(
      (3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
  x = layers.add([x, residual])

  for i in range(8):
    residual = x
    prefix = 'block' + str(i + 5)

    x = Activation('relu', name=prefix + '_sepconv1_act')(x)
    x = SeparableConv2D(
        728, (3, 3), padding='same', use_bias=False,
        name=prefix + '_sepconv1')(x)
    x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
    x = Activation('relu', name=prefix + '_sepconv2_act')(x)
    x = SeparableConv2D(
        728, (3, 3), padding='same', use_bias=False,
        name=prefix + '_sepconv2')(x)
    x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
    x = Activation('relu', name=prefix + '_sepconv3_act')(x)
    x = SeparableConv2D(
        728, (3, 3), padding='same', use_bias=False,
        name=prefix + '_sepconv3')(x)
    x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

    x = layers.add([x, residual])

  residual = Conv2D(
      1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
  residual = BatchNormalization()(residual)

  x = Activation('relu', name='block13_sepconv1_act')(x)
  x = SeparableConv2D(
      728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
  x = BatchNormalization(name='block13_sepconv1_bn')(x)
  x = Activation('relu', name='block13_sepconv2_act')(x)
  x = SeparableConv2D(
      1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
  x = BatchNormalization(name='block13_sepconv2_bn')(x)

  x = MaxPooling2D(
      (3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
  x = layers.add([x, residual])

  x = SeparableConv2D(
      1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
  x = BatchNormalization(name='block14_sepconv1_bn')(x)
  x = Activation('relu', name='block14_sepconv1_act')(x)

  x = SeparableConv2D(
      2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
  x = BatchNormalization(name='block14_sepconv2_bn')(x)
  x = Activation('relu', name='block14_sepconv2_act')(x)

  if include_top:
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
  else:
    if pooling == 'avg':
      x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.
  model = Model(inputs, x, name='xception')

  # load weights
  if weights == 'imagenet':
    if include_top:
      weights_path = get_file(
          'xception_weights_tf_dim_ordering_tf_kernels.h5',
          TF_WEIGHTS_PATH,
          cache_subdir='models',
          file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')
    else:
      weights_path = get_file(
          'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
          TF_WEIGHTS_PATH_NO_TOP,
          cache_subdir='models',
          file_hash='b0042744bf5b25fce3cb969f33bebb97')
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  if old_data_format:
    K.set_image_data_format(old_data_format)
  elif weights is not None:
    model.load_weights(weights)
  return model


def preprocess_input(x):
  """Preprocesses a numpy array encoding a batch of images.

  Arguments:
      x: a 4D numpy array consists of RGB values within [0, 255].

  Returns:
      Preprocessed array.
  """
  return imagenet_utils.preprocess_input(x, mode='tf')


# if __name__ == '__main__':
#     # image size = 299 * 299
#     model = Xception(include_top=True, weights='./imagenet_xception.h5')


#     img_path = 'images/000010.jpg'
#     img = image.load_img(img_path, target_size=(299, 299))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     print('Input image shape:', x.shape)

#     preds = model.predict(x)

#     print(np.argmax(preds))
#     print('Predicted:', decode_predictions(preds, 1))


if __name__ == '__main__':
    # image size = 299 * 299
    model = Xception(include_top=True, weights='./imagenet_xception.h5')

    preds = model.predict(np.ones((1,299,299,3)) * 0.5)
    print(preds)
    print(np.argmax(preds))
    #print('Predicted:', decode_predictions(preds, 1))



    # img_path = 'elephant.jpg'
    # img = image.load_img(img_path, target_size=(299, 299))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # print('Input image shape:', x.shape)


# output
# [[2.44039271e-04 3.40558181e-04 2.15881926e-04 2.18360859e-04
#   2.88262585e-04 3.51772498e-04 1.39083553e-04 4.87553130e-04
#   1.78387156e-04 4.48726409e-04 3.20674240e-04 2.66706105e-04
#   3.06589471e-04 4.18273557e-04 2.00233408e-04 3.22850014e-04
#   7.58993905e-04 1.70717234e-04 3.65104701e-04 3.29575443e-04
#   1.87462647e-04 3.42177832e-03 1.57531677e-03 9.12522897e-04
#   4.74808650e-04 5.03750867e-04 3.02915170e-04 4.55694390e-04
#   6.59965444e-04 1.76039888e-04 1.54873880e-04 3.50133661e-04
#   2.32449209e-04 1.31015753e-04 2.25184005e-04 1.97169618e-04
#   1.99169852e-04 2.19373265e-04 1.17472874e-03 1.84896853e-04
#   2.65799608e-04 2.60540663e-04 5.73144061e-04 3.21989821e-04
#   3.50478251e-04 3.86057189e-04 3.34256969e-04 5.15017891e-04
#   1.53190907e-04 1.79798517e-04 3.44967120e-04 7.03162805e-04
#   7.87434285e-04 4.21380508e-04 2.14891857e-04 3.61554848e-04
#   3.10995063e-04 2.04628275e-04 1.29632099e-04 8.39248416e-04
#   2.74911668e-04 2.76182080e-04 2.06329321e-04 4.80624294e-04
#   4.94690612e-04 2.69681681e-04 4.02257137e-04 2.63020571e-04
#   2.59118213e-04 3.97932658e-04 2.05551391e-04 1.31746510e-03
#   2.75826606e-04 1.34255807e-03 1.79195180e-04 6.80175959e-04
#   3.13495810e-04 3.64334264e-04 1.46027037e-03 9.52783914e-04
#   4.19965101e-04 3.85436579e-04 2.51578924e-04 1.94609733e-04
#   1.51887696e-04 2.77458894e-04 3.00190499e-04 3.23872897e-04
#   3.22439068e-04 5.81079745e-04 3.34015000e-04 4.10893816e-04
#   1.16701215e-03 3.62874067e-04 3.90285801e-04 2.26313728e-04
#   3.79570789e-04 1.57272807e-04 2.77454412e-04 2.67574447e-04
#   2.48215889e-04 2.22690447e-04 2.06891666e-04 4.18494135e-04
#   2.22274481e-04 8.56392217e-05 1.63585035e-04 3.13926197e-04
#   2.55956664e-04 2.12145053e-04 3.58709745e-04 7.93615077e-03
#   4.01076104e-04 5.57611114e-04 2.31016937e-04 2.04064418e-04
#   2.35709755e-04 8.38462496e-04 1.87619764e-04 1.72218439e-04
#   3.08697781e-04 1.55434886e-04 4.16235998e-04 1.85059034e-04
#   2.28378223e-04 1.75888345e-04 3.64472769e-04 1.20374968e-03
#   1.53327931e-03 4.84266930e-04 2.21745693e-04 2.03516349e-04
#   5.04454365e-04 2.49830220e-04 3.45752982e-04 2.43297181e-04
#   1.50585809e-04 2.24065225e-04 2.56275554e-04 2.86073046e-04
#   1.85481505e-04 5.48519893e-04 1.57625313e-04 3.40887840e-04
#   4.06851992e-04 2.09394930e-04 6.88385917e-04 2.52083264e-04
#   2.52348196e-04 4.54790570e-04 7.23221019e-05 3.64280742e-04
#   4.06965817e-04 2.91487435e-04 1.51777349e-04 1.77145936e-04
#   2.52168393e-04 3.16359248e-04 1.41630066e-04 1.50666237e-04
#   1.69904451e-04 1.96407389e-04 1.52202061e-04 7.17518677e-04
#   1.92121079e-04 2.42713781e-04 1.71669119e-04 2.34745239e-04
#   2.21430295e-04 1.74707078e-04 2.60429719e-04 1.54904890e-04
#   3.15618381e-04 2.74902093e-04 2.36956927e-04 3.11621814e-04
#   1.57982577e-04 1.72291693e-04 2.26382370e-04 2.84159905e-04
#   2.24678719e-04 3.60317208e-04 2.57711770e-04 3.12268996e-04
#   1.87667552e-04 4.04387130e-04 3.41737730e-04 2.92282697e-04
#   3.13932047e-04 3.04615358e-04 8.05544958e-04 1.81015159e-04
#   1.32337300e-04 3.13142780e-04 4.27515100e-04 5.10894461e-04
#   2.50709854e-04 3.71323520e-04 2.49028322e-04 4.90829232e-04
#   2.96531915e-04 1.22216574e-04 1.83282216e-04 2.50772369e-04
#   1.79878873e-04 1.28711443e-04 2.67514115e-04 1.87228768e-04
#   3.16528225e-04 1.29724533e-04 2.30620164e-04 1.22380967e-04
#   1.02163598e-04 1.19242010e-04 4.09552973e-04 1.21525969e-04
#   4.17467090e-04 3.25925328e-04 1.63337405e-04 2.49359029e-04
#   5.74716309e-04 2.79885280e-04 3.90140136e-04 3.64086271e-04
#   2.21971131e-04 1.90079198e-04 2.28959718e-04 2.78563559e-04
#   3.74788215e-04 1.98245267e-04 6.89025183e-05 1.64592333e-04
#   1.82392861e-04 2.49615987e-04 3.15534387e-04 5.02826471e-04
#   1.89281171e-04 1.45130107e-04 1.47488070e-04 3.39709572e-04
#   2.66446412e-04 1.94611304e-04 8.37609696e-05 2.75810948e-04
#   1.98132067e-04 6.36467361e-04 2.36530890e-04 3.72744486e-04
#   2.56353756e-04 3.93280032e-04 2.74396996e-04 3.63109633e-04
#   4.57760849e-04 1.84386183e-04 6.00714702e-04 1.58156094e-04
#   2.32384817e-04 3.20812513e-04 3.13669880e-04 1.34746355e-04
#   2.37498520e-04 2.30278514e-04 3.57826153e-04 1.52256558e-04
#   1.59467207e-04 2.95665581e-04 2.17578636e-04 9.29895323e-05
#   2.16281580e-04 3.29788774e-04 4.52707987e-04 1.84365272e-04
#   2.74440274e-04 2.57835898e-04 2.31737504e-04 2.12123530e-04
#   1.92458145e-04 2.31324244e-04 1.13956849e-04 4.33975161e-04
#   1.71093139e-04 3.51679744e-04 1.79462557e-04 4.91213228e-04
#   4.44191042e-04 5.74295118e-04 1.57494957e-04 1.36551680e-04
#   2.34418389e-04 2.19174282e-04 1.59293937e-04 1.41067532e-04
#   2.82207067e-04 4.24366823e-04 1.11401474e-04 1.03822393e-04
#   2.80887383e-04 3.55053053e-04 1.91373707e-04 3.53992975e-04
#   3.83352744e-04 3.61665559e-04 7.56208145e-04 5.84480702e-04
#   2.38649081e-04 3.10221338e-04 1.24729460e-03 4.15439688e-04
#   5.48211450e-04 1.49512940e-04 7.23765464e-04 6.26679743e-04
#   5.70744800e-04 6.21201587e-04 1.14783971e-03 6.55618322e-04
#   2.89018877e-04 2.15038119e-04 9.44400148e-04 5.81019616e-04
#   4.26979823e-04 2.16945322e-04 7.92749634e-05 2.13641688e-04
#   1.53813060e-04 1.35564769e-04 3.10822361e-04 1.68038358e-04
#   3.54396121e-04 2.15596141e-04 3.15417681e-04 3.87888780e-04
#   3.59496218e-04 1.80806397e-04 1.72377186e-04 1.09466193e-04
#   1.63547214e-04 2.43061688e-04 1.27891093e-04 1.76975285e-04
#   2.38554319e-04 2.94405327e-04 2.47226388e-04 1.63817182e-04
#   2.35763044e-04 3.08199204e-04 2.37016473e-04 2.59984547e-04
#   2.57825479e-04 1.93464002e-04 2.93404999e-04 2.76442181e-04
#   1.70701664e-04 2.80819193e-04 5.37016487e-04 1.71333173e-04
#   3.71195521e-04 2.54716491e-04 3.85011343e-04 3.86854197e-04
#   1.20281868e-04 5.43218106e-04 2.53400009e-04 1.91137151e-04
#   3.93087743e-04 3.43329710e-04 2.18025918e-04 3.27854883e-04
#   1.78854913e-04 3.01071675e-04 1.33301743e-04 1.75242487e-04
#   8.84092442e-05 2.21244234e-04 1.09776433e-04 1.87992948e-04
#   2.29415527e-04 1.31361056e-04 3.13597353e-04 1.81626543e-04
#   2.38912893e-04 2.19335940e-04 3.36327445e-04 1.34159229e-04
#   1.66972066e-04 2.29641548e-04 1.45203623e-04 1.04638020e-04
#   1.68155864e-04 2.00599898e-04 1.21051839e-04 1.62631040e-04
#   1.68168772e-04 3.30573152e-04 1.87161218e-04 1.79293551e-04
#   1.44042991e-04 2.15156688e-04 6.85554405e-04 2.75474484e-03
#   6.32824667e-04 4.85791650e-04 1.44409831e-03 4.87420039e-04
#   8.29513185e-04 1.70392892e-03 9.72181297e-05 1.60706899e-04
#   1.71611740e-04 3.51346028e-03 3.41578561e-04 3.01660824e-04
#   7.81773881e-04 3.30832368e-03 7.01052486e-04 7.02633406e-05
#   2.75496393e-03 1.16162153e-03 4.37784102e-03 2.55846977e-03
#   5.55265462e-04 2.39953457e-04 1.19896489e-03 1.68642192e-03
#   3.25620844e-04 3.43664666e-04 6.73498143e-04 7.61065807e-04
#   2.14721920e-04 1.00785040e-03 4.31458291e-04 6.42476138e-04
#   2.43200222e-03 5.79370710e-04 4.85498604e-04 1.58467027e-03
#   5.08891302e-04 5.06193319e-04 1.03690848e-03 3.36999103e-04
#   9.14381875e-04 3.72775597e-04 1.41716702e-03 5.78549516e-04
#   2.70854944e-04 5.39649220e-04 3.56324948e-03 1.28810550e-03
#   2.61602923e-04 3.71837028e-04 1.90678154e-04 1.91083609e-03
#   2.19096866e-04 2.40710491e-04 1.37729687e-04 5.36862412e-04
#   1.53137383e-03 6.75573945e-04 1.47513449e-04 1.15220761e-03
#   4.42913501e-04 2.50017794e-04 3.29706818e-03 4.93907719e-04
#   3.24281771e-03 2.23342981e-03 1.47672647e-04 1.95681874e-04
#   1.57487739e-04 9.41135804e-04 1.29173324e-03 6.22291816e-04
#   1.89144223e-04 2.14256775e-02 1.37624767e-04 1.80048519e-04
#   1.24519269e-04 1.52670417e-03 8.37434607e-04 3.97237367e-04
#   5.14673593e-04 3.72313545e-03 1.78159459e-03 2.00092923e-04
#   5.32159640e-04 1.40057888e-03 6.00600848e-04 2.37316405e-03
#   3.30932927e-03 1.30007465e-04 3.86291242e-04 2.24924507e-03
#   5.92372788e-04 6.39961450e-04 1.62056193e-03 2.10169164e-04
#   3.23865184e-04 2.76658189e-04 5.95014775e-04 1.00520691e-02
#   2.76926352e-04 3.96435207e-04 6.47807901e-04 2.72396579e-03
#   9.66820517e-04 1.63427740e-03 6.27857342e-04 5.14242565e-03
#   7.10885040e-04 1.61024160e-04 5.49553079e-04 3.40635655e-04
#   1.71188638e-02 1.13265275e-03 1.21782732e-03 9.31849179e-04
#   7.68956612e-04 1.19271129e-03 7.23556732e-04 5.58019150e-04
#   4.59414849e-04 5.08046942e-04 7.06899387e-04 8.91712727e-04
#   2.51193764e-04 2.71938654e-04 7.18676369e-04 2.68921652e-03
#   1.15373300e-03 1.07583427e-03 9.51006915e-03 1.66921108e-03
#   3.24091496e-04 3.54896678e-04 6.17161440e-03 5.76931459e-04
#   2.99004198e-04 6.79980556e-04 4.58826078e-04 1.14240625e-03
#   9.15436132e-04 9.22743347e-04 1.49405433e-03 2.01336760e-03
#   3.13789118e-04 1.09742640e-03 2.10035825e-03 2.29676371e-04
#   2.96365964e-04 1.49502959e-02 1.73694512e-03 5.71505679e-03
#   6.21789251e-04 1.09906297e-03 5.17464883e-04 3.21293686e-04
#   6.06367888e-04 6.70823327e-04 9.92170069e-04 2.11752811e-03
#   3.61424696e-04 4.57374641e-04 1.85860801e-04 2.70950166e-03
#   1.90211853e-04 3.11637094e-04 7.71213265e-04 4.30888077e-03
#   2.48159311e-04 1.71133535e-04 1.51328812e-03 1.70044426e-04
#   1.03887310e-03 4.28658997e-04 1.81979046e-03 1.90670806e-04
#   2.47326388e-04 8.32991733e-04 6.37254154e-04 2.87808245e-04
#   1.59904681e-04 1.92098727e-04 1.59584597e-04 1.54527486e-03
#   9.67290811e-03 1.89355528e-03 6.81848440e-04 9.84192174e-03
#   1.95199309e-03 2.79221265e-03 1.61356397e-03 9.62076709e-04
#   3.17211612e-03 2.77718552e-03 4.92741994e-04 6.15699217e-04
#   8.63544736e-03 1.79459667e-03 6.01864012e-04 8.97102407e-04
#   1.12507418e-02 1.13463541e-03 8.05017189e-04 1.41175813e-04
#   3.44532612e-03 2.27107434e-03 2.86640041e-03 7.17224495e-04
#   4.08838212e-04 3.63718922e-04 1.88916747e-03 3.51631461e-04
#   1.51292072e-04 3.86320823e-03 2.26312870e-04 1.82594580e-03
#   2.55135819e-03 7.10132299e-04 6.20356575e-03 7.84598698e-04
#   1.22294354e-03 6.33454765e-04 1.34979584e-03 1.02405054e-02
#   1.32575035e-04 1.51044616e-04 5.22934739e-03 4.12756926e-04
#   4.42252523e-04 4.84381942e-03 9.51871858e-04 2.88472022e-03
#   4.49564541e-03 3.64590460e-03 1.31984780e-04 3.93917412e-03
#   1.29539834e-03 1.30201341e-04 5.82714158e-04 3.53167008e-04
#   3.46774526e-04 6.14737766e-03 5.38542692e-04 9.42293438e-04
#   5.92572428e-03 4.96419962e-04 2.73713900e-04 1.58624130e-03
#   6.52334827e-04 6.14074583e-04 8.10835976e-03 1.45772833e-03
#   3.01242690e-04 4.76241024e-04 2.38483350e-04 4.15955437e-04
#   1.73990033e-04 6.05492445e-04 2.53232720e-04 6.80139870e-04
#   5.95471065e-04 2.43580304e-04 8.47464986e-03 2.52640632e-04
#   1.75258704e-03 3.23539804e-04 1.46721967e-03 1.35794445e-03
#   7.02530204e-04 1.75756810e-04 3.19360232e-04 2.50553538e-04
#   7.92405161e-04 4.80376789e-03 4.24322300e-03 2.60650762e-04
#   1.48688466e-03 6.08615391e-03 8.48063733e-04 2.74493382e-03
#   3.58053227e-03 1.08728008e-02 7.51594198e-04 3.11489194e-03
#   2.50405073e-03 2.98836996e-04 3.03115277e-03 9.93129652e-05
#   8.57855193e-04 1.52714289e-04 2.62172340e-04 1.04848854e-03
#   7.40964839e-04 3.01324879e-04 3.35531833e-04 5.66651695e-04
#   3.97482747e-03 3.02727334e-04 3.24054243e-04 5.40738134e-03
#   1.99600309e-03 1.74735067e-03 4.16404131e-04 2.57937470e-04
#   2.41175905e-04 1.91600891e-04 1.83858720e-04 2.02687341e-04
#   1.53357093e-03 7.50707113e-04 5.89464232e-03 2.89280713e-03
#   3.13441269e-04 1.46778277e-03 2.50423350e-03 5.67096751e-04
#   2.54716870e-04 2.09235528e-04 5.57711115e-04 2.65238853e-03
#   3.79515509e-03 1.31471039e-04 6.81035686e-04 5.79811807e-04
#   5.60818648e-04 6.56650460e-04 1.90014637e-03 3.59991624e-04
#   2.79195723e-04 1.06012053e-03 6.51205657e-04 7.68870348e-03
#   1.79520401e-03 7.08936073e-04 1.68755665e-04 3.10383213e-04
#   5.61488036e-04 7.70802668e-04 5.42372523e-04 2.19337919e-04
#   5.80050470e-03 2.45732343e-04 2.00131419e-03 2.75640952e-04
#   1.31916616e-03 1.93891500e-03 2.68832478e-03 1.85424672e-03
#   8.59084306e-04 5.79205295e-03 1.13422910e-04 1.95587272e-04
#   6.66073931e-04 7.99535948e-04 1.15683470e-02 6.48438989e-04
#   1.75184919e-04 1.72660832e-04 1.68688642e-03 2.44080042e-03
#   1.07478257e-03 5.51242381e-03 1.33144960e-04 3.41837504e-03
#   1.40640116e-03 8.35915911e-04 8.28811200e-04 2.51796492e-03
#   9.17644240e-04 1.16176838e-02 8.00915004e-04 8.92807613e-04
#   4.92670480e-03 5.62897825e-04 2.26286915e-03 3.74127558e-04
#   2.06146878e-03 1.62867887e-03 4.91169514e-03 2.71408062e-04
#   9.39140213e-04 4.70765051e-04 2.07995670e-03 8.69304780e-03
#   6.25818968e-03 9.23961197e-05 3.64605366e-04 2.70366698e-04
#   2.05429766e-04 6.08613307e-04 6.33282703e-04 3.14117409e-04
#   2.05359166e-03 4.77162335e-04 2.70487362e-04 6.87717809e-04
#   9.72806418e-04 1.01295742e-03 6.47979206e-04 3.48857633e-04
#   1.34422851e-04 9.54246556e-04 5.35605766e-04 2.07111283e-04
#   4.38236911e-03 3.91419715e-04 5.94303478e-04 2.97574996e-04
#   3.21045285e-04 2.72554898e-04 9.69987886e-04 1.40099216e-03
#   1.20086619e-03 5.45166386e-03 3.39739694e-04 2.05601275e-04
#   4.24584199e-04 2.93800636e-04 3.96339269e-03 4.40720498e-04
#   2.89386662e-04 3.33809119e-04 5.05516306e-04 3.48564074e-03
#   4.33468580e-04 4.04639170e-04 4.37844265e-03 8.01445567e-04
#   4.67839837e-03 9.89656546e-05 8.23049690e-04 1.56387757e-03
#   3.72532697e-04 5.17696375e-04 7.55272282e-04 2.96095037e-04
#   3.47010442e-03 1.75263465e-03 3.73059371e-03 4.69183782e-04
#   3.31608858e-03 4.37630573e-04 3.68323876e-04 4.85901226e-04
#   7.28685409e-03 4.39446094e-03 3.57634295e-03 5.73831145e-04
#   1.01755653e-03 4.92050662e-04 2.72850477e-04 9.39253485e-04
#   1.67715293e-03 3.28631519e-04 2.64884229e-03 2.07235059e-03
#   3.94351606e-04 3.26313631e-04 5.81823755e-04 2.41195969e-03
#   4.05046099e-04 4.19012457e-03 8.88122304e-04 8.33809376e-04
#   2.46450858e-04 1.17411240e-04 2.20397866e-04 4.64310142e-04
#   6.10933057e-04 4.72636078e-04 6.53434661e-04 7.68031343e-04
#   4.06978652e-03 4.88885329e-04 1.55612128e-04 1.06893806e-03
#   8.12137034e-04 7.29008112e-04 8.67866154e-04 4.51727479e-04
#   4.68844955e-04 8.07281875e-04 8.67321680e-04 9.89009277e-04
#   1.05266939e-04 6.93605398e-04 1.63193312e-04 1.36720177e-04
#   2.80598091e-04 4.64318553e-04 2.23133902e-04 2.38256808e-03
#   5.94309764e-03 1.48056808e-03 8.14453233e-04 1.20447704e-03
#   1.86055514e-03 2.49076029e-03 1.58159179e-03 1.38495804e-03
#   6.94809016e-04 5.92178665e-04 1.47550609e-02 9.60179837e-04
#   3.27270536e-04 8.24608316e-04 1.22308114e-03 8.99795908e-04
#   5.48558193e-04 4.39190073e-04 1.16698514e-03 7.40525662e-04
#   1.17043375e-04 7.70649000e-04 4.76852380e-04 2.14960193e-04
#   6.93711778e-03 2.66100949e-04 3.38351820e-04 8.31563491e-04
#   9.57816257e-04 4.72916989e-04 5.26931603e-04 4.00880672e-04
#   3.36762343e-04 4.21837147e-04 2.71521596e-04 3.66676133e-04
#   1.58203315e-04 5.59787382e-04 5.03910531e-04 2.03677409e-04
#   2.81147048e-04 2.08629208e-04 2.48097349e-04 2.57000123e-04
#   5.41470770e-04 5.36441803e-04 3.75619769e-04 4.23420221e-04
#   5.43418981e-04 3.43837251e-04 4.11142159e-04 4.98211884e-04
#   5.65560418e-04 8.87888367e-04 1.47216837e-04 7.74699947e-05
#   8.43115326e-04 1.80234027e-03 6.08525821e-04 7.68800965e-04
#   4.62411321e-04 4.50784806e-04 9.68802487e-04 1.56956725e-04
#   8.68423260e-04 9.14228382e-04 8.55212857e-04 3.80445679e-04
#   1.16079836e-03 2.58800603e-04 3.73504910e-04 3.26367473e-04
#   3.27318616e-04 1.77301685e-04 4.78650024e-03 5.90185518e-04
#   2.22818553e-03 6.91886235e-04 2.18028930e-04 8.06220109e-04
#   4.49926185e-04 1.74662593e-04 4.86382807e-04 2.12543266e-04
#   4.74094442e-04 3.88993649e-04 1.94331704e-04 2.20697169e-04
#   3.99687036e-04 3.56433215e-04 2.73243786e-04 2.75110477e-04
#   6.36506767e-04 3.93930095e-04 1.46968901e-04 4.29410633e-04
#   4.12151188e-04 1.84499382e-04 3.63839499e-04 2.84115056e-04
#   1.59979027e-04 2.18668327e-04 2.11138700e-04 9.21817991e-05
#   3.32705793e-04 1.06426356e-04 3.64252453e-04 7.31761858e-04]]
# 473
