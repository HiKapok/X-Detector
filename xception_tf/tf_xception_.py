import tensorflow as tf

__weights_dict = dict()

is_train = True

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    input_1         = tf.placeholder(tf.float32,  shape = (None, 299, 299, 3), name = 'input_1')
    block1_conv1    = convolution(input_1, group=1, strides=[2, 2], padding='VALID', name='block1_conv1')
    block1_conv1_bn = batch_normalization(block1_conv1, variance_epsilon=0.0010000000474974513, name='block1_conv1_bn')
    block1_conv1_act = tf.nn.relu(block1_conv1_bn, name = 'block1_conv1_act')
    block1_conv2    = convolution(block1_conv1_act, group=1, strides=[1, 1], padding='VALID', name='block1_conv2')
    block1_conv2_bn = batch_normalization(block1_conv2, variance_epsilon=0.0010000000474974513, name='block1_conv2_bn')
    block1_conv2_act = tf.nn.relu(block1_conv2_bn, name = 'block1_conv2_act')
    block2_sepconv1 = separable_convolution(block1_conv2_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block2_sepconv1')
    conv2d_1        = convolution(block1_conv2_act, group=1, strides=[2, 2], padding='SAME', name='conv2d_1')
    block2_sepconv1_bn = batch_normalization(block2_sepconv1, variance_epsilon=0.0010000000474974513, name='block2_sepconv1_bn')
    batch_normalization_1 = batch_normalization(conv2d_1, variance_epsilon=0.0010000000474974513, name='batch_normalization_1')
    block2_sepconv2_act = tf.nn.relu(block2_sepconv1_bn, name = 'block2_sepconv2_act')
    block2_sepconv2 = separable_convolution(block2_sepconv2_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block2_sepconv2')
    block2_sepconv2_bn = batch_normalization(block2_sepconv2, variance_epsilon=0.0010000000474974513, name='block2_sepconv2_bn')
    block2_pool     = tf.nn.max_pool(block2_sepconv2_bn, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='block2_pool')
    add_1           = block2_pool +batch_normalization_1
    block3_sepconv1_act = tf.nn.relu(add_1, name = 'block3_sepconv1_act')
    conv2d_2        = convolution(add_1, group=1, strides=[2, 2], padding='SAME', name='conv2d_2')
    block3_sepconv1 = separable_convolution(block3_sepconv1_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block3_sepconv1')
    batch_normalization_2 = batch_normalization(conv2d_2, variance_epsilon=0.0010000000474974513, name='batch_normalization_2')
    block3_sepconv1_bn = batch_normalization(block3_sepconv1, variance_epsilon=0.0010000000474974513, name='block3_sepconv1_bn')
    block3_sepconv2_act = tf.nn.relu(block3_sepconv1_bn, name = 'block3_sepconv2_act')
    block3_sepconv2 = separable_convolution(block3_sepconv2_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block3_sepconv2')
    block3_sepconv2_bn = batch_normalization(block3_sepconv2, variance_epsilon=0.0010000000474974513, name='block3_sepconv2_bn')
    block3_pool     = tf.nn.max_pool(block3_sepconv2_bn, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='block3_pool')
    add_2           = block3_pool +batch_normalization_2
    block4_sepconv1_act = tf.nn.relu(add_2, name = 'block4_sepconv1_act')
    conv2d_3        = convolution(add_2, group=1, strides=[2, 2], padding='SAME', name='conv2d_3')
    block4_sepconv1 = separable_convolution(block4_sepconv1_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block4_sepconv1')
    batch_normalization_3 = batch_normalization(conv2d_3, variance_epsilon=0.0010000000474974513, name='batch_normalization_3')
    block4_sepconv1_bn = batch_normalization(block4_sepconv1, variance_epsilon=0.0010000000474974513, name='block4_sepconv1_bn')
    block4_sepconv2_act = tf.nn.relu(block4_sepconv1_bn, name = 'block4_sepconv2_act')
    block4_sepconv2 = separable_convolution(block4_sepconv2_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block4_sepconv2')
    block4_sepconv2_bn = batch_normalization(block4_sepconv2, variance_epsilon=0.0010000000474974513, name='block4_sepconv2_bn')
    block4_pool     = tf.nn.max_pool(block4_sepconv2_bn, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='block4_pool')
    add_3           = block4_pool +batch_normalization_3
    block5_sepconv1_act = tf.nn.relu(add_3, name = 'block5_sepconv1_act')
    block5_sepconv1 = separable_convolution(block5_sepconv1_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block5_sepconv1')
    block5_sepconv1_bn = batch_normalization(block5_sepconv1, variance_epsilon=0.0010000000474974513, name='block5_sepconv1_bn')
    block5_sepconv2_act = tf.nn.relu(block5_sepconv1_bn, name = 'block5_sepconv2_act')
    block5_sepconv2 = separable_convolution(block5_sepconv2_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block5_sepconv2')
    block5_sepconv2_bn = batch_normalization(block5_sepconv2, variance_epsilon=0.0010000000474974513, name='block5_sepconv2_bn')
    block5_sepconv3_act = tf.nn.relu(block5_sepconv2_bn, name = 'block5_sepconv3_act')
    block5_sepconv3 = separable_convolution(block5_sepconv3_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block5_sepconv3')
    block5_sepconv3_bn = batch_normalization(block5_sepconv3, variance_epsilon=0.0010000000474974513, name='block5_sepconv3_bn')
    add_4           = block5_sepconv3_bn +add_3
    block6_sepconv1_act = tf.nn.relu(add_4, name = 'block6_sepconv1_act')
    block6_sepconv1 = separable_convolution(block6_sepconv1_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block6_sepconv1')
    block6_sepconv1_bn = batch_normalization(block6_sepconv1, variance_epsilon=0.0010000000474974513, name='block6_sepconv1_bn')
    block6_sepconv2_act = tf.nn.relu(block6_sepconv1_bn, name = 'block6_sepconv2_act')
    block6_sepconv2 = separable_convolution(block6_sepconv2_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block6_sepconv2')
    block6_sepconv2_bn = batch_normalization(block6_sepconv2, variance_epsilon=0.0010000000474974513, name='block6_sepconv2_bn')
    block6_sepconv3_act = tf.nn.relu(block6_sepconv2_bn, name = 'block6_sepconv3_act')
    block6_sepconv3 = separable_convolution(block6_sepconv3_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block6_sepconv3')
    block6_sepconv3_bn = batch_normalization(block6_sepconv3, variance_epsilon=0.0010000000474974513, name='block6_sepconv3_bn')
    add_5           = block6_sepconv3_bn +add_4
    block7_sepconv1_act = tf.nn.relu(add_5, name = 'block7_sepconv1_act')
    block7_sepconv1 = separable_convolution(block7_sepconv1_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block7_sepconv1')
    block7_sepconv1_bn = batch_normalization(block7_sepconv1, variance_epsilon=0.0010000000474974513, name='block7_sepconv1_bn')
    block7_sepconv2_act = tf.nn.relu(block7_sepconv1_bn, name = 'block7_sepconv2_act')
    block7_sepconv2 = separable_convolution(block7_sepconv2_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block7_sepconv2')
    block7_sepconv2_bn = batch_normalization(block7_sepconv2, variance_epsilon=0.0010000000474974513, name='block7_sepconv2_bn')
    block7_sepconv3_act = tf.nn.relu(block7_sepconv2_bn, name = 'block7_sepconv3_act')
    block7_sepconv3 = separable_convolution(block7_sepconv3_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block7_sepconv3')
    block7_sepconv3_bn = batch_normalization(block7_sepconv3, variance_epsilon=0.0010000000474974513, name='block7_sepconv3_bn')
    add_6           = block7_sepconv3_bn +add_5
    block8_sepconv1_act = tf.nn.relu(add_6, name = 'block8_sepconv1_act')
    block8_sepconv1 = separable_convolution(block8_sepconv1_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block8_sepconv1')
    block8_sepconv1_bn = batch_normalization(block8_sepconv1, variance_epsilon=0.0010000000474974513, name='block8_sepconv1_bn')
    block8_sepconv2_act = tf.nn.relu(block8_sepconv1_bn, name = 'block8_sepconv2_act')
    block8_sepconv2 = separable_convolution(block8_sepconv2_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block8_sepconv2')
    block8_sepconv2_bn = batch_normalization(block8_sepconv2, variance_epsilon=0.0010000000474974513, name='block8_sepconv2_bn')
    block8_sepconv3_act = tf.nn.relu(block8_sepconv2_bn, name = 'block8_sepconv3_act')
    block8_sepconv3 = separable_convolution(block8_sepconv3_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block8_sepconv3')
    block8_sepconv3_bn = batch_normalization(block8_sepconv3, variance_epsilon=0.0010000000474974513, name='block8_sepconv3_bn')
    add_7           = block8_sepconv3_bn +add_6
    block9_sepconv1_act = tf.nn.relu(add_7, name = 'block9_sepconv1_act')
    block9_sepconv1 = separable_convolution(block9_sepconv1_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block9_sepconv1')
    block9_sepconv1_bn = batch_normalization(block9_sepconv1, variance_epsilon=0.0010000000474974513, name='block9_sepconv1_bn')
    block9_sepconv2_act = tf.nn.relu(block9_sepconv1_bn, name = 'block9_sepconv2_act')
    block9_sepconv2 = separable_convolution(block9_sepconv2_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block9_sepconv2')
    block9_sepconv2_bn = batch_normalization(block9_sepconv2, variance_epsilon=0.0010000000474974513, name='block9_sepconv2_bn')
    block9_sepconv3_act = tf.nn.relu(block9_sepconv2_bn, name = 'block9_sepconv3_act')
    block9_sepconv3 = separable_convolution(block9_sepconv3_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block9_sepconv3')
    block9_sepconv3_bn = batch_normalization(block9_sepconv3, variance_epsilon=0.0010000000474974513, name='block9_sepconv3_bn')
    add_8           = block9_sepconv3_bn +add_7
    block10_sepconv1_act = tf.nn.relu(add_8, name = 'block10_sepconv1_act')
    block10_sepconv1 = separable_convolution(block10_sepconv1_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block10_sepconv1')
    block10_sepconv1_bn = batch_normalization(block10_sepconv1, variance_epsilon=0.0010000000474974513, name='block10_sepconv1_bn')
    block10_sepconv2_act = tf.nn.relu(block10_sepconv1_bn, name = 'block10_sepconv2_act')
    block10_sepconv2 = separable_convolution(block10_sepconv2_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block10_sepconv2')
    block10_sepconv2_bn = batch_normalization(block10_sepconv2, variance_epsilon=0.0010000000474974513, name='block10_sepconv2_bn')
    block10_sepconv3_act = tf.nn.relu(block10_sepconv2_bn, name = 'block10_sepconv3_act')
    block10_sepconv3 = separable_convolution(block10_sepconv3_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block10_sepconv3')
    block10_sepconv3_bn = batch_normalization(block10_sepconv3, variance_epsilon=0.0010000000474974513, name='block10_sepconv3_bn')
    add_9           = block10_sepconv3_bn +add_8
    block11_sepconv1_act = tf.nn.relu(add_9, name = 'block11_sepconv1_act')
    block11_sepconv1 = separable_convolution(block11_sepconv1_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block11_sepconv1')
    block11_sepconv1_bn = batch_normalization(block11_sepconv1, variance_epsilon=0.0010000000474974513, name='block11_sepconv1_bn')
    block11_sepconv2_act = tf.nn.relu(block11_sepconv1_bn, name = 'block11_sepconv2_act')
    block11_sepconv2 = separable_convolution(block11_sepconv2_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block11_sepconv2')
    block11_sepconv2_bn = batch_normalization(block11_sepconv2, variance_epsilon=0.0010000000474974513, name='block11_sepconv2_bn')
    block11_sepconv3_act = tf.nn.relu(block11_sepconv2_bn, name = 'block11_sepconv3_act')
    block11_sepconv3 = separable_convolution(block11_sepconv3_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block11_sepconv3')
    block11_sepconv3_bn = batch_normalization(block11_sepconv3, variance_epsilon=0.0010000000474974513, name='block11_sepconv3_bn')
    add_10          = block11_sepconv3_bn +add_9
    block12_sepconv1_act = tf.nn.relu(add_10, name = 'block12_sepconv1_act')
    block12_sepconv1 = separable_convolution(block12_sepconv1_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block12_sepconv1')
    block12_sepconv1_bn = batch_normalization(block12_sepconv1, variance_epsilon=0.0010000000474974513, name='block12_sepconv1_bn')
    block12_sepconv2_act = tf.nn.relu(block12_sepconv1_bn, name = 'block12_sepconv2_act')
    block12_sepconv2 = separable_convolution(block12_sepconv2_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block12_sepconv2')
    block12_sepconv2_bn = batch_normalization(block12_sepconv2, variance_epsilon=0.0010000000474974513, name='block12_sepconv2_bn')
    block12_sepconv3_act = tf.nn.relu(block12_sepconv2_bn, name = 'block12_sepconv3_act')
    block12_sepconv3 = separable_convolution(block12_sepconv3_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block12_sepconv3')
    block12_sepconv3_bn = batch_normalization(block12_sepconv3, variance_epsilon=0.0010000000474974513, name='block12_sepconv3_bn')
    add_11          = block12_sepconv3_bn +add_10
    block13_sepconv1_act = tf.nn.relu(add_11, name = 'block13_sepconv1_act')
    conv2d_4        = convolution(add_11, group=1, strides=[2, 2], padding='SAME', name='conv2d_4')
    block13_sepconv1 = separable_convolution(block13_sepconv1_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block13_sepconv1')
    batch_normalization_4 = batch_normalization(conv2d_4, variance_epsilon=0.0010000000474974513, name='batch_normalization_4')
    block13_sepconv1_bn = batch_normalization(block13_sepconv1, variance_epsilon=0.0010000000474974513, name='block13_sepconv1_bn')
    block13_sepconv2_act = tf.nn.relu(block13_sepconv1_bn, name = 'block13_sepconv2_act')
    block13_sepconv2 = separable_convolution(block13_sepconv2_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block13_sepconv2')
    block13_sepconv2_bn = batch_normalization(block13_sepconv2, variance_epsilon=0.0010000000474974513, name='block13_sepconv2_bn')
    block13_pool    = tf.nn.max_pool(block13_sepconv2_bn, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='block13_pool')
    add_12          = block13_pool +batch_normalization_4
    block14_sepconv1 = separable_convolution(add_12, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block14_sepconv1')
    block14_sepconv1_bn = batch_normalization(block14_sepconv1, variance_epsilon=0.0010000000474974513, name='block14_sepconv1_bn')
    block14_sepconv1_act = tf.nn.relu(block14_sepconv1_bn, name = 'block14_sepconv1_act')
    block14_sepconv2 = separable_convolution(block14_sepconv1_act, strides = [1, 1, 1, 1], padding = 'SAME', name = 'block14_sepconv2')
    block14_sepconv2_bn = batch_normalization(block14_sepconv2, variance_epsilon=0.0010000000474974513, name='block14_sepconv2_bn')
    block14_sepconv2_act = tf.nn.relu(block14_sepconv2_bn, name = 'block14_sepconv2_act')
    avg_pool        = tf.nn.avg_pool(block14_sepconv2_act, [1] + block14_sepconv2_act.get_shape().as_list()[1:-1] + [1], strides = [1] * 4, padding = 'VALID', name = 'avg_pool')
    avg_pool_flatten = tf.contrib.layers.flatten(avg_pool)
    predictions     = tf.layers.dense(avg_pool_flatten, 1000, kernel_initializer = tf.constant_initializer(__weights_dict['predictions']['weights']), bias_initializer = tf.constant_initializer(__weights_dict['predictions']['bias']), use_bias = True)
    predictions_activation = tf.nn.softmax(predictions, name = 'predictions_activation')
    return input_1, block14_sepconv2_act


def separable_convolution(input, name, **kwargs):
    depthwise = tf.Variable(__weights_dict[name]['depthwise_filter'], trainable = is_train, name = name + "_df")
    pointwise = tf.Variable(__weights_dict[name]['pointwise_filter'], trainable = is_train, name = name + "_pf")
    #print(name + "_df", __weights_dict[name]['depthwise_filter'].shape)
    #print(name + "_pf", __weights_dict[name]['pointwise_filter'].shape)
    layer = tf.nn.separable_conv2d(input, depthwise, pointwise, **kwargs)
    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable = is_train, name = name + "_bias")
        layer = layer + b
    return layer

def batch_normalization(input, name, **kwargs):
    mean = tf.Variable(__weights_dict[name]['mean'], name = name + "_mean", trainable = is_train)
    variance = tf.Variable(__weights_dict[name]['var'], name = name + "_var", trainable = is_train)
    offset = tf.Variable(__weights_dict[name]['bias'], name = name + "_bias", trainable = is_train) if 'bias' in __weights_dict[name] else None
    scale = tf.Variable(__weights_dict[name]['scale'], name = name + "_scale", trainable = is_train) if 'scale' in __weights_dict[name] else None
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, name = name, **kwargs)


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    #print(name + "_weight", __weights_dict[name]['weights'].shape)
    if group == 1:
        layer = tf.nn.convolution(input, w, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer
