import numpy as np
import sys
import os
import tensorflow as tf
import tf_xception_


input_placeholder, output = tf_xception_.KitModel('./xception.npy')

for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(var.op.name)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    predict = sess.run(output, feed_dict = {input_placeholder : np.ones((1,299,299,3)) * 0.5})
    print(predict)
    print(np.argmax(predict))
