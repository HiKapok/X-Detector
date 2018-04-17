import tensorflow as tf
import numpy as np
import roi_pooling_op
import roi_pooling_op_grad
import tensorflow as tf



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


class PSROIAlignGradTest(tf.test.TestCase):
  def testPSROIAlignGrad(self):
    with tf.device('/gpu:0'):
        array = np.random.rand(32, 10, 10, 3)
        data = tf.convert_to_tensor(array, dtype=tf.float32)
        rois = tf.convert_to_tensor([[0, 1, 1, 4, 4], [0, 3, 3, 4, 4]], dtype=tf.float32)

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
        W = weight_variable([3, 3, 3, 1])
        h = conv2d(data, W)
        inputs_features = tf.constant(map_to_pool, dtype=tf.float32)
        [y, argmax] = roi_pooling_op.roi_pool(inputs_features, rois, 2, 2, 1.0/3)

        y_data = tf.convert_to_tensor(np.ones((2, 6, 6, 1)), dtype=tf.float32)
        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
            #print(sess.run(y).shape)
            #print(sess.run(tf.gradients(pool_result[0], [inputs_features])))
            print(tf.test.compute_gradient_error(inputs_features, [1, 16, 5, 5], y, [2,2,2,5], delta=0.0001, x_init_value=np.array(map_to_pool)))
            # _, jaccobian = tf.test.compute_gradient(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool))
            # y = sess.run(pool_result[0])
            # print(jaccobian.shape)
            # print(np.reshape(np.matmul(jaccobian, np.ones_like(y.flatten())), np.array(map_to_pool).shape))
            # print(tf.test.compute_gradient(inputs_features, [1, 16, 5, 5], pool_result[0], [1, 3, 4, 4], delta=0.0001, x_init_value=np.array(map_to_pool)))

if __name__ == "__main__":
  tf.test.main()


