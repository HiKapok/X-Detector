
# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a RON model using a given dataset."""
import os

import tensorflow as tf
from scipy.misc import imread, imsave, imshow, imresize

import time

from dataset import dataset_factory
from preprocessing import preprocessing_factory
from preprocessing import anchor_manipulator

slim = tf.contrib.slim

DATA_FORMAT = 'NHWC' #'NCHW'

# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'model_dir', './logs/',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer(
    'num_readers', 2,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 1,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 7200,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.0001, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'momentum',
    'The name of the optimizer, one of "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.002, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.96, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.5,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc_0712', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'data_dir', '../PASCAL/tfrecords/VOC2007/TF/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'light_head_faster_rcnn', 'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', 'light_head_resnet_faster_rcnn', 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', 320, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')
# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model/model.ckpt',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', 'resnet',#None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None, #'ron_320_vgg/reverse_module, ron_320_vgg/conv6, ron_320_vgg/conv7',#None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True, #False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    print(tf.gfile.Glob('./debug/example_01?.jpg'))
    if not FLAGS.data_dir:
        raise ValueError('You must supply the dataset directory with --data_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        global_step = slim.create_global_step()

        #print(tf.gfile.Glob('./debug/example_01?.jpg'))

        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = lambda image_, shape_, glabels_, gbboxes_ : preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=True)(image_, glabels_, gbboxes_, out_shape=[FLAGS.train_image_size] * 2, data_format=DATA_FORMAT)

        anchor_creator = anchor_manipulator.AnchorCreator([FLAGS.train_image_size] * 2,
                                                        layers_shapes = [(38, 38), (19, 19), (10, 10)],
                                                        anchor_scales = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                                                        extra_anchor_scales = [[0.15], [0.35], [0.55]],
                                                        anchor_ratios = [[2, .5], [2, .5, 3, 1./3], [2, .5, 3, 1./3]],
                                                        layer_steps = [8, 16, 32])

        all_anchors = anchor_creator.get_all_anchors()[0]

        # sess = tf.Session()
        # print(all_anchors)
        # print(sess.run(all_anchors))
        anchor_operator = anchor_manipulator.AnchorEncoder(all_anchors,
                                        num_classes = FLAGS.num_classes,
                                        ignore_threshold = 0.,
                                        prior_scaling=[0.1, 0.1, 0.2, 0.2])
        #anchor_encoder_fn = lambda
        next_iter, _ = dataset_factory.get_dataset(FLAGS.dataset_name,
                                                                    FLAGS.dataset_split_name,
                                                                    FLAGS.data_dir,
                                                                    image_preprocessing_fn,
                                                                    file_pattern = None,
                                                                    reader = None,
                                                                    batch_size = FLAGS.batch_size,
                                                                    num_readers = FLAGS.num_readers,
                                                                    num_preprocessing_threads = FLAGS.num_preprocessing_threads,
                                                                    anchor_encoder = anchor_operator.encode_all_anchors)


        sess = tf.Session()
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        count = 0
        start_time = time.time()
        try:
            while not coord.should_stop():
                count += 1
                _ = sess.run([next_iter])
                if count % 10 == 0:
                    time_elapsed = time.time() - start_time
                    print('time: {}'.format(time_elapsed/10.))
                    start_time = time.time()
        except tf.errors.OutOfRangeError:
            log.info('Queue Done!')
        finally:
            pass

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

        for i in range(6):
            list_from_batch = sess.run(next_iter)
            # imsave('./debug/example_%03d.jpg' % (i,), list_from_batch[0][0])
            # imsave('./debug/example_%03d_.jpg' % (i,), list_from_batch[1][0])
            image = list_from_batch[-1]
            shape = list_from_batch[-2]
            glabels = list_from_batch[:len(all_anchors)]
            gtargets = list_from_batch[len(all_anchors):2 * len(all_anchors)]
            gscores = list_from_batch[2 * len(all_anchors):3 * len(all_anchors)]

            imsave('./debug/example_%03d.jpg' % (i,), image[0])

            print(image.shape, shape.shape, glabels[0].shape, gtargets[0].shape, gscores[0].shape)


if __name__ == '__main__':
    tf.app.run()

# global_step = slim.create_global_step()

#         print(tf.gfile.Glob('./debug/example_01?.jpg'))

#         preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
#         image_preprocessing_fn = lambda image_, shape_, glabels_, gbboxes_ : preprocessing_factory.get_preprocessing(
#             preprocessing_name, is_training=True)(image_, glabels_, gbboxes_, out_shape=[FLAGS.train_image_size] * 2, data_format=DATA_FORMAT)

#         #anchor_encoder_fn = lambda
#         image, shape, glabels, gbboxes, initializer = dataset_factory.get_dataset(FLAGS.dataset_name,
#                                                                     FLAGS.dataset_split_name,
#                                                                     FLAGS.data_dir,
#                                                                     image_preprocessing_fn,
#                                                                     file_pattern = None,
#                                                                     reader = None,
#                                                                     batch_size = FLAGS.batch_size,
#                                                                     num_readers = FLAGS.num_readers,
#                                                                     num_preprocessing_threads = FLAGS.num_preprocessing_threads,
#                                                                     num_anchors = 5,
#                                                                     anchor_encoder = None)

#         sess = tf.Session()
#         sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))
#         sess.run(initializer)
#         _shape, _glabels, _gbboxes, eval_image = sess.run([shape, glabels, gbboxes, image])
#         print(_shape, _glabels, _gbboxes)

        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # count = 0
        # try:
        #     while not coord.should_stop():
        #         _shape, _glabels, _gbboxes, eval_image = sess.run([shape, glabels, gbboxes, image])
        #         print(_shape, _glabels, _gbboxes)
        #         imsave('./debug/example_%03d.jpg' % (count,), eval_image)
        #         #imsave('./debug/example_preprocess_%03d.jpg' % (count,), eval_image_preprocess)
        #         count += 1
        #         #break
        # except tf.errors.OutOfRangeError:
        #     log.info('Queue Done!')
        # finally:
        #     pass

        # # Wait for threads to finish.
        # coord.join(threads)
        # sess.close()
