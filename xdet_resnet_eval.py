from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

#from scipy.misc import imread, imsave, imshow, imresize
import tensorflow as tf
import numpy as np

from net import xdet_body
from utility import train_helper
from utility import eval_helper
from utility import metrics

from dataset import dataset_factory
from preprocessing import preprocessing_factory
from preprocessing import anchor_manipulator
from preprocessing import common_preprocessing

# hardware related configuration
tf.app.flags.DEFINE_integer(
    'num_readers', 16,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 48,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')
# scaffold related configuration
tf.app.flags.DEFINE_string(
    'data_dir', '../PASCAL/VOC_TF/VOC2007TEST_TF/',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc_2007', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'model_dir', './logs/',
    'The directory where the model will be stored.')
tf.app.flags.DEFINE_string(
    'debug_dir', './Debug/',
    'The directory where the debug files will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 10,
    'The frequency with which summaries are saved, in seconds.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', 320,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'resnet_size', 50,
    'The size of the ResNet model to use.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'weight_decay', 0.0005, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float(
    'neg_threshold', 0.5, 'Matching threshold for the negtive examples in the loss function.')
tf.app.flags.DEFINE_float(
    'select_threshold', 0.01, 'Class-specific confidence score threshold for selecting a box.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.4, 'Matching threshold in NMS algorithm.')
tf.app.flags.DEFINE_integer(
    'nms_topk_percls', 200, 'Number of object for each class to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'nms_topk', 200, 'Number of total object to keep after NMS.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model/resnet50',#None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'model_scope', 'xdet_resnet',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_boolean(
    'run_on_cloud', True,
    'Wether we will train on cloud (checkpoint will be found in the "data_dir/cloud_checkpoint_path").')
tf.app.flags.DEFINE_string(
    'cloud_checkpoint_path', 'resnet50/model.ckpt',
    'The path to a checkpoint from which to fine-tune.')

FLAGS = tf.app.flags.FLAGS

from dataset import dataset_common
def gain_translate_table():
    label2name_table = {}
    for class_name, labels_pair in dataset_common.VOC_LABELS.items():
        label2name_table[labels_pair[0]] = class_name
    return label2name_table
label2name_table = gain_translate_table()

def input_pipeline():
    image_preprocessing_fn = lambda image_, shape_, glabels_, gbboxes_ : preprocessing_factory.get_preprocessing(
        'xdet_resnet', is_training=False)(image_, glabels_, gbboxes_, out_shape=[FLAGS.train_image_size] * 2, data_format=('NCHW' if FLAGS.data_format=='channels_first' else 'NHWC'))

    anchor_creator = anchor_manipulator.AnchorCreator([FLAGS.train_image_size] * 2,
                                                    layers_shapes = [(40, 40)],
                                                    anchor_scales = [[0.05, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85]],
                                                    extra_anchor_scales = [[]],
                                                    anchor_ratios = [[1., 2., 3., .5, 0.3333]],
                                                    layer_steps = [8])

    def input_fn():
        all_anchors, num_anchors_list = anchor_creator.get_all_anchors()

        anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(all_anchors,
                                        num_classes = FLAGS.num_classes,
                                        allowed_borders = [0.1],
                                        ignore_threshold = FLAGS.match_threshold, # only update labels for positive examples
                                        prior_scaling=[0.1, 0.1, 0.2, 0.2])

        num_readers_to_use = FLAGS.num_readers if FLAGS.run_on_cloud else 2
        num_preprocessing_threads_to_use = FLAGS.num_preprocessing_threads if FLAGS.run_on_cloud else 2

        list_from_batch, _ = dataset_factory.get_dataset(FLAGS.dataset_name,
                                                FLAGS.dataset_split_name,
                                                FLAGS.data_dir,
                                                image_preprocessing_fn,
                                                file_pattern = None,
                                                reader = None,
                                                batch_size = 1,
                                                num_readers = num_readers_to_use,
                                                num_preprocessing_threads = num_preprocessing_threads_to_use,
                                                num_epochs = 1,
                                                method = 'eval',
                                                anchor_encoder = anchor_encoder_decoder.encode_all_anchors)

        return list_from_batch[-1], {'targets': list_from_batch[:-1],
                                    'decode_fn': lambda pred : anchor_encoder_decoder.decode_all_anchors([pred])[0],
                                    'num_anchors_list': num_anchors_list}
    return input_fn

def modified_smooth_l1(bbox_pred, bbox_targets, bbox_inside_weights = 1., bbox_outside_weights = 1., sigma = 1.):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    """
    sigma2 = sigma * sigma

    inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                              tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

    outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

    return outside_mul

if not FLAGS.run_on_cloud:
    from scipy.misc import imread, imsave, imshow, imresize
    from utility import draw_toolbox

def save_image_with_bbox(image, labels_, scores_, bboxes_):
    if not hasattr(save_image_with_bbox, "counter"):
        save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
    save_image_with_bbox.counter += 1

    img_to_draw = np.copy(image)#common_preprocessing.np_image_unwhitened(image))
    if not FLAGS.run_on_cloud:
        img_to_draw = draw_toolbox.bboxes_draw_on_img(img_to_draw, labels_, scores_, bboxes_, thickness=2)
        imsave(os.path.join(FLAGS.debug_dir, '{}.jpg').format(save_image_with_bbox.counter), img_to_draw)
    return save_image_with_bbox.counter#np.array([save_image_with_bbox.counter])

#[feature_h, feature_w, num_anchors, 4]
# only support batch_size 1
def bboxes_eval(org_image, image_shape, bbox_img, cls_pred_logits, bboxes_pred, glabels_raw, gbboxes_raw, isdifficult, num_classes):
    # Performing post-processing on CPU: loop-intensive, usually more efficient.
    cls_pred_prob = tf.nn.softmax(tf.reshape(cls_pred_logits, [-1, num_classes]))
    bboxes_pred = tf.reshape(bboxes_pred, [-1, 4])
    glabels_raw = tf.reshape(glabels_raw, [-1])
    gbboxes_raw = tf.reshape(gbboxes_raw, [-1])
    gbboxes_raw = tf.boolean_mask(gbboxes_raw, glabels_raw > 0)
    glabels_raw = tf.boolean_mask(glabels_raw, glabels_raw > 0)
    isdifficult = tf.reshape(isdifficult, [-1])

    with tf.device('/device:CPU:0'):
        selected_scores, selected_bboxes = eval_helper.tf_bboxes_select([cls_pred_prob], [bboxes_pred], FLAGS.select_threshold, num_classes, scope='xdet_v1_select')

        selected_bboxes = eval_helper.bboxes_clip(bbox_img, selected_bboxes)
        selected_scores, selected_bboxes = eval_helper.filter_boxes(selected_scores, selected_bboxes, 0.03, image_shape, [FLAGS.train_image_size] * 2, keep_top_k = FLAGS.nms_topk * 2)

        # Resize bboxes to original image shape.
        selected_bboxes = eval_helper.bboxes_resize(bbox_img, selected_bboxes)

        selected_scores, selected_bboxes = eval_helper.bboxes_sort(selected_scores, selected_bboxes, top_k=FLAGS.nms_topk * 2)

        # Apply NMS algorithm.
        selected_scores, selected_bboxes = eval_helper.bboxes_nms_batch(selected_scores, selected_bboxes,
                                 nms_threshold=FLAGS.nms_threshold,
                                 keep_top_k=FLAGS.nms_topk)

        # label_scores, pred_labels, bboxes_pred = eval_helper.xdet_predict(bbox_img, cls_pred_prob, bboxes_pred, image_shape, FLAGS.train_image_size, FLAGS.nms_threshold, FLAGS.select_threshold, FLAGS.nms_topk, num_classes, nms_mode='union')

        dict_metrics = {}
        # Compute TP and FP statistics.
        num_gbboxes, tp, fp = eval_helper.bboxes_matching_batch(selected_scores.keys(), selected_scores, selected_bboxes, glabels_raw, gbboxes_raw, isdifficult)

        # FP and TP metrics.
        tp_fp_metric = metrics.streaming_tp_fp_arrays(num_gbboxes, tp, fp, selected_scores)
        # for c in tp_fp_metric[0].keys():
        #     dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c],
        #                                     tp_fp_metric[1][c])
        metrics_name = ('nobjects', 'ndetections', 'tp', 'fp', 'scores')
        for c in tp_fp_metric[0].keys():
            for _ in range(len(tp_fp_metric[0][c])):
                dict_metrics['tp_fp_%s_%s' % (label2name_table[c], metrics_name[_])] = (tp_fp_metric[0][c][_],
                                                tp_fp_metric[1][c][_])


        # Add to summaries precision/recall values.
        aps_voc07 = {}
        aps_voc12 = {}
        for c in tp_fp_metric[0].keys():
            # Precison and recall values.
            prec, rec = metrics.precision_recall(*tp_fp_metric[0][c])

            # Average precision VOC07.
            v = metrics.average_precision_voc07(prec, rec)
            op = tf.summary.scalar('AP_VOC07/%s' % c, v)
            # op = tf.Print(op, [v], 'AP_VOC07/%s' % c)
            #tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            aps_voc07[c] = v

            # Average precision VOC12.
            v = metrics.average_precision_voc12(prec, rec)
            op = tf.summary.scalar('AP_VOC12/%s' % c, v)
            # op = tf.Print(op, [v], 'AP_VOC12/%s' % c)
            #tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            aps_voc12[c] = v

        # Mean average precision VOC07.
        summary_name = 'AP_VOC07/mAP'
        mAP = tf.add_n(list(aps_voc07.values())) / len(aps_voc07)
        mAP = tf.Print(mAP, [mAP], summary_name)
        op = tf.summary.scalar(summary_name, mAP)
        #tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # Mean average precision VOC12.
        summary_name = 'AP_VOC12/mAP'
        mAP = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
        mAP = tf.Print(mAP, [mAP], summary_name)
        op = tf.summary.scalar(summary_name, mAP)
        #tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        labels_list = []
        for k, v in selected_scores.items():
            labels_list.append(tf.ones_like(v, tf.int32) * k)
        save_image_op = tf.py_func(save_image_with_bbox,
                                    [org_image,
                                    tf.concat(labels_list, axis=0),
                                    #tf.convert_to_tensor(list(selected_scores.keys()), dtype=tf.int64),
                                    tf.concat(list(selected_scores.values()), axis=0),
                                    tf.concat(list(selected_bboxes.values()), axis=0)],
                                    tf.int64, stateful=True)

        #dict_metrics['save_image_with_bboxes'] = save_image_count#tf.tuple([save_image_count, save_image_count_update_op])
    # for i, v in enumerate(l_precisions):
    #     summary_name = 'eval/precision_at_recall_%.2f' % LIST_RECALLS[i]
    #     op = tf.summary.scalar(summary_name, v, collections=[])
    #     op = tf.Print(op, [v], summary_name)
    #     tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    return dict_metrics, save_image_op

def xdet_model_fn(features, labels, mode, params):
    """Our model_fn for ResNet to be used with our Estimator."""
    num_anchors_list = labels['num_anchors_list']
    num_feature_layers = len(num_anchors_list)

    shape = labels['targets'][-1]
    if mode != tf.estimator.ModeKeys.TRAIN:
        org_image = labels['targets'][-2]
        isdifficult = labels['targets'][-3]
        bbox_img = labels['targets'][-4]
        gbboxes_raw = labels['targets'][-5]
        glabels_raw = labels['targets'][-6]

    glabels = labels['targets'][:num_feature_layers][0]
    gtargets = labels['targets'][num_feature_layers : 2 * num_feature_layers][0]
    gscores = labels['targets'][2 * num_feature_layers : 3 * num_feature_layers][0]

    with tf.variable_scope(params['model_scope'], default_name = None, values = [features], reuse=tf.AUTO_REUSE):
        backbone = xdet_body.xdet_resnet_v2(params['resnet_size'], params['data_format'])
        multi_merged_feature = backbone(inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

        cls_pred, location_pred = xdet_body.xdet_head(multi_merged_feature, params['num_classes'], num_anchors_list[0], (mode == tf.estimator.ModeKeys.TRAIN), data_format=params['data_format'])

    if params['data_format'] == 'channels_first':
        cls_pred = tf.transpose(cls_pred, [0, 2, 3, 1])
        location_pred = tf.transpose(location_pred, [0, 2, 3, 1])
        #org_image = tf.transpose(org_image, [0, 2, 3, 1])
    # batch size is 1
    shape = tf.squeeze(shape, axis = 0)
    glabels = tf.squeeze(glabels, axis = 0)
    gtargets = tf.squeeze(gtargets, axis = 0)
    gscores = tf.squeeze(gscores, axis = 0)
    cls_pred = tf.squeeze(cls_pred, axis = 0)
    location_pred = tf.squeeze(location_pred, axis = 0)
    if mode != tf.estimator.ModeKeys.TRAIN:
        org_image = tf.squeeze(org_image, axis = 0)
        isdifficult = tf.squeeze(isdifficult, axis = 0)
        gbboxes_raw = tf.squeeze(gbboxes_raw, axis = 0)
        glabels_raw = tf.squeeze(glabels_raw, axis = 0)
        bbox_img = tf.squeeze(bbox_img, axis = 0)

    bboxes_pred = labels['decode_fn'](location_pred)#(tf.reshape(location_pred, location_pred.get_shape().as_list()[:-1] + [-1, 4]))#(location_pred)#

    eval_ops, save_image_op = bboxes_eval(org_image, shape, bbox_img, cls_pred, bboxes_pred, glabels_raw, gbboxes_raw, isdifficult, params['num_classes'])

    _ = tf.identity(save_image_op, name='save_image_with_bboxes_op')

    cls_pred = tf.reshape(cls_pred, [-1, params['num_classes']])
    location_pred = tf.reshape(location_pred, [-1, 4])
    glabels = tf.reshape(glabels, [-1])
    gscores = tf.reshape(gscores, [-1])
    gtargets = tf.reshape(gtargets, [-1, 4])

    # raw mask for positive > 0.5, and for negetive < 0.3
    # each positive examples has one label
    positive_mask = glabels > 0#tf.logical_and(glabels > 0, gscores > 0.)
    fpositive_mask = tf.cast(positive_mask, tf.float32)
    n_positives = tf.reduce_sum(fpositive_mask)
    # negtive examples are those max_overlap is still lower than neg_threshold, note that some positive may also has lower jaccard
    # note those gscores is 0 is either be ignored during anchors encode or anchors have 0 overlap with all ground truth
    negtive_mask = tf.logical_and(tf.logical_and(tf.logical_not(tf.logical_or(positive_mask, glabels < 0)), gscores < params['neg_threshold']), gscores > 0.)
    #negtive_mask = tf.logical_and(tf.logical_and(tf.logical_not(positive_mask), gscores < params['neg_threshold']), gscores > 0.)
    #negtive_mask = tf.logical_and(gscores < params['neg_threshold'], tf.logical_not(positive_mask))
    fnegtive_mask = tf.cast(negtive_mask, tf.float32)
    n_negtives = tf.reduce_sum(fnegtive_mask)

    n_neg_to_select = tf.cast(params['negative_ratio'] * n_positives, tf.int32)
    n_neg_to_select = tf.minimum(n_neg_to_select, tf.cast(n_negtives, tf.int32))

    # hard negative mining for classification
    predictions_for_bg = tf.nn.softmax(cls_pred)[:, 0]
    #negtive_mask = tf.Print(negtive_mask,[n_positives])
    prob_for_negtives = tf.where(negtive_mask,
                           0. - predictions_for_bg,
                           # ignore all the positives
                           0. - tf.ones_like(predictions_for_bg))
    topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=n_neg_to_select)
    selected_neg_mask = prob_for_negtives > topk_prob_for_bg[-1]

    # # random select negtive examples for classification
    # selected_neg_mask = tf.random_uniform(tf.shape(gscores), minval=0, maxval=1.) < tf.where(
    #                                                                                     tf.greater(n_negtives, 0),
    #                                                                                     tf.divide(tf.cast(n_neg_to_select, tf.float32), n_negtives),
    #                                                                                     tf.zeros_like(tf.cast(n_neg_to_select, tf.float32)),
    #                                                                                     name='rand_select_negtive')
    # include both selected negtive and all positive examples
    final_mask = tf.stop_gradient(tf.logical_or(tf.logical_and(negtive_mask, selected_neg_mask), positive_mask))
    total_examples = tf.reduce_sum(tf.cast(final_mask, tf.float32))

    # add mask for glabels and cls_pred here
    glabels = tf.boolean_mask(tf.clip_by_value(glabels, 0, FLAGS.num_classes), tf.stop_gradient(final_mask))
    cls_pred = tf.boolean_mask(cls_pred, tf.stop_gradient(final_mask))
    location_pred = tf.boolean_mask(location_pred, tf.stop_gradient(positive_mask))
    gtargets = tf.boolean_mask(gtargets, tf.stop_gradient(positive_mask))

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.cond(n_positives > 0., lambda: tf.losses.sparse_softmax_cross_entropy(labels=glabels, logits=cls_pred), lambda: 0.)
    #cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=glabels, logits=cls_pred)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy_loss')
    tf.summary.scalar('cross_entropy_loss', cross_entropy)

    loc_loss = tf.cond(n_positives > 0., lambda: modified_smooth_l1(location_pred, tf.stop_gradient(gtargets), sigma=1.), lambda: tf.zeros_like(location_pred))
    #loc_loss = modified_smooth_l1(location_pred, tf.stop_gradient(gtargets))
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=-1))
    loc_loss = tf.identity(loc_loss, name='location_loss')
    tf.summary.scalar('location_loss', loc_loss)
    tf.losses.add_loss(loc_loss)

    with tf.control_dependencies([save_image_op]):
        # Add weight decay to the loss. We exclude the batch norm variables because
        # doing so leads to a small improvement in accuracy.
        loss = 1.3 * (cross_entropy + loc_loss) + params['weight_decay'] * tf.add_n(
          [tf.nn.l2_loss(v) for v in tf.trainable_variables()
           if 'batch_normalization' not in v.name])
        total_loss = tf.identity(loss, name='total_loss')

    predictions = {
        'classes': tf.argmax(cls_pred, axis=-1),
        'probabilities': tf.reduce_max(tf.nn.softmax(cls_pred, name='softmax_tensor'), axis=-1),
        'bboxes_predict': tf.reshape(bboxes_pred, [-1, 4]) }

    summary_hook = tf.train.SummarySaverHook(
                        save_secs=FLAGS.save_summary_steps,
                        output_dir=FLAGS.model_dir,
                        summary_op=tf.summary.merge_all())

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                        evaluation_hooks = [summary_hook],
                        loss=loss, eval_metric_ops=eval_ops)#=eval_ops)
    else:
        raise ValueError('This script only support predict mode!')

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False, intra_op_parallelism_threads = FLAGS.num_cpu_threads, inter_op_parallelism_threads = FLAGS.num_cpu_threads, gpu_options = gpu_options)

    # Set up RunConfig
    run_config = tf.estimator.RunConfig().replace(
                                        save_checkpoints_secs=None).replace(
                                        save_checkpoints_steps=None).replace(
                                        save_summary_steps=FLAGS.save_summary_steps).replace(
                                        keep_checkpoint_max=5).replace(
                                        log_step_count_steps=FLAGS.log_every_n_steps).replace(
                                        session_config=config)

    xdetector = tf.estimator.Estimator(
        model_fn=xdet_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'resnet_size': FLAGS.resnet_size,
            'data_format': FLAGS.data_format,
            'model_scope': FLAGS.model_scope,
            'num_classes': FLAGS.num_classes,
            'negative_ratio': FLAGS.negative_ratio,
            'match_threshold': FLAGS.match_threshold,
            'neg_threshold': FLAGS.neg_threshold,
            'weight_decay': FLAGS.weight_decay,
        })


    tensors_to_log = {
        'ce_loss': 'cross_entropy_loss',
        'loc_loss': 'location_loss',
        'total_loss': 'total_loss',
        'saved_image_index':'save_image_with_bboxes_op'
    }

    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps)

    print('Starting evaluate cycle.')
    xdetector.evaluate(input_fn=input_pipeline(), hooks=[logging_hook], checkpoint_path=train_helper.get_latest_checkpoint_for_evaluate(FLAGS))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
