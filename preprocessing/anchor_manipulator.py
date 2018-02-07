import math

import tensorflow as tf
import numpy as np


class AnchorEncoder(object):
    def __init__(self, anchors, num_classes, ignore_threshold, prior_scaling):
        super(AnchorEncoder, self).__init__()
        self._labels = None
        self._bboxes = None
        self._anchors = anchors
        self._num_classes = num_classes
        self._ignore_threshold = ignore_threshold
        self._prior_scaling = prior_scaling

    def center2point(self, center_y, center_x, height, width):
        return center_y - height / 2., center_x - width / 2., center_y + height / 2., center_x + width / 2.,

    def point2center(self, ymin, xmin, ymax, xmax):
        height, width = (ymax - ymin), (xmax - xmin)
        return ymin + height / 2., xmin + width / 2., height, width

    def encode_anchor(self, anchor):
        assert self._labels is not None, 'must provide labels to encode anchors.'
        assert self._bboxes is not None, 'must provide bboxes to encode anchors.'
        # y, x, h, w are all in range [0, 1] relative to the original image size
        yref, xref, href, wref = tf.expand_dims(anchor[0], axis=-1), tf.expand_dims(anchor[1], axis=-1), anchor[2], anchor[3]
        # for the shape of ymin, xmin, ymax, xmax
        # [[[anchor_0, anchor_1, anchor_2, ...], [anchor_0, anchor_1, anchor_2, ...], [anchor_0, anchor_1, anchor_2, ...], ...],
        # [[anchor_0, anchor_1, anchor_2, ...], [anchor_0, anchor_1, anchor_2, ...], [anchor_0, anchor_1, anchor_2, ...], ...],
        #                                   .
        #                                   .
        # [[anchor_0, anchor_1, anchor_2, ...], [anchor_0, anchor_1, anchor_2, ...], [anchor_0, anchor_1, anchor_2, ...], ...]]
        ymin, xmin, ymax, xmax = self.center2point(yref, xref, href, wref)

        vol_anchors = (xmax - xmin) * (ymax - ymin)

        # store every jaccard score while loop all ground truth, will update depends the score of anchor and current ground_truth
        gt_labels = tf.zeros_like(ymin, dtype=tf.int64)
        gt_scores = tf.zeros_like(ymin, dtype=tf.float32)

        gt_ymin = tf.zeros_like(ymin, dtype=tf.float32)
        gt_xmin = tf.zeros_like(ymin, dtype=tf.float32)
        gt_ymax = tf.ones_like(ymin, dtype=tf.float32)
        gt_xmax = tf.ones_like(ymin, dtype=tf.float32)

        max_mask = tf.cast(tf.zeros_like(ymin, dtype=tf.int32), tf.bool)

        def safe_divide(numerator, denominator):
            return tf.where(
                tf.greater(denominator, 0),
                tf.divide(numerator, denominator),
                tf.zeros_like(denominator))

        def jaccard_with_anchors(bbox):
            """Compute jaccard score between a box and the anchors.
            """
            # the inner square
            inner_ymin = tf.maximum(ymin, bbox[0])
            inner_xmin = tf.maximum(xmin, bbox[1])
            inner_ymax = tf.minimum(ymax, bbox[2])
            inner_xmax = tf.minimum(xmax, bbox[3])
            h = tf.maximum(inner_ymax - inner_ymin, 0.)
            w = tf.maximum(inner_xmax - inner_xmin, 0.)

            inner_vol = h * w
            union_vol = vol_anchors - inner_vol \
                + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            jaccard = safe_divide(inner_vol, union_vol)
            return jaccard

        def condition(i, gt_labels, gt_scores,
                      gt_ymin, gt_xmin, gt_ymax, gt_xmax, max_mask):
            return tf.less(i, tf.shape(self._labels))[0]

        def body(i, gt_labels, gt_scores,
                 gt_ymin, gt_xmin, gt_ymax, gt_xmax, max_mask):
            """Body: update gture labels, scores and bboxes.
            Follow the original SSD paper for that purpose:
              - assign values when jaccard > 0.5;
              - only update if beat the score of other bboxes.
            """
            # get i_th groud_truth(label && bbox)
            label = self._labels[i]
            bbox = self._bboxes[i]
            # current ground_truth's overlap with all others' anchors
            jaccard = jaccard_with_anchors(bbox)
            # the index of the max overlap for current ground_truth
            max_jaccard = tf.reduce_max(jaccard)
            cur_max_indice_mask = tf.equal(jaccard, max_jaccard)

            # the locations where current overlap is higher than before
            greater_than_current_mask = tf.greater(jaccard, gt_scores)
            # we will update these locations as well as the current max_overlap location for this ground_truth
            locations_to_update = tf.logical_or(greater_than_current_mask, cur_max_indice_mask)
            # but we will ignore those locations where is the max_overlap for any ground_truth before
            locations_to_update_with_mask = tf.logical_and(locations_to_update, tf.logical_not(max_mask))
            # for current max_overlap
            # for those current overlap is higher than before
            # for those locations where is not the max_overlap for any before ground_truth
            # update scores, so the terminal scores are either those max_overlap along the way or the max_overlap for any ground_truth
            gt_scores = tf.where(locations_to_update_with_mask, jaccard, gt_scores)

            # !!! because the difference of rules for score and label update !!!
            # !!! so before we get the negtive examples we must use labels as positive mask first !!!
            # for current max_overlap
            # for current jaccard higher than before and higher than threshold (those scores are lower than is is ignored)
            # for those locations where is not the max_overlap for any before ground_truth
            # update labels, so the terminal labels are either those with max_overlap and higher than threshold along the way or the max_overlap for any ground_truth
            locations_to_update_labels = tf.logical_and(tf.logical_or(tf.greater(tf.cast(greater_than_current_mask, tf.float32) * jaccard, self._ignore_threshold), cur_max_indice_mask), tf.logical_not(max_mask))
            locations_to_update_labels_mask = tf.cast(tf.logical_and(locations_to_update_labels, label < self._num_classes), tf.float32)

            gt_labels = tf.cast(locations_to_update_labels_mask, tf.int64) * label + (1 - tf.cast(locations_to_update_labels_mask, tf.int64)) * gt_labels
            #gt_scores = tf.where(mask, jaccard, gt_scores)
            # update ground truth for each anchors depends on the mask
            gt_ymin = locations_to_update_labels_mask * bbox[0] + (1 - locations_to_update_labels_mask) * gt_ymin
            gt_xmin = locations_to_update_labels_mask * bbox[1] + (1 - locations_to_update_labels_mask) * gt_xmin
            gt_ymax = locations_to_update_labels_mask * bbox[2] + (1 - locations_to_update_labels_mask) * gt_ymax
            gt_xmax = locations_to_update_labels_mask * bbox[3] + (1 - locations_to_update_labels_mask) * gt_xmax

            # update max_mask along the way
            max_mask = tf.logical_or(max_mask, cur_max_indice_mask)

            return [i+1, gt_labels, gt_scores,
                    gt_ymin, gt_xmin, gt_ymax, gt_xmax, max_mask]
        # Main loop definition.
        # iterate betwween all ground_truth to encode anchors
        i = 0
        [i, gt_labels, gt_scores,
         gt_ymin, gt_xmin,
         gt_ymax, gt_xmax, max_mask] = tf.while_loop(condition, body,
                                               [i, gt_labels, gt_scores,
                                                gt_ymin, gt_xmin,
                                                gt_ymax, gt_xmax, max_mask], parallel_iterations=16, back_prop=False, swap_memory=True)
        # transform to center / size for later regression target calculating
        gt_cy = (gt_ymax + gt_ymin) / 2.
        gt_cx = (gt_xmax + gt_xmin) / 2.
        gt_h = gt_ymax - gt_ymin
        gt_w = gt_xmax - gt_xmin
        # get regression target for smooth_l1_loss
        # the prior_scaling (in fact is 5 and 10) is use for balance the regression loss of center and with(or height)
        # (x-x_ref)/x_ref * 10 + log(w/w_ref) * 5
        gt_cy = (gt_cy - yref) / href / self._prior_scaling[0]
        gt_cx = (gt_cx - xref) / wref / self._prior_scaling[1]
        gt_h = tf.log(gt_h / href) / self._prior_scaling[2]
        gt_w = tf.log(gt_w / wref) / self._prior_scaling[3]

        # now gt_localizations is our regression object
        return gt_labels, tf.stack([gt_cy, gt_cx, gt_h, gt_w], axis=-1), gt_scores
    def encode_all_anchors(self, labels, bboxes):
        self._labels = labels
        self._bboxes = bboxes

        ground_labels = []
        anchor_regress_targets = []
        ground_scores = []

        for layer_index, anchor in enumerate(self._anchors):
            ground_label, anchor_regress_target, ground_score = self.encode_anchor(anchor)
            ground_labels.append(ground_label)
            anchor_regress_targets.append(anchor_regress_target)
            ground_scores.append(ground_score)
        return ground_labels, anchor_regress_targets, ground_scores, len(self._anchors)

    # return a list, of which each is:
    #   shape: [feature_h, feature_w, num_anchors, 4]
    #   order: ymin, xmin, ymax, xmax
    def decode_all_anchors(self, pred_location):
        assert len(self._anchors) == len(pred_location), 'predict location not equals to anchor priors.'
        pred_bboxes = []
        for index, location_ in enumerate(pred_location):
            # each location_:
            #   shape: [feature_h, feature_w, num_anchors, 4]
            #   order: cy, cx, h, w
            anchor = self._anchors[index]
            yref, xref, href, wref = tf.expand_dims(anchor[0], axis=-1), tf.expand_dims(anchor[1], axis=-1), anchor[2], anchor[3]
            # batch_size, feature_h, feature_w, num_anchors, 4
            location_ = tf.reshape(location_, [-1] + anchor[0].get_shape().as_list() + href.get_shape().as_list() + [4])

            #print(yref.get_shape().as_list())
            def decode_impl(each_location):
                #each_location = tf.reshape(each_location, yref.get_shape().as_list()[:-1] + [4])
                pred_h = tf.exp(each_location[:, :, :, -2] * self._prior_scaling[2]) * href
                pred_w = tf.exp(each_location[:, :, :, -1] * self._prior_scaling[3]) * wref
                pred_cy = each_location[:, :, :, 0] * self._prior_scaling[0] * href + yref
                pred_cx = each_location[:, :, :, 1] * self._prior_scaling[1] * wref + xref
                return tf.stack(self.center2point(pred_cy, pred_cx, pred_h, pred_w), axis=-1)

            pred_bboxes.append(tf.map_fn(decode_impl, location_))

        return pred_bboxes


class AnchorCreator(object):
    def __init__(self, img_shape, layers_shapes, anchor_scales, extra_anchor_scales, anchor_ratios, layer_steps):
        super(AnchorCreator, self).__init__()
        # img_shape -> (height, width)
        self._img_shape = img_shape
        self._layers_shapes = layers_shapes
        self._anchor_scales = anchor_scales
        self._extra_anchor_scales = extra_anchor_scales
        self._anchor_ratios = anchor_ratios
        self._layer_steps = layer_steps
        self._anchor_offset = [0.5] * len(self._layers_shapes)

    def get_layer_anchors(self, layer_shape, anchor_scale, extra_anchor_scale, anchor_ratio, layer_step, offset = 0.5):
        ''' assume layer_shape[0] = 6, layer_shape[1] = 5
        x_on_layer = [[0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4]]
        y_on_layer = [[0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1],
                       [2, 2, 2, 2, 2],
                       [3, 3, 3, 3, 3],
                       [4, 4, 4, 4, 4],
                       [5, 5, 5, 5, 5]]
        '''
        x_on_layer, y_on_layer = tf.meshgrid(tf.range(layer_shape[1]), tf.range(layer_shape[0]))

        y_on_image = (tf.cast(y_on_layer, tf.float32) + offset) * layer_step / self._img_shape[0]
        x_on_image = (tf.cast(x_on_layer, tf.float32) + offset) * layer_step / self._img_shape[1]

        num_anchors = len(anchor_scale) * len(anchor_ratio) + len(extra_anchor_scale)

        list_h_on_image = []
        list_w_on_image = []

        global_index = 0
        for _, scale in enumerate(extra_anchor_scale):
            # h_on_image[global_index] = scale
            # w_on_image[global_index] = scale
            list_h_on_image.append(scale)
            list_w_on_image.append(scale)
            global_index += 1
        for scale_index, scale in enumerate(anchor_scale):
            for ratio_index, ratio in enumerate(anchor_ratio):
                # h_on_image[global_index] = scale  / math.sqrt(ratio)
                # w_on_image[global_index] = scale  * math.sqrt(ratio)
                list_h_on_image.append(scale  / math.sqrt(ratio))
                list_w_on_image.append(scale  * math.sqrt(ratio))
                global_index += 1
        # shape:
        # y_on_image, x_on_image: layers_shapes[0] * layers_shapes[1]
        # h_on_image, w_on_image: num_anchors
        return y_on_image, x_on_image, tf.constant(list_h_on_image, dtype=tf.float32), tf.constant(list_w_on_image, dtype=tf.float32), num_anchors

    def get_all_anchors(self):
        all_anchors = []
        num_anchors = []
        for layer_index, layer_shape in enumerate(self._layers_shapes):
            anchors_this_layer = self.get_layer_anchors(layer_shape,
                                                        self._anchor_scales[layer_index],
                                                        self._extra_anchor_scales[layer_index],
                                                        self._anchor_ratios[layer_index],
                                                        self._layer_steps[layer_index],
                                                        self._anchor_offset[layer_index])
            all_anchors.append(anchors_this_layer[:-1])
            num_anchors.append(anchors_this_layer[-1])
        return all_anchors, num_anchors

# procedure from Detectron of Facebook
# 1. for each location i in a (H, W) grid:
#      generate A anchor boxes centered on cell i
#      apply predicted bbox deltas to each of the A anchors at cell i
# 2. clip predicted boxes to image (may result in proposals with zero area that will be removed in the next step)
# 3. remove predicted boxes with either height or width < threshold
# 4. sort all (proposal, score) pairs by score from highest to lowest
# 5. take the top pre_nms_topN proposals before NMS (e.g. 6000)
# 6. apply NMS with a loose threshold (0.7) to the remaining proposals
# 7. take after_nms_topN (e.g. 300) proposals after NMS
# 8. return the top proposals
class BBoxUtils(object):
    @staticmethod
    def tf_bboxes_nms(scores, labels, bboxes, nms_threshold = 0.5, keep_top_k = 200, mode = 'min', scope=None):
        with tf.name_scope(scope, 'tf_bboxes_nms', [scores, labels, bboxes]):
            num_anchors = tf.shape(scores)[0]
            def nms_proc(scores, labels, bboxes):
                # sort all the bboxes
                scores, idxes = tf.nn.top_k(scores, k = num_anchors, sorted = True)
                labels, bboxes = tf.gather(labels, idxes), tf.gather(bboxes, idxes)

                ymin = bboxes[:, 0]
                xmin = bboxes[:, 1]
                ymax = bboxes[:, 2]
                xmax = bboxes[:, 3]

                vol_anchors = (xmax - xmin) * (ymax - ymin)

                nms_mask = tf.cast(tf.ones_like(scores, dtype=tf.int8), tf.bool)
                keep_mask = tf.cast(tf.zeros_like(scores, dtype=tf.int8), tf.bool)

                def safe_divide(numerator, denominator):
                    return tf.where(tf.greater(denominator, 0), tf.divide(numerator, denominator), tf.zeros_like(denominator))

                def get_scores(bbox, nms_mask):
                    # the inner square
                    inner_ymin = tf.maximum(ymin, bbox[0])
                    inner_xmin = tf.maximum(xmin, bbox[1])
                    inner_ymax = tf.minimum(ymax, bbox[2])
                    inner_xmax = tf.minimum(xmax, bbox[3])
                    h = tf.maximum(inner_ymax - inner_ymin, 0.)
                    w = tf.maximum(inner_xmax - inner_xmin, 0.)
                    inner_vol = h * w
                    this_vol = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if mode == 'union':
                        union_vol = vol_anchors - inner_vol  + this_vol
                    elif mode == 'min':
                        union_vol = tf.minimum(vol_anchors, this_vol)
                    else:
                        raise ValueError('unknown mode to use for nms.')
                    return safe_divide(inner_vol, union_vol) * tf.cast(nms_mask, tf.float32)

                def condition(index, nms_mask, keep_mask):
                    return tf.logical_and(tf.reduce_sum(tf.cast(nms_mask, tf.int32)) > 0, tf.less(index, keep_top_k))

                def body(index, nms_mask, keep_mask):
                    # at least one True in nms_mask
                    indices = tf.where(nms_mask)[0][0]
                    bbox = bboxes[indices]
                    this_mask = tf.one_hot(indices, num_anchors, on_value=False, off_value=True, dtype=tf.bool)
                    keep_mask = tf.logical_or(keep_mask, tf.logical_not(this_mask))
                    nms_mask = tf.logical_and(nms_mask, this_mask)

                    nms_scores = get_scores(bbox, nms_mask)

                    nms_mask = tf.logical_and(nms_mask, nms_scores < nms_threshold)
                    return [index+1, nms_mask, keep_mask]

                index = 0
                [index, nms_mask, keep_mask] = tf.while_loop(condition, body, [index, nms_mask, keep_mask])
                return tf.boolean_mask(scores, keep_mask), tf.boolean_mask(labels, keep_mask), tf.boolean_mask(bboxes, keep_mask)

            return tf.cond(tf.less(num_anchors, 1), lambda: (scores, labels, bboxes), lambda: nms_proc(scores, labels, bboxes))

    @staticmethod
    def tf_bboxes_nms_by_class(scores, labels, bboxes, num_classes, nms_threshold = 0.5, keep_top_k = 200, mode = 'min', scope=None):
        with tf.name_scope(scope, 'tf_bboxes_nms_by_class', [scores, labels, bboxes]):
            num_anchors = tf.shape(scores)[0]
            def nms_proc(scores, labels, bboxes):
                # sort all the bboxes
                scores, idxes = tf.nn.top_k(scores, k = num_anchors, sorted = True)
                labels, bboxes = tf.gather(labels, idxes), tf.gather(bboxes, idxes)

                ymin = bboxes[:, 0]
                xmin = bboxes[:, 1]
                ymax = bboxes[:, 2]
                xmax = bboxes[:, 3]

                vol_anchors = (xmax - xmin) * (ymax - ymin)

                total_keep_mask = tf.cast(tf.zeros_like(scores, dtype=tf.int8), tf.bool)

                def safe_divide(numerator, denominator):
                    return tf.where(tf.greater(denominator, 0), tf.divide(numerator, denominator), tf.zeros_like(denominator))

                def get_scores(bbox, nms_mask):
                    # the inner square
                    inner_ymin = tf.maximum(ymin, bbox[0])
                    inner_xmin = tf.maximum(xmin, bbox[1])
                    inner_ymax = tf.minimum(ymax, bbox[2])
                    inner_xmax = tf.minimum(xmax, bbox[3])
                    h = tf.maximum(inner_ymax - inner_ymin, 0.)
                    w = tf.maximum(inner_xmax - inner_xmin, 0.)
                    inner_vol = h * w
                    this_vol = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if mode == 'union':
                        union_vol = vol_anchors - inner_vol  + this_vol
                    elif mode == 'min':
                        union_vol = tf.minimum(vol_anchors, this_vol)
                    else:
                        raise ValueError('unknown mode to use for nms.')
                    return safe_divide(inner_vol, union_vol) * tf.cast(nms_mask, tf.float32)

                def condition(index, nms_mask, keep_mask):
                    return tf.logical_and(tf.reduce_sum(tf.cast(nms_mask, tf.int32)) > 0, tf.less(index, keep_top_k))

                def body(index, nms_mask, keep_mask):
                    # at least one True in nms_mask
                    indices = tf.where(nms_mask)[0][0]
                    bbox = bboxes[indices]
                    this_mask = tf.one_hot(indices, num_anchors, on_value=False, off_value=True, dtype=tf.bool)
                    keep_mask = tf.logical_or(keep_mask, tf.logical_not(this_mask))
                    nms_mask = tf.logical_and(nms_mask, this_mask)

                    nms_scores = get_scores(bbox, nms_mask)

                    nms_mask = tf.logical_and(nms_mask, nms_scores < nms_threshold)
                    return [index+1, nms_mask, keep_mask]
                def nms_loop_for_each(cls_index, total_keep_mask):
                    index = 0
                    nms_mask = tf.equal(tf.cast(cls_index, tf.int64), labels)
                    keep_mask = tf.cast(tf.zeros_like(scores, dtype=tf.int8), tf.bool)

                    [_, _, keep_mask] = tf.while_loop(condition, body, [index, nms_mask, keep_mask])
                    total_keep_mask = tf.logical_or(total_keep_mask, keep_mask)

                    return cls_index + 1, total_keep_mask
                cls_index = 1
                [_, total_keep_mask] = tf.while_loop(lambda cls_index, _: tf.less(cls_index, num_classes), nms_loop_for_each, [cls_index, total_keep_mask])
                indices_to_select = tf.where(total_keep_mask)
                select_mask = tf.cond(tf.less(tf.shape(indices_to_select)[0], keep_top_k + 1),
                                    lambda: total_keep_mask,
                                    lambda: tf.logical_and(total_keep_mask, tf.range(tf.cast(tf.shape(total_keep_mask)[0], tf.int64), dtype=tf.int64) < indices_to_select[keep_top_k][0]))
                return tf.boolean_mask(scores, select_mask), tf.boolean_mask(labels, select_mask), tf.boolean_mask(bboxes, select_mask)

            return tf.cond(tf.less(num_anchors, 1), lambda: (scores, labels, bboxes), lambda: nms_proc(scores, labels, bboxes))

    @staticmethod
    def filter_boxes(scores, labels, bboxes, min_size_ratio, image_shape, net_input_shape):
        """Only keep boxes with both sides >= min_size and center within the image.
        min_size_ratio is the ratio relative to net input shape
        """
        # Scale min_size to match image scale
        min_size = tf.maximum(0.0001, min_size_ratio * tf.sqrt(tf.cast(image_shape[0] * image_shape[1], tf.float32) / (net_input_shape[0] * net_input_shape[1])))

        ymin = bboxes[:, 0]
        xmin = bboxes[:, 1]

        ws = bboxes[:, 3] - xmin
        hs = bboxes[:, 2] - ymin

        x_ctr = xmin + ws / 2.
        y_ctr = ymin + hs / 2.

        keep_mask = tf.logical_and(tf.greater(ws, min_size), tf.greater(hs, min_size))
        keep_mask = tf.logical_and(keep_mask, tf.greater(x_ctr, 0.))
        keep_mask = tf.logical_and(keep_mask, tf.greater(y_ctr, 0.))
        keep_mask = tf.logical_and(keep_mask, tf.less(x_ctr, 1.))
        keep_mask = tf.logical_and(keep_mask, tf.less(y_ctr, 1.))

        return tf.boolean_mask(scores, keep_mask), tf.boolean_mask(labels, keep_mask), tf.boolean_mask(bboxes, keep_mask)
