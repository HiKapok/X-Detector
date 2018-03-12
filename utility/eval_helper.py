import tensorflow as tf

def tf_bboxes_nms(scores, labels, bboxes, nms_threshold = 0.5, select_threshold = 0., keep_top_k = 200, mode = 'min', scope=None):
    with tf.name_scope(scope, 'tf_bboxes_nms', [scores, labels, bboxes]):
        # get the cls_score for the most-likely class
        scores = tf.reduce_max(scores, -1)
        # apply threshold
        bbox_mask = tf.greater(scores, select_threshold)
        scores, labels, bboxes = tf.boolean_mask(scores, bbox_mask), tf.boolean_mask(labels, bbox_mask), tf.boolean_mask(bboxes, bbox_mask)
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

def tf_bboxes_nms_by_class(scores, labels, bboxes, nms_threshold = 0.5, select_threshold = 0., keep_top_k = 200, mode = 'min', scope=None):
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

            nms_mask = tf.cast(tf.ones_like(scores, dtype=tf.int8), tf.bool)
            nms_mask = tf.logical_and(nms_mask, tf.greater(scores, select_threshold))
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

                this_keep_mask = tf.one_hot(idxes[indices], num_anchors, on_value=True, off_value=False, dtype=tf.bool)
                keep_mask = tf.logical_or(keep_mask, this_keep_mask)

                nms_mask = tf.logical_and(nms_mask, this_mask)

                nms_scores = get_scores(bbox, nms_mask)

                nms_mask = tf.logical_and(nms_mask, nms_scores < nms_threshold)
                return [index+1, nms_mask, keep_mask]

            index = 0
            [index, nms_mask, keep_mask] = tf.while_loop(condition, body, [index, nms_mask, keep_mask])
            return keep_mask
        def nms_by_cls_proc(scores, labels, bboxes):
            total_keep_mask = tf.map_fn(lambda _scores: nms_proc(_scores, labels, bboxes),
                                    tf.transpose(scores, perm=[1, 0]), parallel_iterations=10,
                                    back_prop=False,
                                    swap_memory=False,
                                    dtype=tf.bool,
                                    infer_shape=True)
            total_keep_mask = tf.transpose(total_keep_mask, perm=[1, 0])
            # scores in the keep places
            keep_scores = scores * tf.cast(total_keep_mask, scores.dtype)
            # get the max one in case one bbox is kept twice for different classes
            max_mask_scores = tf.reduce_max(keep_scores, -1)
            new_labels = tf.argmax(keep_scores, -1)
            # ignore bboxes those not been kept
            keep_mask = max_mask_scores > 0.
            return tf.boolean_mask(max_mask_scores, keep_mask), tf.boolean_mask(new_labels, keep_mask), tf.boolean_mask(bboxes, keep_mask)

        return tf.cond(tf.less(num_anchors, 1), lambda: (scores, labels, bboxes), lambda: nms_by_cls_proc(scores, labels, bboxes))

def tf_bboxes_nms_by_class_v1(scores, labels, bboxes, nms_threshold = 0.5, select_threshold = 0., keep_top_k = 200, mode = 'min', scope=None):
    with tf.name_scope(scope, 'tf_bboxes_nms_by_class', [scores, labels, bboxes]):
        scores = tf.reduce_max(scores, -1)
        bbox_mask = tf.greater(scores, select_threshold)
        scores, labels, bboxes = tf.boolean_mask(scores, bbox_mask), tf.boolean_mask(labels, bbox_mask), tf.boolean_mask(bboxes, bbox_mask)
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
            [_, total_keep_mask] = tf.while_loop(lambda cls_index, _: tf.less(cls_index, FLAGS.num_classes), nms_loop_for_each, [cls_index, total_keep_mask])
            indices_to_select = tf.where(total_keep_mask)
            select_mask = tf.cond(tf.less(tf.shape(indices_to_select)[0], keep_top_k + 1),
                                lambda: total_keep_mask,
                                lambda: tf.logical_and(total_keep_mask, tf.range(tf.cast(tf.shape(total_keep_mask)[0], tf.int64), dtype=tf.int64) < indices_to_select[keep_top_k][0]))
            return tf.boolean_mask(scores, select_mask), tf.boolean_mask(labels, select_mask), tf.boolean_mask(bboxes, select_mask)

        return tf.cond(tf.less(num_anchors, 1), lambda: (scores, labels, bboxes), lambda: nms_proc(scores, labels, bboxes))

def get_shape(x, rank=None):
    """Returns the dimensions of a Tensor as list of integers or scale tensors.

    Args:
      x: N-d Tensor;
      rank: Rank of the Tensor. If None, will try to guess it.
    Returns:
      A list of `[d1, d2, ..., dN]` corresponding to the dimensions of the
        input tensor.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

def pad_axis(x, offset, size, axis=0, name=None):
    """Pad a tensor on an axis, with a given offset and output size.
    The tensor is padded with zero (i.e. CONSTANT mode). Note that the if the
    `size` is smaller than existing size + `offset`, the output tensor
    was the latter dimension.

    Args:
      x: Tensor to pad;
      offset: Offset to add on the dimension chosen;
      size: Final size of the dimension.
    Return:
      Padded tensor whose dimension on `axis` is `size`, or greater if
      the input vector was larger.
    """
    with tf.name_scope(name, 'pad_axis'):
        shape = get_shape(x)
        rank = len(shape)
        # Padding description.
        new_size = tf.maximum(size-offset-shape[axis], 0)
        pad1 = tf.stack([0]*axis + [offset] + [0]*(rank-axis-1))
        pad2 = tf.stack([0]*axis + [new_size] + [0]*(rank-axis-1))
        paddings = tf.stack([pad1, pad2], axis=1)
        x = tf.pad(x, paddings, mode='CONSTANT')
        # Reshape, to get fully defined shape if possible.
        # TODO: fix with tf.slice
        shape[axis] = tf.maximum(size, shape[axis])
        x = tf.reshape(x, tf.stack(shape))
        return x

def filter_boxes(scores, bboxes, min_size_ratio, image_shape, net_input_shape, keep_top_k = 100, scope=None):
    """Only keep boxes with both sides >= min_size and center within the image.
    min_size_ratio is the ratio relative to net input shape
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) and isinstance(bboxes, dict):
        with tf.name_scope(scope, 'filter_boxes_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = filter_boxes(scores[c], bboxes[c], min_size_ratio, image_shape, net_input_shape)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'filter_boxes', [scores, bboxes]):
        # Scale min_size to match image scale
        min_size = tf.maximum(0.0001, min_size_ratio * tf.sqrt(tf.cast(image_shape[0] * image_shape[1], tf.float32) / (net_input_shape[0] * net_input_shape[1])))

        ymin = bboxes[:, 0]
        xmin = bboxes[:, 1]
        ymax = bboxes[:, 2]
        xmax = bboxes[:, 3]

        ws = xmax - xmin
        hs = ymax - ymin
        x_ctr = xmin + ws / 2.
        y_ctr = ymin + hs / 2.

        keep_mask = tf.logical_and(tf.greater(ws, min_size), tf.greater(hs, min_size))
        keep_mask = tf.logical_and(keep_mask, tf.greater(x_ctr, 0.))
        keep_mask = tf.logical_and(keep_mask, tf.greater(y_ctr, 0.))
        keep_mask = tf.logical_and(keep_mask, tf.less(x_ctr, 1.))
        keep_mask = tf.logical_and(keep_mask, tf.less(y_ctr, 1.))

        scores = pad_axis(tf.boolean_mask(scores, keep_mask), 0, keep_top_k, axis=0)
        bboxes = pad_axis(tf.boolean_mask(bboxes, keep_mask), 0, keep_top_k, axis=0)

        return scores, bboxes

# Copyright 2017 Paul Balanca. All Rights Reserved.
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
def bboxes_sort(scores, bboxes, top_k=100, scope=None):
    """Sort bounding boxes by decreasing order and keep only the top_k.
    If inputs are dictionnaries, assume every key is a different class.
    Assume a batch-type input.

    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      top_k: Top_k boxes to keep.
    Return:
      scores, bboxes: Sorted Tensors/Dictionaries of shape Batch x Top_k x 1|4.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_sort_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_sort(scores[c], bboxes[c], top_k=tf.minimum(tf.shape(scores[c])[0], top_k))
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_sort', [scores, bboxes]):
        # Sort scores...
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)
        #return scores, tf.gather(bboxes, idxes)
        return pad_axis(scores, 0, top_k, axis=0), pad_axis(tf.gather(bboxes, idxes), 0, top_k, axis=0)



def bboxes_clip(bbox_ref, bboxes, scope=None):
    """Clip bounding boxes to a reference box.
    Batch-compatible if the first dimension of `bbox_ref` and `bboxes`
    can be broadcasted.

    Args:
      bbox_ref: Reference bounding box. Nx4 or 4 shaped-Tensor;
      bboxes: Bounding boxes to clip. Nx4 or 4 shaped-Tensor or dictionary.
    Return:
      Clipped bboxes.
    """
    # Bboxes is dictionary.
    if isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_clip_dict'):
            d_bboxes = {}
            for c in bboxes.keys():
                d_bboxes[c] = bboxes_clip(bbox_ref, bboxes[c])
            return d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_clip'):
        # Easier with transposed bboxes. Especially for broadcasting.
        bbox_ref = tf.transpose(bbox_ref)
        bboxes = tf.transpose(bboxes)
        # Intersection bboxes and reference bbox.
        ymin = tf.maximum(bboxes[0], bbox_ref[0])
        xmin = tf.maximum(bboxes[1], bbox_ref[1])
        ymax = tf.minimum(bboxes[2], bbox_ref[2])
        xmax = tf.minimum(bboxes[3], bbox_ref[3])
        # Double check! Empty boxes when no-intersection.
        #      _____
        #     |     |
        #     |_____|
        #                ______
        #               |      |
        #               |______|
        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)
        bboxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax], axis=0))
        return bboxes

def bboxes_resize(bbox_ref, bboxes, name=None):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform. Useful for
    updating a collection of boxes after cropping an image.
    """
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes

def bboxes_resize(bbox_ref, bboxes, name=None):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform. Useful for
    updating a collection of boxes after cropping an image.
    """
    # Bboxes is dictionary.
    if isinstance(bboxes, dict):
        with tf.name_scope(name, 'bboxes_resize_dict'):
            d_bboxes = {}
            for c in bboxes.keys():
                d_bboxes[c] = bboxes_resize(bbox_ref, bboxes[c])
            return d_bboxes

    # Tensors inputs.
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes

def bboxes_nms(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Should only be used on single-entries. Use batch version otherwise.

    Args:
      scores: N Tensor containing float scores.
      bboxes: N x 4 Tensor containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      classes, scores, bboxes Tensors, sorted by score.
        Padded with zero if necessary.
    """
    with tf.name_scope(scope, 'bboxes_nms_single', [scores, bboxes]):
        # Apply NMS algorithm.
        idxes = tf.image.non_max_suppression(bboxes, scores,
                                             keep_top_k, nms_threshold)
        scores = tf.gather(scores, idxes)
        bboxes = tf.gather(bboxes, idxes)
        # Pad results.
        scores = pad_axis(scores, 0, keep_top_k, axis=0)
        bboxes = pad_axis(bboxes, 0, keep_top_k, axis=0)
        return scores, bboxes


def bboxes_nms_batch(scores, bboxes, nms_threshold=0.5, keep_top_k=200,
                     scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Use only on batched-inputs. Use zero-padding in order to batch output
    results.

    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      scores, bboxes Tensors/Dictionaries, sorted by score.
        Padded with zero if necessary.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_nms_batch_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_nms_batch(scores[c], bboxes[c],
                                        nms_threshold=nms_threshold,
                                        keep_top_k=keep_top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_nms_batch'):
        return bboxes_nms(scores, bboxes, nms_threshold, keep_top_k)

def xdet_predict_clswise(predictions_layer, localizations_layer,
                               select_threshold=None,
                               num_classes=21,
                               ignore_class=0,
                               scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'ssd_bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = tfe.get_shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = tfe.get_shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer,
                                         tf.stack([l_shape[0], -1, l_shape[-1]]))
        # just consider those legal bboxes
        # localizations_mask = (localizations_layer[:, :, 0] < localizations_layer[:, :, 2])
        # localizations_mask = tf.logical_and(localizations_mask, (localizations_layer[:, :, 1] < localizations_layer[:, :, 3]))
        # localizations_mask = tf.Print(localizations_mask,[localizations_mask], message='localizations_mask: ', summarize=30)
        d_scores = {}
        d_bboxes = {}
        for c in range(0, num_classes):
            if c != ignore_class:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater(scores, select_threshold), scores.dtype)
                # fmask = tf.cast(tf.logical_and(tf.greater(scores, select_threshold), localizations_mask), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes


def tf_bboxes_select_layer(predictions_layer, localizations_layer,
                           select_threshold=None,
                           num_classes=21,
                           scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: prediction layer;
      localizations_layer: localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size N x 1 | 4. Each key corresponding to a class.
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        d_scores = {}
        d_bboxes = {}
        for c in range(1, num_classes):
            # Remove boxes under the threshold.
            scores = predictions_layer[:, c]
            fmask = tf.cast(tf.greater(scores, select_threshold), scores.dtype)
            scores = scores * fmask
            bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
            # Append to dictionary.
            d_scores[c] = scores
            d_bboxes[c] = bboxes

        return d_scores, d_bboxes


def tf_bboxes_select(predictions, localizations,
                     select_threshold=None,
                     num_classes=21,
                     scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions: List of SSD prediction layers;
      localizations: List of localization layers;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    with tf.name_scope(scope, 'bboxes_select',
                       [predictions, localizations]):
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions)):
            scores, bboxes = tf_bboxes_select_layer(predictions[i],
                                                    localizations[i],
                                                    select_threshold,
                                                    num_classes)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        # Concat results.
        d_scores = {}
        d_bboxes = {}
        for c in l_scores[0].keys():
            ls = [s[c] for s in l_scores]
            lb = [b[c] for b in l_bboxes]
            d_scores[c] = tf.concat(ls, axis=1)
            d_bboxes[c] = tf.concat(lb, axis=1)
        return d_scores, d_bboxes

# all input are flaten
def xdet_predict(bbox_img, cls_pred_prob, bboxes_pred, input_image_size, train_image_size, nms_threshold, select_threshold, nms_topk, num_classes, nms_mode='union'):
    # remove bboxes that are not foreground
    pred_labels = tf.argmax(cls_pred_prob, -1)
    label_scores = cls_pred_prob
    non_background_mask = tf.greater(pred_labels, 0)

    label_scores, pred_labels, bboxes_pred = tf.boolean_mask(label_scores, non_background_mask), tf.boolean_mask(pred_labels, non_background_mask), tf.boolean_mask(bboxes_pred, non_background_mask)

    bboxes_pred = bboxes_clip(bbox_img, bboxes_pred)

    label_scores, pred_labels, bboxes_pred = filter_boxes(label_scores, pred_labels, bboxes_pred, 0.03, input_image_size, [train_image_size] * 2)

    #label_scores, pred_labels, bboxes_pred = tf_bboxes_nms_by_class(label_scores, pred_labels, bboxes_pred, nms_threshold=nms_threshold, keep_top_k=nms_topk, mode = nms_mode)
    label_scores, pred_labels, bboxes_pred = tf_bboxes_nms(label_scores, pred_labels, bboxes_pred, nms_threshold=nms_threshold, select_threshold = select_threshold, keep_top_k=nms_topk, mode = nms_mode)

    # Resize bboxes to original image shape.
    bboxes_pred = bboxes_resize(bbox_img, bboxes_pred)

    num_anchors = tf.shape(label_scores)[0]
    label_scores, idxes = tf.nn.top_k(label_scores, k = num_anchors, sorted = True)
    pred_labels, bboxes_pred = tf.gather(pred_labels, idxes), tf.gather(bboxes_pred, idxes)

    return label_scores, pred_labels, bboxes_pred


# Copyright 2017 Paul Balanca. All Rights Reserved.
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

def safe_divide(numerator, denominator):
    return tf.where(tf.greater(denominator, 0), tf.divide(numerator, denominator), tf.zeros_like(denominator))

def bboxes_jaccard(bbox_ref, bboxes, name=None):
    """Compute jaccard score between a reference box and a collection
    of bounding boxes.

    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with Jaccard scores.
    """
    with tf.name_scope(name, 'bboxes_jaccard'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        #print(bboxes.get_shape().as_list())
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = -inter_vol \
            + (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1]) \
            + (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
        jaccard = safe_divide(inter_vol, union_vol)
        return jaccard
def bboxes_matching(label, scores, bboxes,
                    glabels, gbboxes, gdifficults,
                    matching_threshold=0.5, scope=None):
    """Matching a collection of detected boxes with groundtruth values.
    Does not accept batched-inputs.
    The algorithm goes as follows: for every detected box, check
    if one grountruth box is matching. If none, then considered as False Positive.
    If the grountruth box is already matched with another one, it also counts
    as a False Positive. We refer the Pascal VOC documentation for the details.

    Args:
      rclasses, rscores, rbboxes: N(x4) Tensors. Detected objects, sorted by score;
      glabels, gbboxes: Groundtruth bounding boxes. May be zero padded, hence
        zero-class objects are ignored.
      matching_threshold: Threshold for a positive match.
    Return: Tuple of:
       n_gbboxes: Scalar Tensor with number of groundtruth boxes (may difer from
         size because of zero padding).
       tp_match: (N,)-shaped boolean Tensor containing with True Positives.
       fp_match: (N,)-shaped boolean Tensor containing with False Positives.
    """
    with tf.name_scope(scope, 'bboxes_matching_single',
                       [scores, bboxes, glabels, gbboxes]):
        rsize = tf.size(scores)
        rshape = tf.shape(scores)
        rlabel = tf.cast(label, glabels.dtype)
        # Number of groundtruth boxes.
        gdifficults = tf.cast(gdifficults, tf.bool)
        n_gbboxes = tf.count_nonzero(tf.logical_and(tf.equal(glabels, label),
                                                    tf.logical_not(gdifficults)))
        # Grountruth matching arrays.
        gmatch = tf.zeros(tf.shape(glabels), dtype=tf.bool)
        grange = tf.range(tf.size(glabels), dtype=tf.int32)
        # True/False positive matching TensorArrays.
        sdtype = tf.bool
        ta_tp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)
        ta_fp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)

        # Loop over returned objects.
        def m_condition(i, ta_tp, ta_fp, gmatch):
            r = tf.less(i, rsize)
            return r

        def m_body(i, ta_tp, ta_fp, gmatch):
            # Jaccard score with groundtruth bboxes.
            rbbox = bboxes[i]
            jaccard = bboxes_jaccard(rbbox, gbboxes)
            jaccard = jaccard * tf.cast(tf.equal(glabels, rlabel), dtype=jaccard.dtype)

            # Best fit, checking it's above threshold.
            idxmax = tf.cast(tf.argmax(jaccard, axis=0), tf.int32)
            jcdmax = jaccard[idxmax]
            match = jcdmax > matching_threshold
            existing_match = gmatch[idxmax]
            not_difficult = tf.logical_not(gdifficults[idxmax])

            # TP: match & no previous match and FP: previous match | no match.
            # If difficult: no record, i.e FP=False and TP=False.
            tp = tf.logical_and(not_difficult,
                                tf.logical_and(match, tf.logical_not(existing_match)))
            ta_tp = ta_tp.write(i, tp)
            fp = tf.logical_and(not_difficult,
                                tf.logical_or(existing_match, tf.logical_not(match)))
            ta_fp = ta_fp.write(i, fp)
            # Update grountruth match.
            mask = tf.logical_and(tf.equal(grange, idxmax),
                                  tf.logical_and(not_difficult, match))
            gmatch = tf.logical_or(gmatch, mask)

            return [i+1, ta_tp, ta_fp, gmatch]
        # Main loop definition.
        i = 0
        [i, ta_tp_bool, ta_fp_bool, gmatch] = \
            tf.while_loop(m_condition, m_body,
                          [i, ta_tp_bool, ta_fp_bool, gmatch],
                          parallel_iterations=1,
                          back_prop=False)
        # TensorArrays to Tensors and reshape.
        tp_match = tf.reshape(ta_tp_bool.stack(), rshape)
        fp_match = tf.reshape(ta_fp_bool.stack(), rshape)

        # Some debugging information...
        # tp_match = tf.Print(tp_match,
        #                     [n_gbboxes,
        #                      tf.reduce_sum(tf.cast(tp_match, tf.int64)),
        #                      tf.reduce_sum(tf.cast(fp_match, tf.int64)),
        #                      tf.reduce_sum(tf.cast(gmatch, tf.int64))],
        #                     'Matching (NG, TP, FP, GM): ')
        return n_gbboxes, tp_match, fp_match

def bboxes_matching_batch(labels, scores, bboxes,
                          glabels, gbboxes, gdifficults,
                          matching_threshold=0.5, scope=None):
    """Matching a collection of detected boxes with groundtruth values.
    Batched-inputs version.

    Args:
      rclasses, rscores, rbboxes: BxN(x4) Tensors. Detected objects, sorted by score;
      glabels, gbboxes: Groundtruth bounding boxes. May be zero padded, hence
        zero-class objects are ignored.
      matching_threshold: Threshold for a positive match.
    Return: Tuple or Dictionaries with:
       n_gbboxes: Scalar Tensor with number of groundtruth boxes (may difer from
         size because of zero padding).
       tp: (B, N)-shaped boolean Tensor containing with True Positives.
       fp: (B, N)-shaped boolean Tensor containing with False Positives.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_matching_batch_dict'):
            d_n_gbboxes = {}
            d_tp = {}
            d_fp = {}
            for c in labels:
                n, tp, fp = bboxes_matching_batch(c, scores[c], bboxes[c],
                                                 glabels, gbboxes, gdifficults,
                                                 matching_threshold)
                d_n_gbboxes[c] = n
                d_tp[c] = tp
                d_fp[c] = fp
            return d_n_gbboxes, d_tp, d_fp

    with tf.name_scope(scope, 'bboxes_matching_batch',
                       [scores, bboxes, glabels, gbboxes]):
        r = tf.map_fn(lambda x: bboxes_matching(tf.expand_dims(labels, axis=0), x[0], x[1],
                                                x[2], x[3], x[4],
                                                matching_threshold),

                      (tf.expand_dims(scores, axis=0), tf.expand_dims(bboxes, axis=0), tf.expand_dims(glabels, axis=0), tf.expand_dims(gbboxes, axis=0), tf.expand_dims(gdifficults, axis=0)),
                      dtype=(tf.int64, tf.bool, tf.bool),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=True,
                      infer_shape=True)
        return r[0], r[1], r[2]#, scores
