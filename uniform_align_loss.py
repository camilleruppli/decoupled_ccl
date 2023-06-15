import json
from collections import Counter

import numpy as np
import tensorflow as tf
from autologging import logged


class UniformAlignLoss:
    def __init__(self, temperature=1.0, l_uniform=1.0, l_alignment=1.0, decoupled=True):
        self.temperature = temperature
        self.l_uniform = l_uniform
        self.l_alignment = l_alignment
        self.decoupled = decoupled

    def __call__(self, hidden, masking_labels=None):
        hidden = tf.math.l2_normalize(hidden, -1)
        hidden1, hidden2 = hidden[0], hidden[1]
        uniform = (1 / 2) * (self.get_uniform(self.get_pdist(hidden1)) +
                             self.get_uniform(self.get_pdist(hidden2)))
        positive_pairs_distance = tf.reduce_sum(tf.square(hidden1 - hidden2), axis=1)
        alignment = tf.reduce_mean(positive_pairs_distance)
        return [self.l_uniform * uniform, self.l_alignment * alignment]

    @staticmethod
    def get_pdist(hidden):
        pairwise_distances_squared = (
                tf.math.add(
                    tf.math.reduce_sum(tf.math.square(hidden), axis=[1], keepdims=True),
                    tf.math.reduce_sum(
                        tf.math.square(tf.transpose(hidden)), axis=[0], keepdims=True
                    ),
                )
                - 2.0 * tf.matmul(hidden, tf.transpose(hidden))
        )
        return pairwise_distances_squared

    def get_uniform(self, pairwise_distances_squared):
        uniform_metric = tf.exp(-self.temperature * pairwise_distances_squared)
        if self.decoupled:
            uniform_metric *= (1 - tf.eye(len(pairwise_distances_squared)))
        uniform = tf.math.log(tf.reduce_mean(uniform_metric))
        return uniform


@logged
class ConditionalUniformAlignLoss(UniformAlignLoss):
    def __init__(self, conditional_uniformity, kernel_func="delta_y",
                 global_uniformity_unlabeled_only=False, temperature=1.0, l_uniform=1.0,
                 l_alignment=1.0, decoupled=True):

        self.conditional_uniformity = conditional_uniformity
        self.kernel_func = kernel_func
        self.global_uniformity_unlabeled_only = global_uniformity_unlabeled_only
        self.kernel = getattr(self, self.kernel_func)
        self.mask = None
        self.weights = None
        self.neg_weights = None
        super().__init__(temperature, l_uniform, l_alignment, decoupled)

    def delta_y(self, x, y):
        return np.array([[tf.squeeze(y_i) == tf.squeeze(y_j) for y_j in y] for y_i in x],
                        dtype=int) * self.alpha

    def mask_update(self, shape_0, shape_1, mask_indices, symetry_update=True):
        mask = tf.ones((shape_0, shape_1))
        mask_updates_zi = tf.zeros((len(mask_indices), mask.shape[1]))
        mask_updates_zj = tf.zeros((len(mask_indices), mask.shape[1]))
        batch_size = shape_0 // 2
        for index in mask_indices:
            mask = tf.transpose(
                tf.tensor_scatter_nd_update(tf.transpose(mask), [index],
                                            tf.zeros((1, shape_0))))
            if symetry_update:
                mask = tf.transpose(
                    tf.tensor_scatter_nd_update(tf.transpose(mask), [index + batch_size],
                                                tf.zeros((1, shape_0))))
        for i in range(len(mask_indices)):
            mask_updates_zi = tf.tensor_scatter_nd_update(mask_updates_zi,
                                                          [[i, mask_indices[i, 0]]],
                                                          [1])
            mask_updates_zj = tf.tensor_scatter_nd_update(mask_updates_zj,
                                                          [[i, mask_indices[i, 0] +
                                                            batch_size]],
                                                          [1])
        mask = tf.tensor_scatter_nd_update(mask, mask_indices, mask_updates_zj)
        mask = tf.tensor_scatter_nd_update(mask, mask_indices + batch_size, mask_updates_zi)

        return mask

    def set_neg_weights(self, masking_labels_1, masking_labels_2, mask_indices):
        labels = self.kernel(masking_labels_1, masking_labels_2).astype(np.float32)
        inf_norm = tf.reduce_max(labels)
        self.neg_weights = (inf_norm - labels)
        self.neg_weights = tf.transpose(
            tf.tensor_scatter_nd_update(tf.transpose(self.neg_weights),
                                        mask_indices,
                                        tf.zeros(
                                            (len(mask_indices), self.neg_weights.shape[1]))))
        for index in mask_indices:
            up = 1 - tf.one_hot(index[0], len(self.neg_weights))
            if self.global_uniformity_unlabeled_only:
                for i in range(len(self.neg_weights)):
                    if [i] not in mask_indices:
                        up = tf.tensor_scatter_nd_update(up, [[i]], [0])
            self.neg_weights = tf.tensor_scatter_nd_update(self.neg_weights, [index],
                                                           tf.expand_dims(up, axis=0))

    def get_conditional_uniformity(self, p_dist, masking_labels_1, masking_labels_2, mask_indices):
        self.set_neg_weights(masking_labels_1, masking_labels_2, mask_indices)
        if tf.reduce_all(tf.reduce_sum(self.neg_weights, axis=1) == 0):
            return 0
        neg_weights_before_update = self.neg_weights
        weights_sum = tf.reduce_sum(self.neg_weights, axis=1)
        neg_weights_divide = self.neg_weights / tf.expand_dims(
            tf.reduce_sum(self.neg_weights, axis=1), 1)
        zero_rows = tf.where(weights_sum == 0)
        self.neg_weights = \
            tf.tensor_scatter_nd_update(neg_weights_divide,
                                        zero_rows,
                                        tf.zeros((len(zero_rows),
                                                  neg_weights_divide.shape[1])))
        unlabeled_negative_update = tf.gather_nd(neg_weights_before_update, mask_indices)
        self.neg_weights = tf.tensor_scatter_nd_update(self.neg_weights,
                                                       mask_indices,
                                                       unlabeled_negative_update)

        if self.compute_uniformity_exp:
            loss = tf.math.log((1 / len(masking_labels_1)) * tf.reduce_sum(
                tf.exp(-self.temperature * p_dist) * self.neg_weights))
        else:
            loss = (1 / len(masking_labels_1)) * tf.reduce_sum(
                -self.temperature * p_dist * self.neg_weights)
        return loss

    def __call__(self, hidden, masking_labels=None):
        if isinstance(hidden[0], list):
            hidden = [hidden[0][0], hidden[1][0]]
        hidden = tf.math.l2_normalize(hidden, -1)
        hidden1, hidden2 = hidden[0], hidden[1]
        batch_size = len(masking_labels)
        pairwise_distance_squared = self.get_pdist(tf.concat([hidden1, hidden2], 0))

        all_labels, mask_indices = self.get_all_labels_and_unlabeled(masking_labels, hidden)
        kernel_labels = self.kernel(all_labels, all_labels).astype(
            np.float32)
        self.mask = self.mask_update(kernel_labels.shape[0],
                                     kernel_labels.shape[0],
                                     mask_indices)
        self.weights = self.mask * kernel_labels * (1 - tf.eye(kernel_labels.shape[0]))
        delta_ij = tf.concat([tf.one_hot(tf.range(batch_size, 2 * batch_size), 2 * batch_size),
                              tf.one_hot(tf.range(batch_size), 2 * batch_size)], axis=0)
        self.weights = delta_ij + (1 - delta_ij) * self.weights

        weights = self.weights / tf.expand_dims(tf.reduce_sum(self.weights, axis=1), 1)
        alignment = (1 / len(hidden1)) * tf.reduce_sum(weights * pairwise_distance_squared)

        if self.conditional_uniformity:
            uniform = (1 / 2) * (
                    self.get_conditional_uniformity(self.get_pdist(hidden1),
                                                    masking_labels,
                                                    masking_labels,
                                                    mask_indices) +
                    self.get_conditional_uniformity(self.get_pdist(hidden1),
                                                    masking_labels,
                                                    masking_labels,
                                                    mask_indices))
        else:
            pairwise_distance_1 = self.get_pdist(hidden1)
            pairwise_distance_2 = self.get_pdist(hidden2)
            if self.global_uniformity_unlabeled_only:
                for i in range(len(pairwise_distance_1)):
                    if [i] not in mask_indices:
                        pairwise_distance_1 = self.pairwise_distance_update(i, pairwise_distance_1)
                        pairwise_distance_2 = self.pairwise_distance_update(i,
                                                                            pairwise_distance_2)
            uniform = (1 / 2) * (self.get_uniform(pairwise_distance_1) +
                                 self.get_uniform(pairwise_distance_2))

        return [self.l_uniform * uniform, self.l_alignment * alignment]

    @staticmethod
    def pairwise_distance_update(index, pairwise_distance):
        update = 1e9 * tf.ones((1, len(pairwise_distance)))
        pairwise_distance = tf.tensor_scatter_nd_update(pairwise_distance, [[index]], update)
        pairwise_distance = tf.transpose(
            tf.tensor_scatter_nd_update(tf.transpose(pairwise_distance),
                                        [[index]],
                                        update))
        return pairwise_distance

    def get_all_labels_and_unlabeled(self, masking_labels, hidden=None):
        unlabeled = tf.where(masking_labels == -1)
        mask_indices = tf.expand_dims(unlabeled[:, 0], axis=1)
        all_labels = tf.concat([masking_labels, masking_labels], 0)
        return all_labels, mask_indices


@logged
class ConfidenceCondUniformAlignLoss(ConditionalUniformAlignLoss):
    def __init__(self, conditional_uniformity, kernel_func="delta_y_confidence",
                 global_uniformity_unlabeled_only=False, temperature=1.0, l_uniform=1.0,
                 l_alignment=1.0, decoupled=True):
        super().__init__(conditional_uniformity, kernel_func, global_uniformity_unlabeled_only,
                         temperature, l_uniform, l_alignment, decoupled)

    @staticmethod
    def delta_y_confidence_delta_alpha(x, y):
        alpha = np.array([[min(y_i[1], y_j[1]) for y_j in y] for y_i in x])
        return np.array([[tf.squeeze(y_i[0]) == tf.squeeze(y_j[0]) for y_j in y] for y_i in x],
                        dtype=int) * alpha * (1 - (alpha == 0.5))

    def get_all_labels_and_unlabeled(self, masking_labels, hidden=None):
        """
        Labels are of the following format [majority voting output, confidence]
        :param hidden:
        :param masking_labels:
        :return:
        """
        self.__log.debug(masking_labels)
        unlabeled = tf.where(masking_labels == -1)
        mask_indices = tf.expand_dims(tf.unique(unlabeled[:, 0])[0], axis=1)
        all_labels = tf.concat([masking_labels, masking_labels], 0)
        return all_labels, mask_indices


class CondCoherentNNAlignUniform(ConfidenceCondUniformAlignLoss):
    def __init__(self, coherent_id_path, coherent_label_path, nearest_neighbor, metadata_file_path,
                 conditional_uniformity, kernel_func="delta_y_confidence",
                 global_uniformity_unlabeled_only=False, temperature=1.0, l_uniform=1.0,
                 l_alignment=1.0, decoupled=True):

        self.coherent_id_path = coherent_id_path
        self.coherent_label_path = coherent_label_path
        self.nearest_neighbor = nearest_neighbor
        self.metadata_file_path = metadata_file_path
        self.feature_queue = None
        self.coherent_labels = np.load(self.coherent_label_path)
        self.mean_annotator = self.get_mean_annotator_number()
        super().__init__(conditional_uniformity, kernel_func, global_uniformity_unlabeled_only,
                         temperature, l_uniform, l_alignment, decoupled)

    def nearest_neighbour(self, projections):
        support_similarities = tf.matmul(
            projections, self.feature_queue, transpose_b=True
        )
        nearest = tf.argsort(support_similarities,
                             axis=1,
                             direction="DESCENDING")[:, :self.nearest_neighbor]
        pseudo_labels = tf.gather(self.coherent_labels, nearest, axis=0)
        return pseudo_labels

    def get_mean_annotator_number(self):
        with open(self.metadata_file_path) as f:
            db_metadata = json.load(f)
        annot_num = 0
        count = 0
        for mongo_id in db_metadata["db"]:
            if len(db_metadata["db"][mongo_id]) != 1:
                annot_num += len(db_metadata["db"][mongo_id])
                count += 1
        return annot_num / count

    def update_with_pseudo_labels(self, masking_labels, hidden):
        pseudo_labels_1 = self.nearest_neighbour(hidden[0])
        pseudo_labels_2 = self.nearest_neighbour(hidden[1])
        for i, label in enumerate(masking_labels):
            if label[1] == 1 / self.mean_annotator:
                pseudo_labels = np.concatenate(
                    [pseudo_labels_1[i, :, 0], pseudo_labels_2[i, :, 0]], axis=0)
                pseudo_labels = np.concatenate(
                    [pseudo_labels, [label[0], label[0]]],
                    axis=0)
                pseudo_labels_number = 2 * (
                        self.nearest_neighbor + 1)
                majority_voting = Counter(pseudo_labels).most_common(1)[0][0]
                confidence = sum(
                    [p == majority_voting for p in pseudo_labels]) / pseudo_labels_number
                label[1] = confidence
                label[0] = majority_voting
        return masking_labels

    def get_all_labels_and_unlabeled(self, masking_labels, hidden=None):
        """
        Labels are of the following format [majority voting output, confidence]
        :param hidden:
        :param masking_labels:
        :return:
        """
        masking_labels = self.update_with_pseudo_labels(masking_labels, hidden)
        return super().get_all_labels_and_unlabeled(masking_labels, None)
