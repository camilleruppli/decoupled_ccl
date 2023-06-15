import numpy as np
import tensorflow as tf


class ContrastiveLoss:

    def __init__(self, temperature=1.0, large_num=1e9, decoupled=True):
        self.temperature = temperature
        self.large_num = large_num
        self.decoupled = decoupled

    def get_labels(self, batch_size, masking_labels=None):
        labels = tf.concat([tf.eye(batch_size)], axis=1)
        labels = tf.concat([labels, tf.zeros((batch_size, batch_size))], axis=1)
        return labels

    @staticmethod
    def compute_decoupled_softmax(labels, logits):
        decoupled_negative = (1 - labels) * tf.exp(logits)
        decoupled_softmax = tf.exp(logits) / tf.reduce_sum(decoupled_negative,
                                                           axis=1,
                                                           keepdims=True)
        prod = labels * tf.math.log(decoupled_softmax + 1e-30)
        return - tf.reduce_sum(prod, axis=1)

    @staticmethod
    def compute_softmax(labels, logits):
        return tf.nn.softmax_cross_entropy_with_logits(labels, logits)

    def __call__(self, input_data, masking_labels=None) -> np.float32:
        hidden_list = [tf.math.l2_normalize(hidden, -1) for hidden in input_data]
        batch_size = tf.shape(hidden_list[0])[0]

        mask = tf.one_hot(tf.range(batch_size), batch_size)
        labels = self.get_labels(batch_size, masking_labels)
        loss = 0
        for i in range(len(hidden_list)):
            logits_list = [tf.matmul(hidden_list[i], hidden_list[j],
                                     transpose_b=True) / self.temperature for j in
                           range(len(hidden_list)) if j != i]
            logits_concat = tf.concat(logits_list, 1)
            logits_ii = tf.matmul(hidden_list[i], hidden_list[i],
                                  transpose_b=True) / self.temperature
            logits_ii = logits_ii - mask * self.large_num
            logits_concat = tf.concat([logits_concat, logits_ii], 1)
            if not self.decoupled:
                loss += self.compute_softmax(labels, logits_concat)
            else:
                loss += self.compute_decoupled_softmax(labels, logits_concat)

            loss = tf.reduce_mean(loss)
        return loss
