import tensorflow as tf
import numpy as np

class VariationalConv2d(tf.keras.layers.Layer):
    def __init__(self, kernel_size, stride, padding='SAME', threshold=3.0, kernel_initializer='glorot_normal', bias_initializer='zeros'):
        super(VariationalConv2d, self).__init__()
        assert len(kernel_size) == 4#kernel_size: [filter_height, filter_width, in_channels, out_channels]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.theta = self.add_weight("kernel", shape=self.kernel_size,
                                     initializer=self.kernel_initializer, trainable=True)

        self.log_sigma2 = self.add_weight("log_sigma2", shape=self.kernel_size,
                                 initializer=tf.constant_initializer(-10.0), trainable=True)

    def sparsity(self):
        total_param = np.prod(tf.shape(self.boolean_mask))
        remaining_param = tf.math.count_nonzero(tf.cast(self.boolean_mask, dtype=tf.uint8)).numpy()

        return remaining_param, total_param

    @property
    def log_alpha(self):
        theta = tf.where(tf.math.is_nan(self.theta), tf.zeros_like(self.theta), self.theta)
        log_sigma2 = tf.where(tf.math.is_nan(self.log_sigma2), tf.zeros_like(self.log_sigma2), self.log_sigma2)
        log_alpha = tf.clip_by_value(log_sigma2 - tf.math.log(tf.square(theta) + 1e-10), -20.0, 4.0)
        return tf.where(tf.math.is_nan(log_alpha), self.threshold * tf.ones_like(log_alpha), log_alpha)

    @property
    def boolean_mask(self):
        return self.log_alpha <= self.threshold

    @property
    def sparse_theta(self):
        theta = tf.where(tf.math.is_nan(self.theta), tf.zeros_like(self.theta), self.theta)
        return tf.where(self.boolean_mask, theta, tf.zeros_like(theta))

    @property
    def regularization(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        mdkl = k1 * tf.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * tf.math.log(1 + (tf.exp(-self.log_alpha))) + C

        return -tf.reduce_sum(mdkl)


    @tf.function
    def call(self, input, sparse=False):
        theta = tf.where(tf.math.is_nan(self.theta), tf.zeros_like(self.theta), self.theta)

        if not sparse:
            sigma = tf.sqrt(tf.exp(self.log_alpha) * theta * theta)
            self.weight = theta + tf.random.normal(tf.shape(theta), 0.0, 1.0) * sigma
            output = tf.nn.conv2d(input, self.weight, self.stride, self.padding)

            return output

        else:
            output = tf.nn.conv2d(input, self.sparse_theta, self.stride, self.padding)

            return output




