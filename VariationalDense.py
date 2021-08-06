import tensorflow as tf
import numpy as np

class VariationalDense(tf.keras.layers.Layer):
    def __init__(self, output_dim, use_bias=True, threshold=3.0, kernel_initializer='glorot_normal', bias_initializer='zeros'):
        super(VariationalDense, self).__init__()
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.threshold = threshold

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.theta = self.add_weight("kernel", shape=(int(input_shape[-1]), self.output_dim),
                                     initializer=self.kernel_initializer, trainable=True)

        self.log_sigma2 = self.add_weight("log_sigma2", shape=(int(input_shape[-1]), self.output_dim),
                                 initializer=tf.constant_initializer(-10.0), trainable=True)

        if self.use_bias == True:
            self.bias = self.add_weight("bias", shape=(self.output_dim,),
                                        initializer=self.bias_initializer, trainable=True)

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
        if not sparse:
            theta = tf.where(tf.math.is_nan(self.theta), tf.zeros_like(self.theta), self.theta)
            sigma = tf.sqrt(tf.exp(self.log_alpha) * theta * theta)
            self.weight = theta + tf.random.normal(tf.shape(theta), 0.0, 1.0) * sigma
            output = tf.matmul(input, self.weight)
            if self.use_bias == True:
                output += self.bias

            return output

        else:
            output = tf.matmul(input, self.sparse_theta)
            if self.use_bias == True:
                output += self.bias

            return output


if __name__ == '__main__':
    a = VariationalDense(10)
    print(a(tf.zeros((1, 1)), sparse=True))
    print(a.regularization)






