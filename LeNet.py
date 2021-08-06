import tensorflow as tf
import numpy as np

from VariationalDense import VariationalDense
from VariationalConv2d import VariationalConv2d
from sklearn.utils import shuffle

def rw_schedule(epoch):
    if epoch <= 1:
        return 0
    else:
        return 0.0001 * (epoch - 1)


class VariationalLeNet(tf.keras.Model):
    def __init__(self, n_class=10):
        super().__init__()
        self.n_class = n_class

        self.conv1 = VariationalConv2d((5,5,1,6), stride=1, padding='VALID')
        self.pooling1 = tf.keras.layers.MaxPooling2D(padding='SAME')
        self.conv2 = VariationalConv2d((5,5,6,16), stride=1, padding='VALID')
        self.pooling2 = tf.keras.layers.MaxPooling2D(padding='SAME')

        self.flat = tf.keras.layers.Flatten()
        self.fc1 = VariationalDense(120)
        self.fc2 = VariationalDense(84)
        self.fc3 = VariationalDense(10)

        self.hidden_layer = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]

    @tf.function
    def call(self, x, sparse=False):
        x = self.conv1(x, sparse)
        x = tf.nn.relu(x)
        x = self.pooling1(x)
        x = self.conv2(x, sparse)
        x = tf.nn.relu(x)
        x = self.pooling2(x)
        x = self.flat(x)
        x = self.fc1(x, sparse)
        x = tf.nn.relu(x)
        x = self.fc2(x, sparse)
        x = tf.nn.relu(x)
        x = self.fc3(x, sparse)
        x = tf.nn.softmax(x)

        return x

    def regularization(self):
        total_reg = 0
        for layer in self.hidden_layer:
            total_reg += layer.regularization

        return total_reg

    def count_sparsity(self):
        total_remain, total_param = 0, 0
        for layer in self.hidden_layer:
            a, b = layer.sparsity()
            total_remain += a
            total_param += b

        return 1 - (total_remain/total_param)



if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)

    @tf.function
    def compute_loss(label, pred, reg):
        return criterion(label, pred) + reg


    @tf.function
    def compute_loss2(label, pred):
        return criterion(label, pred)

    def train_step(x, t, epoch):
        with tf.GradientTape() as tape:
            preds = model(x)
            reg = rw_schedule(epoch) * model.regularization()
            loss = compute_loss(t, preds, reg)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)
        train_acc(t, preds)

        return preds

    @tf.function
    def test_step(x, t):
        preds = model(x, sparse=True)
        loss = compute_loss2(t, preds)
        test_loss(loss)
        test_acc(t, preds)

        return preds

    '''
    Load data
    '''
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    y_train = np.eye(10)[y_train].astype(np.float32)
    y_test = np.eye(10)[y_test].astype(np.float32)

    '''
    Build model
    '''
    model = VariationalLeNet()
    criterion = tf.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    '''
    Train model
    '''
    epochs = 10
    batch_size = 100
    n_batches = x_train.shape[0] // batch_size

    train_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.CategoricalAccuracy()
    test_loss = tf.keras.metrics.Mean()
    test_acc = tf.keras.metrics.CategoricalAccuracy()

    for epoch in range(epochs):

        _x_train, _y_train = shuffle(x_train, y_train, random_state=42)

        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            train_step(_x_train[start:end], _y_train[start:end], epoch)

        if epoch % 1 == 0 or epoch == epochs - 1:
            preds = test_step(x_test, y_test)
            print('Epoch: {}, Valid Cost: {:.3f}, Valid Acc: {:.3f}'.format(
                epoch+1,
                test_loss.result(),
                test_acc.result()
            ))
            print("Sparsity: ", model.count_sparsity())






