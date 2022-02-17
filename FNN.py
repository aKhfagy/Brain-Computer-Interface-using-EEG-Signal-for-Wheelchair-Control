import mne
from sklearn.model_selection import train_test_split
import tensorflow as tf


class FNN:
    def __init__(self, features, labels):
        self.data = features
        self.labels = labels
        # self.eeg_epochs = self.epochs.copy().pick_types(eeg=True, meg=False, eog=False)
        # self.eeg_data = self.eeg_epochs.get_data().reshape(len(self.labels), -1)
        return

    def create_layer(self, input_layer, n_neurons, layer_name="", activation_fun=None):
        with tf.name_scope(layer_name):
            n_inputs = int(input_layer.get_shape()[1])
            initial_value = tf.truncated_normal((n_inputs, n_neurons))  # initial value (will updated at each iteration)
            w = tf.Variable(initial_value, name="weight")  # weight vector, initiazed to initial_value
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")  # bias vector
            op = tf.matmul(input_layer, w) + b
            if activation_fun:
                op = activation_fun(op)
            return op

    def make_model(self, n_inputs, n_hidden1, n_hidden2, n_outputs, n_iterations=50,
                   n_batches=33, learn_rate=0.00003):
        X = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs], name='X')
        y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')
        X_train, X_test, y_train, y_test = train_test_split(self.data,
                                                            self.labels, test_size=0.33,
                                                            random_state=42)

        with tf.name_scope("fnn"):  # fuzzy function
            h1 = self.create_layer(X, n_hidden1, layer_name='hl1', activation_fun=tf.nn.relu)
            h2 = self.create_layer(h1, n_hidden2, layer_name='hl2', activation_fun=tf.nn.relu)
            logits = self.create_layer(h2, n_outputs, layer_name='output')

        with tf.name_scope('loss'):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
            loss = tf.reduce_mean(entropy)

        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learn_rate)
            training_op = optimizer.minimize(loss)

        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, y, 1)
            acc = tf.reduce_mean(tf.cast(correct, tf.float32))

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            batch_size = int(len(X_train) / n_batches)
            for iteration_id in range(n_iterations):
                for batch_id in range(n_batches):
                    i = batch_id * batch_size
                    j = (batch_id + 1) * batch_size
                    xx_train, yy_train = X_train.iloc[i:j, :], y_train.iloc[i:j, :]

                    sess.run(training_op, feed_dict={X: xx_train, y: yy_train})

                loss_val = sess.run([loss], feed_dict={X: X_train.iloc[:, :], y: y_train.iloc[:, :]})
                acc_train_val = sess.run([acc],
                                         feed_dict={X: X_train.iloc[:, :], y: y_train.iloc[:, :]})

                res = res.append({'epoch': iteration_id, 'loss': loss_val[0],
                                  'acc_train': acc_train_val[0]}, ignore_index=True)
                if iteration_id % 10 == 0:
                    print(iteration_id, str(round(loss_val[0], 1)), str(round(acc_train_val[0], 3)))

            all_vars = tf.global_variables()
            saver = tf.train.Saver(all_vars)
            saver.save(sess, 'checkpoint/FNN.ckpt')

            return sess

    def restore_model(self):
        with tf.Session() as sess:
            all_vars = tf.global_variables()
            saver = tf.train.Saver(all_vars)
            saver.restore(sess, 'checkpoint/FNN.ckpt')
            return sess
