from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class FNN:
    def __init__(self, features, labels):
        self.data = features
        self.labels = labels
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

    def create_fuzzy_layer(self, input_layer_1, input_layer_2, n_neurons, layer_name=""):
        with tf.name_scope(layer_name):
            # layer h1
            n_inputs = int(input_layer_1.get_shape()[1])
            initial_value = tf.truncated_normal((n_inputs, n_neurons))  # initial value (will updated at each iteration)
            w = tf.Variable(initial_value, name="weight")  # weight vector, initiazed to initial_value
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")  # bias vector
            u = tf.matmul(input_layer_1, w) + b
            # input layer
            n_inputs = int(input_layer_2.get_shape()[1])
            initial_value = tf.truncated_normal((n_inputs, n_neurons))  # initial value (will updated at each iteration)
            w = tf.Variable(initial_value, name="weight")  # weight vector, initiazed to initial_value
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")  # bias vector
            v = tf.matmul(input_layer_2, w) + b
            op = u * v
            return op

    def make_model(self, n_inputs, n_hidden, n_outputs, n_iterations=50,
                   n_batches=33, learn_rate=0.00003):
        X = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs], name='X')
        y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')
        X_train, X_test, y_train, y_test = train_test_split(self.data,
                                                            self.labels, test_size=0.33,
                                                            random_state=42)

        with tf.name_scope("fnn"):
            h1 = self.create_layer(X, n_hidden, layer_name='hl1')
            h2 = self.create_layer(h1, n_hidden, layer_name='hl2', activation_fun=tf.nn.relu)
            h3 = self.create_layer(h2, n_hidden, layer_name='hl3', activation_fun=tf.nn.relu)
            h4 = self.create_layer(h3, n_hidden, layer_name='hl4')
            h5 = self.create_fuzzy_layer(h3, h4, n_hidden, layer_name='hl5')
            h6 = self.create_layer(h5, n_hidden, layer_name='hl6', activation_fun=tf.nn.softmax)
            logits = self.create_layer(h6, n_outputs, layer_name='output')

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
                    xx_train, yy_train = X_train[i:j], y_train[i:j]

                    sess.run(training_op, feed_dict={X: xx_train, y: yy_train})

                loss_val = sess.run([loss], feed_dict={X: X_train, y: y_train})
                acc_train_val = sess.run([acc],
                                         feed_dict={X: X_train, y: y_train})
                if (iteration_id + 1) % 10 == 0:
                    print('iteration id: ', iteration_id + 1,
                          ', loss: ', str(round(loss_val[0], 1)), 
                          ', accuracy: ', str(round(acc_train_val[0], 3)))

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
