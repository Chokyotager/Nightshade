import tensorflow as tf

class Model ():

    # Sol-L: 300, 2
    def __init__ (self, smiles_vocabulary, rnn_size=[32, 8], classification_size=12, dropout=True):

        assert isinstance(smiles_vocabulary, list)
        assert isinstance(rnn_size, tuple) or isinstance(rnn_size, list)
        assert isinstance(classification_size, int)

        self.smiles_vocabulary = smiles_vocabulary
        self.smiles_vocabulary_size = len(smiles_vocabulary)
        self.rnn_size = rnn_size
        self.classification_size = classification_size

        self.dropout = dropout

        """
        Take in SMILES character indices.
        Change to one-hot labels.
        """

        self.input = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.expect = tf.placeholder(shape=[None, self.classification_size], dtype=tf.float32)

        self.batch_size = tf.placeholder(shape=[], dtype=tf.int32)

        """
        Loss weights: implementation to negate
        samples where the HTS result was inconclusive.
        """

        self.loss_weights = tf.ones_like(self.expect, dtype=tf.float32)

        # +1 for padding values
        input_one_hot = tf.one_hot(self.input, depth=self.smiles_vocabulary_size + 1, axis=None)

        """
        Create LSTM layers.
        """

        with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):

            def rnn_cell (size, activation=tf.nn.selu):
                cell = tf.contrib.rnn.GRUCell(size, activation=activation)

                if self.dropout:
                    return tf.contrib.rnn.DropoutWrapper(cell, state_keep_prob=1, input_keep_prob=1, output_keep_prob=0.7)
                else:
                    return cell

            cells = tf.contrib.rnn.MultiRNNCell([rnn_cell(self.rnn_size[0]) for i in range(self.rnn_size[1])])

            """
            This neural network is iteratively
            stateless.

            However, for testing purposes, states
            are still stored as class properties regardless.
            """

            self.initial_state = cells.zero_state(self.batch_size, dtype=tf.float32)

            rnn_output, self.new_state = tf.nn.dynamic_rnn(cells, input_one_hot, initial_state=self.initial_state, parallel_iterations=64, time_major=False)

            # TODO: Attention mechanism, perhaps?

        with tf.variable_scope("Feedforward", reuse=tf.AUTO_REUSE):

            """
            Only the last output of the
            RNN is used.
            """

            last_output = rnn_output[:,-1]

            dense1 = tf.layers.dense(last_output, 512, activation=tf.nn.selu)
            dense2 = tf.layers.dense(dense1, 512, activation=tf.nn.selu)
            dense3 = tf.layers.dense(dense2, 512, activation=tf.nn.selu)
            dense4 = tf.layers.dense(dense3, 512, activation=tf.nn.selu)

            logits = tf.layers.dense(dense4, self.classification_size, activation=None)

            self.output = tf.nn.sigmoid(logits)
            #self.classification = tf.argmax(self.output, axis=None)

        with tf.variable_scope("Trainer", reuse=tf.AUTO_REUSE):

            def clipGradients (x, loss, vars, value):
              grads = x.compute_gradients(loss, var_list=vars)
              clipped = [(tf.clip_by_norm(gradx, value), var) for gradx, var in grads if gradx != None]
              opt = x.apply_gradients(clipped)
              return opt


            """
            Loss: sigmoid cross entropy
            Implementation: TensorFlow logits v2.
            """

            self.individual_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.expect, logits=logits, weights=self.loss_weights)
            self.loss = tf.reduce_mean(self.individual_loss)

            """
            Define global steps and increment operation.
            """

            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
            increment_global_step = tf.assign(self.global_step, self.global_step + 1)

            optimiser = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.92, beta2=0.99)
            self.optimiser = tf.group([clipGradients(optimiser, self.individual_loss, None, 5), increment_global_step])
