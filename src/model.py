import tensorflow as tf

class Model ():

    def __init__ (self, smiles_vocabulary, rnn_size=[300, 2], classification_size=12):

        assert isinstance(smiles_vocabulary, list)
        assert isinstance(rnn_size, tuple) or isinstance(rnn_size, list)
        assert isinstance(classification_size, int)

        self.smiles_vocabulary = smiles_vocabulary
        self.smiles_vocabulary_size = len(smiles_vocabulary)
        self.rnn_size = rnn_size
        self.classification_size = classification_size

        """
        Take in SMILES character indices.
        Change to one-hot labels.
        """

        self.input = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.expect = tf.placeholder(shape=[None, self.classification_size], dtype=tf.float32)

        self.batch_size = tf.placeholder(shape=[], dtype=tf.int32)

        # +1 for padding values
        input_one_hot = tf.one_hot(self.input, depth=self.smiles_vocabulary_size + 1, axis=None)

        """
        Create LSTM layers.
        """

        with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):

            def rnn_cell (size, activation=tf.nn.selu):
                cell = tf.contrib.rnn.GRUCell(size, activation=activation)
                return tf.contrib.rnn.DropoutWrapper(cell, state_keep_prob=0.8, input_keep_prob=0.9, output_keep_prob=0.9)

            cells = tf.contrib.rnn.MultiRNNCell([rnn_cell(self.rnn_size[0]) for i in range(self.rnn_size[1])])

            """
            This neural network is iteratively
            stateless.

            However, for testing purposes, states
            are still stored as class properties regardless.
            """

            self.initial_state = cells.zero_state(self.batch_size, dtype=tf.float32)

            rnn_output, self.new_state = tf.nn.dynamic_rnn(cells, input_one_hot, initial_state=self.initial_state, parallel_iterations=32, time_major=False)

            # TODO: Attention mechanism, perhaps?

        with tf.variable_scope("Feedforward", reuse=tf.AUTO_REUSE):

            """
            Only the last output of the
            RNN is used.
            """

            last_output = rnn_output[:,-1]

            dense1 = tf.layers.dense(last_output, 128, activation=tf.nn.selu)
            dense2 = tf.layers.dense(dense1, 128, activation=tf.nn.selu)

            logits = tf.layers.dense(dense2, self.classification_size, activation=None)

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

            self.individual_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.expect, logits=logits)
            self.loss = tf.reduce_mean(self.individual_loss)

            """
            Define global steps and increment operation.
            """

            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
            increment_global_step = tf.assign(self.global_step, self.global_step + 1)

            optimiser = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.92, beta2=0.99)
            self.optimiser = tf.group([clipGradients(optimiser, self.individual_loss, None, 5), increment_global_step])
