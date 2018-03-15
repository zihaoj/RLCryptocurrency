from rl_cryptocurrency.models.pg_general_discrete import PGGeneralDiscrete

import tensorflow as tf


class PGGeneralDiscreteRNN(PGGeneralDiscrete):
    def _build_policy_network_op(self):
        """
        RNN version
        """

        with tf.variable_scope("policy_network"):
            # RNN part
            # only consider GRU this moment
            net_rnn = self._obs_placeholder
            cell = tf.nn.rnn_cell.GRUCell(num_units=self.get_config("rnn_hidden_size"))
            _, last_state = tf.nn.dynamic_rnn(cell, net_rnn,
                                              sequence_length=None, dtype=tf.float32)

            # MLP part
            net_mlp = last_state
            for layer in range(self.get_config("n_layers")):
                net_mlp = tf.contrib.layers.fully_connected(
                    inputs=net_mlp,
                    num_outputs=self.get_config("layer_size"),
                    activation_fn=self.get_config("activation"),
                    scope="layer_{:d}".format(layer)
                )
                if self.get_config("batch_norm_policy"):
                    net_mlp = tf.layers.batch_normalization(net_mlp,
                                                            training=self._is_training_placeholder,
                                                            name="layer_{:d}_batch_norm".format(layer))

            logits = tf.contrib.layers.fully_connected(
                inputs=net_mlp,
                num_outputs=2,
                activation_fn=None,
                scope="layer_output",
            )

        with tf.variable_scope("policy_sample"):
            # sample from it
            # shape: [None,]
            self._sampled_action = tf.squeeze(tf.multinomial(logits, num_samples=1), axis=1)

            # obtain log-probability if action is provided
            self._logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._action_placeholder,
                                                                            logits=logits)

        return self

    def _add_baseline_op(self):
        """
        RNN version
        """

        with tf.variable_scope("baseline_network"):
            # RNN part
            # only consider GRU this moment
            net_rnn = self._obs_placeholder
            cell = tf.nn.rnn_cell.GRUCell(num_units=self.get_config("rnn_hidden_size"))
            _, last_state = tf.nn.dynamic_rnn(cell, net_rnn,
                                              sequence_length=None, dtype=tf.float32)

            # MLP part
            net_mlp = last_state
            for layer in range(self.get_config("n_layers")):
                net_mlp = tf.contrib.layers.fully_connected(
                    inputs=net_mlp,
                    num_outputs=self.get_config("layer_size"),
                    activation_fn=self.get_config("activation"),
                    scope="layer_{:d}".format(layer)
                )
                if self.get_config("batch_norm_baseline"):
                    net_mlp = tf.layers.batch_normalization(net_mlp,
                                                            training=self._is_training_placeholder,
                                                            name="layer_{:d}_batch_norm".format(layer))

            baseline = tf.contrib.layers.fully_connected(
                inputs=net_mlp,
                num_outputs=1,
                activation_fn=None,
                scope="layer_output",
            )
            self._baseline = tf.squeeze(baseline, axis=1)

        with tf.variable_scope("baseline_optimize"):
            self._baseline_target_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,),
                                                               name="baseline_target_placeholder")

            baseline_loss = tf.losses.mean_squared_error(
                labels=self._baseline_target_placeholder,
                predictions=self._baseline,
                scope="baseline_loss",
            )
            self._baseline_loss = baseline_loss   # cache for tensorboard

            lr = self.get_config("learning_rate")
            self._baseline_train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(baseline_loss)

        return self
