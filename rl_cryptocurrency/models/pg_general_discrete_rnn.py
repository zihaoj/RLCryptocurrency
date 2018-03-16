from rl_cryptocurrency.models.pg_general_discrete import PGGeneralDiscrete

import tensorflow as tf


def _convert_last_state(last_state, keep_all=True):
    """
    convert state returned from cell to tensor
    for non-LSTM cell, this is straightforward
    This is mainly for LSTM cell

    keep_all is meant for LSTM cell:
    If True, both c, h would be encoded
    If False, only h would be encoded
    """

    if type(last_state) == tf.nn.rnn_cell.LSTMStateTuple:
        # LSTM case
        if keep_all:
            return tf.concat([last_state.c, last_state.h], axis=1)
        else:
            return last_state.h
    else:
        # non-LSTM case
        return last_state


class PGGeneralDiscreteRNN(PGGeneralDiscrete):
    def _build_policy_network_op(self):
        """
        RNN version
        """

        with tf.variable_scope("policy_network"):
            # RNN part
            net_rnn = self._obs_placeholder
            if self.get_config("rnn_cell") == "GRU":
                cell = tf.nn.rnn_cell.GRUCell(num_units=self.get_config("rnn_hidden_size"))
            elif self.get_config("rnn_cell") == "LSTM":
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.get_config("rnn_hidden_size"))
            else:
                raise NotImplementedError("Unknown rnn_cell!")
            _, last_state = tf.nn.dynamic_rnn(cell, net_rnn,
                                              sequence_length=None, dtype=tf.float32)

            # MLP part
            net_mlp = _convert_last_state(last_state, keep_all=False)
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
            self._logits = logits

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
            net_rnn = self._obs_placeholder
            if self.get_config("rnn_cell") == "GRU":
                cell = tf.nn.rnn_cell.GRUCell(num_units=self.get_config("rnn_hidden_size"))
            elif self.get_config("rnn_cell") == "LSTM":
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.get_config("rnn_hidden_size"))
            else:
                raise NotImplementedError("Unknown rnn_cell!")
            _, last_state = tf.nn.dynamic_rnn(cell, net_rnn,
                                              sequence_length=None, dtype=tf.float32)

            # MLP part
            net_mlp = _convert_last_state(last_state, keep_all=False)
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
