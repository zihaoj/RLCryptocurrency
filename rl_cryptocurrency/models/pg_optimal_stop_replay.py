from rl_cryptocurrency.models.pg_optimal_stop import PGOptimalStop

import tensorflow as tf
import numpy as np


class PGOptimalStopReplay(PGOptimalStop):
    """
    This class is to utilize previous time window, but with MLP
    """

    def _add_placeholders_op(self):
        """
        Sequence of adjusted price gap
        """

        with tf.variable_scope("placeholder"):
            obs_dim = 1

            self._obs_placeholder = tf.placeholder(dtype=tf.float32,
                                                   shape=(None, self.get_config("rnn_maxlen"), obs_dim),
                                                   name="obs_placeholder")

            self._action_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,),
                                                      name="action_placeholder")

            self._advantage_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,),
                                                         name="advantage_placeholder")

            self._is_training_placeholder = tf.placeholder(tf.bool, shape=(), name="is_training_placeholder")

            if self.get_config("use_return"):
                self._return_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,), name="return_placeholder")

        return self

    def _transform_obs(self, obs_env_buffer):
        """
        Sequence of adjusted price gap
        """

        assert len(obs_env_buffer) == self.get_config("rnn_maxlen"), "ERROR!"

        def get_timestamp_feature(obs_env):
            """
            Obtain the feature from each time-stamp
            """

            _, obs_market, _ = obs_env

            price_0 = obs_market[0, 0, self._price_index]
            price_1 = obs_market[1, 0, self._price_index]
            price_low, price_high = min(price_0, price_1), max(price_0, price_1)
            price_gap_adjusted = price_high * (1. - self._fee_exchange) - \
                                 price_low / ((1. - self._fee_exchange) * (1. - self._fee_transfer))

            return [price_gap_adjusted]

        # shape: [time-stamp, features]
        return np.array(map(get_timestamp_feature, obs_env_buffer))

    def _build_policy_network_op(self):
        """
        MLP across the whole window
        """

        with tf.variable_scope("policy_network"):
            net_mlp = tf.layers.flatten(self._obs_placeholder)
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
        MLP across whole window
        """

        with tf.variable_scope("baseline_network"):
            net_mlp = tf.layers.flatten(self._obs_placeholder)
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
