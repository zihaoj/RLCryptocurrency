# Consider discrete action space

from rl_cryptocurrency.models.pg_general import PGGeneral

import tensorflow as tf
import numpy as np


class PGGeneralDiscrete(PGGeneral):
    def _add_placeholders_op(self):
        """
        General setup for discrete action space
        Two observation considered this moment: price gap and buffer
        """

        with tf.variable_scope("placeholder"):
            obs_dim = 2

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

    def _build_policy_network_op(self):
        """
        A general discrete action space (in MLP setup)
        Stuff are hard-coded for now

        2 action space (5 has been tried before and it fails ...)
        0: do nothing
        1: all-in (95% actually)
        """

        with tf.variable_scope("policy_network"):
            net = tf.layers.flatten(self._obs_placeholder)
            for layer in range(self.get_config("n_layers")):
                net = tf.contrib.layers.fully_connected(
                    inputs=net,
                    num_outputs=self.get_config("layer_size"),
                    activation_fn=self.get_config("activation"),
                    scope="layer_{:d}".format(layer)
                )
                if self.get_config("batch_norm_policy"):
                    net = tf.layers.batch_normalization(net, training=self._is_training_placeholder,
                                                        name="layer_{:d}_batch_norm".format(layer))

            logits = tf.contrib.layers.fully_connected(
                inputs=net,
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
        Almost same as policy network
        """

        with tf.variable_scope("baseline_network"):
            net = tf.layers.flatten(self._obs_placeholder)
            for layer in range(self.get_config("n_layers")):
                net = tf.contrib.layers.fully_connected(
                    inputs=net,
                    num_outputs=self.get_config("layer_size"),
                    activation_fn=self.get_config("activation"),
                    scope="layer_{:d}".format(layer)
                )
                if self.get_config("batch_norm_baseline"):
                    net = tf.layers.batch_normalization(net, training=self._is_training_placeholder,
                                                        name="layer_{:d}_batch_norm".format(layer))

            baseline = tf.contrib.layers.fully_connected(
                inputs=net,
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

    def _transform_obs(self, obs_env_buffer):

        assert len(obs_env_buffer) == self.get_config("rnn_maxlen"), "ERROR!"

        def get_timestamp_feature(obs_env):
            """
            Obtain the feature from each time-stamp
            """

            _, obs_market, obs_buffer = obs_env

            price_0 = obs_market[0, 0, self._price_index]
            price_1 = obs_market[1, 0, self._price_index]
            price_low, price_high = min(price_0, price_1), max(price_0, price_1)
            price_gap_adjusted = price_high * (1. - self._fee_exchange) - \
                                 price_low / ((1. - self._fee_exchange) * (1. - self._fee_transfer))

            # DEBUG
            assert obs_buffer.shape == (1,), "Unexpected buffer observation!"

            return [price_gap_adjusted, obs_buffer[0]]

        # shape: [time-stamp, features]
        return np.array(map(get_timestamp_feature, obs_env_buffer))

    def _transform_action(self, action, obs_env):
        """
        Action here represents the action one will perform at exchange-0

        2 action space:
        0: do nothing
        1: all-in (95%)
        """

        # get observation from environment
        obs_portfolio, obs_market, _ = obs_env
        price_matrix = obs_market[:, :, self._price_index]

        if action == 0:
            action_final = 0.
        else:
            if price_matrix[0, 0] < price_matrix[1, 0]:
                price_adjusted = price_matrix[0, 0] / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
                bound = np.min((obs_portfolio[1, 1], obs_portfolio[0, 0] / price_adjusted))
                action_final = 0.95 * bound
            else:
                price_adjusted = price_matrix[1, 0] / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
                bound = np.min((obs_portfolio[0, 1], obs_portfolio[1, 0] / price_adjusted))
                action_final = -0.95 * bound

        purchase_matrix = np.array([[action_final], [-action_final]])

        # next, we derive the transfer matrix based on purchase matrix
        # The idea is we immediately balance accounts across all exchanges
        transfer_matrix = np.zeros(shape=(self._n_exchange, self._n_exchange, self._n_currency), dtype=np.float32)

        # get index of element in purchase_matrix
        def _get_index(condition):
            x_list, y_list = np.where(condition)
            return zip(x_list, y_list)

        indices_buy = _get_index(purchase_matrix > 0)
        indices_sell = _get_index(purchase_matrix < 0)

        for currency in range(self._n_currency):
            queue_buy = map(lambda xy: [xy[0], purchase_matrix[xy[0], xy[1]]],
                            filter(lambda xy: xy[1] == currency, indices_buy))
            queue_sell = map(lambda xy: [xy[0], -purchase_matrix[xy[0], xy[1]]],
                             filter(lambda xy: xy[1] == currency, indices_sell))

            while len(queue_buy) > 0 and len(queue_sell) > 0:
                exchange_from = queue_buy[0][0]
                exchange_to = queue_sell[0][0]

                if queue_buy[0][1] > queue_sell[0][1]:
                    amount = queue_sell[0][1]
                    queue_buy[0][1] -= amount
                    queue_sell.pop(0)
                else:
                    amount = queue_buy[0][1]
                    queue_buy.pop(0)
                    queue_sell[0][1] -= amount
                    if np.isclose(queue_sell[0][1], 0.):
                        queue_sell.pop(0)

                transfer_matrix[exchange_from, exchange_to, currency] += amount

        # done. return
        return purchase_matrix, transfer_matrix
