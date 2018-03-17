# Almost same as pg_general_discrete_rnn, except we use 3 discrete actions instead of 2

from rl_cryptocurrency.models.pg_general_discrete_rnn import PGGeneralDiscreteRNN, _convert_last_state

import tensorflow as tf
import numpy as np


class PGGeneralDiscrete3RNN(PGGeneralDiscreteRNN):
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
                num_outputs=3,
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

    def _transform_action(self, action, obs_env):
        """
        Action here represents the action one will perform at exchange-0

        3 action space:
        0: do nothing
        1: buy 50% from low-side
        2: buy 95% from low-side
        """

        # validity check
        assert 0 <= action <= 2, "Invalid action {:d}!".format(action)

        # get observation from environment
        obs_portfolio, obs_market, _ = obs_env
        price_matrix = obs_market[:, :, self._price_index]

        # get purchase action
        if action == 0:
            action_final = 0.
        else:
            if price_matrix[0, 0] < price_matrix[1, 0]:
                # 0 is low-side
                price_adjusted = price_matrix[0, 0] / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
                bound = np.min((obs_portfolio[1, 1], obs_portfolio[0, 0] / price_adjusted))
            else:
                # 1 is low-side
                price_adjusted = price_matrix[1, 0] / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
                bound = -np.min((obs_portfolio[0, 1], obs_portfolio[1, 0] / price_adjusted))

            if action == 1:
                action_final = 0.5 * bound
            else:
                action_final = 0.95 * bound

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
