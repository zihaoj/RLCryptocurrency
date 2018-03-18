from rl_cryptocurrency.models.pg_general import PGGeneral

import tensorflow as tf
import numpy as np


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


class PGGeneralDiscreteExperimentLongTerm(PGGeneral):
    """
    what is in obs_placeholder:
    0. adjusted price gap
    1. buffer
    2. bound
    3. price average

    what is in action space (action at exchange-0)
    0: do nothing
    1: all-in (95%), +(buy) if low-side, -(sell) if high-side
    2: all-in (95%), sell if low-side, buy if high-side
    """

    def _add_placeholders_op(self):

        with tf.variable_scope("placeholder"):
            obs_dim = 4

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
        Action is composed of two parts:

        1. if "stuck" condition (buffer is zero and bound is zero) is met, then we either take action 2 (with certain
           probability), or do nothing (0)
        2. if "stuck" condition NOT met, then we proceed as normal between 0 and 1

        When proceed as normal between 0 and 1:
        1. only adjusted price gap is used for RNN
        2. for buffer and bound, only number from current time-stamp is used
        3. average price would not be used
        """

        with tf.variable_scope("policy_network"):

            # stage-1: opposite trading

            with tf.variable_scope("stage_1"):
                adjust_const = (1. - self._fee_exchange) - 1. / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
                adjusted_price_gap_ref = self._obs_placeholder[:, -1, 3] * adjust_const
                adjusted_price_gap = self._obs_placeholder[:, -1, 0]

                # log_w = tf.get_variable("log_w_human", shape=(), dtype=tf.float32,
                #                         initializer=tf.constant_initializer(0.))
                # log_b = tf.get_variable("log_b_human", shape=(), dtype=tf.float32,
                #                         initializer=tf.constant_initializer(0.))

                log_w = tf.get_variable("log_w_human", shape=(), dtype=tf.float32,
                                        initializer=tf.constant_initializer(-0.693))
                log_b = tf.get_variable("log_b_human", shape=(), dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.))

                w = tf.exp(log_w)
                b = tf.exp(log_b)

                self._w_stage_1 = w
                self._b_stage_1 = b

                # 1: opposite trading
                # 0: do nothing
                logits_stage1 = tf.stack([-w * b * adjusted_price_gap_ref, -w * adjusted_price_gap], axis=1,
                                         name="logits_stage1")  # [batch_size, 2]

                sampled_action_stage1 = tf.squeeze(tf.multinomial(logits_stage1, num_samples=1)*2, axis=1,
                                                   name="sampled_action_stage1")

                action_placeholder_stage1 = tf.cast(tf.equal(self._action_placeholder, 2),
                                                    dtype=tf.int32, name="action_placeholder_stage1")

                logprob_stage1 = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action_placeholder_stage1,
                                                                                 logits=logits_stage1,
                                                                                 name="logprob_stage1")

                # # DEBUG
                # self._debug_adjusted_price_gap = adjusted_price_gap
                # self._debug_adjusted_price_gap_ref = adjusted_price_gap_ref
                # self._debug_prob_action2 = tf.nn.softmax(logits_stage1)[:, 1]

            # stage-2: normal trading

            with tf.variable_scope("stage_2"):
                # RNN part
                net_rnn = self._obs_placeholder[:, :, 0:1]
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
                net_mlp = tf.concat([net_mlp, self._obs_placeholder[:, -1, 1:3]], axis=1, name="mlp_input")
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

                logits_stage2 = tf.contrib.layers.fully_connected(
                    inputs=net_mlp,
                    num_outputs=2,
                    activation_fn=None,
                    scope="layer_output",
                )

                sampled_action_stage2 = tf.squeeze(tf.multinomial(logits_stage2, num_samples=1), axis=1,
                                                   name="sampled_action_stage2")

                action_placeholder_stage2 = tf.where(self._action_placeholder > 1,
                                                     tf.zeros(tf.shape(self._action_placeholder), dtype=tf.int32),
                                                     self._action_placeholder,
                                                     name="action_placeholder_stage2")
                logprob_stage2 = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action_placeholder_stage2,
                                                                                 logits=logits_stage2,
                                                                                 name="logprob_stage2")

            # combine stage-1 and stage-2 #

            with tf.variable_scope("combined"):
                # stuck condition
                condition_buffer_zero = tf.less(tf.abs(self._obs_placeholder[:, -1, 1] - 0.), 1e-4,
                                                name="condition_buffer_zero")
                condition_bound_zero = tf.less(tf.abs(self._obs_placeholder[:, -1, 2] - 0.), 1e-4,
                                               name="condition_bound_zero")
                condition_stuck = tf.logical_and(condition_buffer_zero, condition_bound_zero, name="condition_stuck")

                sampled_action_combined = tf.where(condition_stuck, x=sampled_action_stage1, y=sampled_action_stage2,
                                                   name="sampled_action_combined")
                logprob_combined = tf.where(condition_stuck, x=logprob_stage1, y=logprob_stage2,
                                            name="logprob_combined")

                # # DEBUG
                # self._condition_buffer_zero = condition_buffer_zero
                # self._condition_bound_zero = condition_bound_zero
                # self._condition_stuck = condition_stuck

            with tf.variable_scope("policy_sample"):
                self._sampled_action = sampled_action_combined
                self._logprob = logprob_combined

        return self

    def _add_baseline_op(self):
        """
        Use adjusted price gap as input to RNN
        Use current buffer and bound as input to MLP along with last state from RNN
        """

        with tf.variable_scope("baseline_network"):
            # RNN part
            net_rnn = self._obs_placeholder[:, :, 0:1]
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
            net_mlp = tf.concat([net_mlp, self._obs_placeholder[:, -1, 1:3]], axis=1, name="mlp_input")
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

    def _transform_obs(self, obs_env_buffer):

        assert len(obs_env_buffer) == self.get_config("rnn_maxlen"), "ERROR!"

        def get_timestamp_feature(obs_env):
            """
            Obtain the feature from each time-stamp
            """

            obs_portfolio, obs_market, obs_buffer = obs_env
            price_matrix = obs_market[:, :, self._price_index]

            # adjusted price gap

            price_0 = price_matrix[0, 0]
            price_1 = price_matrix[1, 0]
            price_low, price_high = sorted([price_0, price_1])
            price_gap_adjusted = price_high * (1. - self._fee_exchange) - \
                                 price_low / ((1. - self._fee_exchange) * (1. - self._fee_transfer))

            # buffer

            assert obs_buffer.shape == (1,), "Unexpected buffer observation!"
            buffer_to_use = obs_buffer[0]

            # bound

            if price_0 < price_1:
                # 0-buy, 1-sell
                # positive
                price_adjusted = price_0 / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
                bound_to_use = np.min((obs_portfolio[1, 1], obs_portfolio[0, 0] / price_adjusted))
            else:
                # 0-sell, 1-buy
                # negative
                price_adjusted = price_1 / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
                bound_to_use = -np.min((obs_portfolio[0, 1], obs_portfolio[1, 0] / price_adjusted))

            # price average

            price_average = (price_0 + price_1) / 2.

            return [price_gap_adjusted, buffer_to_use, bound_to_use, price_average]

        # shape: [time-stamp, features]
        return np.array(map(get_timestamp_feature, obs_env_buffer))

    def _transform_action(self, action, obs_env):

        # get observation from environment
        obs_portfolio, obs_market, _ = obs_env
        price_matrix = obs_market[:, :, self._price_index]

        if action == 0:
            action_final = 0.
        else:
            if action == 1:
                buy_condition = price_matrix[0, 0] < price_matrix[1, 0]
            else:
                buy_condition = price_matrix[0, 0] > price_matrix[1, 0]

            if buy_condition:
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

    def _add_extra_summary(self):
        super(PGGeneralDiscreteExperiment, self)._add_extra_summary()

        tf.summary.scalar("w_stage_1", self._w_stage_1)
        tf.summary.scalar("b_stage_1", self._b_stage_1)
