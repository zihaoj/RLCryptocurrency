from rl_cryptocurrency.models.pg_optimal_stop import PGOptimalStop

import tensorflow as tf
import numpy as np


class PGOptimalStopMoreFeatures(PGOptimalStop):
    """
    Here we do feature engineering to expand beyond just price gap between two exchanges
    Notice that we are still using information from current time-stamp only
    """

    def _add_placeholders_op(self):
        """
        A basic version: price gap only
        """

        with tf.variable_scope("placeholder"):
            # # trial 1
            # obs_dim = self._n_exchange * (self._n_currency + 1) + self._n_exchange * self._n_currency * self._d_market

            # # trial 2
            # obs_dim = self._n_exchange * self._n_currency * self._d_market

            # # trial 3
            # obs_dim = 3

            # trial 4
            obs_dim = 1

            # # trial 5
            # obs_dim = 2

            # # trial 6
            # obs_dim = 1

            # # DEBUG
            # obs_dim = 2

            self._obs_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim),
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
        Now we also add other information of current time-stamp
        Notice that no buffer information should be used, since in the training time there will be no buffer at all
        """

        _, obs_market, _ = obs_env_buffer[-1]

        # # trial 1: simply dump all information
        # # FAIL
        # return np.concatenate((obs_portfolio.reshape((-1,)), obs_market.reshape((-1,))))

        # # trial 2: no portfolio, since that is also useless during the training
        # return obs_market.reshape((-1,))

        # trial 3: price gap under different scenario and price average
        # price_matrix = obs_market[:, :, self._price_index]
        # price_gap_1 = price_matrix[1, 0] * (1. - self._fee_exchange) - price_matrix[0, 0] \
        #               / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
        # price_gap_2 = price_matrix[0, 0] * (1. - self._fee_exchange) - price_matrix[1, 0] \
        #               / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
        # price_avg = (price_matrix[0, 0] + price_matrix[1, 0]) / 2.
        # return np.array([price_gap_1, price_gap_2, price_avg])

        # trial 4: adjusted price gap
        price_0 = obs_market[0, 0, self._price_index]
        price_1 = obs_market[1, 0, self._price_index]
        price_low, price_high = min(price_0, price_1), max(price_0, price_1)
        price_gap_adjusted = price_high * (1. - self._fee_exchange) - \
                             price_low / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
        return np.array([price_gap_adjusted])

        # # trial 5: adjusted price gap + average price
        # price_0 = obs_market[0, 0, self._price_index]
        # price_1 = obs_market[1, 0, self._price_index]
        # price_low, price_high = min(price_0, price_1), max(price_0, price_1)
        # price_gap_adjusted = price_high * (1. - self._fee_exchange) - \
        #                      price_low / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
        # price_avg = (price_high + price_low) / 2.
        # return np.array([price_gap_adjusted, price_avg])

        # # trial 6: adjusted price gap / average price
        # price_0 = obs_market[0, 0, self._price_index]
        # price_1 = obs_market[1, 0, self._price_index]
        # price_low, price_high = min(price_0, price_1), max(price_0, price_1)
        # price_gap_adjusted = price_high * (1. - self._fee_exchange) - \
        #                      price_low / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
        # price_avg = (price_high + price_low) / 2.
        # return np.array([100.0 * price_gap_adjusted / price_avg])

        # # DEBUG
        # price_0 = obs_market[0, 0, self._price_index]
        # price_1 = obs_market[1, 0, self._price_index]
        # price_low, price_high = min(price_0, price_1), max(price_0, price_1)
        # price_gap_adjusted = price_high * (1. - self._fee_exchange) - \
        #                      price_low / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
        # return np.array([price_gap_adjusted, (price_low + price_high) / 2.])



