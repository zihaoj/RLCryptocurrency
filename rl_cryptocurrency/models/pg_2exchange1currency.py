# A special situation where we can do things rather efficiently
# Here we will only consider 2 exchanges and 1 crypto-currency

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

from gym_rlcrptocurrency.envs import RLCrptocurrencyEnv
from rl_cryptocurrency.utils.general import get_logger, export_plot
from rl_cryptocurrency.models.pg_basics import build_mlp, PG


class PG2Exchange1Currency(PG):
    """
    Wrapper class with models, training and evaluation procedures enclosed.
    """

    def __init__(self, env, env_aux, config, logger=None):
        """
        Constructor
        """

        # initialize through base class
        super(PG2Exchange1Currency, self).__init__(env, env_aux, config, logger)

        # assertion on environment
        assert self._n_exchange == 2, "PG2Exchange1Currency only supports 2 exchanges!"
        assert self._n_currency == 1, "PG2Exchange1Currency only supports 1 crypto-currency!"

    ########################################
    # Methods that build up the operations #
    ########################################

    def _add_placeholders_op(self):
        """
        In this case, our action space can be simplified to just one dimension
        Here we choose it to be the purchase amount on whatever first exchange

        :return: Self, for chaining
        """

        with tf.variable_scope("placeholder"):
            # # Obs here refers to the one that is directly fed into NN
            # # In this implementation, we simply flatten everything into one row
            # obs_dimension = self._n_exchange * (self._n_currency + 1)  # portfolio
            # obs_dimension += self._n_exchange * self._n_currency * self._d_market  # market info
            # obs_dimension += self._n_currency  # buffer
            # self._obs_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, obs_dimension),
            #                                        name="obs_placeholder")
            self._obs_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="obs_placeholder")

            # Action here refers to the one directly sampled out of distribution as parameterized by policy NN
            self._action_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1),
                                                      name="action_placeholder")

            # advantage
            self._advantage_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,), name="advantage_placeholder")

        return self

    def _build_policy_network_op(self):
        """
        The policy network output a value between -1 to 1
        Positive value means "buy-in", negative value means "sell-off"
        The absolute value represents the fraction of maximal feasible transaction amount given current observation

        Notice that the NN direct output is just the mean of continuous distribution. Combined with std, we use
        truncated gaussian distribution to sample actual action, with truncation applied between -1 and 1 as well

        :return: Self, for chaining
        """

        with tf.variable_scope("policy_kernel"):
            # action logits as output of policy network
            action_logits__mean = build_mlp(
                mlp_input=self._obs_placeholder,
                output_size=1,
                scope="policy_network",
                n_layers=self.get_config("n_layers"),
                size=self.get_config("layer_size"),
                hidden_activation=self.get_config("activation"),
                output_activation=tf.nn.sigmoid,
            )
            action_logits__mean = 2 * action_logits__mean - 1

        with tf.variable_scope("policy_sample"):
            # define variable log of std
            # A good initialization is critical in speeding up training ...
            action_logits__logstd = tf.get_variable(name="log_std", shape=(1,), dtype=tf.float32,
                                                    initializer=tf.constant_initializer(-2.5))
            action_logits__std = tf.exp(action_logits__logstd, name="std")
            self._logstd = action_logits__logstd

            # gaussian distribution without truncation
            gauss_kernel_nontruncated = tf.distributions.Normal(loc=action_logits__mean, scale=action_logits__std,
                                                                name="gauss_kernel_nontruncated")

            # get boundary -- will follow the same shape as action_logits__mean
            cdf_low = gauss_kernel_nontruncated.cdf(-0.95, name="cdf_low")
            cdf_high = gauss_kernel_nontruncated.cdf(0.95, name="cdf_high")

            # sampling based on truncated gaussian
            random_seed = tf.random_uniform(shape=[self.get_config("batch_size"), 1],
                                            minval=0., maxval=1., dtype=tf.float32,
                                            name="random_seed")
            random_seed_scaled = random_seed * (cdf_high - cdf_low) + cdf_low

            # invert it -- this is the truncated sample
            truncated_sample = gauss_kernel_nontruncated.quantile(random_seed_scaled, name="truncated_sample")

            # clip it a bit, to avoid numerics overflow
            # TODO: check if this is really needed
            # self._sampled_action = tf.clip_by_value(truncated_sample, clip_value_min=-1., clip_value_max=1.)
            self._sampled_action = truncated_sample

            # on the other hand, compute the probability given the action
            logprob = gauss_kernel_nontruncated.log_prob(self._action_placeholder) - tf.log(cdf_high - cdf_low)
            self._logprob = tf.squeeze(logprob, axis=1)

        return self

    ######################
    # TensorBoard Stuffs #
    ######################

    def _add_summary(self):
        """
        Tensorboard stuff.

        :return Self, for chaining
        """

        # extra placeholders to log stuff from python
        self._avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self._max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self._std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")
        self._eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")
        self._grad_norm_placeholder = tf.placeholder(tf.float32, shape=(), name="grad_norm")
        self._loss_placeholder = tf.placeholder(tf.float32, shape=(), name="loss")

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self._avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self._max_reward_placeholder)
        tf.summary.scalar("Std Reward", self._std_reward_placeholder)
        tf.summary.scalar("Eval Reward", self._eval_reward_placeholder)
        tf.summary.scalar("Grad Norm", self._grad_norm_placeholder)
        tf.summary.scalar("Loss", self._loss_placeholder)

        # Parameters
        tf.summary.scalar("Param Std", tf.reshape(tf.exp(self._logstd), shape=()))

        # logging
        self._merged = tf.summary.merge_all()
        self._file_writer = tf.summary.FileWriter(self.get_config("output_path"), self._sess.graph)

        return self

    #########
    # Utils #
    #########

    def _obs_transformer(self, obs_env):
        """
        Transformation from environment observation to NN input observation

        :param obs_env: The observation numpy array (not tensor!) as provided by environment
        :return: The observation numpy array (not tensor!) to be fed into self._obs_placeholder
        """

        obs_portfolio, obs_market, obs_buffer = obs_env

        # a simplified version: just look at price gap
        price_gap = obs_market[0, 0, self._price_index] - obs_market[1, 0, self._price_index]
        return np.array([price_gap])

        # more complete version: take all available information in!
        # return np.concatenate((obs_portfolio.reshape((-1,)), obs_market.reshape((-1,)), obs_buffer.reshape((-1,))))

    def _action_transformer(self, action, obs_env):
        """
        Transformation from NN output action logits to the actual action acceptable by environment

        The output from NN is the action logits after sampling, which is a fraction number between 0 to 1, representing
        scale between smallest possible transaction amount and largest transaction amount

        :param action: The action numpy array (not tensor!) as returned by NN output
        :param obs_env: The observation numpy array (not tensor!) as current environment state
                        This is to provide constraint on the actual action
        :return: The action numpy array (not tensor!) to be fed into env.step()
        """

        # action space check
        assert action.shape == (1,), "Invalid action shape for class PG2Exchange1Currency"

        # get observation from environment
        obs_portfolio, obs_market, _ = obs_env
        price_matrix = obs_market[:, :, self._price_index]

        # obtain the bound
        if action[0] >= 0:
            price_adjusted = price_matrix[0, 0] / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
            bound = np.min((obs_portfolio[1, 1], obs_portfolio[0, 0] / price_adjusted))
        else:
            price_adjusted = price_matrix[1, 0] / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
            bound = np.min((obs_portfolio[0, 1], obs_portfolio[1, 0] / price_adjusted))

        # get purchase matrix
        action_final = action[0] * bound
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
