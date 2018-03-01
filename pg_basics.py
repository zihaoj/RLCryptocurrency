# First version of policy gradients
# This basically implements REINFORCE algorithm, using MC sampling, with baseline term included
# Action space constrained to crypto-currency purchase only. After each purchase, one would transfer the crypto-currency
# from buy sides to sell sides immediately for balancing.
# Code adapted from CS234 2018 winter quarter assignment 3

import os
import tensorflow as tf
import numpy as np
import pandas as pd

from gym_rlcrptocurrency.envs import RLCrptocurrencyEnv

from config import config
from utils.general import get_logger, export_plot


def build_mlp(mlp_input, output_size, scope, n_layers, size, hidden_activation, output_activation=None):
    """
    A generic function to build a simple feed forward network

    :param mlp_input: Input tensor
    :param output_size: Size of output
    :param scope: Scope for this network
    :param n_layers: Number of hidden layers
    :param size: Size of hidden units on each layer
    :param hidden_activation: Activation for hidden layers
    :param output_activation: Activation for output layer. None means linear transformation only.
    :return: Output tensor from NN
    """

    with tf.variable_scope(scope):
        net = mlp_input
        for i_layer in range(n_layers):
            net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=size, activation_fn=hidden_activation,
                                                    scope="layer_{:d}".format(i_layer))
        net = tf.contrib.layers.fully_connected(inputs=net, num_outputs=output_size, activation_fn=output_activation,
                                                scope="output")

    return net


class PG(object):
    """
    Wrapper class with models, training and evaluation procedures enclosed.
    """

    def __init__(self, env, config, logger=None):
        """
        Initialize Policy Gradient Class

        :param env: OpenAI environment. Have to be RLCrptocurrencyEnv currently
        :param config: Configuration with all necessary hyper-parameters
        :param logger: Logging instance
        """

        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # cache provided objects
        self._config = config
        self._logger = logger
        if logger is None:
            self._logger = get_logger(config.log_path)
        self._env = env

        # check environment
        assert isinstance(self._env, RLCrptocurrencyEnv), "Only RLCrptocurrencyEnv supported for now!"

        # short cuts to environment setup
        self._n_exchange = self._env.n_exchange
        self._n_currency = self._env.n_currency
        self._d_market = len(self._env.market_obs_attributes)
        self._price_index = self._env.market_obs_attributes.index("Weighted_Price")
        self._fee_exchange = self._env.fee_exchange
        self._fee_transfer = self._env.fee_transfer

        # internal states
        self._sess = None
        self._built = False

    ###########
    # Getters #
    ###########

    def get_config(self, name):
        """
        Uniform interface of accessing hyper-parameter. If requested key is not available, error should be thrown

        :param name: Attribute key
        :return: Attribute value
        """

        return getattr(self._config, name)

    #####################
    # Public Interfaces #
    #####################

    def build(self):
        """
        Build model by adding all necessary variables

        :return Self, for chaining
        """

        if self._built:
            self._logger.warning("Model already built before!")

        # add placeholders
        self._add_placeholders_op()
        # create policy net
        self._build_policy_network_op()
        # add policy gradient loss
        self._add_loss_op()
        # add optimizer for the main networks
        self._add_optimizer_op()

        if self.get_config("use_baseline"):
            self._add_baseline_op()

        self._built = True

        return self

    def initialize(self):
        """
        Assumes the graph has been constructed (have called self.build())
        Creates a tf Session and run initializer of variables

        :return Self, for chaining
        """

        assert self._built, "Please run build() prior to initialize()!"

        # create tf session
        self._sess = tf.Session()

        # tensorboard stuff
        self._add_summary()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self._sess.run(init)

        return self

    def sample_path(self, env, init_portfolio, start_date, num_episodes=None):
        """
        Sample path with current policy network

        :param env: environment object
        :param num_episodes: Number of episodes. If None, then keep samping until batch_size is reached
        :param init_portfolio: Initial portfolio
        :param start_date: The starting date of path sampling, in string format "yyyy-mm-dd"
                           Notice that each episode is at most one day. After one episode is finished, we would jump
                           to the start of next day in next episode
        :return: tuple of (paths, total_rewards):
                 * paths: List of path, each of which is a dictionary with:
                     * path["observations"] List of observations as returned by environment
                     * path["actions"] List of actions as output by policy network
                     * path["rewards"] List of rewards along the path
                 * total_rewards: List of total rewards, one for each path
        """

        episode = 0
        t = 0

        episode_rewards = []
        paths = []

        # loop over episodes
        while num_episodes or t < self.get_config("batch_size"):
            # update initial time to next day
            init_time = pd.to_datetime(start_date) + pd.Timedelta(episode, unit="d")

            # initialization for current episode
            obs, _, _, _ = env.init(init_portfolio, init_time)
            observations, actions, rewards = [], [], []
            episode_reward = 0

            # loop over time-stamps
            for step in range(self.get_config("max_ep_len")):
                # note current obs
                observations.append(obs)

                # run policy
                action = self._sess.run(
                    self._sampled_action,
                    feed_dict={self._obs_placeholder: self._obs_transformer(observations[-1])[None]}
                )[0]

                # run environment to get next step
                obs, reward, done, info = env.step(self._action_transformer(action, obs))
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1

                # episode termination condition
                if done or step == self.get_config("max_ep_len") - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.get_config("batch_size"):
                    break

            # summarize on episode
            path = {
                "observation": observations,
                "actions": actions,
                "rewards": rewards,
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    def train(self, init_portfolio, start_date):
        """
        Performs training

        :param init_portfolio: Initial portfolio
        :param start_date: Starting date
        :return Self, for chaining
        """

        last_eval = 0
        last_record = 0

        self._init_averages()
        scores_eval = []  # list of scores computed at iteration time

        for t in range(self.get_config("num_batches")):
            # Initialization for each batch
            # Notice that we shift by one day forward in each batch
            init_portfolio_batch = np.copy(init_portfolio)
            start_date_bath = pd.to_datetime(start_date) + pd.Timedelta(1, unit="d")

            # sample paths for current batch
            paths, total_rewards = self.sample_path(self._env, init_portfolio_batch, start_date_bath)

            # update evaluation scores
            scores_eval += total_rewards

            # concatenate all episodes along time-stamp dimension
            observations = reduce(lambda x, y: x+y, map(lambda path: path["observations"], paths))
            actions = reduce(lambda x, y: x+y, map(lambda path: path["actions"], paths))
            rewards = reduce(lambda x, y: x+y, map(lambda path: path["rewards"], paths))

            # get advantages
            returns = self._get_returns(paths)
            advantages = self._calculate_advantage(returns, observations)

            # update baseline network, if applicable
            if self.get_config("use_baeline"):
                self._update_baseline(returns, observations)

            # update policy network
            self._sess.run(
                self._train_op,
                feed_dict={
                    self._obs_placeholder: np.array(map(self._obs_transformer, observations)),
                    self._action_placeholder: np.array(actions),
                    self._advantage_placeholder: advantages,
                }
            )

            # tf stuff
            if t % self.get_config("summary_freq") == 0:
                self._update_averages(total_rewards, scores_eval)
                self._record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self._logger.info(msg)

            if self.get_config("record") and (last_record > self.get_config("record_freq")):

                raise NotImplementedError("No recording available!")

                # self._logger.info("Recording...")
                # last_record = 0
                # self._record()

        self._logger.info("- Training done.")
        export_plot(scores_eval, "Score", config.env_name, self.get_config("plot_output"))

    def evaluate(self, env, num_episodes, **init):
        """
        Evaluates the return for num_episodes episodes.

        :param env: An external environment
        :param num_episodes: Number of episodes to run
        :param init: Initialization parameters ins sample_path()
        :return Average rewards
        """

        paths, rewards = self.sample_path(env, num_episodes=num_episodes, **init)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self._logger.info(msg)

        return avg_reward

    ########################################
    # Methods that build up the operations #
    ########################################

    def _add_placeholders_op(self):
        """
        Define placeholders needed for calculating the loss
        Remember loss is basically the product of log policy probability with advantage, evaluated on paths generated by
        current policy. However, In policy gradient, gradient is only with respect to the policy term. Therefore,
        advantage should be provided as a placeholder, instead of operations from policy network. Similarly, observation
        and actions along the path should also be provided as placeholder since we only want a partial derivative on
        policy network wrt parameters instead of full derivative.

        :return: Self, for chaining
        """

        with tf.variable_scope("placeholder"):
            # Obs here refers to the one that is directly fed into NN
            # In this implementation, we simply flatten everything into one row
            obs_dimension = self._n_exchange * (self._n_currency + 1)  # portfolio
            obs_dimension += self._n_exchange * self._n_currency * self._d_market  # market info
            obs_dimension += self._n_currency  # buffer
            self._obs_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, obs_dimension),
                                                   name="obs_placeholder")

            # Action here refers to the one that is direct output of NN
            # a reduced parameterization of action space
            action_dimension = self._n_exchange * (self._n_currency + 1)
            self._action_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, action_dimension),
                                                      name="action_placeholder")

            # advantage
            self._advantage_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,), name="advantage_placeholder")

        return self

    def _build_policy_network_op(self):
        """
        Build the policy network, construct the tensorflow operation to sample
        actions from the policy network outputs, and compute the log probabilities
        of the taken actions (for computing the loss later).
        These operations are stored in self._sampled_action and self._logprob.

        :return: Self, for chaining
        """

        # action logits as output of policy network
        action_dimension = self._action_placeholder.get_shape().as_list()[1]
        action_logits__mean = build_mlp(
            mlp_input=self._obs_placeholder,
            output_size=action_dimension,
            scope="policy_network",
            n_layers=self.get_config("n_layers"),
            size=self.get_config("layer_size"),
            hidden_activation=self.get_config("activation"),
            output_activation=None,
        )

        with tf.variable_scope("policy_sample"):
            with tf.variable_scope("policy_normalize"):
                # normalize action logits
                # specifically, we do one separate normalization per exchange
                action_logits__mean_packed = tf.reshape(action_logits__mean, shape=(-1, self._n_exchange, self._n_currency+1))
                action_logits__mean_packed__mean, action_logits__mean_packed__var = tf.nn.moments(
                    action_logits__mean_packed,
                    axes=-1,
                    keep_dims=True,
                )
                action_logits__mean_packed = action_logits__mean_packed - action_logits__mean_packed__mean
                action_logits__mean_packed = action_logits__mean_packed / tf.sqrt(action_logits__mean_packed__var)
                # unpack back to original shape
                action_logits__mean = tf.layers.flatten(action_logits__mean_packed)

            # sample from it
            action_logits__logstd = tf.get_variable(name="log_std", shape=(action_dimension,), dtype=tf.float32)
            self._sampled_action = action_logits__mean + \
                                   tf.exp(action_logits__logstd) * tf.random_normal(shape=tf.shape(action_logits__mean))

            # compute log probability
            helper = tf.contrib.distributions.MultivariateNormalDiag(
                loc=action_logits__mean,
                scale_diag=tf.exp(action_logits__logstd),
            )
            self._logprob = helper.log_prob(self._action_placeholder)

        return self

    def _add_loss_op(self):
        """
        Sets the loss of a batch, the loss is a scalar
        Recall the update for REINFORCE with advantage:
        θ = θ + α ∇_θ log π_θ(s_t, a_t) A_t

        Think about how to express this update as minimizing a
        loss (so that tensorflow will do the gradient computations
        for you).

        Set the loss to self._loss

        :return: Self, for chaining
        """

        with tf.variable_scope("policy_loss"):
            self._loss = -tf.reduce_mean(self._logprob * self._advantage_placeholder)

        return self

    def _add_optimizer_op(self):
        """
        Add optimizer given the loss
        Set to self._train_op

        :return: Self, for chaining
        """

        with tf.variable_scope("policy_optimize"):
            self._train_op = tf.train.AdamOptimizer(learning_rate=self.get_config("learning_rate")).minimize(self._loss)

        return self

    def _add_baseline_op(self):
        """
        This will do all the baseline jobs, including
        1. Define target placeholder
        2. Obtain output of baseline network
        3. Minimize baseline loss

        :return: Self, for chaining
        """

        self._baseline = tf.reshape(build_mlp(
            mlp_input=self._obs_placeholder,
            output_size=1,
            scope="baseline_network",
            n_layers=self.get_config("n_layers"),
            size=self.get_config("layer_size"),
            hidden_activation=self.get_config("activation"),
            output_activation=None,
        ), shape=(-1,))

        with tf.variable_scope("baseline_optimize"):
            self._baseline_target_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,),
                                                               name="baseline_target_placeholder")

            baseline_loss = tf.losses.mean_squared_error(
                labels=self._baseline_target_placeholder,
                predictions=self._baseline,
                scope="baseline_loss",
            )

            self._baseline_train_op = tf.train.AdamOptimizer(learning_rate=self.get_config("learning_rate"))\
                .minimize(baseline_loss)

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

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self._avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self._max_reward_placeholder)
        tf.summary.scalar("Std Reward", self._std_reward_placeholder)
        tf.summary.scalar("Eval Reward", self._eval_reward_placeholder)

        # logging
        self._merged = tf.summary.merge_all()
        self._file_writer = tf.summary.FileWriter(self.get_config("output_path"), self._sess.graph)

        return self

    def _init_averages(self):
        """
        Defines extra attributes for tensorboard.

        :return Self, for chaining
        """

        self._avg_reward = 0.
        self._max_reward = 0.
        self._std_reward = 0.
        self._eval_reward = 0.

        return self

    def _update_averages(self, rewards, scores_eval):
        """
        Update the averages.

        :param rewards: deque
        :param scores_eval: list
        :return: Self, for chaining
        """

        self._avg_reward = np.mean(rewards)
        self._max_reward = np.max(rewards)
        self._std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self._eval_reward = scores_eval[-1]

    def _record_summary(self, t):
        """
        Add summary to tfboard
        :return: Self, for chaining
        """

        fd = {
            self._avg_reward_placeholder: self._avg_reward,
            self._max_reward_placeholder: self._max_reward,
            self._std_reward_placeholder: self._std_reward,
            self._eval_reward_placeholder: self._eval_reward,
        }
        summary = self._sess.run(self._merged, feed_dict=fd)
        # tensorboard stuff
        self._file_writer.add_summary(summary, t)

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

        return np.concatenate((obs_portfolio.reshape((-1,)), obs_market.reshape((-1,)), obs_buffer.reshape((-1,))))

    def _obs_transformer_inverse(self, obs):
        """
        Inverse of _obs_transformer

        :param obs: Observation numpy array to be fed into policy network
        :return: Observation numpy array as provided by environment
        """

        obs = np.copy(obs)

        obs_portfolio = obs[:self._n_exchange * (self._n_currency + 1)]
        obs_market = obs[self._n_exchange * (self._n_currency + 1):self._n_exchange * (self._n_currency + 1)+self._n_exchange * self._n_currency * self._d_market]
        obs_buffer = obs[self._n_exchange * (self._n_currency + 1)+self._n_exchange * self._n_currency * self._d_market:self._n_exchange * (self._n_currency + 1)+self._n_exchange * self._n_currency * self._d_market+self._n_currency]

        # we specify all dimensions explicitly
        # so that error would be thrown out if input is incompatible
        obs_portfolio = obs_portfolio.reshape((self._n_exchange, self._n_currency + 1))
        obs_market = obs_market.reshape((self._n_exchange, self._n_currency, self._d_market))
        obs_buffer = obs_buffer.reshape((self._n_currency,))

        return obs_portfolio, obs_market, obs_buffer

    def _action_transformer(self, action, obs_env):
        """
        Transformation from NN output action logits to the actual action acceptable by environment

        :param action: The action numpy array (not tensor!) as returned by NN output
        :param obs_env: The observation numpy array (not tensor!) as current environment state
                        This is to provide constraint on the actual action
        :return: The action numpy array (not tensor!) to be fed into env.step()
        """

        # convert from array to matrix
        # row representing exchanges, column representing currencies
        action_matrix = action.reshape((-1, self._n_currency+1))
        assert action_matrix.shape[0] == self._n_exchange, "How do you turn this on?!"

        # constraint each row by softmax
        action_shifted = action_matrix - np.max(action_matrix, axis=1, keepdims=True)
        action_exp = np.exp(action_shifted)
        action_softmax = action_exp / np.sum(action_exp, axis=1, keepdims=True)
        action_softmax = action_softmax[:, :-1]  # last column corresponds to dummy slack variable

        # obtain actual action, given price matrix and cash array
        def kernel(portfolio_matrix, price_matrix, logits_matrix):
            """
            :param portfolio_matrix: (n_exchange, n_currency+1). First column is the available cash.
            :param price_matrix: (n_exchange, n_currency)
            :param logits_matrix: (n_exchange, n_currency). Should already be bounded between 0 and 1
            :return: (n_exchange, n_currency)
            """

            cash_matrix = portfolio_matrix[:, :1]
            currency_matrix = portfolio_matrix[:, 1:]

            t_matrix = cash_matrix * logits_matrix / price_matrix
            b_matrix = t_matrix + currency_matrix

            alpha = 1.0 * np.sum(currency_matrix, axis=0) / np.sum(b_matrix, axis=0)
            alpha_matrix = alpha.reshape((1, -1))

            bprime_matrix = alpha_matrix * b_matrix
            xfinal_matrix = bprime_matrix - currency_matrix

            return xfinal_matrix

        # theoretically, we need to adjust the price according to whether we are buying or selling
        # however, we are unable to get the purchase matrix unless we know the price matrix
        # what a chicken & egg problem ...
        # so here is the heuristics
        # given the fee is small, we assume the sign of purchase matrix is insensitive to fee
        # so we first obtain purchase matrix as if there is no fee, and set the sign based on that
        # then we adjust the price matrix according to sign and compute purchase matrix again

        obs_portfolio, obs_market, _ = obs_env
        price_matrix = obs_market[:, :, self._price_index]

        purchase_matrix_nofee = kernel(obs_portfolio, price_matrix, action_softmax)
        mask_buy = purchase_matrix_nofee > 0
        mask_sell = purchase_matrix_nofee < 0

        price_matrix_adjusted = np.copy(price_matrix)
        price_matrix_adjusted[mask_buy] /= ((1.0 - self._fee_exchange) * (1.0 - self._fee_transfer))
        price_matrix_adjusted[mask_sell] *= (1.0 - self._fee_exchange)

        purchase_matrix = kernel(obs_portfolio, price_matrix_adjusted, action_softmax)

        # TODO: let's hope this would work ...
        # Otherwise a iteration would be needed here ...
        assert np.all((purchase_matrix > 0) == mask_buy), "Incompatible buy!"
        assert np.all((purchase_matrix < 0) == mask_sell), "Incompatible sell!"

        # next, we derive the transfer matrix based on purchase matrix
        # The idea is we immediately balance accounts across all exchanges
        transfer_matrix = np.zeros(shape=(self._n_exchange, self._n_exchange, self._n_currency), dtype=np.float32)

        indices_buy = map(lambda xy: tuple(xy), np.where(purchase_matrix > 0))
        indices_sell = map(lambda xy: tuple(xy), np.where(purchase_matrix < 0))

        for currency in range(self._n_currency):
            queue_buy = map(lambda xy: (xy[0], purchase_matrix[xy[0], xy[1]]),
                            filter(lambda xy: xy[1] == currency, indices_buy))
            queue_sell = map(lambda xy: (xy[0], -purchase_matrix[xy[0], xy[1]]),
                             filter(lambda xy: xy[1] == currency, indices_sell))

            while len(queue_buy) > 0 and len(queue_sell) > 0:
                exchange_from = queue_buy[0][0]
                exchange_to = queue_buy[0][0]

                if queue_buy[0][1] > queue_sell[0][1]:
                    amount = queue_sell[0][1]
                    queue_buy[0][1] -= amount
                    queue_sell.pop(0)
                else:
                    amount = queue_buy[0][1]
                    queue_buy.pop(0)
                    queue_sell[0][1] -= amount
                    if queue_sell[0][1] == 0:
                        queue_sell.pop(0)

                transfer_matrix[exchange_from, exchange_to, currency] += amount

        # done. return
        return purchase_matrix, transfer_matrix

    def _get_returns(self, paths):
        """
        Calculate the returns G_t for each timestep

        :param paths: The path as returned by sample_path()
        :return 1-D numpy array of all returns
        """

        all_returns = []
        for path in paths:
            rewards = path["rewards"]
            returns_reversed = [0.]
            for r in reversed(rewards):
                last_return = returns_reversed[-1]
                returns_reversed.append(r + self.get_config("gamma") * last_return)
            returns = returns_reversed[:0:-1]
            all_returns.append(returns)

        returns = np.concatenate(all_returns)
        return returns

    def _calculate_advantage(self, returns, observations):
        """
        Calculate the advantage using current baseline network

        :param returns: 1-D numpy array of returns for each time-step
        :param observations: List of observations for each time-step
        :return 1-D numpy array of advantages for each time-step
        """

        adv = returns

        if self.get_config("use_baseline"):
            baselines = self._sess.run(
                self._baseline,
                feed_dict={
                    self._obs_placeholder: np.array(map(self._obs_transformer, observations)),
                }
            )
            adv = returns - baselines

        if self.get_config("normalize_advantage"):
            mean = np.mean(adv)
            std = np.std(adv)
            adv = (adv - mean) / std

        return adv

    def _update_baseline(self, returns, observations):
        """
        Update the baseline network

        :param returns: 1-D numpy array of returns for each time-step
        :param observations: List of observations for each time-step
        :return Self, for chaining
        """

        self._sess.run(
            self._baseline_train_op,
            feed_dict={
                self._obs_placeholder: np.array(map(self._obs_transformer, observations)),
                self._baseline_target_placeholder: returns,
            }
        )

        return self





