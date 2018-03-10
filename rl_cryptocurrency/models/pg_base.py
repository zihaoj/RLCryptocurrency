# Skeleton of policy gradients
# This is REINFORCE algorithm, with advantage implemented
# Adapted from CS234 2018 winter quarter assignment 3


import os
import tensorflow as tf
import numpy as np

from gym_rlcrptocurrency.envs import RLCrptocurrencyEnv
from rl_cryptocurrency.utils.general import get_logger


class PGBase(object):
    """
    Wrapper class with models, training and evaluation procedures enclosed.
    """

    def __init__(self, env, env_aux, config, logger=None):
        """
        Initialize Policy Gradient Class

        :param env: OpenAI environment. Have to be RLCrptocurrencyEnv currently
        :param env_aux: Auxiliary environment. For example, this can be the inverse "env" such that agent will not
                        be fooled by the situation when price at one exchange is always higher than another
                        But in general, this can be any perturbation of default environment for improving agent
                        robustness. It can be None, in case one does not want it.
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

        # auxiliary environment
        self._env_aux = env_aux
        if self._env_aux is not None:
            self._logger.info("You have added an auxiliary environment. No check is performed on it, "
                              "but you must make sure it is consistent with default environment!")

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

    def sample_path(self, env, init_portfolio, start_date, **options):
        """
        Sample path with current policy network with initialization as specified in input

        :param env: environment object
        :param init_portfolio: Initial portfolio
        :param start_date: Starting date. Acceptable input includes:
                           1. String. In this case, environment will be initialized explicitly on start_date.
                                      Notice that initialization is slow in this way
                           2. Int. In this case, environment will be reset to whenever it was explicitly initialized
                                   last time, and then moved forward by this amount of time-stamp
        :param options: Any other options


        :return: tuple of (paths, total_rewards):
                 * paths: List of episode, each of which is a dictionary with:
                     * path["observations"] List of observations at each time-step
                     * path["actions"] List of actions at each time-step
                     * path["rewards"] List of rewards at each time-step
                 * total_rewards: List of total rewards, one for each path
        """

        raise NotImplementedError("Please implement this in your sub-class!")

    def train(self, init_portfolio, start_date, **options):
        """
        Performs training

        :param init_portfolio: Initial portfolio
        :param start_date: Starting date
        :param options: Any other options
        :return Self, for chaining
        """

        raise NotImplementedError("Please implement this in your sub-class!")

    def evaluate(self, env, n):
        """
        Perform evaluation

        :param env: Enviornment that has been well initialized
        :param n: Number of time-steps to evaluate
        :return Evaluation result
        """

        raise NotImplementedError("Please implement this in your sub-class")

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

        raise NotImplementedError("Please implement this in your sub-class!")

    def _build_policy_network_op(self):
        """
        Build the policy network, construct the tensorflow operation to sample
        actions from the policy network outputs, and compute the log probabilities
        of the taken actions (for computing the loss later).
        These operations are stored in self._sampled_action and self._logprob.

        :return: Self, for chaining
        """

        raise NotImplementedError("Please implement this in your sub-class!")

    def _add_loss_op(self):
        """
        Sets the loss of a batch. The loss is a scalar

        Expecting self._logprob and self._advantage_placeholder available in the class
        Set self._loss

        :return: Self, for chaining
        """

        with tf.variable_scope("policy_loss"):
            self._loss = -tf.reduce_mean(self._logprob * self._advantage_placeholder)

        return self

    def _add_optimizer_op(self):
        """
        Add optimizer given the loss

        Expecting self._loss
        set self._grad_norm, self._train_op

        :return: Self, for chaining
        """

        with tf.variable_scope("policy_optimize"):
            lr = self.get_config("learning_rate")
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)

            list_grad_and_var = optimizer.compute_gradients(self._loss)
            list_grad, list_var = zip(*list_grad_and_var)

            self._grad_norm = tf.global_norm(list_grad)
            self._train_op = optimizer.apply_gradients(list_grad_and_var)

        return self

    def _add_baseline_op(self):
        """
        This will do all the baseline jobs, including
        1. Define target placeholder
        2. Obtain output of baseline network
        3. Minimize baseline loss

        :return: Self, for chaining
        """

        raise NotImplementedError("Please implement this in your sub-class")

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

        # any extra summary that does not depends on inputs (typically parameters)
        self._add_extra_summary()

        # logging
        self._merged = tf.summary.merge_all()
        self._file_writer = tf.summary.FileWriter(self.get_config("output_path"), self._sess.graph)

        return self

    def _add_extra_summary(self):
        """
        Add any additional summary information to be presented on Tensorboard
        Notice that only variables that do not depend on placeholder inputs can be added here. Typically you might
        want to put parameters or hyper-parameters here

        :return: Self, fo chaining
        """

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

    def _record_summary(self, t, grad_norm, loss, fd_extra={}):
        """
        Add summary to tfboard
        :return: Self, for chaining
        """

        fd = {
            self._avg_reward_placeholder: self._avg_reward,
            self._max_reward_placeholder: self._max_reward,
            self._std_reward_placeholder: self._std_reward,
            self._eval_reward_placeholder: self._eval_reward,
            self._grad_norm_placeholder: grad_norm,
            self._loss_placeholder: loss,
        }
        for key, value in fd_extra.items():
            fd[key] = value

        summary = self._sess.run(self._merged, feed_dict=fd)
        # tensorboard stuff
        self._file_writer.add_summary(summary, t)

        return self

    #########
    # Utils #
    #########

    def _transform_obs(self, obs_env):
        """
        Transformation from environment observation to NN input observation
        This is where feature engineering happens

        :param obs_env: The observation numpy array (not tensor!) as provided by environment
        :return: The observation numpy array (not tensor!) to be fed into self._obs_placeholder
        """

        raise NotImplementedError("Please implement this in your sub-class!")

    def _transform_action(self, action, obs_env):
        """
        Transformation from NN output action logits to the actual action acceptable by environment

        :param action: The action numpy array (not tensor!) as returned by NN output
        :param obs_env: The observation numpy array (not tensor!) as current environment state
                        This is to provide constraint on the actual action
        :return: The action numpy array (not tensor!) to be fed into env.step()
        """

        raise NotImplementedError("Please implement this in your sub-class!")

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

        Expecting self._baseline and self._obs_placeholder

        :param returns: 1-D numpy array of returns for each time-step
        :param observations: List of observations for each time-step
        :return 1-D numpy array of advantages for each time-step
        """

        adv = returns

        if self.get_config("use_baseline"):
            baselines = self._sess.run(
                self._baseline,
                feed_dict={
                    self._obs_placeholder: np.array(map(self._transform_obs, observations)),
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

        Expecting self._obs_placeholder, self._baseline_target_placeholder, self._baseline_train_op, self._baseline_loss

        :param returns: 1-D numpy array of returns for each time-step
        :param observations: List of observations for each time-step
        :return Self, for chaining
        """

        _, baseline_loss = \
            self._sess.run(
                [self._baseline_train_op, self._baseline_loss],
                feed_dict={
                    self._obs_placeholder: np.array(map(self._transform_obs, observations)),
                    self._baseline_target_placeholder: returns,
                }
            )

        return baseline_loss





