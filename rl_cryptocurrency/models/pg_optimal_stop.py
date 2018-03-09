# A special case where we reduce the problem into an optimal stopping problem

from rl_cryptocurrency.models.pg_base import PGBase

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm


class PGOptimalStop(PGBase):
    def __init__(self, env, env_aux, config, logger=None):
        """
        Only support special case of 2 exchanges and 1 currency this moment
        """

        # initialize through base class
        super(PGOptimalStop, self).__init__(env, env_aux, config, logger)

        # assertion on environment
        assert self._n_exchange == 2, "PG2Exchange1Currency only supports 2 exchanges!"
        assert self._n_currency == 1, "PG2Exchange1Currency only supports 1 crypto-currency!"

    def sample_path(self, env, init_portfolio, start_date, **options):
        """
        Not too different from the general path sampling. But we speed up by utilizing stopping dynamics
        """

        # Declare additional options needed by this implementation
        # Horizon of each episode
        max_ep_len = options["max_ep_len"]
        # List of integer, indicating the list of first time-stamp for each episode wrt start_date
        loop_list = options["loop_list"]

        # Initialize environment
        # t_init is the time-stamp w.r.t market reset index
        if type(start_date) == str:
            env.init(init_portfolio, start_date)
            t_init = 0
        else:
            assert type(start_date) == int, "Invalid start_date {:s}!".format(str(start_date))
            t_init = start_date

        # Loop through episodes
        paths = []
        episode_rewards = []
        for t in loop_list:
            # reset environment
            env.init(init_portfolio, None)
            obs, _, _, _ = env.move_market(t_init + t)

            # unroll the episode
            observations, actions, rewards = [], [], []
            episode_reward = 0
            count = 0

            while count < max_ep_len:
                # current observation
                observations.append(obs)

                # sample an action
                action = self._sess.run(
                    self._sampled_action,
                    feed_dict={self._obs_placeholder: self._transform_obs(obs)[None]}
                )[0]
                actions.append(action)

                # move one step forward
                # this is where optimal stopping problem is special
                if action == 1:
                    action_env = self._transform_action(action, obs)
                    _, reward, _, _ = env.step(action_env)

                    rewards.append(reward)
                    episode_reward += reward

                    break
                else:
                    assert action == 0, "How do you turn this on?!"

                    obs, reward, _, _ = env.move_market(1)

                    rewards.append(reward)
                    episode_reward += reward

                # continue to next iteration
                count += 1

            # prepare episode
            path = {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
            }
            paths.append(path)
            episode_rewards.append(episode_reward)

        # return
        return paths, episode_rewards

    def train(self, init_portfolio, start_date, **options):
        """
        A basic version
        """

        # collect all options needed
        end_date = options["end_date"]
        env_eval_list = options["env_eval_list"]  # make sure this has been initialized!
        train_size = self.get_config("train_size")
        batch_size = self.get_config("batch_size")
        num_epoch = self.get_config("num_epoch")
        max_ep_len = self.get_config("max_ep_len")

        # prepare training pool
        train_pool = self._generate_train_pool(start_date, end_date, train_size)

        # initialize env market
        env_list = [self._env]
        if self._env_aux is not None:
            env_list.append(self._env_aux)
        for env in env_list:
            env.init(init_portfolio, start_date)

        # average rewards
        self._init_averages()
        scores_eval = []  # list of scores computed at iteration time

        # loop through epoch
        # one epoch is one pass through all the training episodes
        counter = 0
        for epoch in range(num_epoch):
            train_batches = self._generate_train_batch(train_pool, batch_size)

            # loop through batches
            # one batch is the most basic unit corresponding to one parameter update
            for batch in train_batches:
                for env in env_list:
                    paths, total_rewards = \
                        self.sample_path(env, init_portfolio, 0, max_ep_len=max_ep_len, loop_list=batch)

                    # update evaluation score
                    scores_eval += total_rewards

                    # concatenate all episodes to 1-D array
                    observations = reduce(lambda x, y: x + y, map(lambda path: path["observations"], paths))
                    actions = reduce(lambda x, y: x + y, map(lambda path: path["actions"], paths))

                    # get advantage
                    returns = self._get_returns(paths)
                    advantages = self._calculate_advantage(returns, observations)

                    # update baseline network, if applicable
                    if self.get_config("use_baseline"):
                        self._update_baseline(returns, observations)

                    # update policy network
                    _, grad_norm, loss = \
                        self._sess.run(
                            [self._train_op, self._grad_norm, self._loss],
                            feed_dict={
                                self._obs_placeholder: np.array(map(self._transform_obs, observations)),
                                self._action_placeholder: np.array(actions),
                                self._advantage_placeholder: advantages,
                            }
                        )

                    # tf stuff
                    if counter % self.get_config("summary_freq") == 0:
                        self._update_averages(total_rewards, scores_eval)
                        self._record_summary(counter, grad_norm, loss)

                    # compute reward statistics for this batch and log
                    avg_reward = np.mean(total_rewards)
                    sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
                    msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
                    self._logger.info(msg)

                    # update counter
                    counter += 1

                if counter % self.get_config("eval_freq") == 0:
                    # perform one evaluation at end of one epoch
                    self._logger.info("\n===> Evaluation ... <===\n")

                    for env_eval in env_eval_list:
                        self._logger.info("Evaluating on environment {:s}".format(env_eval.name))
                        env_eval.init(init_portfolio, None)
                        real_eval_score = self.evaluate(env_eval, 7*1440)  # TODO: hard-coded, run for 7-days as test
                        self._logger.info("Accumulated return is {:.4f}".format(real_eval_score["accumulated_reward"][-1]))
                        self._logger.info("\n")

        # finish training
        self._logger.info("- Training done.")
        return self

    def evaluate(self, env, n):
        """
        Instead of using sample_path(), we do the normal simulation to test return for a pre-defined period of time
        """

        obs = env.get_observation()

        counter = 0
        reward_list = []
        accumulated_reward_list = [0.]

        pbar = tqdm(total=n, desc="Loop over time-stamps")

        while counter < n:
            # sample an action
            action = self._sess.run(
                self._sampled_action,
                feed_dict={self._obs_placeholder: self._transform_obs(obs)[None]}
            )[0]

            obs, reward, _, _ = env.step(self._transform_action(action, obs))

            # update
            counter += 1
            reward_list.append(reward)
            accumulated_reward_list.append(accumulated_reward_list[-1] + reward)
            pbar.update(1)

        return {
            "reward": reward_list,
            "accumulated_reward": accumulated_reward_list
        }

    def _add_placeholders_op(self):
        """
        A basic version: price gap only
        """

        with tf.variable_scope("placeholder"):
            self._obs_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1),
                                                   name="obs_placeholder")

            self._action_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,),
                                                      name="action_placeholder")

            self._advantage_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,),
                                                         name="advantage_placeholder")

        return self

    def _build_policy_network_op(self):
        """
        A basic version: MLP
        """

        with tf.variable_scope("policy_network"):
            net = self._obs_placeholder
            for layer in range(self.get_config("n_layers")):
                net = tf.contrib.layers.fully_connected(
                    inputs=net,
                    num_outputs=self.get_config("layer_size"),
                    activation_fn=self.get_config("activation"),
                    scope="layer_{:d}".format(layer)
                )
            logits = tf.contrib.layers.fully_connected(
                inputs=net,
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
        This will do all the baseline jobs, including
        1. Define target placeholder
        2. Obtain output of baseline network
        3. Minimize baseline loss

        :return: Self, for chaining
        """

        with tf.variable_scope("baseline_network"):
            net = self._obs_placeholder
            for layer in range(self.get_config("n_layers")):
                net = tf.contrib.layers.fully_connected(
                    inputs=net,
                    num_outputs=self.get_config("layer_size"),
                    activation_fn=self.get_config("activation"),
                    scope="layer_{:d}".format(layer)
                )
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

            lr = self.get_config("learning_rate")
            self._baseline_train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(baseline_loss)

        return self

    def _transform_obs(self, obs_env):
        """
        Basic version: just price gap
        """

        obs_portfolio, obs_market, _ = obs_env
        price_gap = obs_market[0, 0, self._price_index] - obs_market[1, 0, self._price_index]

        return np.array([price_gap])

    def _transform_action(self, action, obs_env):
        """
        If action is 1, then we all in, otherwise, do nothing
        """

        # get observation from environment
        obs_portfolio, obs_market, _ = obs_env
        price_matrix = obs_market[:, :, self._price_index]

        # obtain the bound
        if action >= 0:
            price_adjusted = price_matrix[0, 0] / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
            bound = np.min((obs_portfolio[1, 1], obs_portfolio[0, 0] / price_adjusted))
        else:
            price_adjusted = price_matrix[1, 0] / ((1. - self._fee_exchange) * (1. - self._fee_transfer))
            bound = np.min((obs_portfolio[0, 1], obs_portfolio[1, 0] / price_adjusted))

        if action == 1:
            action_final = 0.95
        else:
            action_final = 0.

        # get purchase matrix
        action_final = action_final * bound
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

    def _generate_train_pool(self, start_date, end_date, n):
        """
        Generate training pools constrained by start_date and end_date

        Here is just a basic version: Assume we generate n-episodes continuously

        :param start_date: Str. Lower bound of starting date
        :param end_date: Str. Upper bound of starting date
        :param n: Size of training pool
        :return: A list of integer, representing the number of time-stamps from start_date
        """

        max_delta = int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).total_seconds() / 60.)
        max_ep_len = self.get_config("max_ep_len")

        train_pool = range(0, max_delta, max_ep_len)
        if len(train_pool) > n:
            train_pool = train_pool[:n]
        else:
            self._logger.warn(
                "Unable to generate training pool of length {:d}. Actual length is {:d}".format(n, len(train_pool))
            )

        return train_pool

    def _generate_train_batch(self, train_pool, batch_size):
        """
        Generate training batches.
        This should be called for every epoch

        :param train_pool: training pool as generated from _generate_train_pool
        :param batch_size: Batch size
        :return: List of batch. Each batch is a list of integers
        """

        train_pool_shuffled = np.random.permutation(train_pool)

        output = []
        index = 0
        while index < len(train_pool):
            batch = train_pool_shuffled[index:index+batch_size]
            output.append(batch)
            index += batch_size

        return output










