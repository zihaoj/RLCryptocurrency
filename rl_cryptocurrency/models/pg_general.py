# Now, instead of an optimal stopping problem, we just treat it as a normal MDP dynamics
# Inherited from optimal stopping setup

from rl_cryptocurrency.models.pg_optimal_stop import PGOptimalStop
from collections import deque


class PGGeneral(PGOptimalStop):

    def sample_path(self, env, init_portfolio, start_date, **options):
        """
        Almost same as optimal stopping setup, but in a general setup now
        """

        # Declare additional options needed by this implementation
        # Horizon of each episode
        max_ep_len = options["max_ep_len"]
        # List of integer, indicating the list of first time-stamp for each episode wrt start_date
        loop_list = options["loop_list"]
        # buffer max length
        max_buffer_len = self.get_config("rnn_maxlen")

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
            env.move_market(t_init + t)

            # unroll the episode
            observations, actions, rewards = [], [], []
            episode_reward = 0.
            count = 0

            # initialize the history buffer with maxlen - 1 time-stamps before starting point
            observations_buffer = deque(maxlen=max_buffer_len)
            observations_buffer_list = []

            obs, _, _, _ = env.move_market(-max_buffer_len+1)
            for _ in range(max_buffer_len - 1):
                observations_buffer.append(obs)
                obs, _, _, _ = env.move_market(1)

            while count < max_ep_len:
                # current observation
                observations_buffer.append(obs)
                observations_buffer_list.append(list(observations_buffer))
                observations.append(obs)

                # sample an action
                action = self._sess.run(
                    self._sampled_action,
                    feed_dict={
                        self._obs_placeholder: self._transform_obs(observations_buffer)[None],
                        self._is_training_placeholder: False,
                    }
                )[0]

                # # DEBUG
                # action, condition_stuck, debug_adjusted_price_gap, debug_adjusted_price_gap_ref, debug_prob_action2 = self._sess.run(
                #     [self._sampled_action, self._condition_stuck, self._debug_adjusted_price_gap, self._debug_adjusted_price_gap_ref, self._debug_prob_action2],
                #     feed_dict={
                #         self._obs_placeholder: self._transform_obs(observations_buffer)[None],
                #         self._is_training_placeholder: False,
                #     }
                # )
                #
                # action = action[0]
                # condition_stuck = condition_stuck[0]
                # debug_adjusted_price_gap = debug_adjusted_price_gap[0]
                # debug_adjusted_price_gap_ref = debug_adjusted_price_gap_ref[0]
                # debug_prob_action2 = debug_prob_action2[0]
                #
                # if not condition_stuck:
                #     assert action < 2, "Invalid action -- 1!"
                # else:
                #     assert action in [0, 2], "Invalid action -- 2!"
                #
                #     print "{:.4f}, {:.4f}, {:.4f}".format(debug_adjusted_price_gap, debug_adjusted_price_gap_ref, debug_prob_action2)
                #
                #     if action == 2:
                #         print "HERE!!!"

                actions.append(action)

                # move one step forward
                action_env = self._transform_action(action, obs)
                obs, reward, _, _ = env.step(action_env)

                rewards.append(reward)
                episode_reward += reward

                # continue to next iteration
                count += 1

            # prepare episode
            path = {
                "observations": observations,
                "observations_buffer_list": observations_buffer_list,
                "actions": actions,
                "rewards": rewards,
            }
            paths.append(path)
            episode_rewards.append(episode_reward)

        # return
        return paths, episode_rewards
