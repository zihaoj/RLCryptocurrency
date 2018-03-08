import gym
from gym_rlcrptocurrency.envs import Market
import numpy as np

from rl_cryptocurrency.models.pg_basics import PG
from rl_cryptocurrency.models.pg_2exchange1currency import PG2Exchange1Currency
from rl_cryptocurrency.tests.config import config


# setup market data
data_path = "/Users/qzeng/Dropbox/MyDocument/Mac-ZQ/CS/CS234/Material2018/project/data/"
markets = [
    [Market("{:s}/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv".format(data_path))],
    [Market("{:s}/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv".format(data_path))],
]

env_name = "rlcrptocurrency-v1"

# setup environment
env = gym.make(env_name).set_name("DefaultEnv")
env.set_markets(markets)

# setup reversed environment
env_aux = gym.make(env_name).set_name("ReverseEnv")
env_aux.set_markets(markets[::-1])

# initialize environment
init_portfolio = np.array(
    [
        [10000, 1],
        [10000, 1],
    ],
    dtype=np.float64
)
start_date = "2015-3-1"
# start_date = "2015-8-23"
# start_date = "2017-1-1"
# start_date = "2017-12-5"

# create model
# agent = PG(env, config).build().initialize()
agent = PG2Exchange1Currency(env, env_aux, config).build().initialize()

# training job
agent = agent.train(init_portfolio, start_date)

# evaluation

print "\n===> Evaluation <===\n"

env_eval = gym.make(env_name).set_name("EvalEnv")
env_eval.set_markets(markets)

agent.evaluate(env_eval, num_episodes=None, init_portfolio=init_portfolio, start_date="2017-12-5")

# # example: make a single sampling of path
# paths, episode_rewards = agent.sample_path(env, init_portfolio, start_date)
# print "episode_rewards:"
# print episode_rewards
# print "observation:"
# print env.get_observation()
