import gym
from gym_rlcrptocurrency.envs import Market
import numpy as np

from rl_cryptocurrency.models.pg_basics import PG
from rl_cryptocurrency.models.config import config


# setup market data
data_path = "/Users/qzeng/Dropbox/MyDocument/Mac-ZQ/CS/CS234/Material2018/project/data/"
markets = [
    [Market("{:s}/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv".format(data_path))],
    [Market("{:s}/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv".format(data_path))],
]

# setup environment
env = gym.make("rlcrptocurrency-v0")
env.set_markets(markets)

# initialize environment
init_portfolio = np.array(
    [
        [10000, 1],
        [10000, 1],
    ],
    dtype=np.float64
)
start_date = "2017-1-1"

# create model
agent = PG(env, config).build().initialize()

# training job
agent = agent.train(init_portfolio, start_date)


# # example: make a single sampling of path
# paths, episode_rewards = agent.sample_path(env, init_portfolio, start_date)
# print "episode_rewards:"
# print episode_rewards
# print "observation:"
# print env.get_observation()
