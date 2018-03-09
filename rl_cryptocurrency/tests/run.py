import gym
from gym_rlcrptocurrency.envs import Market
import numpy as np

from rl_cryptocurrency.models.pg_optimal_stop import PGOptimalStop
from rl_cryptocurrency.tests.config import config

# setup market data
data_path = "/Users/qzeng/Dropbox/MyDocument/Mac-ZQ/CS/CS234/Material2018/project/data/"
markets = [
    "{:s}/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv".format(data_path),
    "{:s}/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv".format(data_path),
]

# setup environment
env = gym.make(config.env_name).set_name("DefaultEnv")
env.set_markets([[Market(markets[0])], [Market(markets[1])]])

# setup reversed environment
env_aux = gym.make(config.env_name).set_name("ReverseEnv")
env_aux.set_markets([[Market(markets[1])], [Market(markets[0])]])

# initialize environment
init_portfolio = np.array(
    [
        [10000, 1],
        [10000, 1],
    ],
    dtype=np.float64
)
start_date = "2015-2-1"

# setup evaluation environment
env_eval = gym.make(config.env_name).set_name("EvalEnvDefault")
env_eval.set_markets([[Market(markets[0])], [Market(markets[1])]]).init(init_portfolio, "2015-9-1")

env_eval_reverse = gym.make(config.env_name).set_name("EvalEnvReverse")
env_eval_reverse.set_markets([[Market(markets[1])], [Market(markets[0])]]).init(init_portfolio, "2015-9-1")

# create model
agent = PGOptimalStop(env, env_aux, config).build().initialize()

# training job
agent = agent.train(init_portfolio, start_date, end_date="2015-8-1", env_eval_list=[env_eval, env_eval_reverse])

# # example: make a single sampling of path
# paths, episode_rewards = agent.sample_path(env, init_portfolio, start_date)
# print "episode_rewards:"
# print episode_rewards
# print "observation:"
# print env.get_observation()
