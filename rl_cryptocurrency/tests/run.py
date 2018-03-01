import gym
from gym_rlcrptocurrency.envs import Market

from rl_cryptocurrency.models.pg_basics import PG


# setup market data
data_path = "/Users/qzeng/Dropbox/MyDocument/Mac-ZQ/CS/CS234/Material2018/project/data/"
markets = [
    [Market("{:s}/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv".format(data_path))],
    [Market("{:s}/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv".format(data_path))],
]

# setup environment
env = gym.make("rlcrptocurrency-v0")
env.set_markets(markets)

# initial
