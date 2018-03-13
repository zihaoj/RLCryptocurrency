import gym
from gym_rlcrptocurrency.envs import Market
import numpy as np
import argparse
import dill

from rl_cryptocurrency.models.pg_optimal_stop_more_features import PGOptimalStopMoreFeatures
from rl_cryptocurrency.models.pg_optimal_stop_rnn import PGOptimalStopRNN
from rl_cryptocurrency.tests.config import Config


def main(args):

    assert args.mode in ["train", "test"], "Invalid input mode {:s}!".format(args.mode)

    # training / evaluation period #

    # train_start_date = "2015-2-1"
    # train_end_date = "2015-8-1"
    # eval_start_date = "2015-9-1"

    train_start_date = "2017-8-1"
    train_end_date = "2017-11-1"
    # eval_start_date = "2017-12-5"
    eval_start_date = "2017-11-15"
    # eval_start_date = "2017-11-5"

    # choose what model to use #

    model_class = PGOptimalStopMoreFeatures

    # setup market data #

    data_path = "/Users/qzeng/Dropbox/MyDocument/Mac-ZQ/CS/CS234/Material2018/project/data/"
    markets = [
        "{:s}/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv".format(data_path),
        "{:s}/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv".format(data_path),
    ]

    # setup config #

    config = Config("rlcrptocurrency-v1", "test")

    # setup environment #

    env = gym.make(config.env_name).set_name("DefaultEnv")
    env.set_markets([[Market(markets[0])], [Market(markets[1])]])

    # setup reversed environment #

    env_aux = gym.make(config.env_name).set_name("ReverseEnv")
    env_aux.set_markets([[Market(markets[1])], [Market(markets[0])]])

    # initialize environment #

    init_portfolio = np.array(
        [
            [10000, 1],
            [10000, 1],
        ],
        dtype=np.float64
    )

    # setup evaluation environment #

    env_eval = gym.make(config.env_name).set_name("EvalEnvDefault")
    env_eval.set_markets([[Market(markets[0])], [Market(markets[1])]]).init(init_portfolio, eval_start_date)
    # DEBUG
    env_eval.debug = True

    env_eval_reverse = gym.make(config.env_name).set_name("EvalEnvReverse")
    env_eval_reverse.set_markets([[Market(markets[1])], [Market(markets[0])]]).init(init_portfolio, eval_start_date)
    # DEBUG
    env_eval_reverse.debug = True

    if args.mode == "train":

        # create model #

        agent = model_class(env, env_aux, config, overwrite=True).build().initialize(restore_id=args.model)

        # training job #

        agent = agent.train(init_portfolio, train_start_date, end_date=train_end_date,
                            env_eval_list=[env_eval, env_eval_reverse])

        return
    else:
        agent = model_class(env, env_aux, config, overwrite=False).build().initialize(restore_id=args.model)

        # collect all tests
        eval_result = []
        for _ in range(args.num_test):
            env_eval.init(init_portfolio, None)
            eval_result.append(agent.evaluate(env_eval, 7*1440))
        dill.dump(eval_result, open("eval_result.dill", "w"))

        # print out mean of total accumulated return
        total_return_list = map(lambda index: eval_result[index]["accumulated_reward"][-1], range(len(eval_result)))
        print "Total return: {:.4f} +/- {:.4f}".format(np.mean(total_return_list), np.std(total_return_list))


if __name__ == "__main__":
    # parse
    parser = argparse.ArgumentParser(description="Run training / evaluation job")
    parser.add_argument("--mode", dest="mode", action="store", type=str, required=True,
                        help="Mode of command. Either \"train\" or \"test\". "
                             "If \"test\", must specify id of model being restored")
    parser.add_argument("--model", dest="model", action="store", type=int, default=None,
                        help="Model id to be restored. Default is None, in which case will be randomly initialized")
    parser.add_argument("--num_test", dest="num_test", action="store", type=int, default=1,
                        help="Number of iterations to be performed for evaluation. Only activated if mode is test")
    args = parser.parse_args()

    # run
    main(args)



# # example: make a single sampling of path
# paths, episode_rewards = agent.sample_path(env, init_portfolio, start_date)
# print "episode_rewards:"
# print episode_rewards
# print "observation:"
# print env.get_observation()
