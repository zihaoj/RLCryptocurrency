import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import matplotlib.patches as patches
import numpy as np
import dill


def show_action(file_path):
    record = dill.load(open(file_path))

    price_index = 6  # TODO: hard-coded

    obs_list = record[0]["obs_list"][:]
    action_list = record[0]["action_list"][:]

    ex_A_price = np.array(map(lambda obs: obs[1][0, 0, price_index], obs_list))
    ex_B_price = np.array(map(lambda obs: obs[1][1, 0, price_index], obs_list))
    price_gap = ex_A_price - ex_B_price

    # action on exchange-A
    action_raw = np.array(map(lambda action: action[0][0, 0], action_list))
    mask_A_buy = action_raw < 0.

    action_A_buy = np.copy(action_raw)
    action_A_buy[~mask_A_buy] = 0.

    action_B_buy = np.copy(action_raw)
    action_B_buy[mask_A_buy] = 0.

    ##########################################################

    e1, = plt.plot(range(ex_A_price.shape[0]), ex_A_price, 'r')
    e2, = plt.plot(range(ex_B_price.shape[0]), ex_B_price, 'b')

    plt.title("Price and Action History")
    plt.legend([e1, e2], ["bitstamp", "coinBase"])

    plt.ylim( ( min(min(ex_A_price), min(ex_B_price) )*0.98, max(max(ex_A_price), max(ex_B_price) )*1.04  )  )

    ## plot actions

    for ia in range(len(action_A_buy)):
        plt.gca().add_patch( patches.Rectangle( (ia-0.1, max(ex_B_price)*1.02 ), 0.2, action_A_buy[ia]*(max(ex_B_price))*0.01, facecolor="b", edgecolor="b"  ) )

    for ib in range(len(action_B_buy)):
        plt.gca().add_patch( patches.Rectangle( (ib-0.1, max(ex_B_price)*1.02 ), 0.2, action_B_buy[ib]*(max(ex_B_price))*0.01, facecolor="r", edgecolor="r"  ) )

    plt.xlabel('Time Stamp (min)')
    plt.ylabel('Price')

    # plt.savefig( "{!s}ActionHistory_{!s}.png".format(basedir,ip) )
    plt.savefig("ActionHistory.png")
    plt.close()

    gap, = plt.plot(range(price_gap.shape[0]), price_gap, 'black')
    plt.title("Price Gap and Action History")
    plt.legend([e1], ["Price Gap = P(bitstamp)-P(coinBase)"])
    plt.ylim( ( min(price_gap)-20, max(price_gap )+20  ))

    for ia in range(len(action_A_buy)):
        plt.gca().add_patch( patches.Rectangle( (ia-0.1, max(price_gap) ), 0.2, action_A_buy[ia]*15, facecolor="b", edgecolor="b"  ) )

    for ib in range(len(action_B_buy)):
        plt.gca().add_patch( patches.Rectangle( (ib-0.1, max(price_gap) ), 0.2, action_B_buy[ib]*15, facecolor="r", edgecolor="r"  ) )

    # plt.savefig( "{!s}GapHistory_{!s}.png".format(basedir,ip) )
    plt.savefig("GapHistory.png")


def accum_return_plot(f_greedy, f_model):

    # process result returned by greedy
    def process_greedy(input_path):
        obj = dill.load(open(input_path))

        output = []
        for day in range(7):
            check_point = map(lambda index: obj["reward_accum_list"][index][day][1],
                              range(len(obj["reward_accum_list"])))
            output.append((day, np.mean(check_point), np.std(check_point)))

        return output

    # process result returned by model
    def process_model(input_path):
        obj = dill.load(open(input_path))
        n_run = len(obj)

        output = []
        for day in range(7):
            t = (day + 1) * 1440
            check_point = map(lambda index: 100 * obj[index]["accumulated_reward"][t] / 20000., range(n_run))
            output.append((day, np.mean(check_point), np.std(check_point)))

        return output

    # make the plot
    def add_curve(data, name, color):
        days, returns_mean, returns_std = zip(*data)

        y_mean = np.array(returns_mean)
        y_error = np.array(returns_std)

        plt.plot(days, y_mean, color=color, label=name)
        plt.fill_between(days, y_mean-y_error, y_mean+y_error, facecolor=color, alpha=0.5)

        # return plt.errorbar(x=days, y=returns_mean, yerr=returns_std, color=color, label=name)

    plt.figure()
    add_curve(process_greedy(f_greedy), "Greedy", "r")
    add_curve(process_model(f_model), "Optimal-stop", "b")

    plt.xlabel("Number of days")
    plt.ylabel("Accumulated Return [%]")
    plt.legend()

    plt.savefig("accum_return_plot.png")


if __name__ == "__main__":
    show_action("eval_result.dill")

    # dir_base = "/Users/qzeng/Dropbox/MyDocument/Mac-ZQ/CS/CS234/Material2018/project/"
    # accum_return_plot(
    #     "{:s}/gym-rlcrptocurrency/gym_rlcrptocurrency/tests/poster/20171201_20runs/result_run_policy.dill".format(dir_base),
    #     "{:s}/RLCryptocurrency/rl_cryptocurrency/tests/poster/optimal-stop-default/20171201_20run/eval_result.dill".format(dir_base)
    # )
