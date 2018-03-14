import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import numpy as np

basedir = "results/rlcrptocurrency-v0RNN320batch_500batchsize_100maxeplen_10replays_maxtrans0.1_1Layers_64Node/"

def ShowAction(hf):
    observations = hf.get("observation_full")
    actions = hf.get("action")
    constraint_ups = hf.get("constraint_up")
    constraint_dns = hf.get("constraint_dn")

    print actions, observations, constraint_dns, constraint_ups


    for ip in range(observations.shape[0]):
        ex_A_price = observations[ip,:,0]/observations[ip,:,1]
        ex_B_price = observations[ip,:,0]+ex_A_price

        if "RNN" in basedir:
            ex_A_price = ex_A_price[100:]
            ex_B_price = ex_B_price[100:]
        else:
            ex_A_price = ex_A_price[10:]
            ex_B_price = ex_B_price[10:]
            

        price_gap = ex_A_price-ex_B_price

        action = actions[ip]

        action_A_buy = action* (action>0)
        action_B_buy = action* (action<0)

        action_A_buy = action_A_buy[:,0]
        action_B_buy = action_B_buy[:,0]

        e1, = plt.plot(range(ex_A_price.shape[0]), ex_A_price, 'r')
        e2, = plt.plot(range(ex_B_price.shape[0]), ex_B_price, 'b')

        plt.title("Price and Action History")
        plt.legend([e1, e2], ["bitstamp", "coinBase"])


        plt.ylim( ( min(min(ex_A_price), min(ex_B_price) )*0.98, max(max(ex_A_price), max(ex_B_price) )*1.04  )  )

        ## plot actions

        for ia in range(len(action_A_buy)):
            plt.gca().add_patch( patches.Rectangle( (ia-0.1, max(ex_B_price)*1.02 ), 0.2, action_A_buy[ia]*(max(ex_B_price))*0.04, facecolor="b", edgecolor="b"  ) )

        for ib in range(len(action_B_buy)):
            plt.gca().add_patch( patches.Rectangle( (ib-0.1, max(ex_B_price)*1.02 ), 0.2, action_B_buy[ib]*(max(ex_B_price))*0.04, facecolor="r", edgecolor="r"  ) )

        plt.xlabel('Time Stamp (min)')
        plt.ylabel('Price')

        plt.savefig( "{!s}ActionHistory_{!s}.png".format(basedir,ip) )
        plt.close()


        gap, = plt.plot(range(price_gap.shape[0]), price_gap, 'black')
        plt.title("Price Gap and Action History")
        plt.legend([e1], ["Price Gap = P(bitstamp)-P(coinBase)"])
        plt.ylim( ( min(price_gap)-20, max(price_gap )+20  ))

        for ia in range(len(action_A_buy)):
            plt.gca().add_patch( patches.Rectangle( (ia-0.1, max(price_gap) ), 0.2, action_A_buy[ia]*1, facecolor="b", edgecolor="b"  ) )

        for ib in range(len(action_B_buy)):
            plt.gca().add_patch( patches.Rectangle( (ib-0.1, max(price_gap) ), 0.2, action_B_buy[ib]*1, facecolor="r", edgecolor="r"  ) )

        plt.savefig( "{!s}GapHistory_{!s}.png".format(basedir,ip) )


hf = h5py.File(basedir+'performance.h5', 'r')
ShowAction(hf)
