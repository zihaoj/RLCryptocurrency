import tensorflow as tf
import time


class config():
    # Change env_name for the different experiments
    #

    # output config

    record = False
    env_name = "rlcrptocurrency-v0"
    output_path = "results/" + env_name + "/{:d}".format(int(time.time())) + "/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"
    record_path = output_path
    record_freq = 5
    summary_freq = 1

    # model and training config

    num_batches = 300  # number of batches trained on
    batch_size = 1000  # number of steps used to compute each policy update
    max_ep_len = 100  # maximum episode length 1440 minutes = 1day
    learning_rate = 1e-2  # default is 1e-3
    gamma = 0.99  # the discount factor
    use_baseline = True
    normalize_advantage = True

    batch_start_over = True  # whether we start over again for each batch

    decay_lr = True  # whether we use decaying learning rate; If true, then "learning_rate" will be the initial value
    decay_rate = 0.63  # decay rate, only enabled if decay_lr is true
    decay_steps = 40   # decay steps, only enabled if decay_lr is true

    # parameters for the policy and baseline models

    n_layers = 1
    layer_size = 64
    activation = staticmethod(tf.nn.leaky_relu)

    # since we start new episodes for each batch

    assert max_ep_len <= batch_size
    if max_ep_len < 0:
        max_ep_len = batch_size
