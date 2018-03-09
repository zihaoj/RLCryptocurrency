import tensorflow as tf
import time


class config():
    # Change env_name for the different experiments
    #

    # output config

    env_name = "rlcrptocurrency-v1"
    output_path = "results/" + env_name + "/{:d}".format(int(time.time())) + "/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"
    record_path = output_path
    # record_freq = 5
    # summary_freq = 1

    # training config

    train_size = 12960  # 3 month
    batch_size = 32
    num_epoch = 10
    max_ep_len = 10  # 10 minute
    mix_reverse = True  # whether we mix reversed episode with normal one into the same batch

    gamma = 1.0
    learning_rate = 5e-3

    eval_freq = 100

    # policy network config
    n_layers = 1
    layer_size = 64
    activation = staticmethod(tf.nn.leaky_relu)

    # baseline config
    use_baseline = True
    normalize_advantage = True
