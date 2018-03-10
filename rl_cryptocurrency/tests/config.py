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

    # MLP network config

    n_layers = 1
    layer_size = 32
    activation = staticmethod(tf.nn.leaky_relu)
    # activation = staticmethod(tf.nn.tanh)

    batch_norm_policy = True   # Whether we apply batch normalization on policy network.
    batch_norm_baseline = False  # Whether we apply batch normalization on baseline network

    # baseline config

    use_baseline = True
    normalize_advantage = True

    # RNN config (if enabled)
    rnn_maxlen = 10  # length of buffer, including current time-stamp. Thus must be at least 1.
    assert rnn_maxlen >= 1, "Invalid buffer max len!"
    rnn_hidden_size = 32


