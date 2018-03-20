import tensorflow as tf

class config():
  

    # Change env_name for the different experiments
    #
    #
    # for cartpole
    # output config

    # model and training config
    network_type = "RNN"
    use_only_price_info = False
    use_discrete_activation = False
    num_batches = 200 # number of batches trained on 
    batch_size = 500 # number of steps used to compute each policy update
    max_ep_len = 100 # maximum episode length 1440 minutes = 1day
    learning_rate = 1e-3
    gamma         = 0.99 # the discount factor
    max_quantity_per_transaction =0.7
    use_baseline = True
    normalize_advantage=True
    # parameters for the policy and baseline models
    n_layers = 1
    layer_size = 64
    activation=staticmethod(tf.nn.relu)
    replaysteps = 10

    postfix ="{!s}{!s}batch_{!s}batchsize_{!s}maxeplen_{!s}replays_maxtrans{!s}_{!s}Layers_{!s}Node".format(network_type, num_batches, batch_size, max_ep_len, 
                                                                                                            replaysteps, max_quantity_per_transaction, n_layers, 
                                                                                                            layer_size )

    #postfix ="{!s}{!s}batch_{!s}batchsize_{!s}maxeplen_{!s}replays_maxtrans{!s}_{!s}Layers_{!s}Node".format(network_type, 320, 500, max_ep_len, 
    #                                                                                                        10, 0.1, n_layers, 
    #                                                                                                        layer_size )

    #postfix += use_discrete_activation*"_use_discrete_activation"

    # output config
    record = True
    env_name="rlcrptocurrency-v0"
    output_path  = "results/" + env_name + postfix+"/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path 
    record_freq = 5
    summary_freq = 1

    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0: max_ep_len = batch_size
