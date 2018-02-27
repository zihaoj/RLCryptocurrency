import tensorflow as tf

class config():
  

    # Change env_name for the different experiments
    #
    #
    # for cartpole
    # output config


    # for inverted pendulum
    # output config
    record = False
    env_name="rlcrptocurrency-v0"
    output_path  = "results/" + env_name + "/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path 
    record_freq = 5
    summary_freq = 1

    
    # model and training config
    
    use_only_price_info = True
    num_batches = 50 # number of batches trained on 
    batch_size = 600 # number of steps used to compute each policy update
    max_ep_len = 300 # maximum episode length 1440 minutes = 1day
    learning_rate = 1e-4
    gamma              = 1.0 # the discount factor
    max_quantity_per_transaction =0.3
    use_baseline = True
    normalize_advantage=True
    # parameters for the policy and baseline models
    n_layers = 4
    layer_size = 64
    activation=staticmethod(tf.nn.relu)
    replaysteps = 10

    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0: max_ep_len = batch_size


