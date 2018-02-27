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
    num_batches = 20 # number of batches trained on 
    batch_size = 10 # number of steps used to compute each policy update
    max_ep_len = 10 # maximum episode length 1440 minutes = 1day
    learning_rate = 3e-2
    gamma              = 1.0 # the discount factor
    use_baseline = True
    normalize_advantage=True
    # parameters for the policy and baseline models
    n_layers = 1
    layer_size = 10
    activation=staticmethod(tf.nn.relu)

    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0: max_ep_len = batch_size

