import tensorflow as tf
import time


class Config(object):
    # training config

    train_size = 2592  # 6 month
    batch_size = 5
    num_epoch = 10
    max_ep_len = 100
    mix_reverse = True  # whether we mix reversed episode with normal one into the same batch

    gamma = 1.0
    learning_rate = 5e-3

    eval_freq = 100

    # MLP network config

    n_layers = 1
    layer_size = 32
    activation = staticmethod(tf.nn.leaky_relu)
    # activation = staticmethod(tf.nn.tanh)

    batch_norm_policy = False   # Whether we apply batch normalization on policy network.
    batch_norm_baseline = False  # Whether we apply batch normalization on baseline network

    # baseline config

    use_baseline = True
    normalize_advantage = True

    # RNN config (if enabled)
    rnn_maxlen = 60  # length of buffer, including current time-stamp. Thus must be at least 1.
    assert rnn_maxlen >= 1, "Invalid buffer max len!"
    rnn_hidden_size = 16
    rnn_cell = "LSTM"

    def __init__(self, env_name, config_name=None):
        self._env_name = env_name

        if config_name is None:
            config_name = str(time.time())
        self._config_name = config_name

    @property
    def env_name(self):
        return self._env_name

    @property
    def config_name(self):
        return self._config_name

    @property
    def output_path(self):
        return "results/{:s}/{:s}/".format(self.env_name, self.config_name)

    @property
    def model_output(self):
        return "{:s}/model_caches/".format(self.output_path)

    @property
    def log_path(self):
        return "{:s}/log.txt".format(self.output_path)

    @property
    def plot_output(self):
        return "{:s}/scores.png".format(self.output_path)

    @property
    def record_path(self):
        return self.output_path

