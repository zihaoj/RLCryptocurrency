# Utility functions

import tensorflow as tf


def add_exploration_entropy(class_obj, tau):
    """
    Add exploration term through entropy
    see: https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/a3c/estimators.py
    The idea is, entropy is maximized when p=0.5 (pure exploration) and minimized with p=0/1 (pure deterministic)
    Thus to encourage exploration, we want to INCREASE entropy (with some factor)

    :param class_obj: The class of RL agent
    :param tau: regularization parameters
    :return The new class of augmented RL agent
    """

    class RLEnhancedExplorationEntropy(class_obj):
        def _add_loss_op(self):
            with tf.variable_scope("policy_loss"):
                # the normal loss (to be maximized)
                loss_to_max = self._logprob * self._advantage_placeholder

                # entropy-based exploration
                # important: use v2, otherwise back propagation will only happen on logits
                entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.nn.softmax(self._logits, axis=1),
                    logits=self._logits,
                )

                # the actual loss
                self._loss = -tf.reduce_mean(loss_to_max + tau * entropy)

            return self

    return RLEnhancedExplorationEntropy


def add_exploration_ment(class_obj, tau):
    """
    This is in principle equivalent to entropy-based exploration, except difference in implementation
    In _entropy implementation, the gradient is computed on full entropy (with summation over all possible actions)
    directly
    In this implementation, sampling method will be used instead
    """

    class RLEnhancedExplorationMent(class_obj):
        def _add_loss_op(self):
            with tf.variable_scope("policy_loss"):
                loss_to_max = self._logprob * (self._advantage_placeholder - tau) - 0.5 * tau * tf.square(self._logprob)
                self._loss = -tf.reduce_mean(loss_to_max)

            return self

    return RLEnhancedExplorationMent


def add_exploration_urex(class_obj, tau):
    """
    Attempts to add exploration following UREX method:
    https://arxiv.org/abs/1611.09321

    assuming return_placeholder is available
    also assuming a fixed-length episode, as specified in the config by "max_ep_len"

    Here we have to assume the episode length is fixed and accessible from configuration
    """

    assert tau > 0, "Invalid tau {:.4f}!".format(tau)

    class RLEnhancedExplorationUrex(class_obj):
        def _add_loss_op(self):
            assert self.get_config("use_return"), "Please turn on use_return in your config first!"

            with tf.variable_scope("policy_loss"):
                # REINFORCE part as usual
                loss_to_max = self._logprob * self._advantage_placeholder

                # the UREX part
                return_reshape = tf.reshape(self._return_placeholder, shape=(-1, self.get_config("max_ep_len")),
                                            name="return_reshape")
                logprob_reshape = tf.reshape(self._logprob, shape=(-1, self.get_config("max_ep_len")),
                                             name="logprob_reshape")

                return_per_episode = return_reshape[:, 0]
                logprob_per_episode = tf.reduce_sum(logprob_reshape, axis=1)
                weight_unscaled_per_episode = return_per_episode / tau - logprob_per_episode
                weight_per_episode = tf.nn.softmax(weight_unscaled_per_episode)

                weight = tf.reshape(weight_per_episode, shape=(-1, 1)) * tf.ones(shape=(self.get_config("max_ep_len"),))
                weight = tf.reshape(weight, shape=(-1,))
                weight_forward_only = tf.stop_gradient(weight, "weight_forward_only")

                urex = self._logprob * tau * weight_forward_only

                # add them together
                self._loss = -tf.reduce_mean(loss_to_max + urex)

            return self

    return RLEnhancedExplorationUrex

