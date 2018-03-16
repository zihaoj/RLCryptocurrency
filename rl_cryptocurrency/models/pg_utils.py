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

