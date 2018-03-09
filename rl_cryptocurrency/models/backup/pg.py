#import logging
import numpy as np
import tensorflow as tf
import gym
from gym_rlcrptocurrency.envs import Market
import os
#import inspect
from rl_cryptocurrency.utils.general import get_logger, export_plot
from rl_cryptocurrency.tests.config import config
from datetime import timedelta, date


def build_mlp(
          mlp_input, 
          output_size,
          scope, 
          n_layers=config.n_layers, 
          size=config.layer_size, 
          output_activation=None):
  '''
  Build a feed forward network (multi-layer-perceptron, or mlp) 
  with 'n_layers' hidden layers, each of size 'size' units.
  Use tf.nn.relu nonlinearity between layers. 
  Args:
          mlp_input: the input to the multi-layer perceptron
          output_size: the output size for action

          scope: the scope of the neural network
          n_layers: the number of layers of the network
          size: the size of each layer:
          output_activation: the activation of output layer
  Returns:
          The tensor output of the network
  
  TODO: Implement this function. This will be similar to the linear 
  model you implemented for Assignment 2. 
  "tf.layers.dense" or "tf.contrib.layers.fully_connected" may be helpful.

  A network with n layers has n 
    linear transform + nonlinearity
  operations before a final linear transform for the output layer 
  (followed by the output activation, if it is not None).

  '''
  #######################################################
  #########   YOUR CODE HERE - 7-20 lines.   ############
  with tf.variable_scope(scope):

    dense=[]
    flatten = tf.contrib.layers.flatten(mlp_input, scope='flatten')

    for il in range(n_layers):
      
      thisdense = None
      if il ==0:
        thisdense = tf.contrib.layers.fully_connected(inputs = flatten, num_outputs = size, activation_fn = config.activation, scope = "dense"+str(il))
      else:
        thisdense = tf.contrib.layers.fully_connected(inputs = dense[il-1], num_outputs = size, activation_fn = config.activation, scope = "dense"+str(il))

      dense.append(thisdense)

    output = tf.contrib.layers.fully_connected(inputs = dense[-1], num_outputs = output_size, activation_fn = output_activation, scope = "output", weights_initializer = tf.zeros_initializer(), biases_initializer = tf.zeros_initializer() )
  
    return output


def RepackActionForEnv(env, action):

  purchase = action[0:env._n_exchange*env._n_currency]
  transfer = action[env._n_exchange*env._n_currency:]

  purchase = np.reshape( purchase, (env._n_exchange, env._n_currency )  )
  transfer = np.reshape( transfer, (env._n_exchange, env._n_exchange, env._n_currency )  )

  output = (purchase, transfer)
  return output



class PG(object):
  """
  Abstract Class for implementing a Policy Gradient Based Algorithm
  """
  def __init__(self, env, config, logger=None):
    """
    Initialize Policy Gradient Class
  
    Args:
            env: the open-ai environment
            config: class with hyperparameters
            logger: logger instance from logging module

    You do not need to implement anything in this function. However,
    you will need to use self.discrete, self.observation_dim,
    self.action_dim, and self.lr in other methods.
    
    """
    # directory for training outputs
    if not os.path.exists(config.output_path):
      os.makedirs(config.output_path)
            
    # store hyper-params
    self.config = config
    self.logger = logger
    if logger is None:
      self.logger = get_logger(config.log_path)
    self.env = env
  
    # discrete action space or continuous action space
    self.discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # get observation dimension
    self.observation_dim =0
    self.observation_dim += env._n_exchange *(env._n_currency+1)  ## portfolio

    if config.use_only_price_info:
      self.observation_dim += env._n_exchange * env._n_currency
    else:
      self.observation_dim += env._n_exchange * env._n_currency * len(env.market_obs_attributes)


    print "obs dim", self.observation_dim

    # get action dimension
    # due to constraints for purchase action we need only n_exchange-1
    # due to constraints for transfer action we need only n_exchange**2*n_currency

    self.action_dim = 0
    self.action_purchase_dim = 0
    self.action_transfer_dim = 0
    ## first box is purchase
    ## second box is transfer

    self.action_purchase_dim = (env._n_exchange-1)
    self.action_transfer_dim = (env._n_exchange-1)*env._n_exchange*env._n_currency/2
    
    self.action_dim += self.action_purchase_dim
    self.action_dim += self.action_transfer_dim

    print "action dim", self.action_dim

    self.lr = self.config.learning_rate
    self.replaysteps = self.config.replaysteps

    self.build()

  
  def add_placeholders_op(self):
    """
    Adds placeholders to the graph
    Set up the observation, action, and advantage placeholder
  
    TODO: add placeholders:
    self.observation_placeholder, type = tf.float32
    self.action_placeholder, type depends on the self.discrete
    self.advantage_placeholder, type = tf.float32
  
    HINT: In the case of a continuous action space, an action will be specified 
    by self.action_dim float32 numbers.  
    """
    #######################################################
    #########   YOUR CODE HERE - 8-12 lines.   ############
    self.observation_placeholder = tf.placeholder(tf.float32, shape=(None, self.replaysteps, self.observation_dim) )
    self.action_constraint_up_placeholder   = tf.placeholder(tf.float32, shape=(self.env._n_currency+1 ) )
    self.action_constraint_dn_placeholder   = tf.placeholder(tf.float32, shape=(self.env._n_currency+1 ) )
    if self.discrete:
      self.action_placeholder = tf.placeholder(tf.int64,   shape=(None))
    else:
      self.action_placeholder = tf.placeholder(tf.float32, shape=(None, self.action_dim))
  
    # Define a placeholder for advantages
    self.advantage_placeholder = tf.placeholder( tf.float32, shape=(None))
    #######################################################
    #########          END YOUR CODE.          ############
  
  
  def build_policy_network_op(self, scope = "policy_network"):
    """
    Build the policy network, construct the tensorflow operation to sample 
    actions from the policy network outputs, and compute the log probabilities
    of the taken actions (for computing the loss later). These operations are 
    stored in self.sampled_action and self.logprob. Must handle both settings
    of self.discrete.

    TODO:
    Discrete case:
        logits: the logits for each action
            HINT: use build_mlp
        self.sampled_action: sample from these logits
            HINT: use tf.multinomial + tf.squeeze
        self.logprob: compute the log probabilities of the taken actions
            HINT: 1. tf.nn.sparse_softmax_cross_entropy_with_logits computes 
                     the *negative* log probabilities of labels, given logits.
                  2. taken actions are different than sampled actions!

    Continuous case:
        To build a policy in a continuous action space domain, we will have the
        model output the means of each action dimension, and then sample from
        a multivariate normal distribution with these means and trainable standard
        deviation.

        That is, the action a_t ~ N( mu(o_t), sigma)
        where mu(o_t) is the network that outputs the means for each action 
        dimension, and sigma is a trainable variable for the standard deviations.
        N here is a multivariate gaussian distribution with the given parameters.

        action_means: the predicted means for each action dimension.
            HINT: use build_mlp
        log_std: a trainable variable for the log standard deviations.
        --> think about why we use log std as the trainable variable instead of std
        self.sampled_actions: sample from the gaussian distribution as described above
            HINT: use tf.random_normal
        self.lobprob: the log probabilities of the taken actions
            HINT: use tf.contrib.distributions.MultivariateNormalDiag

    """
    #######################################################
    #########   YOUR CODE HERE - 5-10 lines.   ############
    
    self.action_means =          build_mlp(mlp_input=self.observation_placeholder,
                                           output_size = self.action_dim,
                                           scope=scope,
                                           n_layers=self.config.n_layers, size=self.config.layer_size, output_activation=None)
    
    self.log_std =               tf.get_variable("logsigma", shape=[self.action_dim], dtype = tf.float32, initializer=tf.initializers.constant([-2,-2]) )

    self.sampled_action = tf.random_normal( shape=[self.action_dim], mean=self.action_means, stddev=tf.exp(self.log_std) )
    self.sampled_action = tf.clip_by_value(self.sampled_action, self.action_constraint_dn_placeholder, self.action_constraint_up_placeholder)
  
    mvn = tf.contrib.distributions.MultivariateNormalDiag(self.action_means, tf.exp(self.log_std) )
    self.logprob        =   mvn.log_prob(self.action_placeholder)
    #######################################################
    #########          END YOUR CODE.          ############
            
  
  
  def add_loss_op(self):
    """
    Compute the loss for a given batch. 
    Recall the update for REINFORCE with advantage:
    Think about how to express this update as minimizing a 
    loss (so that tensorflow will do the gradient computations
    for you).
    
    You only have to reference fields of self that you have
    already set in previous methods.

    """
    ######################################################
    #########   YOUR CODE HERE - 1-2 lines.   ############
    self.loss = -tf.reduce_mean(self.logprob * self.advantage_placeholder)
    #######################################################
    #########          END YOUR CODE.          ############
  
  
  def add_optimizer_op(self):
    """
    Sets the optimizer using AdamOptimizer
    TODO: Set self.train_op
    HINT: Use self.lr, and minimize self.loss
    """
    ######################################################
    #########   YOUR CODE HERE - 1-2 lines.   ############

    with tf.variable_scope("policy_network"):    
      var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "policy_network")
      adam_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
      self.train_op = adam_optimizer.minimize(self.loss) 

    #######################################################
    #########          END YOUR CODE.          ############
  
  
  def add_baseline_op(self, scope = "baseline"):
    """
    Build the baseline network within the scope

    In this function we will build the baseline network.
    Use build_mlp with the same parameters as the policy network to
    get the baseline estimate. You also have to setup a target
    placeholder and an update operation so the baseline can be trained.
    
    Args:
            scope: the scope of the baseline network
  
    TODO: Set 
    self.baseline,
        HINT: use build_mlp
    self.baseline_target_placeholder,
    self.update_baseline_op,
        HINT: first construct a loss. Use tf.losses.mean_squared_error.

    """
    ######################################################
    #########   YOUR CODE HERE - 4-8 lines.   ############

    with tf.variable_scope("baseline"):    
      self.baseline = tf.squeeze(build_mlp(mlp_input=self.observation_placeholder, output_size = 1, scope="baseline",
                                           n_layers=self.config.n_layers, size=self.config.layer_size, output_activation=None) )

      self.baseline_target_placeholder = tf.placeholder(tf.float32, shape=(None))

      loss = tf.losses.mean_squared_error( self.baseline_target_placeholder, self.baseline)
      self.baseline_loss = loss
      
      var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope)
      adam_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
      self.update_baseline_op = adam_optimizer.minimize(loss)
    #######################################################
    #########          END YOUR CODE.          ############
  
  def build(self):
    """
    Build model by adding all necessary variables

    You don't have to change anything here - we are just calling
    all the operations you already defined to build the tensorflow graph.
    """
  
    # add placeholders
    self.add_placeholders_op()
    # create policy net
    self.build_policy_network_op()
    # add square loss
    self.add_loss_op()
    # add optmizer for the main networks
    self.add_optimizer_op()
  
    if self.config.use_baseline:
      self.add_baseline_op()
  
  def initialize(self):
    """
    Assumes the graph has been constructed (have called self.build())
    Creates a tf Session and run initializer of variables

    You don't have to change or use anything here.
    """
    # create tf session
    self.sess = tf.Session()
    # tensorboard stuff
    self.add_summary()
    # initiliaze all variables
    init = tf.global_variables_initializer()
    self.sess.run(init)
  
  
  def add_summary(self):
    """
    Tensorboard stuff. 

    You don't have to change or use anything here.
    """
    # extra placeholders to log stuff from python
    self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
    self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
    self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")
  
    self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")
  
    # extra summaries from python -> placeholders
    tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
    tf.summary.scalar("Max Reward", self.max_reward_placeholder)
    tf.summary.scalar("Std Reward", self.std_reward_placeholder)
    tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)
            
    # logging
    self.merged = tf.summary.merge_all()
    self.file_writer = tf.summary.FileWriter(self.config.output_path,self.sess.graph) 

  def init_averages(self):
    """
    Defines extra attributes for tensorboard.

    You don't have to change or use anything here.
    """
    self.avg_reward = 0.
    self.max_reward = 0.
    self.std_reward = 0.
    self.eval_reward = 0.
  

  def update_averages(self, rewards, scores_eval):
    """
    Update the averages.

    You don't have to change or use anything here.
  
    Args:
            rewards: deque
            scores_eval: list
    """
    self.avg_reward = np.mean(rewards)
    self.max_reward = np.max(rewards)
    self.std_reward = np.sqrt(np.var(rewards) / len(rewards))
  
    if len(scores_eval) > 0:
      self.eval_reward = scores_eval[-1]
  
  
  def record_summary(self, t):
    """
    Add summary to tfboard

    You don't have to change or use anything here.
    """
  
    fd = {
      self.avg_reward_placeholder: self.avg_reward, 
      self.max_reward_placeholder: self.max_reward, 
      self.std_reward_placeholder: self.std_reward, 
      self.eval_reward_placeholder: self.eval_reward, 
    }
    summary = self.sess.run(self.merged, feed_dict=fd)
    # tensorboard stuff
    self.file_writer.add_summary(summary, t)
  
  
  def sample_path(self, env, sampling_date, num_episodes = None):
    """
    Sample path for the environment.
  
    Args:
            num_episodes:   the number of episodes to be sampled 
              if none, sample one batch (size indicated by config file)
    Returns:
        paths: a list of paths. Each path in paths is a dictionary with
            path["observation"] a numpy array of ordered observations in the path
            path["actions"] a numpy array of the corresponding actions in the path
            path["reward"] a numpy array of the corresponding rewards in the path
        total_rewards: the sum of all rewards encountered during this "path"

    Use 2017 data 
    Initialize to 1 BTC and 10k at each exchange
    """

    episode = 0
    episode_rewards = []
    paths = []
    t = 0

    while (num_episodes or t < self.config.batch_size):



      init_portfolio = np.array([ [10000, 1],
                                  [10000, 1], ],
                                dtype=np.float64)

      thisdate = (sampling_date + timedelta(episode)).strftime("%Y-%m-%d")

      state, _, _, _ = env.init( init_portfolio, thisdate)
      states, actions, rewards, constraints = [], [], [], []
      episode_reward = 0

      for step in range(self.config.max_ep_len):
        
        ### flatten the state array
        state_flatten = np.array([])

        portfolio_array = state[0]

        cash_1 = portfolio_array[0,0]/ np.max(state[1][:,:,-1])
        cash_2 = portfolio_array[1,0]/ np.max(state[1][:,:,-1])
        btc_1  = portfolio_array[0,1]
        btc_2  = portfolio_array[1,1]

        #buffering at 0.05 bitcoins
        cash_1 = (cash_1>0.05)* cash_1
        cash_2 = (cash_2>0.05)* cash_2
        btc_1 = (btc_1>0.05)* btc_1
        btc_2 = (btc_2>0.05)* btc_2


        cash_1 = ( state[1][0,:,-1]< state[1][1,:,-1] )* cash_1
        cash_2 = ( state[1][1,:,-1]< state[1][0,:,-1] )* cash_2

        cash_1_up_constraint = np.min( [cash_1, btc_2,  config.max_quantity_per_transaction*2]  )/2*(1-env._fee_transfer)*(1-env._fee_exchange )  ## minmum of cash_1 and BTC_2
        cash_1_dn_constraint = -np.min( [cash_2, btc_1, config.max_quantity_per_transaction*2]  )/2*(1-env._fee_transfer)*(1-env._fee_exchange )  ## minmum of cash_2 and BTC_1

        btc_1 = (portfolio_array[0,1]>portfolio_array[1,1]+1e-4)*btc_1
        btc_2 = (portfolio_array[1,1]>portfolio_array[0,1]+1e-4)*btc_2

        btc_1_up_constraint =  np.min([btc_1/2, config.max_quantity_per_transaction])*(1-env._fee_transfer)  #BTC_1
        btc_1_dn_constraint = -np.min([btc_2/2, config.max_quantity_per_transaction])*(1-env._fee_transfer)  #BTC_2

        
        portfolio_up_constraints = np.array([cash_1_up_constraint, btc_1_up_constraint])
        portfolio_dn_constraints = np.array([cash_1_dn_constraint, btc_1_dn_constraint])

        constraints.append(  (portfolio_up_constraints, portfolio_dn_constraints) )

                                          
        for isubarray in range(2):
          if config.use_only_price_info:
            if isubarray ==0:
              state_flatten = np.concatenate( (state_flatten, state[isubarray].flatten()) )
            else:
              state_flatten = np.concatenate( (state_flatten, state[isubarray][:,:,-1].flatten()) )
          else:
            state_flatten = np.concatenate( (state_flatten, state[isubarray].flatten()) )


        if step ==0:
          for r in range(self.replaysteps):
            states.append(state_flatten)

        else:
          states.append(state_flatten)


        action_accept = False
        n_failed_sampling = 0

        action_means = self.sess.run(self.action_means, feed_dict={self.observation_placeholder : [states[-self.replaysteps:]]})
        log_std      = self.sess.run(self.log_std, feed_dict={self.observation_placeholder : [states[-self.replaysteps:]]})
                                        
        while not action_accept:
          action  = self.sess.run(self.sampled_action, feed_dict={self.observation_placeholder : [states[-self.replaysteps:]],
                                                                  self.action_constraint_up_placeholder : portfolio_up_constraints,
                                                                  self.action_constraint_dn_placeholder : portfolio_dn_constraints,})[0]

          ## purchase action has last element constrained to balance other exchange activities
          purchase_action = action[0: self.action_purchase_dim]
          if ( np.sum(purchase_action)>0 ):
            purchase_action = np.append( purchase_action, -np.sum(purchase_action))*(1-env._fee_transfer) #to confirm 
          else:
            purchase_action = np.append( purchase_action, -np.sum(purchase_action))/(1-env._fee_transfer)
          
          transfer_flat = action[ self.action_purchase_dim:]
          transfer_action = np.zeros( (env._n_exchange, env._n_exchange, env._n_currency) )
          itransfer = -1

          for iccur  in range(env._n_currency ):
            for iex in range( env._n_exchange ):
              for jex in range( iex, env._n_exchange ):
                # transfer matrix diagonal always 0
                if iex == jex:
                  continue
                else:
                  itransfer +=1
                  transfer_action[ iex, jex,  iccur] = transfer_flat[itransfer]
                  transfer_action[ jex, iex,  iccur] = -transfer_flat[itransfer]

          action_topack = np.concatenate( (purchase_action, transfer_action.flatten())  )
          action_repacked = RepackActionForEnv(env, action_topack)

          ### repacking flat action array for environment
          try:
            state, reward, done, info = env.step(action_repacked)
            action_accept = True
          except AssertionError:
            n_failed_sampling += 1
            if n_failed_sampling>100:
              reward = 0
              done = True
              info = None
              action_accept = True
              print "give up at step ", step,len(paths), 
              
        actions.append(action)
        rewards.append(reward)
        episode_reward += reward
        t += 1
        if (done or step == self.config.max_ep_len-1):
          episode_rewards.append(episode_reward)  
          break
        if (not num_episodes) and t == self.config.batch_size:
          break

        if step % 500==0:
          print "sampling step", step, 
          print "constraint purchase:", round(cash_1_up_constraint,3), round(cash_1_dn_constraint,3),  "constraint transfer:", round(btc_1_up_constraint,3), round(btc_1_dn_constraint,3), 
          print "action", action, "means", action_means, "log_std", np.exp(log_std)
          print "final portfolio state", state[0].flatten()
        
      path = {"observation" : np.array(states), 
              "reward" : np.array(rewards), 
              "action" : np.array(actions)}

      paths.append(path)
      
      episode += 1

      if num_episodes and episode >= num_episodes:
        break        

    return paths, episode_rewards
  
  
  def get_returns(self, paths):
    """
    Calculate the returns G_t for each timestep
  
    Args:
      paths: recorded sampled path.  See sample_path() for details.
  
    After acting in the environment, we record the observations, actions, and
    rewards. 

    """

    all_returns = []
    for path in paths:

      rewards = path["reward"]
      #######################################################
      #########   YOUR CODE HERE - 5-10 lines.   ############

      path_returns = []

      for ir in range(len(rewards)):
        path_return = 0

        for jr in range(ir, len(rewards)):
          path_return += config.gamma**(jr-ir)*rewards[jr]

      
        path_returns.append(path_return)

      
      #######################################################
      #########          END YOUR CODE.          ############
      all_returns.append(path_returns)

    returns = np.concatenate(all_returns)
  
    return returns
  
  
  def calculate_advantage(self, returns, observations):
    """
    Calculate the advantage
    Args:
            returns: all discounted future returns for each step
            observations: observations
              Calculate the advantages, using baseline adjustment if necessary,
              and normalizing the advantages if necessary.
              If neither of these options are True, just return returns.

    TODO:
    If config.use_baseline = False and config.normalize_advantage = False,
    then the "advantage" is just going to be the returns (and not actually
    an advantage). 

    if config.use_baseline, then we need to evaluate the baseline and subtract
      it from the returns to get the advantage. 
      HINT: 1. evaluate the self.baseline with self.sess.run(...

    if config.normalize_advantage:
      after doing the above, normalize the advantages so that they have a mean of 0
      and standard deviation of 1.
  
    """
    adv = returns

    #######################################################
    #########   YOUR CODE HERE - 5-10 lines.   ############
    if self.config.use_baseline:

      baseline = self.sess.run(self.baseline, {self.observation_placeholder:observations})

      print "return and baseline shape", returns.shape, baseline.shape
      
      adv = adv - baseline

    if self.config.normalize_advantage:

      adv = (adv-np.mean(adv))/np.std(adv)

    #######################################################
    #########          END YOUR CODE.          ############

    print "advantages shape", adv.shape
    
    return adv
  
  
  def update_baseline(self, returns, observations):
    """
    Update the baseline

    TODO:
      apply the baseline update op with the observations and the returns.
    """
    #######################################################
    #########   YOUR CODE HERE - 1-5 lines.   ############
    self.sess.run(self.update_baseline_op,
                  feed_dict={self.observation_placeholder : observations,
                             self.baseline_target_placeholder: returns})

    #######################################################
    #########          END YOUR CODE.          ############
  
  
  def train(self):
    """
    Performs training

    You do not have to change or use anything here, but take a look
    to see how all the code you've written fits together!
    """
    last_eval = 0 
    last_record = 0
    scores_eval = []
    
    self.init_averages()
    scores_eval = [] # list of scores computed at iteration time

    start_date = date(2017, 1, 1)
    #print "start_date", start_date

  
    for t in range(self.config.num_batches):
      

      thisdate = (start_date + timedelta(t))
        
      # collect a minibatch of samples
      paths, total_rewards = self.sample_path(self.env, thisdate)

      #print paths, total_rewards

      #print "sampled epoch", t, "len of paths", len(paths)
      
      scores_eval = scores_eval + total_rewards
      observations = np.concatenate([path["observation"] for path in paths])
      actions = np.concatenate([path["action"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      # compute Q-val estimates (discounted future returns) for each time step
      returns = self.get_returns(paths)

      #print "observation shape", observations.shape
      #print observations
      observations_stack = []

      for ip in range(len(paths)):

        for io in range(paths[ip]["observation"].shape[0]-self.replaysteps+1):
          tmp_stack = []

          for r in range(self.replaysteps):
            
            tmp_stack.append( paths[ip]["observation"][ io+r  ] )

          observations_stack.append( tmp_stack )

      observations_stack = np.array(observations_stack)
      advantages = self.calculate_advantage(returns, observations_stack)

      # run training operations
      if self.config.use_baseline:
        self.update_baseline(returns, observations_stack)

      self.sess.run(self.train_op, feed_dict={
        self.observation_placeholder : observations_stack, 
        self.action_placeholder : actions,
        self.advantage_placeholder : advantages})

      loss = self.sess.run(self.loss, feed_dict={
        self.observation_placeholder : observations_stack, 
        self.action_placeholder : actions,
        self.advantage_placeholder : advantages})

      print "loss", loss
  
      # tf stuff
      if (t % self.config.summary_freq == 0):
        self.update_averages(total_rewards, scores_eval)
        self.record_summary(t)

      # compute reward statistics for this batch and log
      avg_reward = np.mean(total_rewards)
      sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
      msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
      self.logger.info(msg)
  
      if  self.config.record and (last_record > self.config.record_freq):
        self.logger.info("Recording...")
        last_record =0
        self.record()

      print "finished this iteration", t
  
    self.logger.info("- Training done.")
    export_plot(scores_eval, "Score", config.env_name, self.config.plot_output)


  def evaluate(self, env=None, num_episodes=1):
    """
    Evaluates the return for num_episodes episodes.
    Not used right now, all evaluation statistics are computed during training 
    episodes.
    """
    if env==None: env = self.env
    paths, rewards = self.sample_path(env, num_episodes)
    avg_reward = np.mean(rewards)
    sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
    msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
    self.logger.info(msg)
    return avg_reward
     
  
  def record(self):
     """
     Re create an env and record a video for one episode
     """
     env = gym.make(self.config.env_name)
     env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
     self.evaluate(env, 1)
  

  def run(self):
    """
    Apply procedures of training for a PG.
    """
    # initialize
    self.initialize()
    # record one game at the beginning
    if self.config.record:
        self.record()
    # model
    self.train()
    # record one game at the end
    if self.config.record:
      self.record()
          
if __name__ == '__main__':

  data_path = "../gym-rlcrptocurrency"
  markets = [
    [Market("{:s}/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv".format(data_path))],
    [Market("{:s}/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv".format(data_path))],
  ]
  
  env = gym.make(config.env_name)
  env.set_markets(markets)
  print "environment", env
  
  # train model
  model = PG(env, config)
  model.run()
