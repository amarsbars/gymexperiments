import argparse
import gym
from gym.spaces import Box, Discrete
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, l1
from keras.constraints import maxnorm, unitnorm
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import theano.tensor as T
import numpy as np
from buffer import Buffer
import RecordData as recordData
import csv
import pdb
from keras.callbacks import History
from keras.models import load_model
import matplotlib.pyplot as plt
from theano import shared
import sys
import theano

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=200)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--batch_norm', action="store_true", default=False)
parser.add_argument('--no_batch_norm', action="store_false", dest="batch_norm")
parser.add_argument('--max_norm', type=int)
parser.add_argument('--unit_norm', action='store_true', default=False)
parser.add_argument('--l2_reg', type=float)
parser.add_argument('--l1_reg', type=float)
parser.add_argument('--replay_size', type=int, default=100000)
parser.add_argument('--train_repeat', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--max_timesteps', type=int, default=200)
parser.add_argument('--activation', choices=['tanh', 'relu'], default='relu')
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
parser.add_argument('--optimizer_lr', type=float, default=0.001)
parser.add_argument('--noise', choices=['linear_decay', 'exp_decay', 'fixed', 'covariance'], default='linear_decay')
parser.add_argument('--noise_scale', type=float, default=0.01)
parser.add_argument('--display', action='store_true', default=True)
parser.add_argument('--no_display', dest='display', action='store_false')
parser.add_argument('--feature_norm', action='store_false', default=False)
parser.add_argument('--gym_record')
parser.add_argument('environment')
args = parser.parse_args()

assert K._BACKEND == 'theano', "only works with Theano as backend"

# create environment
env = gym.make('Pendulum-v0')
assert isinstance(env.observation_space, Box), "observation space must be continuous"
assert isinstance(env.action_space, Box), "action space must be continuous"
assert len(env.action_space.shape) == 1
num_actuators = env.action_space.shape[0]
print "num_actuators:", num_actuators

# start monitor for OpenAI Gym
if args.gym_record:
  env.monitor.start(args.gym_record)

# optional norm constraint
if args.max_norm:
  W_constraint = maxnorm(args.max_norm)
elif args.unit_norm:
  W_constraint = unitnorm()
else:
  W_constraint = None

# optional regularizer
def regularizer():
  if args.l2_reg:
    return l2(args.l2_reg)
  elif args.l1_reg:
    return l1(args.l1_reg)
  else:
    return None

# helper functions to use with layers
if num_actuators == 1:
  # simpler versions for single actuator case
  def _L(x):
    return K.exp(x)

  def _P(x):
    return x**2

  def _A(t):
    m, p, u = t
    return -(u - m)**2 * p

  def _Q(t):
    v, a = t
    return v + a

else:
  # use Theano advanced operators for multiple actuator case
  def _L(x):
    # initialize with zeros
    batch_size = x.shape[0]
    a = T.zeros((batch_size, num_actuators, num_actuators))
    # set diagonal elements
    batch_idx = T.extra_ops.repeat(T.arange(batch_size), num_actuators)
    diag_idx = T.tile(T.arange(num_actuators), batch_size)
    b = T.set_subtensor(a[batch_idx, diag_idx, diag_idx], T.flatten(T.exp(x[:, :num_actuators])))
    # set lower triangle
    cols = np.concatenate([np.array(range(i), dtype=np.uint) for i in xrange(num_actuators)])
    rows = np.concatenate([np.array([i]*i, dtype=np.uint) for i in xrange(num_actuators)])
    cols_idx = T.tile(T.as_tensor_variable(cols), batch_size)
    rows_idx = T.tile(T.as_tensor_variable(rows), batch_size)
    batch_idx = T.extra_ops.repeat(T.arange(batch_size), len(cols))
    c = T.set_subtensor(b[batch_idx, rows_idx, cols_idx], T.flatten(x[:, num_actuators:]))
    return c

  def _P(x):
    return K.batch_dot(x, K.permute_dimensions(x, (0,2,1)))

  def _A(t):
    m, p, u = t
    d = K.expand_dims(u - m, -1)
    return -K.batch_dot(K.batch_dot(K.permute_dimensions(d, (0,2,1)), p), d)

  def _Q(t):
    v, a = t
    return v + a

  def _ind(t):
    x_i, ind = t
    return K.concatenate(x_i, ind, axis = 1)

# helper function to produce layers twice
def createLayers():
  # x = Input(shape=(env.observation_space.shape[0] + 1, ), name='x')
  x = Input(shape=(env.observation_space.shape[0], ), name='x')
  u = Input(shape=env.action_space.shape, name='u')
  if args.batch_norm:
    h = BatchNormalization()(x)
  else:
    h = x
  for i in xrange(args.layers):
    h = Dense(args.hidden_size, activation=args.activation, name='h'+str(i+1),
        W_constraint=W_constraint, W_regularizer=regularizer())(h)
    if args.batch_norm and i != args.layers - 1:
      h = BatchNormalization()(h)
  v = Dense(1, name='v', W_constraint=W_constraint, W_regularizer=regularizer())(h)
  m = Dense(num_actuators, name='m', W_constraint=W_constraint, W_regularizer=regularizer())(h)
  l0 = Dense(num_actuators * (num_actuators + 1)/2, name='l0',
        W_constraint=W_constraint, W_regularizer=regularizer())(h)
  l = Lambda(_L, output_shape=(num_actuators, num_actuators), name='l')(l0)
  p = Lambda(_P, output_shape=(num_actuators, num_actuators), name='p')(l)
  a = merge([m, p, u], mode=_A, output_shape=(num_actuators,), name="a")
  if args.feature_norm:
    q = merge([v, a, i], mode=_Q, output_shape=(num_actuators,), name="q")
  else:
    q = merge([v, a], mode=_Q, output_shape=(num_actuators,), name="q")
  return x, u, m, v, q, p, a

def custom_objective(y_true, y_pred):
    mse_loss = K.mean(K.square(y_pred - y_true), axis=-1)
    expert_models = list()
    e_mod = load_model('Pendulum_Simple.hd5')
    w_e = e_mod.get_weights()
    pdb.set_trace()
    expert_models.append(e_mod)

    return K.mean(y_true - y_pred)


x, u, m, v, q, p, a = createLayers()
# wrappers around computational graph
fmu = K.function([K.learning_phase(), x], m)
mu = lambda x: fmu([0, x])

fP = K.function([K.learning_phase(), x], p)
P = lambda x: fP([0, x])

fA = K.function([K.learning_phase(), x, u], a)
A = lambda x, u: fA([0, x, u])

fQ = K.function([K.learning_phase(), x, u], q)
Q = lambda x, u: fQ([0, x, u])

# V() function uses target model weights
fV = K.function([K.learning_phase(), x], v)
V = lambda x: fV([0, x])

# main model
model = Model(input=[x,u], output=q)
model.summary()
# pdb.set_trace()
# new_weights = model.layers[8].get_weights()
# new_weights[0] = new_weights[0] * 0
# model.layers[8].set_weights(new_weights)

if args.optimizer == 'adam':
  optimizer = Adam(args.optimizer_lr)
elif args.optimizer == 'rmsprop':
  optimizer = RMSprop(args.optimizer_lr)
else:
  assert False

if args.feature_norm:
  f_loss = custom_objective
else:
  f_loss = 'mse'
  # f_loss = 'msle'
  # f_loss = 'kullback_leibler_divergence'

model.compile(optimizer=optimizer, loss=f_loss, metrics=['mean_squared_error'])
# pdb.set_trace()
# K.printing.pydotprint(model, 'Network.png')
# # another set of layers for target model
# x, u, m, v, q, p, a = createLayers()

# # target model is initialized from main model
# target_model = Model(input=[x,u], output=q)
# target_model.set_weights(model.get_weights())

savefn = 'Pendulum_General'

########### LOAD TRAINING DATA ##############
try:
  saved_model = load_model(savefn+'.hd5')
  saved_weights = saved_model.get_weights()
  model.set_weights(saved_weights)
  print("Old Model Loaded")
except:
  filenames = ['Pendulum_Simple.csv'] #['Pendulum_Long.csv', 'Pendulum_simple.csv', 'Pendulum_Heavy.csv'] 
  f_id = 0
  for fn in filenames:
    with open(fn, 'rb') as csv_file:
      load_data = np.genfromtxt(csv_file, dtype = 'float')
      try:
        all_data = np.append(all_data, load_data, axis = 0)
        model_id = np.append(model_id, np.full(len(all_data), f_id), axis = 0)
      except:
        all_data = load_data
        model_id = np.full(len(all_data), f_id)
    f_id += 1

  history = History
  pre_state = all_data[:,0:3]
  action = all_data[:,3]
  reward = all_data[:,4]
  post_state = all_data[:,5:]
  # pdb.set_trace()
  plt.ion()
  plt.plot(pre_state[-1000:,0], label = 'cos')
  plt.plot(pre_state[-1000:,1], label = 'sin')
  plt.legend()
  plt.show()

  ############## LOAD EXPERT WEIGHTS ################

  # do i need to do the thing with calculating V and adding it to the outputs?
  # and then train on the resulting data like they do online? - didn't work just training the network

  ########### TRAINING LOOP ###############
  # perform Q-updates
  h = []
  loss = []
  iteration_count = 200 * 200 #200 episodes * 200 timesteps
  learning_rate = 0.999
  # start_index = int(len(all_data) * 0.8) # only take good experience
  start_index = 0
  for k1 in xrange(iteration_count):
    for k2 in xrange(args.train_repeat): #10 times on each iteration
      # shortening by one because no preobs for first step (or no post obs for last step)
      rand_indx = np.random.choice(len(all_data) - start_index, 10)
      rand_indx += start_index
      preobs = all_data[rand_indx,0:3]
      actions = all_data[rand_indx,3]
      rewards = all_data[rand_indx,4]
      postobs = all_data[rand_indx, 5:]
      m_id = model_id[rand_indx]


      # postobs = np.concatenate((postobs, m_id.reshape(-1, 1)), axis = 1)
      # preobs = np.concatenate((preobs, m_id.reshape(-1, 1)), axis = 1)

      # Q-update
      v = V(postobs) # need to compare predicted value with ideal reward
      y = rewards + args.gamma * np.squeeze(v)
      # train model
      h = model.train_on_batch([preobs, actions], y)
      loss.append(h[0])
      #update learning rate
    # new_lr = np.max((0.000001, np.array(model.optimizer.lr.get_value() * learning_rate).astype('float32'))).astype('float32')
    # model.optimizer.lr.set_value(new_lr)
    print('Iterations: %s, Learning Rate: %s, Loss: %s \r' %(k1, model.optimizer.lr.get_value(), loss[-1])),
    sys.stdout.flush()
      # if k % 10000 == 0:
  plt.figure(10)
  # plt.plot(loss[300:])
  plt.semilogy(loss)
  plt.title('Loss - Mean Squared Error')
  plt.show()
  # pdb.set_trace()
  plt.pause(1)

  with open(savefn + '_results_training.csv', 'wb') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow(loss)
      
  model.save(savefn + '.hd5')
########## TEST ON ENVIRONMENTS ##############
env_names = ['Pendulum-v0'] #['PendulumLong-v0', 'Pendulum-v0', 'PendulumHeavy-v0'] #, 'PendulumLongHeavy-v0'
exp_networks = ['Pendulum_Simple.hd5', 'Pendulum_Simple.hd5', 'Pendulum_Heavy.hd5']#['Pendulum_Long.hd5', 'Pendulum_Simple.hd5', 'Pendulum_Heavy.hd5']
# run once with expert and once with general network
exp_reward = []
gen_reward = []
u_dif = []
for i_env in range(len(env_names)):
  observation_start = np.full((args.episodes, env.observation_space.shape[0]),env.reset())
  # [observation_start.append(env.reset()) for i_episodes in xrange(args.episodes)]
  for i_ex in range(1,2):
    if i_ex == 0:
      #load expert network
      exp_model = load_model(exp_networks[i_env])
      exp_weights = exp_model.get_weights()
      exp_model.compile(optimizer=optimizer, loss=f_loss, metrics=['mean_squared_error'])
      exp_activations = theano.function([exp_model.layers[0].input], exp_model.layers[5].output, allow_input_downcast=True)
      print('Expert Network Loaded')
    else:
      gen_model = load_model(savefn + '.hd5')
      gen_weights = gen_model.get_weights()
      model.set_weights(gen_weights)
      print('General Network Loaded')

    total_reward = 0
    reward_hist = []
    for i_episode in xrange(args.episodes):
        observation = env.reset()
        #if you want consistent starting position
        # observation = observation_start[i_episode]
        # env.state = np.array((np.arctan2(observation[0], observation[1]), observation[2]))

        #print "initial state:", observation
        episode_reward = 0
        for t in xrange(args.max_timesteps):
            if args.display:
              env.render()

            # predict the mean action from current observation
            
            x = np.array([observation])
            # if i_ex == 0: # with envId
            #   x = np.squeeze(x).reshape(1, -1)
            #   # pdb.set_trace()
            #   u = exp_activations(x)
            # else:
            #   # x = np.concatenate((x, i_env), axis = 1)
            #   x = np.append(x, i_env).reshape(1,-1)
            #   u = mu(x)[0]

            if i_ex == 0: #without env ID
              x_ex = np.squeeze(x).reshape(1, -1)
              u_ex = exp_activations(x_ex)
            else:
              # x_gen = np.concatenate((x, np.array(i_env).reshape(1,-1)), axis = 1)
              # x_gen = np.append(x, i_env).reshape(1,-1)
              # x_gen = x
              # u_gen = mu(x_gen)[0]
              u_gen = mu(x)[0]
              # u_dif.append(u_ex - u_gen)



            #print "action:", action, "Q:", Q(x, np.array([action])), "V:", V(x)
            #print "action:", action, "advantage:", A(x, np.array([action]))
            #print "mu:", u, "action:", action
            #print "Q(mu):", Q(x, np.array([u])), "Q(action):", Q(x, np.array([action]))

            # take the action and record reward
            # observation, reward, done, info = env.step(u)
            if i_ex == 0:
              observation, reward, done, info = env.step(u_ex)
            else:
              observation, reward, done, info = env.step(u_gen)

            episode_reward += reward
            #print "reward:", reward
            #print "poststate:", observation

            if done:
                break

        print "Episode {} finished after {} timesteps, reward {}".format(i_episode + 1, t + 1, episode_reward)
        total_reward += episode_reward
        reward_hist.append(episode_reward)
    if i_ex == 0:
      exp_reward.append(reward_hist)
    else:
      gen_reward.append(reward_hist)
# for i in range(len(exp_reward)):
  # plt.figure()
  # plt.plot(exp_reward[i], 'r')
  # plt.plot(gen_reward[i], 'b')
  # plt.title(env_names[i])

# plt.figure()
# plt.plot(np.array(u_dif).squeeze(), 'ro')
# plt.title('Difference In U')
# plt.show()
# plt.pause(0)
  # if i_env == 1:
  #   write_mode = 'wb'
  # else:
  #   write_mode = 'a'
with open(savefn + '_results_env.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file, delimiter =  ' ')
    # writer.writerow(env_names[i_env])
    writer.writerow(gen_reward)



# print "Average reward per episode {}".format(total_reward / args.episodes)
if args.gym_record:
  env.monitor.close()
