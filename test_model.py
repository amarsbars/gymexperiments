
from keras.models import load_model
import gym
import theano
import csv
import numpy as np
import pdb
import matplotlib.pyplot as plt




test_model_names = ['LayerSwappingEnvReplay_0.hd5', 'LayerSwappingEnvReplay_1.hd5', 'LayerSwappingEnvReplay_2.hd5']
test_env_names = ['Pendulum-v0', 'PendulumLong-v0', 'PendulumHeavy-v0']

exp_model_names = ['Pendulum_Simple.hd5', 'Pendulum_Long.hd5', 'Pendulum_Heavy.hd5']


episodes = 200
max_timesteps = 200
render = False
Collect = False
Plot = True

# getting a bunch of starting positions
observation_start = []
env = gym.make(test_env_names[0]) 
for i_eps in xrange(episodes):
	observation_start.append(env.reset())
if Collect:
	for i_env in range(len(test_model_names)):
		for i_ex in range(2):
			env = gym.make(test_env_names[i_env]) # load correct environment
			if i_ex == 0:
				model = load_model(test_model_names[i_env]) # load correct general model
			else:
				model = load_model(exp_model_names[i_env]) # load correct expert model
			model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
			if i_ex == 0:
				model_action = theano.function([model.layers[0].input], model.layers[7].output, allow_input_downcast=True)
			else:
				model_action = theano.function([model.layers[0].input], model.layers[5].output, allow_input_downcast=True)
			
			# run in environment
			reward_hist = []
			for i_eps in xrange(episodes):
				# observation = env.reset()
				observation = observation_start[i_eps]
				env.state = np.array((np.arctan2(observation[0], observation[1]), observation[2]))
				episode_reward = 0
				for t in xrange(max_timesteps):
					if render:
						env.render()
					x = np.array([observation])
					if i_ex == 0:
						x = np.append(x, i_env).reshape(1,-1)
					u = model_action(x)[0]
					obs, reward, done, info = env.step(u)
					episode_reward += reward
				reward_hist.append(episode_reward)
				print('Env: %s, Iteration: %s, Reward: %s' %(i_env, i_eps, episode_reward))
			if i_ex == 0:
				fn = test_model_names[i_env].split('.')[0] + '_test.csv'
			else:
				fn = exp_model_names[i_env].split('.')[0] + '_test.csv'
			with open(fn, 'wb') as csv_file:
				writer = csv.writer(csv_file, delimiter = ' ')
				writer.writerow(reward_hist)







if Plot:
	counter  = 0
	for i_env in range(len(test_model_names)):
		#### Making it Pretty ######
		# These are the "Tableau 20" colors as RGB.    
		tableau20 = [(31, 119, 180), (255, 127, 14), (255, 187, 120),    
		         (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
		         (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
		         (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
		         (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    

		# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
		for i in range(len(tableau20)):    
			r, g, b = tableau20[i]    
			tableau20[i] = (r / 255., g / 255., b / 255.)  

		plt.figure(figsize = (12, 9))
		# Remove the plot frame lines. They are unnecessary chartjunk.    
		ax = plt.subplot(111)    
		ax.spines["top"].set_visible(False)    
		ax.spines["bottom"].set_visible(False)    
		ax.spines["right"].set_visible(False)    
		ax.spines["left"].set_visible(False)    

		# Ensure that the axis ticks only show up on the bottom and left of the plot.    
		# Ticks on the right and top of the plot are generally unnecessary chartjunk.    
		ax.get_xaxis().tick_bottom()    
		ax.get_yaxis().tick_left()  

		plt.yticks(fontsize=14)    
		plt.xticks(fontsize=14)    

		plt.title('Comparison of Performance in Different Environments')
		plt.ylabel('Reward for Episode')
		plt.xlabel('Epoch')
		for i_ex in range(2):
			reward_hist = []
			if i_ex == 0:
				fn = test_model_names[i_env].split('.')[0] + '_test.csv'
			else:
				fn = exp_model_names[i_env].split('.')[0] + '_test.csv'
			with open(fn, 'rb') as csv_file:
				reader = csv.reader(csv_file, delimiter = ' ')
				for row in reader:
					for val in row:
						reward_hist.append(val)

			N = 20
			# pdb.set_trace()
			running_mean1 = np.convolve(np.array(reward_hist).astype('float'), np.ones((N,))/N, mode = 'valid')
			if i_ex == 0:
				plt.plot(running_mean1, label = test_model_names[i_env].split('.')[0], color = tableau20[counter])
			else:
				plt.plot(running_mean1, label = exp_model_names[i_env].split('.')[0], color = tableau20[counter])
			counter += 1
		plt.legend()
		plt.show()
plt.legend()
plt.show()






