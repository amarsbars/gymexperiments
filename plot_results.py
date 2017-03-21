import csv
import matplotlib.pyplot as plt
import pdb
import numpy as np


episode_rewards_exp = []
episode_rewards_none = []
episode_rewards_env = []
episode_rewards_multi = []
episode_rewards_data = []

with open('ExperienceReplay_episodeRs.csv') as csv_file:
	reader = csv.reader(csv_file, delimiter = ' ')
	for row in reader:
		for val in row:
			episode_rewards_exp.append(float(val))


with open('NoExperienceReplay_episodeRs.csv') as csv_file:
	reader = csv.reader(csv_file, delimiter = ' ')
	for row in reader:
		for val in row:
			episode_rewards_none.append(float(val))

with open('EnvExperienceReplay_episodeRs.csv') as csv_file:
	reader = csv.reader(csv_file, delimiter = ' ')
	for row in reader:
		for val in row:
			episode_rewards_env.append(float(val))

with open('MultiEnvExperienceReplay_episodeRs.csv') as csv_file:
	reader = csv.reader(csv_file, delimiter = ' ')
	for row in reader:
		for val in row:
			episode_rewards_multi.append(float(val))

with open('Pendulum_general_results_env.csv') as csv_file:
	reader = csv.reader(csv_file, delimiter = ' ')
	for row in reader:
		for val in row:
			val = val.strip('[]').split(',')
			for v in val:
				episode_rewards_data.append(float(v))


episode_rewards_exp = np.array(episode_rewards_exp)
episode_rewards_none = np.array(episode_rewards_none)
episode_rewards_env = np.array(episode_rewards_env)
episode_rewards_multi = np.array(episode_rewards_multi)
episode_rewards_data = np.array(episode_rewards_data)
# indxs_o = range(len(episode_rewards))
# indxs = indxs_o[0::200]
# indxs2 = indxs_o[1::200]
N = 20
running_mean1 = np.convolve(episode_rewards_exp, np.ones((N,))/N, mode = 'valid')
running_mean2 = np.convolve(episode_rewards_none, np.ones((N,))/N, mode = 'valid')
running_mean3 = np.convolve(episode_rewards_env, np.ones((N,))/N, mode = 'valid')
running_mean4 = np.convolve(episode_rewards_multi, np.ones((N,))/N, mode = 'valid')
running_mean5 = np.convolve(episode_rewards_data, np.ones((N,))/N, mode = 'valid')

# running_mean4 = episode_rewards_multi

  
# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
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

plt.title('Comparison of Training Rates')
plt.ylabel('Reward for Episode')
plt.xlabel('Epoch')
# plt.plot(running_mean1, color = tableau20[0], label = 'Expert Replay Buffer')
# plt.plot(running_mean2, color = tableau20[3], label = 'No Replay Buffer')
# plt.plot(running_mean5, color = tableau20[10], label = 'Data Only')
plt.plot(running_mean3, color = tableau20[6], label = 'Expert Replay Buffer with Env')
plt.plot(running_mean4, color = tableau20[10], label = 'Expert Replay Buffer with MultiEnv')

plt.legend()
plt.show()