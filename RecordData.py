import numpy as np
import csv

class RecordData:

	def __init__(self, filename, append = False):
		self.fn = filename
		self.csv_states = list()
		if not append:
			self.test_file_exists()


	def test_file_exists(self):
		try:
			f = open(self.fn, 'wb')
			f.close()
			print('File existed.  Cleared old file.')
		except:
			print('File did not exist. Creating new file.')

	def csv_state_reset(self):
		self.csv_states = list()

	def append(self, pre_obs, action, reward, post_obs):
		csv_state_line = np.append(np.append(np.append(pre_obs, action), reward), post_obs)
		self.csv_states.append(csv_state_line)
		# print(csv_state_line)

	def write(self, break_str = None):
		with open(self.fn, 'a') as csvfile:
			writer = csv.writer(csvfile, delimiter = ' ')
			for line in self.csv_states:
				writer.writerow(line)
			if break_str is not None:
				writer.writerow(break_str)