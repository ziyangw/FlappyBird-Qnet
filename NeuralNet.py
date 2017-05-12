from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

import numpy as np
import random

first = True
count =  0
PIPEGAPSIZE  = 100 
BIRDHEIGHT = 24
MAXYVEL = 9


class NeuralNet(object):
	"""docstring for NeuralNet"""

	def __init__(self, num_inputs, num_hidden1, num_hidden2, num_output, record_size, sub_record, gamma):
		super(NeuralNet, self).__init__()
		self.num_inputs = num_inputs
		self.num_hidden1 = num_hidden1
		self.num_hidden2 = num_hidden2
		self.num_output = num_output
		self.gamma = gamma
		self.record_size = record_size
		self.sub_record = sub_record
		self.record = []
		self.record_idx = 0
		self.build()

	def build(self):
		model = Sequential()
		model.add(Dense(self.num_hidden1, init='lecun_uniform', input_shape=(self.num_inputs,)))
		model.add(Activation('relu'))

		model.add(Dense(self.num_hidden2, init='lecun_uniform'))
		model.add(Activation('relu'))

		model.add(Dense(self.num_output, init='lecun_uniform'))
		model.add(Activation('linear'))

		rms = RMSprop()
		model.compile(loss='mse', optimizer=rms)
		self.model = model


	def fly(self, playerx, playery, lowerPipes, pipeW, playerW):
		# Find closest pipes
		self.pipeW = pipeW
		self.playerW = playerW
		for i in range(len(lowerPipes)):
			if ((lowerPipes[i]['x'] + pipeW) >= (playerx - playerW)):
				self.x, self.y = lowerPipes[i]['x'], lowerPipes[i]['y']
				self.distX = (self.x + pipeW) - playerx
				self.distY = self.y - playery
				break
		self.playery = playery
		self.old_state = np.array([self.distX, self.distY])

		# Forward Prop
		self.curr_output = self.model.predict(self.old_state.reshape(1, self.num_inputs), batch_size=1)[0]
		# Chose Max
		self.choice = (0 if (self.curr_output[0] > self.curr_output[1]) else 1)
		randVal = random.random()

		return (self.choice, self.curr_output)


	def update(self, crash, scored, playery, pipVelX):
		global count
		last_reward = 0
		new_distX = self.distX + pipVelX
		new_distY = self.y - playery
		new_state = np.array([new_distX, new_distY])

		
		if crash: last_reward -= 10000
		elif (self.choice == 1) and (new_distY >= PIPEGAPSIZE/2): last_reward -= 10000
		elif (self.choice == 0) and (new_distY <= BIRDHEIGHT + MAXYVEL): last_reward -= 10000
		else: last_reward += 100

		new_predict = self.model.predict(new_state.reshape(1, self.num_inputs), batch_size = 1)
		if (crash):
			self.curr_output[self.choice] = last_reward
		else:
			self.curr_output[self.choice] =  last_reward + self.gamma * np.max(new_predict)
		if (len(self.record) < self.record_size):
			self.record.append((self.curr_output, self.old_state))
		else:
			self.record[self.record_idx] = (self.curr_output, self.old_state)
			self.record_idx = (self.record_idx + 1) % self.record_size

		if (len(self.record) >= self.sub_record):
			# Back Prop 
			subset = random.sample(self.record, self.sub_record)
			X = []
			Y = []
			for y, x in subset:
				X.append(x) # Old State
				Y.append(y) # New curr_output
			X = np.array(X)
			Y = np.array(Y)
			self.model.fit(X, Y, batch_size=self.sub_record, nb_epoch=1, verbose=0)
		









