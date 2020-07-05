from tron.player import Player, Direction
from tron.game import Tile, Game, PositionPlayer
from tron.window import Window
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

import os
from tron.constant import *

# General parameters
folderName = 'basic'

# Net parameters
BATCH_SIZE = 128
GAMMA = 0.9 # Discount factor

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.batch_size = BATCH_SIZE
		self.gamma = GAMMA
		self.conv1 = nn.Conv2d(1, 32, 6)
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.fc1 = nn.Linear(64*(GameSize - 5)*(GameSize - 5), 512)
		self.fc2 = nn.Linear(512, 3)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.view(-1, 64*(GameSize - 5)*(GameSize - 5))
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


class Ai(Player):

	def __init__(self,epsilon=0):
		super(Ai, self).__init__()
		self.net = Net()
		self.epsilon = epsilon
		# Load network weights if they have been initialized already
		if os.path.isfile('ais/' + folderName + '/' + str(GameSize) + '_ai.bak'):
			self.net.load_state_dict(torch.load('ais/' + folderName + '/' + str(GameSize) + '_ai.bak'))

	"""
	def action(self, map, id):

		game_map = map.state_for_player(id)

		input = np.reshape(game_map, (1, 1, game_map.shape[0], game_map.shape[1]))
		input = torch.from_numpy(input).float()
		output = self.net(input)

		_, predicted = torch.max(output.data, 1)
		predicted = predicted.numpy()
		next_action = predicted[0] + 1

		if random.random() <= self.epsilon:
			next_action = random.randint(1,4)

		if next_action == 1:
			next_direction = Direction.UP
		if next_action == 2:
			next_direction = Direction.RIGHT
		if next_action == 3:
			next_direction = Direction.DOWN
		if next_action == 4:
			next_direction = Direction.LEFT

		return next_direction
	"""

	def action(self, map, last_direction, id):

		game_map = map.state_for_player(id)
		ind = np.unravel_index(np.argmax(game_map, axis=None), game_map.shape)

		start_direction = 0
		while (last_direction == None):
			start_direction = random.randint(1, 4)
			if start_direction == 1 and game_map[ind[0]-1][ind[1]] == 1:
				last_direction = Direction.UP
			elif start_direction == 2 and game_map[ind[0]][ind[1]+1] == 1:
				last_direction = Direction.RIGHT
			elif start_direction == 3 and game_map[ind[0]+1][ind[1]] == 1:
				last_direction = Direction.DOWN
			elif start_direction == 4 and game_map[ind[0]][ind[1]-1] == 1:
				last_direction = Direction.LEFT

		game_map[ind] += start_direction * 10

		input = np.reshape(game_map, (1, 1, game_map.shape[0], game_map.shape[1]))
		input = torch.from_numpy(input).float()
		output = self.net(input)

		_, predicted = torch.max(output.data, 1)
		predicted = predicted.numpy()
		next_action = predicted[0] + 1

		if random.random() <= self.epsilon:
			next_action = random.randint(1,3)

		if next_action == 1:
			next_direction = last_direction
		if next_action == 2:
			if last_direction == Direction.UP:
				next_direction = Direction.RIGHT
			elif last_direction == Direction.RIGHT:
				next_direction = Direction.DOWN
			elif last_direction == Direction.DOWN:
				next_direction = Direction.LEFT
			elif last_direction == Direction.LEFT:
				next_direction = Direction.UP
		if next_action == 3:
			if last_direction == Direction.UP:
				next_direction = Direction.LEFT
			elif last_direction == Direction.RIGHT:
				next_direction = Direction.UP
			elif last_direction == Direction.DOWN:
				next_direction = Direction.RIGHT
			elif last_direction == Direction.LEFT:
				next_direction = Direction.DOWN

		return next_direction, next_action

