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
folderName = 'survivor'

# Net parameters
BATCH_SIZE = 128
GAMMA = 0.9 # Discount factor

# Exploration parameters
EPSILON_START = 1
EPSILON_END = 0.05
DECAY_RATE = 0.999


# Memory parameters
MEM_CAPACITY = 10000

Transition = namedtuple('Transition',('old_state', 'action', 'new_state', 'reward', 'terminal'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.batch_size = BATCH_SIZE
		self.gamma = GAMMA
		self.conv1 = nn.Conv2d(1, 32, 6)
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.fc1 = nn.Linear(64*(GameSize - 5)*(GameSize - 5), 512)
		self.fc2 = nn.Linear(512, 4)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.view(-1, 64*(GameSize - 5)*(GameSize - 5))
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


class Ai(Player):


	def __init__(self):
		super(Ai, self).__init__()

		# Initialize exploration rate
		epsilon = EPSILON_START
		epsilon_temp = float(epsilon)

		self.net = Net()
		self.epsilon = epsilon


		# Initialize neural network parameters and optimizer
		self.optimizer = optim.Adam(self.net.parameters())
		self.criterion = nn.MSELoss()

		# Initialize memory
		self.memory = ReplayMemory(MEM_CAPACITY)

		# Load network weights if they have been initialized already
		if os.path.isfile('ais/' + folderName + '/' + str(GameSize) + '_ai.bak'):
			self.net.load_state_dict(torch.load('ais/' + folderName + '/' + str(GameSize) + '_ai.bak'))

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

		self.decay_epsilon()

		return next_direction


	def decay_epsilon(self):
		# Update exploration rate
		nouv_epsilon = self.epsilon * DECAY_RATE
		if nouv_epsilon > EPSILON_END:
			self.epsilon = nouv_epsilon

	def train(self, game):

		terminal = False

		# Get the initial state for each player
		old_state_p1 = game.history[0].map.state_for_player(1)
		old_state_p1 = np.reshape(old_state_p1, (1, 1, old_state_p1.shape[0], old_state_p1.shape[1]))
		old_state_p1 = torch.from_numpy(old_state_p1).float()
		old_state_p2 = game.history[0].map.state_for_player(2)
		old_state_p2 = np.reshape(old_state_p2, (1, 1, old_state_p2.shape[0], old_state_p2.shape[1]))
		old_state_p2 = torch.from_numpy(old_state_p2).float()

		for historyStep in range(len(game.history) - 1):

			# Get the state for each player
			new_state_p1 = game.history[historyStep + 1].map.state_for_player(1)
			new_state_p1 = np.reshape(new_state_p1, (1, 1, new_state_p1.shape[0], new_state_p1.shape[1]))
			new_state_p1 = torch.from_numpy(new_state_p1).float()
			new_state_p2 = game.history[historyStep + 1].map.state_for_player(2)
			new_state_p2 = np.reshape(new_state_p2, (1, 1, new_state_p2.shape[0], new_state_p2.shape[1]))
			new_state_p2 = torch.from_numpy(new_state_p2).float()

			# Get the action for each player
			if game.history[historyStep].player_one_direction is not None:
				action_p1 = torch.from_numpy(
					np.array([game.history[historyStep].player_one_direction.value - 1], dtype=np.float32)).unsqueeze(0)
				action_p2 = torch.from_numpy(
					np.array([game.history[historyStep].player_two_direction.value - 1], dtype=np.float32)).unsqueeze(0)
			else:
				action_p1 = torch.from_numpy(np.array([0], dtype=np.float32)).unsqueeze(0)
				action_p2 = torch.from_numpy(np.array([0], dtype=np.float32)).unsqueeze(0)

			# Compute the reward for each player
			reward_p1 = +1
			reward_p2 = +1
			if historyStep + 1 == len(game.history) - 1:
				if game.winner is None:
					reward_p1 = 0
					reward_p2 = 0
				elif game.winner == 1:
					reward_p1 = 100
					reward_p2 = -25
				else:
					reward_p1 = -25
					reward_p2 = 100
				terminal = True

			reward_p1 = torch.from_numpy(np.array([reward_p1], dtype=np.float32)).unsqueeze(0)
			reward_p2 = torch.from_numpy(np.array([reward_p2], dtype=np.float32)).unsqueeze(0)

			# Save the transition for each player
			self.memory.push(old_state_p1, action_p1, new_state_p1, reward_p1, terminal)

			# Update old state for each player
			old_state_p1 = new_state_p1
			old_state_p2 = new_state_p2

		# Get a sample for training
		transitions = self.memory.sample(min(len(self.memory), self.net.batch_size))
		batch = Transition(*zip(*transitions))
		old_state_batch = torch.cat(batch.old_state)
		action_batch = torch.cat(batch.action).long()
		new_state_batch = torch.cat(batch.new_state)
		reward_batch = torch.cat(batch.reward)

		# Compute predicted Q-values for each action
		pred_q_values_batch = torch.sum(self.net(old_state_batch).gather(1, action_batch), dim=1)
		pred_q_values_next_batch = self.net(new_state_batch)

		# Compute targeted Q-value for action performed
		target_q_values_batch = torch.cat(tuple(reward_batch[i] if batch[4][i]
												else reward_batch[i] + self.net.gamma * torch.max(pred_q_values_next_batch[i])
												for i in range(len(reward_batch))))

		# zero the parameter gradients
		self.net.zero_grad()

		# Compute the loss
		target_q_values_batch = target_q_values_batch.detach()
		loss = self.criterion(pred_q_values_batch, target_q_values_batch)

		# Do backward pass
		loss.backward()
		self.optimizer.step()

		return loss


	def save_model(self):
		# Update bak
		torch.save(self.net.state_dict(), 'ais/' + folderName + '/' + str(GameSize) + '_ai.bak')
