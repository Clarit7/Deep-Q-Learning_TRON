from tron.player import Player, Direction
from tron.game import Tile, Game, PositionPlayer
from tron.window import Window
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from tron.minimax import MinimaxPlayer
from tron import resnet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import visdom

import os


# General parameters
folderName = 'survivor'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Net parameters
BATCH_SIZE = 128
GAMMA = 0.9 # Discount factor
conv1x1=resnet.conv1x1

# Exploration parameters
EPSILON_START = 1
ESPILON_END = 0.003
DECAY_RATE = 0.999

# Map parameters
MAP_WIDTH = 10
MAP_HEIGHT = 10

# Memory parameters
MEM_CAPACITY = 10000

# Cycle parameters
GAME_CYCLE = 20
DISPLAY_CYCLE = GAME_CYCLE


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.batch_size = BATCH_SIZE
		self.gamma = GAMMA
		self.inplanes = 1
		self.layers = 1

		# self.conv1 = nn.Conv2d(1, 8, 5,padding=2)
		# self.conv2 = nn.Conv2d(8, 32, 3,padding=1)
		# self.conv3 = nn.Conv2d(32, 64, 3,padding=1)

		# self.layer1=self._make_layer(BasicBlock,8,self.layers,stride=2)
		# self.layer2 = self._make_layer(BasicBlock, 64, self.layers)
		# self.layer3 = self._make_layer(BasicBlock, 128, self.layers)
		self.conv1 = nn.Conv2d(1, 8, 7, padding=3, stride=2)
		self.conv2 = nn.Conv2d(8, 32, 5, padding=2, stride=2)
		self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
		self.conv4 = nn.Conv2d(64, 256, 3, padding=1)

		# self.conv1=nn.Conv2d(1, 32, 6)
		# self.conv2 = nn.Conv2d(32, 64, 3)

		self.fc1 = nn.Linear(256 * 1 * 1, 256)
		# self.fc2 = nn.Linear(256, 512)
		# self.fc3 = nn.Linear(512, 256)
		self.fc4 = nn.Linear(256, 64)
		self.fc5 = nn.Linear(64, 4)

		self.maxPool = nn.MaxPool2d(kernel_size=3, stride=1)

		self.dropout = nn.Dropout(p=0.2)

		self.batch_norm = nn.BatchNorm2d(1)
		torch.nn.init.xavier_uniform_(self.fc1.weight)
		# torch.nn.init.xavier_uniform_(self.fc2.weight)
		# torch.nn.init.xavier_uniform_(self.fc3.weight)
		torch.nn.init.xavier_uniform_(self.fc4.weight)
		torch.nn.init.xavier_uniform_(self.fc5.weight)

	def forward(self, x):
		x = x.cuda()

		x = self.batch_norm(x)
		#
		# x = self.layer1(x)
		# x = self.layer2(x)
		# x = self.AvgPool(x)
		# x = self.layer3(x)
		# x = self.AvgPool(x)

		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.maxPool(x)

		x = x.view(-1, 256 * 1 * 1)

		x = self.dropout(F.relu(self.fc1(x)))
		# x = self.dropout(F.relu(self.fc2(x)))
		# x = self.dropout(F.relu(self.fc3(x)))
		x = self.dropout(F.relu(self.fc4(x)))
		x = self.fc5(x)

		return x

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None

		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []

		layers.append(block(self.inplanes, planes, stride, downsample))

		self.inplanes = planes * block.expansion

		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)


class Ai(Player):

	def __init__(self,epsilon=0):
		super(Ai, self).__init__()
		self.net = Net().to(device)
		self.epsilon = epsilon
		# Load network weights if they have been initialized already

		if os.path.isfile('ais/' + folderName  +'/'+ '_ai.bak'):
			self.net.load_state_dict(torch.load('ais/' + folderName +'/' + '_ai.bak'))


	def action(self, map, id):

		game_map = map.state_for_player(id)

		input = np.reshape(game_map, (1, 1, game_map.shape[0], game_map.shape[1]))
		input = torch.from_numpy(input).float()
		output = self.net(input)

		_, predicted = torch.max(output.data, 1)
		predicted = predicted.cpu().numpy()
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


Transition = namedtuple('Transition',('old_state', 'action', 'new_state', 'reward', 'terminal'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.immemory=[]
        self.position = 0

    def reset(self):
        self.memory.append(None)

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def push_old(self,i):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] =i
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def delete_old(self,batch_size):

        del_mem=random.sample(self.memory, batch_size)
        im=list(set(self.memory) - set(del_mem))

        self.memory.append(None)
        self.memory=im

        self.position=self.__len__()

        return del_mem
    def thanos(self):

        self.memory=random.sample(self.memory, int(len(self.memory)*0.2))
        self.position=len(self.memory)

    def __len__(self):
        return len(self.memory)

def train(model):
	writer = SummaryWriter()
	vis = visdom.Visdom()

	vis.close(env="main")
	loss_plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
	du_plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='duration_tracker', legend=['duration'], showlegend=True))
	win_plot = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='rating_tracker', legend=['win_rate'], showlegend=True))
	test_plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='test', legend=['test'], showlegend=True))

	# Initialize neural network parameters and optimizer
	optimizer = optim.Adam(model.parameters())
	criterion = nn.MSELoss()

	# Initialize exploration rate
	epsilon = EPSILON_START
	epsilon_temp = float(epsilon)

	# Initialize memory
	memory = ReplayMemory(MEM_CAPACITY)
	mini_memory= ReplayMemory(MEM_CAPACITY)
	blowup1 = ReplayMemory(MEM_CAPACITY)
	blowup2 = ReplayMemory(MEM_CAPACITY)

	# Initialize the game counter
	game_counter = 0
	move_counter = 0

	win_p1 = 0
	win_p2 = 0
	changeAi = 0
	vs_min_p1_win = 0
	minimax_game = 0
	minnimax_match = 1000
	minimam_match = 1000
	Aiset = True
	otherOpponent = True
	ai='basic'
	rate=0
	# Start training
	player_2 = Ai(epsilon)
	while True:

		# Initialize the game cycle parameters
		cycle_step = 0
		p1_victories = 0
		p2_victories = 0
		null_games = 0
		player_1 = Ai(epsilon)
		if(Aiset):
			player_2 = Ai(epsilon)



		# Play a cycle of games
		while cycle_step < GAME_CYCLE:
			changeAi += 1

			if (game_counter < 3000):
				changeAi = 0

			if (changeAi > minnimax_match):

				if (Aiset):
					#memory.reset()
					player_2 = Ai(epsilon)
					player_2.epsilon = epsilon
					player_2 = MinimaxPlayer(2, 'VORNOI')

					Aiset = False
					otherOpponent = False
					ai = 'minimax'
					minnimax_match = (10000 * rate) + minimam_match

				else:
					#mini_memory.reset()
					rate = vs_min_p1_win / minimax_game
					minnimax_match = 10000 - minnimax_match
					vs_min_p1_win = 0
					minimax_game = 0
					Aiset = True
					player_2 = Ai(epsilon)
					player_2.epsilon = epsilon
					otherOpponent = True
					ai = 'basic Ai'

				changeAi = 0

			# Increment the counters
			game_counter += 1
			cycle_step += 1

			# Initialize the starting positions
			x1 = random.randint(0,MAP_WIDTH-1)
			y1 = random.randint(0,MAP_HEIGHT-1)
			x2 = random.randint(0,MAP_WIDTH-1)
			y2 = random.randint(0,MAP_HEIGHT-1)
			while x1==x2 and y1==y2:
				x1 = random.randint(0,MAP_WIDTH-1)
				y1 = random.randint(0,MAP_HEIGHT-1)

			# Initialize the game
			player_1.epsilon = epsilon
			if(Aiset):
				player_2.epsilon = epsilon
			game = Game(MAP_WIDTH,MAP_HEIGHT, [
						PositionPlayer(1, player_1, [x1, y1]),
						PositionPlayer(2, player_2, [x2, y2]),])

			# Get the initial state for each player
			old_state_p1 = game.map().state_for_player(1)
			old_state_p1 = np.reshape(old_state_p1, (1, 1, old_state_p1.shape[0], old_state_p1.shape[1]))
			old_state_p1 = torch.from_numpy(old_state_p1).float()
			old_state_p2 = game.map().state_for_player(2)
			old_state_p2 = np.reshape(old_state_p2, (1, 1, old_state_p2.shape[0], old_state_p2.shape[1]))
			old_state_p2 = torch.from_numpy(old_state_p2).float()

			# Run the game
			# if VisibleScreen:
			# 	window = Window(game, 40)
			# 	game.main_loop(window)
			# else:
			game.main_loop()

			# Analyze the game
			move_counter += len(game.history)
			terminal = False

			for historyStep in range(len(game.history)-1):

				# Get the state for each player
				new_state_p1 = game.history[historyStep+1].map.state_for_player(1)
				new_state_p1 = np.reshape(new_state_p1, (1, 1, new_state_p1.shape[0], new_state_p1.shape[1]))
				new_state_p1 = torch.from_numpy(new_state_p1).float()

				new_state_p2 = game.history[historyStep+1].map.state_for_player(2)
				new_state_p2 = np.reshape(new_state_p2, (1, 1, new_state_p2.shape[0], new_state_p2.shape[1]))
				new_state_p2 = torch.from_numpy(new_state_p2).float()

				# Get the action for each player
				if game.history[historyStep].player_one_direction is not None:
					action_p1 = torch.from_numpy(np.array([game.history[historyStep].player_one_direction.value-1], dtype=np.float32)).unsqueeze(0)
					action_p2 = torch.from_numpy(np.array([game.history[historyStep].player_two_direction.value-1], dtype=np.float32)).unsqueeze(0)
				else:
					action_p1 = torch.from_numpy(np.array([0], dtype=np.float32)).unsqueeze(0)
					action_p2 = torch.from_numpy(np.array([0], dtype=np.float32)).unsqueeze(0)

				# Compute the reward for each player
				reward_p1 = historyStep
				reward_p2 = historyStep
				if historyStep +1 == len(game.history)-1:
					if game.winner is None:
						null_games += 1
						reward_p1 = 0
						reward_p2 = 0

						# sample1 = blowup1.delete_old(len(blowup1.memory))
						# for i in sample1:
						# 	list1 = list(i)
						# 	list1[3] = 1
						# 	list1[3] = torch.from_numpy(np.array([list1[3]], dtype=np.float32)).unsqueeze(0)
						# 	memory.push(list1[0], list1[1], list1[2], list1[3], list1[4])
						#
						# reward_p1 = torch.from_numpy(np.array([reward_p1], dtype=np.float32)).unsqueeze(0)
						# terminal = True
						# memory.push(old_state_p1, action_p1, new_state_p1, reward_p1, terminal)

					elif game.winner == 1:
						reward_p1 = 100
						reward_p2 = -25
						p1_victories +=1

						# sample1 = blowup1.delete_old(len(blowup1.memory))
						#
						# for i in sample1:
						# 	list1 = list(i)
						# 	list1[3] = list1[3].item()
						# 	list1[3] = torch.from_numpy(np.array([list1[3]], dtype=np.float32)).unsqueeze(0)
						# 	memory.push(list1[0], list1[1], list1[2], list1[3], list1[4])
						#
						# reward_p1 = torch.from_numpy(np.array([reward_p1], dtype=np.float32)).unsqueeze(0)
						# terminal = True
						# memory.push(old_state_p1, action_p1, new_state_p1, reward_p1, terminal)
						if not (Aiset):
							vs_min_p1_win+=1
					else:
						reward_p1 = -25
						reward_p2 = 100
						p2_victories += 1

						# sample1 = blowup1.delete_old(len(blowup1.memory))
						# for i in sample1:
						# 	list1 = list(i)
						# 	list1[3] = (-list1[3].item())
						# 	list1[3] = torch.from_numpy(np.array([list1[3]], dtype=np.float32)).unsqueeze(0)
						#
						# 	memory.push(list1[0], list1[1], list1[2], list1[3], list1[4])
						#
						# reward_p1 = torch.from_numpy(np.array([reward_p1], dtype=np.float32)).unsqueeze(0)
						# terminal = True
						# memory.push(old_state_p1, action_p1, new_state_p1, reward_p1, terminal)
					if not (Aiset):
						minimax_game += 1
					terminal=True

				reward_p1 = torch.from_numpy(np.array([reward_p1], dtype=np.float32)).unsqueeze(0)
				reward_p2 = torch.from_numpy(np.array([reward_p2], dtype=np.float32)).unsqueeze(0)

				#Save the transition for each player
				memory.push(old_state_p1, action_p1, new_state_p1, reward_p1, terminal)
				memory.push(old_state_p2, action_p2, new_state_p2, reward_p2, terminal)

				# if not(otherOpponent) :
				# 	mini_memory.push(old_state_p2, action_p2, new_state_p2, reward_p2, terminal)

				# if not (terminal):
				# 	blowup1.push(old_state_p1, action_p1, new_state_p1, reward_p1, terminal)

				# Update old state for each player
				old_state_p1 = new_state_p1
				old_state_p2 = new_state_p2

			# Update exploration rate
			nouv_epsilon = epsilon*DECAY_RATE
			if nouv_epsilon > ESPILON_END:
				epsilon = nouv_epsilon

			if epsilon==0 and game_counter%100==0 :
				epsilon = epsilon_temp

		# Get a sample for training
		transitions = memory.sample(min(len(memory), model.batch_size))
		# if (otherOpponent):
		# 	transitions = memory.sample(min(len(memory),model.batch_size))
		# if not (otherOpponent):
		#
		# 	transitions = mini_memory.sample(min(len(mini_memory), model.batch_size))
		#print(transitions)
		#print(transitions)
		batch = Transition(*zip(*transitions))
		old_state_batch = torch.cat(batch.old_state)
		action_batch = torch.cat(batch.action).long()
		print(action_batch)
		new_state_batch = torch.cat(batch.new_state)
		reward_batch = torch.cat(batch.reward).to(device)

		# Compute predicted Q-values for each action
		pred_q_values_batch = torch.sum(model(old_state_batch).gather(1, action_batch.to(device)),dim=1)
		pred_q_values_next_batch = model(new_state_batch)

		# Compute targeted Q-value for action performed
		target_q_values_batch = torch.cat(tuple(reward_batch[i] if batch[4][i]
					   else reward_batch[i] + model.gamma * torch.max(pred_q_values_next_batch[i])
					   for i in range(len(reward_batch)))).to(device)

		# zero the parameter gradients

		under_minus_26 = 0
		if (torch.min(pred_q_values_next_batch) < (-25)):
			if (game_counter % 100 == 0):
				under_minus_26 = torch.sum(pred_q_values_next_batch < -25).squeeze()

		model.zero_grad()

		# Compute the loss
		target_q_values_batch = target_q_values_batch.detach()
		#loss = criterion(pred_q_values_batch,target_q_values_batch)
		loss=F.smooth_l1_loss(pred_q_values_batch,target_q_values_batch)

		# Do backward pass
		loss.backward()
		optimizer.step()

		# Update bak
		torch.save(model.state_dict(), 'ais/' + folderName +'/'+'_ai.bak')
		p1_winrate = p1_victories / (GAME_CYCLE)
		# Display results
		if (game_counter%DISPLAY_CYCLE)==0:

			loss_string = str(loss)
			loss_string = loss_string[7:len(loss_string)]
			loss_value = loss_string.split(',')[0]
			print("--- Match", game_counter, "---")
			print("Average duration :", float(move_counter)/float(DISPLAY_CYCLE))
			print("Loss =", loss_value)
			print("Epsilon =", epsilon)
			print("")
			#print("Max duration :", max_du)
			print("score p1 vs p2 =", win_p1, ":", win_p2)
			print("ai state=", ai)
			p1_winrate = p1_victories / (GAME_CYCLE)
			print("mini", minnimax_match)
			#print("old", old_memory.position, "posi", len(old_memory.memory), "mem size")
			print("new", memory.position, "posi", len(memory.memory), "mem size")

			vis_loss = float(loss_value)
			vis.line(X=torch.tensor([game_counter]),
					 Y=torch.tensor([vis_loss]),
					 win=loss_plot,
					 update='append'
					 )
			vis.line(X=torch.tensor([game_counter]),
					 Y=torch.tensor([float(move_counter) / float(DISPLAY_CYCLE)]),
					 win=du_plot,
					 update='append'
					 )
			vis.line(X=torch.tensor([game_counter]),
					 Y=torch.tensor([p1_winrate]),
					 win=win_plot,
					 update='append'
					 )

			vis.line(X=torch.tensor([game_counter]),
					 Y=torch.tensor([under_minus_26]),
					 win=test_plot,
					 update='append'
					 )
			writer.add_scalar('loss_tracker', vis_loss, game_counter)
			writer.add_scalar('duration_tracker', (float(move_counter) / float(DISPLAY_CYCLE)), game_counter)
			writer.add_scalar('ration_tracker', p1_winrate, game_counter)
			writer.add_scalar('test', under_minus_26, game_counter)

			with open('ais/' + folderName +'/'+ '/data.txt', 'a') as myfile:
				myfile.write(str(game_counter) + ', ' + str(float(move_counter)/float(DISPLAY_CYCLE)) + ', ' + loss_value + '\n')

			move_counter = 0



def main():
	model = Net().to(device)
	if os.path.isfile('ais/' + folderName + '/_ai.bak'):
	  	model.load_state_dict(torch.load('ais/' + folderName + '/_ai.bak'))
	train(model)

if __name__ == "__main__":
	main()

