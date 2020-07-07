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

from ais.basic.ai import Ai as BasicAi
from ais.survivor.ai import Ai as SurvivorAi

# General parameters
folderName = 'basic'

# Net parameters
BATCH_SIZE = 128
GAMMA = 0.9 # Discount factor



# Map parameters
MAP_WIDTH = GameSize
MAP_HEIGHT = GameSize


# Cycle parameters
GAME_CYCLE = 20
DISPLAY_CYCLE = GAME_CYCLE


def train():

	# Initialize the game counter
	game_counter = 0
	move_counter = 0

	player_1 = BasicAi()
	player_2 = SurvivorAi()

	# Start training
	while True:

		# Initialize the game cycle parameters
		cycle_step = 0
		# Play a cycle of games
		while cycle_step < GAME_CYCLE:

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

			game = Game(MAP_WIDTH,MAP_HEIGHT, [
						PositionPlayer(1, player_1, [x1, y1]),
						PositionPlayer(2, player_2, [x2, y2]),])


			# Run the game
			if TrainVisibleScreen:
				window = Window(game, 40)
				game.main_loop(window)
			else:
				game.main_loop()

			# Analyze the game
			move_counter += len(game.history)

		basic_loss = player_1.train(game)
		# survivor_loss = player_2.train(game)

		player_1.save_model()
		player_2.save_model()


		# Display results
		if (game_counter%DISPLAY_CYCLE)==0:
			loss_string = str(basic_loss)
			loss_string = loss_string[7:len(loss_string)]
			loss_value = loss_string.split(',')[0]
			print("--- Match", game_counter, "---")
			print("Average duration :", float(move_counter)/float(DISPLAY_CYCLE))
			print("Loss =", loss_value)
			print("Epsilon =", player_1.epsilon)
			print("")
			with open('ais/' + folderName + '/data.txt', 'a') as myfile:
				myfile.write(str(game_counter) + ', ' + str(float(move_counter)/float(DISPLAY_CYCLE)) + ', ' + loss_value + '\n')
			move_counter = 0


def main():
	# if os.path.isfile('ais/' + folderName + '/10_ai.bak'):
	#   	model.load_state_dict(torch.load('ais/' + folderName + '/10_ai.bak'))
	train()

if __name__ == "__main__":
	main()

