from tron.player import Player, Direction
from tron.game import Game, PositionPlayer

from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from Net.DQNNet import Net

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

# General parameters
folderName = 'survivor'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Net parameters
BATCH_SIZE = 128
GAMMA = 0.9  # Discount factor

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


class Ai(Player):

    def __init__(self, epsilon=0):
        super(Ai, self).__init__()
        self.net = Net().to(device)
        self.epsilon = epsilon

    # Load network weights if they have been initialized already

    # if os.path.isfile('ais/' + folderName  +'/'+ '_ai.bak'):
    # 	self.net.load_state_dict(torch.load('ais/' + folderName +'/' + '_ai.bak'))

    def action(self, map, id):

        game_map = map.state_for_player(id)

        input = np.reshape(game_map, (1, 1, game_map.shape[0], game_map.shape[1]))
        input = torch.from_numpy(input).float()
        output = self.net(input)

        _, predicted = torch.max(output.data, 1)
        predicted = predicted.cpu().numpy()
        next_action = predicted[0] + 1

        if random.random() <= self.epsilon:
            next_action = random.randint(1, 4)

        if next_action == 1:
            next_direction = Direction.UP
        if next_action == 2:
            next_direction = Direction.RIGHT
        if next_action == 3:
            next_direction = Direction.DOWN
        if next_action == 4:
            next_direction = Direction.LEFT

        return next_direction


Transition = namedtuple('Transition', ('old_state', 'action', 'new_state', 'reward', 'terminal'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.immemory = []
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

    def push_old(self, i):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = i
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def delete_old(self, batch_size):

        del_mem = random.sample(self.memory, batch_size)
        im = list(set(self.memory) - set(del_mem))

        self.memory.append(None)
        self.memory = im

        self.position = self.__len__()

        return del_mem

    def thanos(self):

        self.memory = random.sample(self.memory, int(len(self.memory) * 0.2))
        self.position = len(self.memory)

    def __len__(self):
        return len(self.memory)


def train(model):
    writer = SummaryWriter()

    # Initialize neural network parameters and optimizer
    optimizer = optim.Adam(model.parameters())

    # Initialize exploration rate
    epsilon = EPSILON_START
    epsilon_temp = float(epsilon)

    # Initialize memory
    memory = ReplayMemory(MEM_CAPACITY)

    # Initialize the game counter
    game_counter = 0
    move_counter = 0
    vs_min_p1_win = 0
    minimax_game = 0

    while True:

        # Initialize the game cycle parameters
        cycle_step = 0
        p1_victories = 0
        p2_victories = 0
        null_games = 0

        player_1 = Ai(epsilon)
        player_2 = Ai(epsilon)

        # Play a cycle of games
        while cycle_step < GAME_CYCLE:
            # Increment the counters
            game_counter += 1
            cycle_step += 1

            # Initialize the starting positions
            x1 = random.randint(0, MAP_WIDTH - 1)
            y1 = random.randint(0, MAP_HEIGHT - 1)
            x2 = random.randint(0, MAP_WIDTH - 1)
            y2 = random.randint(0, MAP_HEIGHT - 1)
            while x1 == x2 and y1 == y2:
                x1 = random.randint(0, MAP_WIDTH - 1)
                y1 = random.randint(0, MAP_HEIGHT - 1)

            # Initialize the game
            player_1.epsilon = epsilon
            player_2.epsilon = epsilon

            game = Game(MAP_WIDTH, MAP_HEIGHT, [
                PositionPlayer(1, player_1, [x1, y1]),
                PositionPlayer(2, player_2, [x2, y2]), ])

            # Get the initial state for each player
            old_state_p1 = game.map().state_for_player(1)
            old_state_p1 = np.reshape(old_state_p1, (1, 1, old_state_p1.shape[0], old_state_p1.shape[1]))
            old_state_p1 = torch.from_numpy(old_state_p1).float()
            old_state_p2 = game.map().state_for_player(2)
            old_state_p2 = np.reshape(old_state_p2, (1, 1, old_state_p2.shape[0], old_state_p2.shape[1]))
            old_state_p2 = torch.from_numpy(old_state_p2).float()

            game.main_loop(model)

            # Analyze the game
            move_counter += len(game.history)
            terminal = False

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
                    action_p1 = torch.from_numpy(np.array([game.history[historyStep].player_one_direction.value - 1],
                                                          dtype=np.float32)).unsqueeze(0)
                    action_p2 = torch.from_numpy(np.array([game.history[historyStep].player_two_direction.value - 1],
                                                          dtype=np.float32)).unsqueeze(0)
                else:
                    action_p1 = torch.from_numpy(np.array([0], dtype=np.float32)).unsqueeze(0)
                    action_p2 = torch.from_numpy(np.array([0], dtype=np.float32)).unsqueeze(0)

                # Compute the reward for each player
                reward_p1 = historyStep
                reward_p2 = historyStep

                if historyStep + 1 == len(game.history) - 1:
                    if game.winner is None:
                        null_games += 1
                        reward_p1 = 0
                        reward_p2 = 0
                    elif game.winner == 1:
                        reward_p1 = 100
                        reward_p2 = -25
                        p1_victories += 1
                    else:
                        reward_p1 = -25
                        reward_p2 = 100
                        p2_victories += 1

                    terminal = True

                reward_p1 = torch.from_numpy(np.array([reward_p1], dtype=np.float32)).unsqueeze(0)
                reward_p2 = torch.from_numpy(np.array([reward_p2], dtype=np.float32)).unsqueeze(0)

                # Save the transition for each player
                memory.push(old_state_p1, action_p1, new_state_p1, reward_p1, terminal)
                memory.push(old_state_p2, action_p2, new_state_p2, reward_p2, terminal)

                # Update old state for each player
                old_state_p1 = new_state_p1
                old_state_p2 = new_state_p2

            # Update exploration rate
            nouv_epsilon = epsilon * DECAY_RATE
            if nouv_epsilon > ESPILON_END:
                epsilon = nouv_epsilon

            if epsilon == 0 and game_counter % 100 == 0:
                epsilon = epsilon_temp

        # Get a sample for training
        transitions = memory.sample(min(len(memory), model.batch_size))

        batch = Transition(*zip(*transitions))

        old_state_batch = torch.cat(batch.old_state)
        action_batch = torch.cat(batch.action).long()
        new_state_batch = torch.cat(batch.new_state)
        reward_batch = torch.cat(batch.reward).to(device)

        # Compute predicted Q-values for each action
        pred_q_values_batch = torch.sum(model(old_state_batch).gather(1, action_batch.to(device)), dim=1)
        pred_q_values_next_batch = model(new_state_batch)

        # Compute targeted Q-value for action performed
        target_q_values_batch = torch.cat(
            tuple(reward_batch[i] if batch[4][i] else reward_batch[i] + model.gamma * torch.max(
                pred_q_values_next_batch[i]) for i in range(len(reward_batch)))).to(device)

        # zero the parameter gradients

        model.zero_grad()

        # Compute the loss
        target_q_values_batch = target_q_values_batch.detach()
        # loss = criterion(pred_q_values_batch,target_q_values_batch)
        loss = F.smooth_l1_loss(pred_q_values_batch, target_q_values_batch)

        # Do backward pass
        loss.backward()
        optimizer.step()

        # Update bak
        torch.save(model.state_dict(), 'save/DQN.bak')
        p1_winrate = p1_victories / (GAME_CYCLE)

        if game_counter % DISPLAY_CYCLE == 0:
            loss_string = str(loss)
            loss_string = loss_string[7:len(loss_string)]
            loss_value = loss_string.split(',')[0]
            vis_loss = float(loss_value)

            writer.add_scalar('loss_tracker', vis_loss, game_counter)
            writer.add_scalar('duration_tracker', (float(move_counter) / float(DISPLAY_CYCLE)), game_counter)
            writer.add_scalar('ration_tracker', p1_winrate, game_counter)

            move_counter = 0


def main():
    model = Net().to(device)
    train(model)


if __name__ == "__main__":
    main()
