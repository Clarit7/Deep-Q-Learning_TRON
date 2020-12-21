from tron.minimax import MinimaxPlayer
from tron import resnet

from collections import namedtuple,deque
import os

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from util import *


# General parameters
folderName = 'survivor'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Net parameters
BATCH_SIZE = 64
GAMMA = 0.9 # Discount factor

# Exploration parameters
EPSILON_START = 1
ESPILON_END = 0.003
DECAY_RATE = 0.999
TAU = 0.001

# Map parameters
MAP_WIDTH = 10
MAP_HEIGHT = 10

# resnet
conv1x1=resnet.conv1x1
BasicBlock=resnet.BasicBlock


# Memory parameters
MEM_CAPACITY = int(1e5)

# Cycle parameters
UPDATE_EVERY = 4
GAME_CYCLE = 20
DISPLAY_CYCLE = GAME_CYCLE

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.inplanes = 3
        self.layers = 3

        # self.conv1 = nn.Conv2d(1, 8, 5,padding=2)
        # self.conv2 = nn.Conv2d(8, 32, 3,padding=1)
        # self.conv3 = nn.Conv2d(32, 64, 3,padding=1)

        # self.layer1 = self._make_layer(BasicBlock, 16, self.layers, stride=2)
        # self.layer2 = self._make_layer(BasicBlock, 64, self.layers)
        # self.layer3 = self._make_layer(BasicBlock, 128, self.layers)

        #
        # self.conv1 = nn.Conv2d(3, 16, 7, padding=3, stride=1)
        # self.conv2 = nn.Conv2d(16, 64, 5, padding=1, stride=1)
        # self.conv3 = nn.Conv2d(64, 128, 5)
        # self.conv4 = nn.Conv2d(64, 256, 3, padding=1)

        self.conv1 = nn.Conv2d(3, 32, 6)
        self.conv2 = nn.Conv2d(32, 64, 3)

        # self.fc1 = nn.Linear(128 * 2* 2, 256)
        # self.fc2 = nn.Linear(256, 512)
        # self.fc3 = nn.Linear(512, 256)
        # self.fc4 = nn.Linear(256, 64)
        # self.fc5 = nn.Linear(64, 4)

        self.fc1 = nn.Linear(64 * 5* 5, 64)
        # self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4)

        self.AvgPool = nn.AvgPool2d(kernel_size=3, stride=1)

        self.dropout = nn.Dropout(p=0.3)

        self.batch_norm = nn.BatchNorm2d(3)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)
        # torch.nn.init.xavier_uniform_(self.fc4.weight)
        # torch.nn.init.xavier_uniform_(self.fc5.weight)

    def forward(self, x):
        x = x.cuda()

        # x = self.batch_norm(x)
        #
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.AvgPool(x)

        # x = self.layer3(x)

        #
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.AvgPool(x)
        # x = F.relu(self.conv3(x))
        # x = self.AvgPool(x)
        # x = F.relu(self.conv4(x))
        # x = self.maxPool(x)

        # x = x.view(-1, 128 * 2 * 2)
        #
        # x = self.fc1(x)
        # x1 = self.dropout(F.relu(x))
        # x1 += x

        #
        # x = self.fc2(x1)
        # x2 = self.dropout(F.relu(x))
        # x2+=x
        #
        # x = self.fc3(x2)
        # x3 = self.dropout(F.relu(x))
        # x3 += x

        # x = self.fc4(x1)
        # x4 = self.dropout(F.relu(x))
        # x4 += x
        #
        # x = self.fc5(x4)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 5 * 5)

        x=self.dropout(F.relu(self.fc1(x)))
        # x=self.dropout(F.relu(self.fc2(x)))
        x=self.fc3(x)


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


class Agent():
    def __init__(self):


        """Initialize an Agent object.

               Params
               =======
                   state_size (int): dimension of each state
                   action_size (int): dimension of each action
                   seed (int): random seed
               """
        # Q- Network
        self.qnetwork_local = Net().to(device)
        self.qnetwork_target = Net().to(device)
        self.action_size=4
        self.steps=0;
        # if os.path.isfile('ais/' + folderName  +'/'+ '_ai.bak'):
        #     self.net.load_state_dict(torch.load('ais/' + folderName +'/' + '_ai.bak'))

        self.optimizer = optim.Adam(self.qnetwork_local.parameters())
        self.epsilon=0
        self.totalloss=0
        # Replay memory
        self.memory = ReplayBuffer(4, MEM_CAPACITY, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        if os.path.isfile('ais/' + folderName + '/local_ai.bak'):
            self.qnetwork_local.load_state_dict(torch.load('ais/' + folderName + '/local_ai.bak'))
        if os.path.isfile('ais/' + folderName + '/target_ai.bak'):
            self.qnetwork_target.load_state_dict(torch.load('ais/' + folderName + '/target_ai.bak'))

    def get_loss(self):
        out_loss=self.totalloss/self.steps
        self.totalloss=0
        self.steps=0

        return out_loss
    def step(self, state, action, reward, next_step, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        #print(len(self.memory))
        # print(self.t_step,"step")

        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.steps += 1
                self.learn(experience, GAMMA)


    def action(self,game_map):

        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """


        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(game_map)
        self.qnetwork_local.train()

        # Epsilon -greedy action selction

        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())+1
        else:
            return random.choice(np.arange(self.action_size)+1)

    def find_file(self, name):
        return '/'.join(self.__module__.split('.')[:-1]) + '/' + name

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_state, dones = experiences

        criterion = torch.nn.MSELoss()
        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        # shape of output from the model (batch_size,action_dim) = (64,4)

        predicted_targets = self.qnetwork_local(states).gather(1, actions-1)
        #################Updates for Double DQN learning###########################
        self.qnetwork_local.eval()
        with torch.no_grad():
            actions_q_local = self.qnetwork_local(next_state).detach().max(1)[1].unsqueeze(1).long()
            labels_next = self.qnetwork_target(next_state).gather(1, actions_q_local)
        self.qnetwork_local.train()
        ############################################################################
        # with torch.no_grad():
        #     labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

        #print("?",labels_next)
        # .detach() ->  Returns a new Tensor, detached from the current graph.

        labels = rewards + (gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(device)
        self.totalloss += loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

class player(Player):
    def __init__(self):
        super(player, self).__init__()

        """Initialize an Agent object.

               Params
               =======
                   state_size (int): dimension of each state
                   action_size (int): dimension of each action
                   seed (int): random seed
               """
    def get_direction(self,next_action):

        if next_action == 1:
            next_direction = Direction.UP
        if next_action == 2:
            next_direction = Direction.RIGHT
        if next_action == 3:
            next_direction = Direction.DOWN
        if next_action == 4:
            next_direction = Direction.LEFT

        return next_direction

    def next_position_and_direction(self, current_position,action):

        direction = self.get_direction(action)
        return (self.next_position(current_position, direction),direction)

    def next_position(self, current_position, direction):

        if direction == Direction.UP:
            return (current_position[0] - 1, current_position[1])
        if direction == Direction.RIGHT:
            return (current_position[0], current_position[1] + 1)
        if direction == Direction.DOWN:
            return (current_position[0] + 1, current_position[1])
        if direction == Direction.LEFT:
            return (current_position[0], current_position[1] - 1)

class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size,):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state","action","reward","next_state","done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)




def train():
    writer = SummaryWriter('runs/Double DQN')


    # Initialize exploration rate

    epsilon = EPSILON_START
    epsilon_temp = float(epsilon)

    # Initialize the game counter
    game_counter = 0
    move_counter = 0


    changeAi = 0


    ai='basic'

    # vs minimax

    win_p1 = 0
    win_p2 = 0
    rate=0
    play_with_minimax=0
    defalt_match=1000
    minimax_match=0
    mini=False
    duel_mini=False
    start_mini=100000



    brain = Agent()
    minimax=MinimaxPlayer(2,'voronoi')
    while True:

        # Initialize the game cycle parameters
        cycle_step = 0
        p1_victories = 0
        p2_victories = 0
        null_games = 0



        # Play a cycle of games
        while cycle_step < GAME_CYCLE:
            #print(cycle_step)
            # Increment the counters
            game_counter += 1
            cycle_step += 1
            changeAi += 1

            '''
            if(game_counter<start_mini):
                changeAi=0

            
            if (changeAi > minimax_match):

                if (mini):
                    minimax_match = 11000 - minimax_match
                    mini = False

                else:
                    duel_mini=False
                    if not(play_with_minimax==0):
                        rate =  win_p1 / play_with_minimax
                    print(rate)
                    if(rate>0.7):
                        print("do i win?")
                        break;
                    minimax_match = (10000 * rate) + defalt_match
                    mini = True
                    play_with_minimax=0

                changeAi = 0
            '''

            # Initialize the starting positions
            x1 = random.randint(0,MAP_WIDTH-1)
            y1 = random.randint(0,MAP_HEIGHT-1)
            x2 = random.randint(0,MAP_WIDTH-1)
            y2 = random.randint(0,MAP_HEIGHT-1)

            while x1==x2 and y1==y2:
                x1 = random.randint(0,MAP_WIDTH-1)
                y1 = random.randint(0,MAP_HEIGHT-1)
            # Initialize the game

            player1 = player()
            player2 = player()
            #
            game = Game(MAP_WIDTH,MAP_HEIGHT, [
                PositionPlayer(1,player1, [x1, y1]),
                PositionPlayer(2,player2, [x2, y2]),])

            # Get the initial state for each player

            old_state_p1 = game.map().state_for_player(1)
            old_state_p1 = pop_up(old_state_p1)
            old_state_p1 = np.reshape(old_state_p1, (1, -1, old_state_p1.shape[1], old_state_p1.shape[2]))
            old_state_p1 = torch.from_numpy(old_state_p1).float()

            # old_state_p1 = game.map().state_for_player(1)
            # old_state_p1 = np.reshape(old_state_p1, (1, 1, old_state_p1.shape[0], old_state_p1.shape[1]))
            # old_state_p1 = torch.from_numpy(old_state_p1).float()

            old_state_p2 = game.map().state_for_player(2)
            old_state_p2 = pop_up(old_state_p2)
            old_state_p2 = np.reshape(old_state_p2, (1, -1, old_state_p2.shape[1], old_state_p2.shape[2]))
            old_state_p2 = torch.from_numpy(old_state_p2).float()

            # old_state_p2 = game.map().state_for_player(2)
            # old_state_p2 = np.reshape(old_state_p2, (1, 1, old_state_p2.shape[0], old_state_p2.shape[1]))
            # old_state_p2 = torch.from_numpy(old_state_p2).float()

            done=False
            move = 0

            while not(done):
                brain.epsilon=epsilon

                if(duel_mini):
                    p1_action = minimax.action(game.map(), 2) - 1
                    p2_action = minimax.action(game.map(), 2) - 1


                elif(mini):
                    p1_action = brain.action(old_state_p1)
                    p2_action = minimax.action(game.map(), 2) - 1
                else:
                    p1_action = brain.action(old_state_p1)
                    p2_action = brain.action(old_state_p2)

                p1_next_state, p1_reward,p2_next_state, p2_reward,done= game.step(p1_action, p2_action)

                move_counter += 1
                move+=1

                p1_next_state = pop_up(p1_next_state)
                p1_next_state = np.reshape(p1_next_state, (1, -1, p1_next_state.shape[1], p1_next_state.shape[2]))
                p1_next_state = torch.from_numpy(p1_next_state).float()

                p2_next_state = pop_up(p2_next_state)
                p2_next_state = np.reshape(p2_next_state, (1, -1, p2_next_state.shape[1], p2_next_state.shape[2]))
                p2_next_state = torch.from_numpy(p2_next_state).float()

                # p1_next_state = np.reshape(p1_next_state, (1, 1, p1_next_state.shape[0], p1_next_state.shape[1]))
                # p1_next_state = torch.from_numpy(p1_next_state).float()
                #
                # p2_next_state = np.reshape(p2_next_state, (1, 1, p2_next_state.shape[0], p2_next_state.shape[1]))
                # p2_next_state = torch.from_numpy(p2_next_state).float()

                if done:
                    if(mini):
                        play_with_minimax += 1

                    if game.winner is None:
                        null_games += 1
                        p1_reward = 0
                        p2_reward = 0

                    elif game.winner == 1:
                        p1_reward = 100
                        p2_reward = -100
                        p1_victories +=1

                        if(mini):
                            win_p1+=1


                    else:
                        p1_reward = -100
                        p2_reward = 100
                        p2_victories += 1


                brain.step(old_state_p1, p1_action, p1_reward, p1_next_state, done)
                brain.step(old_state_p2, p2_action, p2_reward, p2_next_state, done)

                old_state_p1 = p1_next_state
                old_state_p2 = p2_next_state



        nouv_epsilon = epsilon * DECAY_RATE
        if nouv_epsilon > ESPILON_END:
            epsilon = nouv_epsilon

        if epsilon == 0 and game_counter % 100 == 0:
            epsilon = epsilon_temp

            # Update exploration rate


        # Compute the loss



        #p1_winrate = p1_victories / (GAME_CYCLE)
        # Display results
        # Update bak
        torch.save(brain.qnetwork_local.state_dict(), 'ais/' + folderName + '/' + 'local_ai.bak')
        torch.save(brain.qnetwork_target.state_dict(), 'ais/' + folderName + '/' + 'target_ai.bak')

        if(game_counter==start_mini):
            torch.save(brain.qnetwork_local.state_dict(), 'ais/' + folderName + '/' + 'local_ai_200k.bak')
            torch.save(brain.qnetwork_target.state_dict(), 'ais/' + folderName + '/' + 'target_ai_200k.bak')

        if (game_counter%DISPLAY_CYCLE)==0:
            loss=brain.get_loss()
            loss_string = str(loss)
            loss_string = loss_string[7:len(loss_string)]
            loss_value = loss_string.split(',')[0]
            print("--- Match", game_counter, "---")
            print("Average duration :", float(move_counter)/float(DISPLAY_CYCLE))
            print("Loss =", loss_value)
            print("Epsilon =", epsilon)
            # print("Max duration :", max_du)
            # print("score p1 vs p2 =", win_p1, ":", win_p2)
            print("minimax state=", mini)
            # p1_winrate = p1_victories / (GAME_CYCLE)
            print("mini", minimax_match)
            print("")
            # print("old", old_memory.position, "posi", len(old_memory.memory), "mem size")
            # print("new", memory.position, "posi", len(memory.memory), "mem size")
            if not(mini):
                p1_winrate=-1
            else:
                p1_winrate=win_p1/DISPLAY_CYCLE
                win_p1=0

            writer.add_scalar('Training loss', float(loss_value), game_counter)
            writer.add_scalar('Duration', (float(move_counter) / float(DISPLAY_CYCLE)), game_counter)
            writer.add_scalar('Win rate', p1_winrate, game_counter)
            # writer.add_scalar('test', under_minus_26, game_counter)

            # with open('ais/' + folderName +'/'+ '/data.txt', 'a') as myfile:
            #     myfile.write(str(game_counter) + ', ' + str(float(move_counter)/float(DISPLAY_CYCLE)) + ', ' + loss_value + '\n')

            move_counter = 0



def main():
    #model = Net().to(device)
    local_model = Net()
    target_model = Net()

    # if os.path.isfile('ais/' + folderName + '/_ai.bak'):
    #     local_model.load_state_dict(torch.load('ais/' + folderName + '/_ai.bak'))
    train()

if __name__ == "__main__":
    main()

