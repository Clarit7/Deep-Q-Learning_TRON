from collections import namedtuple,deque
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tron.util import *
from Net.DQNNet import Net

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


# Memory parameters
MEM_CAPACITY = int(1e5)

# Cycle parameters
UPDATE_EVERY = 4
GAME_CYCLE = 20
DISPLAY_CYCLE = GAME_CYCLE


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
            action_values = self.qnetwork_local(game_map.to(device))
        self.qnetwork_local.train()

        # Epsilon -greedy action selction

        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

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

        predicted_targets = self.qnetwork_local(states).gather(1, actions)
        #################Updates for Double DQN learning###########################
        self.qnetwork_local.eval()
        with torch.no_grad():
            actions_q_local = self.qnetwork_local(next_state).detach().max(1)[1].unsqueeze(1).long()
            labels_next = self.qnetwork_target(next_state).gather(1, actions_q_local)
        self.qnetwork_local.train()
        ############################################################################
        # with torch.no_grad():
        #     labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

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
    win_p1 = 0

    minimax_match=0
    mini=False

    brain = Agent()

    while True:

        # Initialize the game cycle parameters
        cycle_step = 0
        null_games = 0

        # Play a cycle of games
        while cycle_step < GAME_CYCLE:

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
            x1 = random.randint(0, MAP_WIDTH-1)
            y1 = random.randint(0, MAP_HEIGHT-1)
            x2 = random.randint(0, MAP_WIDTH-1)
            y2 = random.randint(0, MAP_HEIGHT-1)

            while x1 == x2 and y1 == y2:
                x1 = random.randint(0, MAP_WIDTH-1)
                y1 = random.randint(0, MAP_HEIGHT-1)
            # Initialize the game

            player1 = ACPlayer()
            player2 = ACPlayer()
            #
            game = Game(MAP_WIDTH,MAP_HEIGHT, [
                PositionPlayer(1,player1, [x1, y1]),
                PositionPlayer(2,player2, [x2, y2]),])

            # Get the initial state for each player

            # old_state_p1 = game.map().state_for_player(1)
            # old_state_p1 = pop_up(old_state_p1)
            # old_state_p1 = np.reshape(old_state_p1, (1, -1, old_state_p1.shape[1], old_state_p1.shape[2]))
            # old_state_p1 = torch.from_numpy(old_state_p1).float()

            old_state_p1 = game.map().state_for_player(1)
            old_state_p1 = np.reshape(old_state_p1, (1, 1, old_state_p1.shape[0], old_state_p1.shape[1]))
            old_state_p1 = torch.from_numpy(old_state_p1).float()

            # old_state_p2 = game.map().state_for_player(2)
            # old_state_p2 = pop_up(old_state_p2)
            # old_state_p2 = np.reshape(old_state_p2, (1, -1, old_state_p2.shape[1], old_state_p2.shape[2]))
            # old_state_p2 = torch.from_numpy(old_state_p2).float()

            old_state_p2 = game.map().state_for_player(2)
            old_state_p2 = np.reshape(old_state_p2, (1, 1, old_state_p2.shape[0], old_state_p2.shape[1]))
            old_state_p2 = torch.from_numpy(old_state_p2).float()

            done=False
            move = 0

            while not done:
                brain.epsilon = epsilon

                p1_action = brain.action(old_state_p1)
                p2_action = brain.action(old_state_p2)

                p1_next_state, p1_reward,p2_next_state, p2_reward, done, _, _ = game.step(p1_action, p2_action)

                move_counter += 1
                move += 1

                # p1_next_state = pop_up(p1_next_state)
                # p1_next_state = np.reshape(p1_next_state, (1, -1, p1_next_state.shape[1], p1_next_state.shape[2]))
                # p1_next_state = torch.from_numpy(p1_next_state).float()

                # p2_next_state = pop_up(p2_next_state)
                # p2_next_state = np.reshape(p2_next_state, (1, -1, p2_next_state.shape[1], p2_next_state.shape[2]))
                # p2_next_state = torch.from_numpy(p2_next_state).float()

                p1_next_state = np.reshape(p1_next_state, (1, 1, p1_next_state.shape[0], p1_next_state.shape[1]))
                p1_next_state = torch.from_numpy(p1_next_state).float()

                p2_next_state = np.reshape(p2_next_state, (1, 1, p2_next_state.shape[0], p2_next_state.shape[1]))
                p2_next_state = torch.from_numpy(p2_next_state).float()

                if done:

                    if game.winner is None:
                        null_games += 1
                        p1_reward = 0
                        p2_reward = 0

                    elif game.winner == 1:
                        p1_reward = 100
                        p2_reward = -100

                    else:
                        p1_reward = -100
                        p2_reward = 100

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
        # Display results
        # Update bak

        torch.save(brain.qnetwork_target.state_dict(), 'save/DDQN.bak')

        if (game_counter%DISPLAY_CYCLE)==0:
            loss=brain.get_loss()
            loss_string = str(loss)
            loss_string = loss_string[7:len(loss_string)]
            loss_value = loss_string.split(',')[0]

            print('%d Episode: Finished after %d steps' % (game_counter,int(float(move_counter) / float(DISPLAY_CYCLE))))

            if not(mini):
                p1_winrate=-1
            else:
                p1_winrate=win_p1/DISPLAY_CYCLE
                win_p1=0

            writer.add_scalar('Training loss', float(loss_value), game_counter)
            writer.add_scalar('Duration', (float(move_counter) / float(DISPLAY_CYCLE)), game_counter)
            writer.add_scalar('Win rate', p1_winrate, game_counter)

            move_counter = 0


if __name__ == "__main__":
    train()

