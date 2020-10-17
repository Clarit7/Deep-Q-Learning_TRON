import numpy as np
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim

import random
import visdom

from tron.DDQN_player import Player, Direction
from tron.DDQN_game import Tile, Game, PositionPlayer

from torch.utils.tensorboard import SummaryWriter
from tron.minimax import MinimaxPlayer
from tron import resnet
from time import sleep

device = 'cuda' if torch.cuda.is_available() else 'cpu'



ENV = 'CartPole-v0'  # 태스크 이름
GAMMA = 0.99  # 시간할인율
MAX_STEPS = 200  # 1에피소드 당 최대 단계 수
NUM_EPISODES = 1000  # 최대 에피소드 수

lr = 7e-4
eps = 1e-5
alpha = 0.99

NUM_PROCESSES = 32  # 동시 실행 환경 수
NUM_ADVANCED_STEP = 3 # 총 보상을 계산할 때 Advantage 학습을 할 단계 수

# A2C 손실함수 계산에 사용되는 상수
value_loss_coef = 1
entropy_coef = 1
max_grad_norm = 5

MAP_WIDTH = 10
MAP_HEIGHT = 10

SHOW_ITER=100
minimax = MinimaxPlayer(2, 'voronoi')

class RolloutStorage(object):
    '''Advantage 학습에 사용할 메모리 클래스'''

    def __init__(self, num_steps, num_processes):

        self.observations = torch.zeros(num_steps + 1, num_processes,3,12,12)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()

        # 할인 총보상 저장
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.index = 0  # insert할 인덱스

    def insert(self, current_obs, action, reward, mask):
        '''현재 인덱스 위치에 transition을 저장'''
        self.observations[self.index + 1].copy_(current_obs)

        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP  # 인덱스 값 업데이트

    def after_update(self):
        '''Advantage학습 단계만큼 단계가 진행되면 가장 새로운 transition을 index0에 저장'''
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        '''Advantage학습 범위 안의 각 단계에 대해 할인 총보상을 계산'''

        # 주의 : 5번째 단계부터 거슬러 올라오며 계산
        # 주의 : 5번째 단계가 Advantage1, 4번째 단계는 Advantage2가 됨

        self.returns[-1] = next_value

        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] * GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)

        self.dropout = nn.Dropout(p=0.3)
        self.actor1 = nn.Linear(64, 64)
        self.actor2 = nn.Linear(64, 4)  # 행동을 결정하는 부분이므로 출력 갯수는 행동의 가짓수

        self.critic1 = nn.Linear(64, 64)
        self.critic2 = nn.Linear(64, 1)  # 상태가치를 출력하는 부분이므로 출력 갯수는 1개
        self.batch_norm = nn.BatchNorm2d(3)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        torch.nn.init.xavier_uniform_(self.actor1.weight)
        torch.nn.init.xavier_uniform_(self.critic1.weight)

        torch.nn.init.xavier_uniform_(self.actor2.weight)
        torch.nn.init.xavier_uniform_(self.critic2.weight)


    def forward(self, x):
        '''신경망 순전파 계산을 정의'''
        x=x.to(device)
        x=self.batch_norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 4 * 4)


        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        h1 = self.dropout(F.relu(self.actor1(x)))
        actor_output =self.actor2(h1)

        h2=self.dropout(F.relu(self.critic1(x)))
        critic_output = self.critic2(h2)


        return critic_output, actor_output

    def act(self, x):
        '''상태 x로부터 행동을 확률적으로 결정'''
        value, actor_output = self(x)
        # dim=1이므로 행동의 종류에 대해 softmax를 적용
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        return action

    def get_value(self, x):
        '''상태 x로부터 상태가치를 계산'''
        value, actor_output = self(x)

        return value

    def evaluate_actions(self, x, actions):
        '''상태 x로부터 상태가치, 실제 행동 actions의 로그 확률, 엔트로피를 계산'''
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        action_log_probs = log_probs.gather(1, actions.to(device))  # 실제 행동의 로그 확률(log_probs)을 구함

        probs = F.softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대한 계산
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy
# 에이전트의 두뇌 역할을 하는 클래스. 모든 에이전트가 공유한다


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
        next_action=next_action+1
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


class Brain(object):
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic.to(device)  # actor_critic은 Net 클래스로 구현한 신경망
        # self.optimizer = optim.RMSprop(self.actor_critic.parameters(), lr=lr, eps=eps, alpha=alpha)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

    def update(self, rollouts):
        '''Advantage학습의 대상이 되는 5단계 모두를 사용하여 수정'''
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, 3,12, 12),
            rollouts.actions.view(-1, 1))

        # 주의 : 각 변수의 크기
        # rollouts.observations[:-1].view(-1, 4) torch.Size([80, 4])
        # rollouts.actions.view(-1, 1) torch.Size([80, 1])
        # values torch.Size([80, 1])
        # action_log_probs torch.Size([80, 1])
        # entropy torch.Size([])

        values = values.view(num_steps, num_processes,1)  # torch.Size([160, 1]) ->([5, 32, 1])

        action_log_probs = action_log_probs.view(num_steps, num_processes, 1) # torch.Size([160, 1]) ->([5, 32, 1])

        # advantage(행동가치-상태가치) 계산
        # print(rollouts.returns[:-1])
        # print(values)
        # print("?@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # sleep(5)
        # print( rollouts.returns[:-1])
        advantages = (rollouts.returns[:-1].to(device) - values)  # torch.Size([5, 32, 1])

        # Critic의 loss 계산
        value_loss = advantages.pow(2).mean()

        # Actor의 gain 계산, 나중에 -1을 곱하면 loss가 된다
        # print(action_log_probs.mean(),"ac.mean")
        # print(action_log_probs.max(), "ac.max")
        #
        # print(advantages.mean(),"advan.mean")
        # print(advantages.max(),"advan.max")
        action_gain = (action_log_probs*advantages.detach()).mean()
        # detach 메서드를 호출하여 advantages를 상수로 취급

        # 오차함수의 총합
        total_loss = (value_loss * value_loss_coef -
                      action_gain - entropy * entropy_coef)
        return_loss=total_loss
        # 결합 가중치 수정
        self.actor_critic.train()  # 신경망을 학습 모드로 전환
        self.optimizer.zero_grad()  # 경사를 초기화
        total_loss.backward()  # 역전파 계산

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        # 결합 가중치가 한번에 너무 크게 변화하지 않도록, 경사를 0.5 이하로 제한함(클리핑)

        self.optimizer.step()  # 결합 가중치 수정

        return return_loss,value_loss,action_gain,entropy

def pop_up(map):

    my=np.zeros((map.shape[0],map.shape[1]))
    ener=np.zeros((map.shape[0],map.shape[1]))
    wall=np.zeros((map.shape[0],map.shape[1]))

    for i in range(len(map[0])):
        for j in range(len(map[1])):

            if(map[i][j]==-1):
                wall[i][j]=1
            elif (map[i][j] == -2):
                my[i][j] = 1
            elif (map[i][j] == -3):
                ener[i][j] = 1
            elif (map[i][j] == -10):
                ener[i][j] = 10
            elif (map[i][j] == 10):
                my[i][j] = 10

    wall=wall.reshape(1,wall.shape[0],wall.shape[1])
    ener = ener.reshape(1, ener.shape[0], ener.shape[1])
    my = my.reshape(1, my.shape[0], my.shape[1])

    wall=torch.from_numpy(wall)
    ener=torch.from_numpy(ener)
    my=torch.from_numpy(my)

    return np.concatenate((wall,my,ener),axis=0)

def make_game():
    x1 = random.randint(0, MAP_WIDTH - 1)
    y1 = random.randint(0, MAP_HEIGHT - 1)
    x2 = random.randint(0, MAP_WIDTH - 1)
    y2 = random.randint(0, MAP_HEIGHT - 1)

    while x1 == x2 and y1 == y2:
        x1 = random.randint(0, MAP_WIDTH - 1)
        y1 = random.randint(0, MAP_HEIGHT - 1)
    # Initialize the game

    player1 = player()
    player2 = player()
    #
    game = Game(MAP_WIDTH, MAP_HEIGHT, [
        PositionPlayer(1, player1, [x1, y1]),
        PositionPlayer(2, player2, [x2, y2]), ])

    return game

def train():
    '''실행 엔트리 포인트'''
    total_loss_sum2 = 0
    val_loss_sum2 = 0
    entropy_sum2 = 0
    act_loss_sum2 = 0
    max=0
    min=0
    total_loss_sum1 = 0
    val_loss_sum1 = 0
    entropy_sum1 = 0
    act_loss_sum1 = 0

    vis = visdom.Visdom()
    vis.close(env="main")
    # 동시 실행할 환경 수 만큼 env를 생성
    total_loss_plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='total_tracker', legend=['loss'], showlegend=True))
    act_plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='act_tracker', legend=['act'], showlegend=True))
    entropy_plot = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='entropy_tracker', legend=['entropy'], showlegend=True))
    value_plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='value_tracker', legend=['value'], showlegend=True))
    duration_plot = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='duration', legend=['duration'], showlegend=True))


    envs = [make_game() for i in range(NUM_PROCESSES)]


    # 모든 에이전트가 공유하는 Brain 객체를 생성
    # n_in = envs[0].observation_space.shape[0]  # 상태 변수 수는 12x12
    # n_out = envs[0].action_space.n  # 행동 가짓수는 4
    # n_mid = 32

    actor_critic = Net()  # 신경망 객체 생성
    global_brain = Brain(actor_critic)


    rollouts1 = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES)  # rollouts 객체
    episode_rewards1 = torch.zeros([NUM_PROCESSES, 1])  # 현재 에피소드의 보상
    obs_np1 = np.zeros([NUM_PROCESSES,12,12])  # Numpy 배열 # 게임 상황이 12x12임
    reward_np1 = np.zeros([NUM_PROCESSES, 1])  # Numpy 배열
    save_reward1 = np.zeros([NUM_PROCESSES])
    each_step1 = np.zeros(NUM_PROCESSES)  # 각 환경의 단계 수를 기록

    rollouts2 = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES)  # rollouts 객체
    episode_rewards2 = torch.zeros([NUM_PROCESSES, 1])  # 현재 에피소드의 보상
    obs_np2 = np.zeros([NUM_PROCESSES,12, 12])  # Numpy 배열 # 게임 상황이 12x12임
    reward_np2 = np.zeros([NUM_PROCESSES, 1])  # Numpy 배열
    save_reward2 = np.zeros([NUM_PROCESSES])
    each_step2 = np.zeros(NUM_PROCESSES)  # 각 환경의 단계 수를 기록

    episode = 0  # 환경 0의 에피소드 수
    done_np = np.zeros([NUM_PROCESSES, 1])  # Numpy 배열

    # 초기 상태로부터 시작

    obs1 = [pop_up(envs[i].map().state_for_player(1)) for i in range(NUM_PROCESSES)]

    obs1 = np.array(obs1)
    obs1 = torch.from_numpy(obs1).float()  # torch.Size([32, 4])

    current_obs1 = obs1  # 가장 최근의 obs를 저장

    obs2 = [pop_up(envs[i].map().state_for_player(2)) for i in range(NUM_PROCESSES)]

    obs2 = np.array(obs2)
    obs2 = torch.from_numpy(obs2).float()  # torch.Size([32, 4])

    current_obs2 = obs2  # 가장 최근의 obs를 저장

    # advanced 학습에 사용되는 객체 rollouts 첫번째 상태에 현재 상태를 저장
    rollouts1.observations[0].copy_(current_obs1)
    rollouts2.observations[0].copy_(current_obs2)
    gamecount=0
    losscount=0
    duration=0
    # 1 에피소드에 해당하는 반복문
    while(True):  # 전체 for문
        # advanced 학습 대상이 되는 각 단계에 대해 계산
        for step in range(NUM_ADVANCED_STEP):

            # 행동을 선택
            with torch.no_grad():
                action1 = actor_critic.act(rollouts1.observations[step]).to(device)
                action2 = actor_critic.act(rollouts2.observations[step]).to(device)

            # (32,1)→(32,) -> tensor를 NumPy변수로

            actions1 = action1.squeeze(1).cpu().numpy()
            actions2 = action2.squeeze(1).cpu().numpy()

            # 한 단계를 실행

            for i in range(NUM_PROCESSES):
                # if (i == 0):
                #     action1[i] = minimax.action(envs[i].map(), 1)
                #     action2[i] = minimax.action(envs[i].map(), 2)
                # print(actions1[i])
                # print(actions2[i])
                # print(envs[i].map().state_for_player(1))
                # print(envs[i].map().state_for_player(2))
                # print(obs_np2[i])
                obs_np1[i], reward_np1[i],obs_np2[i], reward_np2[i], done_np[i] = envs[i].step(actions1[i],actions2[i])
                # print(obs_np1[i])
                # if(gamecount>10000):
                #     if(i==0):
                #         sleep(3)
                #         print(action1[i])
                #         print(obs_np1[i])
                # print(obs_np2[i])
                each_step1[i] += 1
                each_step2[i] += 1

                if done_np[i]:

                    if envs[i].winner is None:

                        reward_np1[i]=0.0
                        reward_np2[i]=0.0

                    elif envs[i].winner == 1:

                        reward_np1[i] = 100.0
                        reward_np2[i] = -100.0
                    else:
                        reward_np1[i] = -100.0
                        reward_np2[i] = 100.0
                    envs[i]=make_game()

                    if(i==0):
                        gamecount += 1
                        duration += each_step1[i]

                    if (i == 0):
                        if(gamecount%SHOW_ITER==0):

                            vis.line(X=torch.tensor([gamecount]),
                                     Y=torch.tensor([duration/SHOW_ITER]),
                                     win=duration_plot,
                                     update='append'
                                     )
                            duration=0
                        # print(reward_np1[i], "reward")
                        # print(each_step1[i], "step")

                    obs_np1[i] = envs[i].map().state_for_player(1)
                    obs_np2[i] = envs[i].map().state_for_player(2)
                    each_step1[i]=0
                    each_step2[i]=0
                    save_reward1[i] = 0.0;
                    save_reward2[i] = 0.0;
                else:
                    save_reward1[i] += 1.0;
                    save_reward2[i] += 1.0;
                    reward_np1[i] = save_reward1[i] # 그 외의 경우는 보상 0 부여
                    reward_np2[i] = save_reward2[i]


            # 보상을 tensor로 변환하고, 에피소드의 총보상에 더해줌
            reward1 = torch.from_numpy(reward_np1).float()
            episode_rewards1 += reward1

            reward2 = torch.from_numpy(reward_np2).float()
            episode_rewards2 += reward2

            # 각 실행 환경을 확인하여 done이 true이면 mask를 0으로, false이면 mask를 1로
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done_np])


            # current_obs를 업데이트
            obs1 = [pop_up(obs_np1[i]) for i in range(NUM_PROCESSES)]
            obs2 = [pop_up(obs_np2[i]) for i in range(NUM_PROCESSES)]
            # obs1 = torch.from_numpy(pop_up(obs_np1)).float()
            # obs2 = torch.from_numpy(pop_up(obs_np2)).float()
            obs1=torch.tensor(np.array(obs1))
            obs2 = torch.tensor(np.array(obs2))
            current_obs1 = obs1  # 최신 상태의 obs를 저장
            current_obs2 = obs2  # 최신 상태의 obs를 저장

            # 메모리 객체에 현 단계의 transition을 저장
            rollouts1.insert(current_obs1, action1.data, reward1, masks)
            rollouts2.insert(current_obs2, action2.data, reward2, masks)

        # advanced 학습 for문 끝

        # advanced 학습 대상 중 마지막 단계의 상태로 예측하는 상태가치를 계산

        with torch.no_grad():
            next_value1 = actor_critic.get_value(rollouts1.observations[-1]).detach()
            # next_value2 = actor_critic.get_value(rollouts2.observations[-1]).detach()
            # rollouts.observations의 크기는 torch.Size([6, 32, 4])

        # 모든 단계의 할인총보상을 계산하고, rollouts의 변수 returns를 업데이트
        rollouts1.compute_returns(next_value1)
        # rollouts2.compute_returns(next_value2)

        # 신경망 및 rollout 업데이트
        loss1,val1,act1,entro1=global_brain.update(rollouts1)
        # loss2,val2,act2,entro2=global_brain.update(rollouts2)
        losscount+=1

        act_loss_sum1+=act1
        entropy_sum1+=entro1
        val_loss_sum1+=val1
        total_loss_sum1+=loss1
        #
        # act_loss_sum2 += act2
        # entropy_sum2 += entro2
        # val_loss_sum2 += val2
        # total_loss_sum2 += loss2

        if(losscount%SHOW_ITER==0):
            total_loss_sum1 =total_loss_sum1 / SHOW_ITER
            val_loss_sum1=val_loss_sum1 / SHOW_ITER
            act_loss_sum1=act_loss_sum1 / SHOW_ITER
            entropy_sum1= entropy_sum1 / SHOW_ITER

            if(val_loss_sum1>max):
                max=val_loss_sum1
            if (total_loss_sum1 < min):
                min=act_loss_sum1

            print(max,"maxxxxxxxxxxxxxxxxxxxx")
            print(min,"miiiiiiiiiiiiiin")
            print(total_loss_sum1,":total1")
            print(val_loss_sum1, ":val1")
            print(act_loss_sum1, ":act1")
            print(entropy_sum1, ":entropy1",end="\n\n")

            # print(total_loss_sum2/SHOW_ITER, ":total2")
            # print(val_loss_sum2/SHOW_ITER, ":val2")
            # print(act_loss_sum2/SHOW_ITER, ":act2")
            # print(entropy_sum2/SHOW_ITER, ":entropy2",end="\n\n")

            vis.line(X=torch.tensor([losscount]),
                     Y=torch.tensor([total_loss_sum1]),
                     win=total_loss_plot,
                     update='append'
                     )
            vis.line(X=torch.tensor([losscount]),
                     Y=torch.tensor([val_loss_sum1]),
                     win=value_plot,
                     update='append'
                     )
            vis.line(X=torch.tensor([losscount]),
                     Y=torch.tensor([act_loss_sum1]),
                     win=act_plot,
                     update='append'
                     )
            vis.line(X=torch.tensor([losscount]),
                     Y=torch.tensor([entropy_sum1]),
                     win=entropy_plot,
                     update='append'
                     )
            act_loss_sum1 =0
            entropy_sum1 =0
            val_loss_sum1 =0
            total_loss_sum1 =0

            # act_loss_sum2 =0
            # entropy_sum2 =0
            # val_loss_sum2 =0
            # total_loss_sum2 =0

        rollouts1.after_update()
        # rollouts2.after_update()


# main 실행

def main():
    train()


if __name__ == "__main__":
    main()

