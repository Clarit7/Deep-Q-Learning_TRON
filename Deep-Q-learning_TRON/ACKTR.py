import numpy as np
import gym
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import random
import visdom

from object.DDQN_player import Player, Direction, ACPlayer
from object.DDQN_game import Tile, Game, PositionPlayer
from tron.minimax import MinimaxPlayer
from tron import resnet

from time import sleep
from datetime import datetime

from kfac import KFACOptimizer
from util import *

from config import *


minimax = MinimaxPlayer(2, 'voronoi')

class RolloutStorage(object):
    '''Advantage 학습에 사용할 메모리 클래스'''
    def __init__(self, num_steps, num_processes):

        # self.observations = torch.zeros(num_steps + 1, num_processes,3,12,12)
        self.observations = torch.zeros(num_steps + 1, num_processes, 3, 12, 12)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()

        # 할인 총보상 저장
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.index = 0  # insert할 인덱스

    def insert(self, current_obs, action, reward, mask):
        '''현재 인덱스 위치에 transition을 저장'''
        #
        # self.observations[self.index + 1].copy_(current_obs)
        #
        # self.masks[self.index + 1].copy_(mask)
        # self.rewards[self.index].copy_(reward)
        # self.actions[self.index].copy_(action)
        # print(self.index)
        # self.index = (self.index + 1) % NUM_ADVANCED_STEP  # 인덱스 값 업데이트

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

        self.conv1 = nn.Conv2d(3, 32, 6)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.fc1 = nn.Linear(64 * 5* 5, 64)

        self.actor1 = nn.Linear(64, 64)
        self.actor2 = nn.Linear(64, 4)

        self.critic1 = nn.Linear(64, 64)
        self.critic2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        '''신경망 순전파 계산을 정의'''
        x = x.to(device)

        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))

        x = x.view(-1, 64 * 5 * 5)

        x = self.dropout(F.tanh(self.fc1(x)))

        actor_output = self.actor2(F.tanh(self.actor1(x)))
        critic_output = self.critic2(F.tanh(self.critic1(x)))
        #actor_output = actor_output.to('cpu') # Why???
        #critic_output = critic_output.to('cpu')

        return critic_output, actor_output

    def act(self, x):
        '''상태 x로부터 행동을 확률적으로 결정'''
        value, actor_output = self(x)
        # actor_output=torch.clamp_min_(actor_output,min=0)
        # dim=1이므로 행동의 종류에 대해 softmax를 적용
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        return action

    def deterministic_act(self, x):
        '''상태 x로부터 행동을 확률적으로 결정'''
        value, actor_output = self(x)
        # actor_output=torch.clamp_min_(actor_output,min=0)
        # dim=1이므로 행동의 종류에 대해 softmax를 적용
        return torch.argmax(actor_output, dim=1)

    def get_value(self, x):
        '''상태 x로부터 상태가치를 계산'''
        value, actor_output = self(x)

        return value

    def evaluate_actions(self, x, actions):
        '''상태 x로부터 상태가치, 실제 행동 actions의 로그 확률, 엔트로피를 계산'''
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산

        action_log_probs = log_probs.gather(1, actions.detach())  # 실제 행동의 로그 확률(log_probs)을 구함

        probs = F.softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대한 계산
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy


# 에이전트의 두뇌 역할을 하는 클래스. 모든 에이전트가 공유한다
class Brain(object):
    def __init__(self, actor_critic, acktr=False):
        self.actor_critic = actor_critic.to(device)  # actor_critic은 Net 클래스로 구현한 신경망
        #self.optimizer = optim.RMSprop(self.actor_critic.parameters(), lr=lr, eps=eps, alpha=alpha)
        self.acktr = acktr

        if acktr:
            self.optimizer = KFACOptimizer(self.actor_critic)
        else:
            self.optimizer = optim.RMSprop(
                self.actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        '''Advantage학습의 대상이 되는 5단계 모두를 사용하여 수정'''
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, 3, 12, 12).to(device).detach(),
            rollouts.actions.view(-1, 1).to(device).detach())

        # 주의 : 각 변수의 크기
        # rollouts.observations[:-1].view(-1, 4) torch.Size([80, 4])
        # rollouts.actions.view(-1, 1) torch.Size([80, 1])
        # # values torch.Size([80, 1])
        # action_log_probs torch.Size([80, 1])
        # entropy torch.Size([])

        values = values.view(num_steps, num_processes,1)  # torch.Size([160, 1]) ->([5, 32, 1])

        action_log_probs = action_log_probs.view(num_steps, num_processes, 1) # torch.Size([160, 1]) ->([5, 32, 1])

        # advantage(행동가치-상태가치) 계산
        advantages = rollouts.returns[:-1].to(device).detach() - values  # torch.Size([5, 32, 1])

        # Critic의 loss 계산
        value_loss = advantages.pow(2).mean()

        # Actor의 gain 계산, 나중에 -1을 곱하면 loss가 된다

        radvantages = advantages.detach().mean()
        action_gain = (action_log_probs * advantages.detach()).mean()
        # detach 메서드를 호출하여 advantages를 상수로 취급

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Compute fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()

        # 오차함수의 총합
        total_loss = (value_loss * value_loss_coef -
                      action_gain * policy_loss_coef - entropy * entropy_coef)

        # 결합 가중치 수정
        total_loss.backward()  # 역전파 계산

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()  # 결합 가중치 수정

        return total_loss,value_loss,action_gain,entropy,action_log_probs.mean(),radvantages



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
    prob1_loss_sum1=0
    advan_loss_sum1=0

    envs = [make_game() for i in range(NUM_PROCESSES)]

    eventid = datetime.now().strftime('runs/ACKTR-%Y%m-%d%H-%M%S-ent:') + str(entropy_coef) + '-step:' + str(
        NUM_ADVANCED_STEP) + '-process:' + str(NUM_PROCESSES)
    writer = SummaryWriter(eventid)

    # 모든 에이전트가 공유하는 Brain 객체를 생성
    # n_in = envs[0].observation_space.shape[0]  # 상태 변수 수는 12x12
    # n_out = envs[0].action_space.n  # 행동 가짓수는 4
    # n_mid = 32

    actor_critic = Net()  # 신경망 객체 생성

    # actor_critic.load_state_dict(torch.load('ais/A3C/player_1.bak'))

    #actor_critic2 = Net()  # 신경망 객체 생성
    global_brain = Brain(actor_critic, acktr=True)
    #global_brain2 = Brain(actor_critic2)


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

    done_np = np.zeros([NUM_PROCESSES, 1])  # Numpy 배열

    # 초기 상태로부터 시작

    obs1 = [pop_up(envs[i].map().state_for_player(1)) for i in range(NUM_PROCESSES)]
    # obs1 = [envs[i].map().state_for_player(1) for i in range(NUM_PROCESSES)]
    obs1 = np.array(obs1)
    obs1 = torch.from_numpy(obs1).float()  # torch.Size([32, 4])

    current_obs1 = obs1  # 가장 최근의 obs를 저장

    obs2 = [pop_up(envs[i].map().state_for_player(2)) for i in range(NUM_PROCESSES)]
    # obs2 = [envs[i].map().state_for_player(2) for i in range(NUM_PROCESSES)]

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
                action1 = actor_critic.act(rollouts1.observations[step])
                action2 = actor_critic.act(rollouts2.observations[step])

            # (32,1)→(32,) -> tensor를 NumPy변수로
            # actions1 = action1.squeeze(1).to('cpu').numpy()
            # actions2 = action2.squeeze(1).to('cpu').numpy()

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
                # obs_np1[i], reward_np1[i],obs_np2[i], reward_np2[i], done_np[i] = envs[i].step(actions1[i],actions2[i])
                obs_np1[i], reward_np1[i], obs_np2[i], reward_np2[i], done_np[i] = envs[i].step(minimax.action(envs[i].map(), 1),
                                                                                                minimax.action(envs[i].map(), 2))
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
                        reward_np1[i] = 0
                        reward_np2[i] = 0
                    elif envs[i].winner == 1:
                        reward_np1[i] = 10
                        reward_np2[i] = -10
                    else:
                        reward_np1[i] = -10
                        reward_np2[i] = 10

                    if (i == 0):
                        gamecount += 1
                        duration += each_step1[i]

                    if (i == 0):
                        if(gamecount % SHOW_ITER==0):
                            print('%d Episode: Finished after %d steps' % (gamecount, each_step1[i]))
                            writer.add_scalar('Duration', duration/SHOW_ITER, gamecount)
                            duration=0
                        # print(reward_np1[i], "reward")
                        # print(each_step1[i], "step")
                    envs[i] = make_game()
                    obs_np1[i] = envs[i].map().state_for_player(1)
                    obs_np2[i] = envs[i].map().state_for_player(2)
                    each_step1[i] = 0
                    each_step2[i] = 0
                    # save_reward1[i] = 0.0;
                    # save_reward2[i] = 0.0;
                else:
                    # save_reward1[i] += 1.0;
                    # save_reward2[i] += 1.0;
                    # reward_np1[i] = save_reward1[i] # 그 외의 경우는 보상 0 부여
                    # reward_np2[i] = save_reward2[i]
                    reward_np1[i] = 1 # 그 외의 경우는 보상 0 부여
                    reward_np2[i] = 1


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
            # obs1 = [obs_np1[i] for i in range(NUM_PROCESSES)]
            # obs2 = [obs_np2[i] for i in range(NUM_PROCESSES)]
            # obs1 = torch.from_numpy(pop_up(obs_np1)).float()
            # obs2 = torch.from_numpy(pop_up(obs_np2)).float()

            obs1 = torch.tensor(np.array(obs1))
            obs2 = torch.tensor(np.array(obs2))

            current_obs1 = obs1  # 최신 상태의 obs를 저장
            current_obs2 = obs2  # 최신 상태의 obs를 저장

            # 메모리 객체에 현 단계의 transition을 저장
            rollouts1.insert(current_obs1, action1.data, reward1, masks)
            rollouts2.insert(current_obs2, action2.data, reward2, masks)

        # advanced 학습 for문 끝

        # advanced 학습 대상 중 마지막 단계의 상태로 예측하는 상태가치를 계산

        with torch.no_grad():
            next_value1 = actor_critic.get_value(rollouts1.observations[-1])
            next_value2 = actor_critic.get_value(rollouts2.observations[-1])
            # rollouts.observations의 크기는 torch.Size([6, 32, 4])

        # 모든 단계의 할인총보상을 계산하고, rollouts의 변수 returns를 업데이트
        rollouts1.compute_returns(next_value1)
        rollouts2.compute_returns(next_value2)

        # 신경망 및 rollout 업데이트
        loss1,val1,act1,entro1,prob1,advan1=global_brain.update(rollouts1)
        loss2,val2,act2,entro2,prob2,advan2=global_brain.update(rollouts2)
        losscount+=1



        act_loss_sum1+=act1
        entropy_sum1+=entro1
        val_loss_sum1+=val1
        total_loss_sum1+=loss1
        prob1_loss_sum1+=prob1
        advan_loss_sum1+=advan1

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
            prob1_loss_sum1 /=SHOW_ITER
            advan_loss_sum1 /=SHOW_ITER

            if(val_loss_sum1>max):
                max=val_loss_sum1
            if (total_loss_sum1 < min):
                min=act_loss_sum1

            #print(max,"maxxxxxxxxxxxxxxxxxxxx")
            #print(min,"miiiiiiiiiiiiiin")
            #print(total_loss_sum1,":total1")
            #print(val_loss_sum1, ":val1")
            #print(act_loss_sum1, ":act1")
            #print(entropy_sum1, ":entropy1",end="\n\n")

            # print(total_loss_sum2/SHOW_ITER, ":total2")
            # print(val_loss_sum2/SHOW_ITER, ":val2")
            # print(act_loss_sum2/SHOW_ITER, ":act2")
            # print(entropy_sum2/SHOW_ITER, ":entropy2",end="\n\n")
            # print(losscount)
            torch.save(global_brain.actor_critic.state_dict(), './ais/ACKTR/' + 'ACKTR_player.bak')
            # torch.save(global_brain2.actor_critic.state_dict(), 'ais/a3c/' + 'player_2.bak')

            writer.add_scalar('Training loss', total_loss_sum1, losscount)
            writer.add_scalar('Value loss', val_loss_sum1, losscount)
            writer.add_scalar('Action gain', act_loss_sum1, losscount)
            writer.add_scalar('Entropy loss', entropy_sum1, losscount)
            writer.add_scalar('Action log probability', prob1_loss_sum1, losscount)
            writer.add_scalar('Advantage', advan_loss_sum1, losscount)


            act_loss_sum1 =0
            entropy_sum1 =0
            val_loss_sum1 =0
            total_loss_sum1 =0
            prob1_loss_sum1=0
            advan_loss_sum1=0


            # act_loss_sum2 =0
            # entropy_sum2 =0
            # val_loss_sum2 =0
            # total_loss_sum2 =0

        rollouts1.after_update()
        rollouts2.after_update()


# main 실행

def main():
    train()


if __name__ == "__main__":
    main()