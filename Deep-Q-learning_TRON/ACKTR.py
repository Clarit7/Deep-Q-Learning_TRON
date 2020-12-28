from Net.ACNet import *
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# from Net.DQNNet import Net as DQNNET

from Net.kfac import KFACOptimizer
from tron.util import *
from config import *

import argparse

minimax = MinimaxPlayer(2, 'voronoi')
folderName='save'


# DQN = DQNNET()
# DQN.load_state_dict(torch.load(folderName+'/DDQN.bak'))
# DQN.eval()


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

# 에이전트의 두뇌 역할을 하는 클래스. 모든 에이전트가 공유한다
class Brain(object):
    def __init__(self, actor_critic,args, acktr=False):
        self.actor_critic = actor_critic.to(device)  # actor_critic은 Net 클래스로 구현한 신경망
        #self.optimizer = optim.RMSprop(self.actor_critic.parameters(), lr=lr, eps=eps, alpha=alpha)

        self.acktr = acktr

        self.policy_loss_coef = policy_loss_coef if args.p is None else float(args.p)
        self.value_loss_coef = value_loss_coef if args.v is None else float(args.v)

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

        # if self.acktr == False:
        #     nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
        #                              self.max_grad_norm)

        self.optimizer.step()  # 결합 가중치 수정

        return total_loss,value_loss,action_gain,entropy,action_log_probs.mean(),radvantages


def train(args):
    '''실행 엔트리 포인트'''
    max_val = 0
    min_loss = 0
    total_loss_sum1 = 0
    val_loss_sum1 = 0
    entropy_sum1 = 0
    act_loss_sum1 = 0
    prob1_loss_sum1 = 0
    advan_loss_sum1 = 0

    p1_win = 0
    game_draw = 0

    ai_p1=True
    ai_p2=True
    p= "1" if args.p is None else args.p
    v = "1" if args.v is None else args.v
    m = "1" if args.m is None else args.m
    r = "1" if args.r is None else args.r
    unique= "" if args.u is None else args.u

    envs = [make_game(ai_p1,ai_p2) for i in range(NUM_PROCESSES)]

    eventid = datetime.now().strftime('runs/ACKTR-%Y%m-%d%H-%M%S-ent ') + str(entropy_coef) + '-pol ' + p + '-val ' + v + '-step' + str(
        NUM_ADVANCED_STEP) + '-process ' + str(NUM_PROCESSES) + unique + '-model ' + m + '-reward ' + r

    writer = SummaryWriter(eventid)


    if args.m == "2":
        actor_critic = Net2()  # 신경망 객체 생성
    elif args.m == "3":
        actor_critic = Net3()
    else:
        actor_critic = Net()

    global_brain = Brain(actor_critic,args, acktr=True)

    rollouts1 = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES)  # rollouts 객체
    episode_rewards1 = torch.zeros([NUM_PROCESSES, 1])  # 현재 에피소드의 보상
    obs_np1 = np.zeros([NUM_PROCESSES,12,12])  # Numpy 배열 # 게임 상황이 12x12임
    reward_np1 = np.zeros([NUM_PROCESSES, 1])  # Numpy 배열
    each_step1 = np.zeros(NUM_PROCESSES)  # 각 환경의 단계 수를 기록

    rollouts2 = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES)  # rollouts 객체
    episode_rewards2 = torch.zeros([NUM_PROCESSES, 1])  # 현재 에피소드의 보상
    obs_np2 = np.zeros([NUM_PROCESSES,12, 12])  # Numpy 배열 # 게임 상황이 12x12임
    reward_np2 = np.zeros([NUM_PROCESSES, 1])  # Numpy 배열
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
    gamecount = 0
    losscount = 0
    duration = 0

    if args.r == "2":
        reward_constants = reward_cons2
    elif args.r == "3":
        reward_constants = reward_cons3
    else:
        reward_constants = reward_cons1



    # 1 에피소드에 해당하는 반복문
    while True:  # 전체 for문
        # advanced 학습 대상이 되는 각 단계에 대해 계산
        for step in range(NUM_ADVANCED_STEP):
            # 행동을 선택
            with torch.no_grad():
                action1 = actor_critic.act(rollouts1.observations[step])
                action2 = actor_critic.act(rollouts2.observations[step])

            # (32,1)→(32,) -> tensor를 NumPy변수로
            actions1 = action1.squeeze(1).to('cpu').numpy()
            actions2 = action2.squeeze(1).to('cpu').numpy()

            # 한 단계를 실행
            for i in range(NUM_PROCESSES):
                act1 = actions1[i] if ai_p1 else minimax.action(envs[i].map(), 1)
                act2 = actions2[i] if ai_p2 else minimax.action(envs[i].map(), 2)

                obs_np1[i], reward_np1[i], obs_np2[i], reward_np2[i], done_np[i],loser_len,winner_len = envs[i].step(act1,act2)

                each_step1[i] += 1
                each_step2[i] += 1

                if done_np[i]:
                    reward_np1[i],reward_np2[i]=get_reward(envs[i], reward_constants, winner_len, loser_len)
                    if i == 0:
                        gamecount += 1
                        duration += each_step1[i]+loser_len

                        if gamecount % SHOW_ITER == 0:
                            print('%d Episode: Finished after %d steps' % (gamecount, each_step1[i]))
                            writer.add_scalar('Duration', duration/SHOW_ITER, gamecount)
                            duration = 0

                    envs[i] = make_game(ai_p1,ai_p2)

                    obs_np1[i] = envs[i].map().state_for_player(1)
                    obs_np2[i] = envs[i].map().state_for_player(2)
                    each_step1[i] = 0
                    each_step2[i] = 0
                else:
                    reward_np1[i] = -1  # 그 외의 경우는 보상 0 부여
                    reward_np2[i] = -1

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
        loss1, val1, act1, entro1, prob1, advan1 = global_brain.update(rollouts1)
        global_brain.update(rollouts2)
        losscount += 1

        act_loss_sum1 += act1
        entropy_sum1 += entro1
        val_loss_sum1 += val1
        total_loss_sum1 += loss1
        prob1_loss_sum1 += prob1
        advan_loss_sum1 += advan1

        # if(gamecount>2000):
        #     pygame.init()
        #     game = make_game(True, True)
        #     pygame.mouse.set_visible(False)
        #
        #     window = Window(game, 40)
        #
        #     game.main_loop(global_brain.actor_critic, pop_up, window)

        if losscount%SHOW_ITER == 0:
            total_loss_sum1 = total_loss_sum1 / SHOW_ITER
            val_loss_sum1 = val_loss_sum1 / SHOW_ITER
            act_loss_sum1 = act_loss_sum1 / SHOW_ITER
            entropy_sum1 = entropy_sum1 / SHOW_ITER
            prob1_loss_sum1 /= SHOW_ITER
            advan_loss_sum1 /= SHOW_ITER

            if val_loss_sum1 > max_val:
                max_val = val_loss_sum1
            if total_loss_sum1 < min_loss:
                min_loss = act_loss_sum1

            torch.save(global_brain.actor_critic.state_dict(), 'save/' + 'ACKTR_player'+m + unique +'.bak')
            # torch.save(global_brain2.actor_critic.state_dict(), 'ais/a3c/' + 'player_2.bak')

            writer.add_scalar('Training loss', total_loss_sum1, losscount)
            writer.add_scalar('Value loss', val_loss_sum1, losscount)
            writer.add_scalar('Action gain', act_loss_sum1, losscount)
            writer.add_scalar('Entropy loss', entropy_sum1, losscount)
            writer.add_scalar('Action log probability', prob1_loss_sum1, losscount)
            writer.add_scalar('Advantage', advan_loss_sum1, losscount)

            if losscount%200 == 0:
                for i in range(PLAY_WITH_MINIMAX):
                    game = make_game(True, False)
                    game.main_loop(global_brain.actor_critic, pop_up)

                    if game.winner == 1:
                        p1_win += 1
                    elif game.winner is None:
                        game_draw += 1

                writer.add_scalar('minimax rating', p1_win/(PLAY_WITH_MINIMAX - game_draw), losscount)

            p1_win = 0
            game_draw = 0
            act_loss_sum1 = 0
            entropy_sum1 = 0
            val_loss_sum1 = 0
            total_loss_sum1 = 0
            prob1_loss_sum1 = 0
            advan_loss_sum1 = 0

        rollouts1.after_update()
        rollouts2.after_update()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', required=False, help='model structure number')
    parser.add_argument('-r', required=False, help='reward condition number')

    parser.add_argument('-p', required=False, help='policy coefficient')
    parser.add_argument('-v', required=False, help='value coefficient')
    parser.add_argument('-u', required=False, help='unique string')

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
