from torch import optim
# from Net.DQNNet import Net as DQNNET

from Net.kfac import KFACOptimizer
from tron.util import *
from config import *

class RolloutStorage(object):
    '''Advantage 학습에 사용할 메모리 클래스'''
    def __init__(self, num_steps):
        self.observations = torch.zeros(num_steps + 1, 1, MAP_WIDTH + 2, MAP_HEIGHT + 2)
        self.masks = torch.ones(num_steps + 1, 1, 1)
        self.rewards = torch.zeros(num_steps, 1, 1)
        self.actions = torch.zeros(num_steps, 1, 1).long()

        # 할인 총보상 저장
        self.returns = torch.zeros(num_steps + 1, 1)
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


def get_mask(game_map, x, y, mask):
    if game_map[x + 1, y] == 0:
        game_map[x + 1, y] = 1
        mask[x + 1, y] = 1
        mask = get_mask(game_map, x + 1, y, mask)

    if game_map[x - 1, y] == 0:
        game_map[x - 1, y] = 1
        mask[x - 1, y] = 1
        mask = get_mask(game_map, x - 1, y, mask)

    if game_map[x, y + 1] == 0:
        game_map[x, y + 1] = 1
        mask[x, y + 1] = 1
        mask = get_mask(game_map, x, y + 1, mask)

    if game_map[x, y - 1] == 0:
        game_map[x, y - 1] = 1
        mask[x, y - 1] = 1
        mask = get_mask(game_map, x, y - 1, mask)

    return mask


def train_dist(env, global_brain, ac_dist, writer):
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

    rollouts1 = RolloutStorage(NUM_ADVANCED_STEP)  # rollouts 객체
    episode_rewards1 = torch.zeros([1])  # 현재 에피소드의 보상
    obs_np1 = np.zeros([MAP_WIDTH + 2,MAP_HEIGHT + 2])  # Numpy 배열 # 게임 상황이 12x12임
    reward_np1 = np.zeros([1])  # Numpy 배열
    each_step1 = np.zeros([1])  # 각 환경의 단계 수를 기록

    rollouts2 = RolloutStorage(NUM_ADVANCED_STEP)  # rollouts 객체
    episode_rewards2 = torch.zeros([1])  # 현재 에피소드의 보상
    obs_np2 = np.zeros([MAP_WIDTH + 2, MAP_HEIGHT + 2])  # Numpy 배열 # 게임 상황이 12x12임
    reward_np2 = np.zeros([1])  # Numpy 배열
    each_step2 = np.zeros([1])  # 각 환경의 단계 수를 기록

    done_np = np.zeros([1])  # Numpy 배열

    # 초기 상태로부터 시작

    test = env.map().state_for_player(1)
    obs1 = pop_up(env.map().state_for_player(1))
    obs1 = np.array(obs1)
    obs1 = torch.from_numpy(obs1).float()  # torch.Size([32, 4])

    p2_static = torch.zeros((12, 12))
    p2_head = torch.nonzero(obs1[2] == 10).squeeze(0)
    idx = torch.nonzero(obs1[2]).split(1, dim=1)
    p2_static[idx] = 1
    static_obs1 = torch.where(obs1[1] == 0, obs1[0] + p2_static, obs1[1])

    mask1 = get_mask(static_obs1, p2_head[0].item(), p2_head[1].item(), torch.zeros((12, 12))) + p2_static
    current_obs1 = static_obs1  # 가장 최근의 obs를 저장

    obs2 = pop_up(env.map().state_for_player(2))
    obs2 = np.array(obs2)
    obs2 = torch.from_numpy(obs2).float()  # torch.Size([32, 4])

    p1_static = torch.zeros((12, 12))
    p1_head = torch.nonzero(obs2[2] == 10).split(1, dim=1)
    idx = torch.nonzero(obs2[2]).split(1, dim=1)
    p1_static[idx] = 1
    static_obs2 = torch.where(obs2[1] == 0, obs2[0] + p1_static, obs2[1])

    mask2 = get_mask(static_obs2, p1_head[0].item(), p1_head[1].item(), torch.zeros((12, 12))) + p1_static
    current_obs2 = static_obs2  # 가장 최근의 obs를 저장

    # advanced 학습에 사용되는 객체 rollouts 첫번째 상태에 현재 상태를 저장
    rollouts1.observations[0].copy_(current_obs1.unsqueeze(0))
    rollouts2.observations[0].copy_(current_obs2.unsqueeze(0))
    gamecount = 0
    losscount = 0
    duration = 0

    reward_constants = reward_cons1

    # 1 에피소드에 해당하는 반복문
    crash1, crash2 = False, False

    while not crash1 and not crash2:  # 전체 for문
        # advanced 학습 대상이 되는 각 단계에 대해 계산
        for step in range(NUM_ADVANCED_STEP):
            # 행동을 선택
            with torch.no_grad():
                action1 = ac_dist.act(rollouts1.observations[step].unsqueeze(0))
                action2 = ac_dist.act(rollouts2.observations[step].unsqueeze(0))

            # (32,1)→(32,) -> tensor를 NumPy변수로
            actions1 = action1.squeeze(1).to('cpu').numpy()
            actions2 = action2.squeeze(1).to('cpu').numpy()

            # 한 단계를 실행
            act1 = actions1
            act2 = actions2

            obs_np1, reward_np1, obs_np2, reward_np2, done_np, loser_len, winner_len, sep, winner, true_done, crash1, crash2 = env.step_dist(act1,act2)

            each_step1 += 1 if not crash1 else 0
            each_step2 += 1 if not crash2 else 0

            if done_np and true_done:
                # if sep:
                #     train_dist(env, ac_dist, global_brain)
                # reward_np1, reward_np2 = get_reward(env, reward_constants, winner_len, loser_len)
                reward_np1 = -1
                reward_np2 = -1

                gamecount += 1
                duration += each_step1 + loser_len

                if gamecount % SHOW_ITER == 0:
                    print('%d Episode: Finished after %d steps' % (gamecount, each_step1))
                    writer.add_scalar('Duration', duration/SHOW_ITER, gamecount)
                    duration = 0
            else:
                reward_np1 = 1  # 그 외의 경우는 보상 0 부여
                reward_np2 = 1

            # 보상을 tensor로 변환하고, 에피소드의 총보상에 더해줌
            reward1 = torch.as_tensor(reward_np1).float()
            episode_rewards1 += reward1

            reward2 = torch.as_tensor(reward_np2).float()
            episode_rewards2 += reward2

            # 각 실행 환경을 확인하여 done이 true이면 mask를 0으로, false이면 mask를 1로
            masks = torch.FloatTensor([0.0] if done_np else [1.0])

            # current_obs를 업데이트
            obs1 = pop_up(obs_np1)
            obs2 = pop_up(obs_np2)
            # obs1 = [obs_np1[i] for i in range(NUM_PROCESSES)]
            # obs2 = [obs_np2[i] for i in range(NUM_PROCESSES)]

            # obs1 = torch.from_numpy(pop_up(obs_np1)).float()
            # obs2 = torch.from_numpy(pop_up(obs_np2)).float()

            obs1 = torch.tensor(np.array(obs1))
            obs2 = torch.tensor(np.array(obs2))

            p2_static = torch.zeros((12, 12))
            idx = torch.nonzero(obs1[2]).split(1, dim=1)
            p2_static[idx] = 1

            p1_static = torch.zeros((12, 12))
            idx = torch.nonzero(obs2[2]).split(1, dim=1)
            p1_static[idx] = 1

            static_obs1 = torch.where(obs1[1] == 0, obs1[0] + mask1, obs1[1])
            static_obs2 = torch.where(obs2[1] == 0, obs2[0] + mask2, obs2[1])

            current_obs1 = static_obs1  # 최신 상태의 obs를 저장
            current_obs2 = static_obs2  # 최신 상태의 obs를 저장

            # 메모리 객체에 현 단계의 transition을 저장
            if not crash1:
                rollouts1.insert(current_obs1.unsqueeze(0), action1.data, reward1, masks)
            if not crash2:
                rollouts2.insert(current_obs2.unsqueeze(0), action2.data, reward2, masks)

        # advanced 학습 for문 끝

        # advanced 학습 대상 중 마지막 단계의 상태로 예측하는 상태가치를 계산

        with torch.no_grad():
            next_value1 = ac_dist.get_value(rollouts1.observations[-1].unsqueeze(0))
            next_value2 = ac_dist.get_value(rollouts2.observations[-1].unsqueeze(0))
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

            torch.save(global_brain.ac_dist.state_dict(), 'save/' + 'ACKTR_dist'+ '.bak')
            # torch.save(global_brain2.actor_critic.state_dict(), 'ais/a3c/' + 'player_2.bak')

            writer.add_scalar('Training loss', total_loss_sum1, losscount)
            writer.add_scalar('Value loss', val_loss_sum1, losscount)
            writer.add_scalar('Action gain', act_loss_sum1, losscount)
            writer.add_scalar('Entropy loss', entropy_sum1, losscount)
            writer.add_scalar('Action log probability', prob1_loss_sum1, losscount)
            writer.add_scalar('Advantage', advan_loss_sum1, losscount)

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

    return each_step1, each_step2