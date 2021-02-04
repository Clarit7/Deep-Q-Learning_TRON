from torch import optim
# from Net.DQNNet import Net as DQNNET

from Net.kfac import KFACOptimizer
from tron.util import *
from config import *

reward1 = torch.as_tensor(5).float()
reward2 = torch.as_tensor(5).float()
reward1_crash = torch.as_tensor(-1).float()
reward2_crash = torch.as_tensor(-1).float()


def pop_up(map):
    my = np.zeros((map.shape[0],map.shape[1]))
    enem = np.zeros((map.shape[0],map.shape[1]))
    wall = np.zeros((map.shape[0],map.shape[1]))

    find_my_head = False
    find_enem_head = False

    for i in range(len(map[0])):
        for j in range(len(map[1])):
            if map[i][j] == -1:
                wall[i][j]=1
            elif map[i][j] == -2:
                my[i][j] = 1
            elif map[i][j] == -3:
                enem[i][j] = 1
            elif map[i][j] == -10:
                enem[i][j] = 10
                find_enem_head = True
                head_i = i
                head_j = j
                if i == 0 or i == len(map[0]) - 1 or j == 0 or j == len(map[1]):
                    wall[i][j] = 1
            elif map[i][j] == 10:
                my[i][j] = 10
                find_my_head = True
                head_i = i
                head_j = j
                if i == 0 or i == len(map[0]) - 1 or j == 0 or j == len(map[1]):
                    wall[i][j] = 1

    if not find_enem_head:
        enem[head_i][head_j] = 10

    if not find_my_head:
        my[head_i][head_j] = 10

    wall = wall.reshape(1,wall.shape[0],wall.shape[1])
    enem = enem.reshape(1, enem.shape[0], enem.shape[1])
    my = my.reshape(1, my.shape[0], my.shape[1])

    wall = torch.from_numpy(wall)
    enem = torch.from_numpy(enem)
    my = torch.from_numpy(my)

    return np.concatenate((wall,my,enem),axis=0)


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


def train_dist(env, global_brain, ac_dist, writer, distcount, loss_dict,
               rollouts1_dist=None, rollouts2_dist=None, last_act1=None, last_act2=None):
    '''실행 엔트리 포인트'''
    max_val = 0
    min_loss = 0

    # 초기 상태로부터 시작
    each_step1_dist = np.zeros([1])  # 각 환경의 단계 수를 기록
    each_step2_dist = np.zeros([1])  # 각 환경의 단계 수를 기록

    obs1 = pop_up(env.map().state_for_player(1))
    obs1 = np.array(obs1)
    obs1 = torch.from_numpy(obs1).float()  # torch.Size([32, 4])

    p2_static = torch.zeros((12, 12))
    p2_head = torch.nonzero(obs1[2] == 10).squeeze(0)
    idx = torch.nonzero(obs1[2]).split(1, dim=1)
    p2_static[idx] = 1
    static_obs1 = torch.where(obs1[1] == 0, obs1[0] + p2_static, obs1[1])
    mask1 = get_mask(static_obs1, p2_head[0].item(), p2_head[1].item(), torch.zeros((12, 12))) + p2_static

    obs2 = pop_up(env.map().state_for_player(2))
    obs2 = np.array(obs2)
    obs2 = torch.from_numpy(obs2).float()  # torch.Size([32, 4])

    p1_static = torch.zeros((12, 12))
    p1_head = torch.nonzero(obs2[2] == 10).split(1, dim=1)
    idx = torch.nonzero(obs2[2]).split(1, dim=1)
    p1_static[idx] = 1
    static_obs2 = torch.where(obs2[1] == 0, obs2[0] + p1_static, obs2[1])
    mask2 = get_mask(static_obs2, p1_head[0].item(), p1_head[1].item(), torch.zeros((12, 12))) + p1_static

    static_obs1 = torch.cat((torch.zeros((1, 12, 12)), static_obs1.unsqueeze(0)), dim=0)
    static_obs2 = torch.cat((torch.zeros((1, 12, 12)), static_obs2.unsqueeze(0)), dim=0)

    static_obs1[0, p1_head[0].item(), p1_head[1].item()] = 10
    static_obs1[1, p1_head[0].item(), p1_head[1].item()] = 0

    static_obs2[0, p2_head[0].item(), p2_head[1].item()] = 10
    static_obs2[1, p2_head[0].item(), p2_head[1].item()] = 0

    current_obs1 = static_obs1  # 가장 최근의 obs를 저장
    current_obs2 = static_obs2  # 가장 최근의 obs를 저장

    # advanced 학습에 사용되는 객체 rollouts 첫번째 상태에 현재 상태를 저장
    if distcount == 0:
        rollouts1_dist.observations[0].copy_(current_obs1)
        rollouts2_dist.observations[0].copy_(current_obs2)
    else:
        rollouts1_dist.insert(current_obs1, last_act1, torch.as_tensor(reward1_crash).float(), torch.FloatTensor([0.0]))
        rollouts2_dist.insert(current_obs1, last_act2, torch.as_tensor(reward2_crash).float(), torch.FloatTensor([0.0]))

    # 1 에피소드에 해당하는 반복문
    true_done = False

    while not true_done:  # 전체 for문
        # advanced 학습 대상이 되는 각 단계에 대해 계산
        with torch.no_grad():
            action1 = ac_dist.act(rollouts1_dist.observations[rollouts1_dist.index].unsqueeze(0))
            action2 = ac_dist.act(rollouts2_dist.observations[rollouts2_dist.index].unsqueeze(0))

        # (32,1)→(32,) -> tensor를 NumPy변수로
        actions1 = action1.squeeze(1).to('cpu').numpy()
        actions2 = action2.squeeze(1).to('cpu').numpy()

        # 한 단계를 실행
        act1 = actions1
        act2 = actions2

        obs_np1, obs_np2, done_np, sep, true_done, crash1, crash2 = env.step_dist(act1,act2)

        each_step1_dist += 1 if not crash1 else 0
        each_step2_dist += 1 if not crash2 else 0

        # 각 실행 환경을 확인하여 done이 true이면 mask를 0으로, false이면 mask를 1로
        masks1_dist = torch.FloatTensor([1.0])
        masks2_dist = torch.FloatTensor([1.0])

        # 보상을 tensor로 변환하고, 에피소드의 총보상에 더해줌

        if not crash1:
            # current_obs를 업데이트
            obs1 = pop_up(obs_np1)
            obs1 = torch.tensor(np.array(obs1))
            p1_head = torch.nonzero(obs1[1] == 10).squeeze()

            static_obs1 = torch.where(obs1[1] == 0, obs1[0] + mask1, obs1[1])
            static_obs1 = torch.cat((torch.zeros((1, 12, 12)), static_obs1.unsqueeze(0)), dim=0)

            static_obs1[0, p1_head[0].item(), p1_head[1].item()] = 10
            static_obs1[1, p1_head[0].item(), p1_head[1].item()] = 0

            current_obs1 = static_obs1  # 최신 상태의 obs를 저장

            # 메모리 객체에 현 단계의 transition을 저장
            rollouts1_dist.insert(current_obs1, action1.data, reward1, masks1_dist)

        if not crash2:
            obs2 = pop_up(obs_np2)
            obs2 = torch.tensor(np.array(obs2))
            p2_head = torch.nonzero(obs2[1] == 10).squeeze()

            static_obs2 = torch.where(obs2[1] == 0, obs2[0] + mask2, obs2[1])
            static_obs2 = torch.cat((torch.zeros((1, 12, 12)), static_obs2.unsqueeze(0)), dim=0)

            static_obs2[0, p2_head[0].item(), p2_head[1].item()] = 10
            static_obs2[1, p2_head[0].item(), p2_head[1].item()] = 0

            current_obs2 = static_obs2  # 최신 상태의 obs를 저장

            rollouts2_dist.insert(current_obs2, action2.data, reward2, masks2_dist)

            # advanced 학습 끝
        if rollouts1_dist.index == 0:
            with torch.no_grad():
                # advanced 학습 대상 중 마지막 단계의 상태로 예측하는 상태가치를 계산
                next_value1 = ac_dist.get_value(rollouts1_dist.observations[-1].unsqueeze(0))
                # rollouts.observations의 크기는 torch.Size([6, 32, 4])

            rollouts1_dist.compute_returns(next_value1)
            rollouts1_dist.after_update()
            loss1, val1, act1, entro1, prob1, advan1 = global_brain.update(rollouts1_dist)

            loss_dict['update'] += 1
            loss_dict['act_gain'] += act1
            loss_dict['entropy'] += entro1
            loss_dict['value'] += val1
            loss_dict['loss'] += loss1
            loss_dict['prob'] += prob1
            loss_dict['advan'] += advan1

            if loss_dict['update'] % SHOW_ITER == 0:
                loss_dict['act_gain'] /= SHOW_ITER
                loss_dict['entropy'] /= SHOW_ITER
                loss_dict['value'] /= SHOW_ITER
                loss_dict['loss'] /= SHOW_ITER
                loss_dict['prob'] /= SHOW_ITER
                loss_dict['advan'] /= SHOW_ITER

                if loss_dict['value'] > max_val:
                    max_val = loss_dict['value']
                if loss_dict['loss'] < min_loss:
                    min_loss = loss_dict['loss']

                torch.save(global_brain.ac_dist.state_dict(), 'save/' + 'ACKTR_dist' + '.bak')

                writer.add_scalar('Training loss', loss_dict['loss'], loss_dict['update'])
                writer.add_scalar('Value loss', loss_dict['value'], loss_dict['update'])
                writer.add_scalar('Action gain', loss_dict['act_gain'], loss_dict['update'])
                writer.add_scalar('Entropy loss', loss_dict['entropy'], loss_dict['update'])
                writer.add_scalar('Action log probability', loss_dict['prob'], loss_dict['update'])
                writer.add_scalar('Advantage', loss_dict['advan'], loss_dict['update'])

                loss_dict['act_gain'] = 0
                loss_dict['entropy'] = 0
                loss_dict['value'] = 0
                loss_dict['loss'] = 0
                loss_dict['prob'] = 0
                loss_dict['advan'] = 0

        if rollouts2_dist.index == 0:
            with torch.no_grad():
                next_value2 = ac_dist.get_value(rollouts2_dist.observations[-1].unsqueeze(0))

            rollouts2_dist.compute_returns(next_value2)
            rollouts2_dist.after_update()
            global_brain.update(rollouts2_dist)

    loss_dict['duration'] += (each_step1_dist + each_step2_dist) / 2

    if distcount % SHOW_ITER == 0:
        loss_dict['duration'] /= SHOW_ITER
        writer.add_scalar('Static duration', loss_dict['duration'], distcount)
        loss_dict['duration'] = 0

    print('Static %d Episode: Finished after %d, %d steps' % (distcount, each_step1_dist, each_step2_dist))
    return each_step1_dist, each_step2_dist, loss_dict, action1.data, action2.data