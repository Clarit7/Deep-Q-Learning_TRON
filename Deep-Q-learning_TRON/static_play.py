from tron.util import *
from torch.utils.tensorboard import SummaryWriter
from Net.ACNet import *
from ACKTR import Brain
import argparse

def main(args):
    iter=5000
    folderName = 'ex_saves2'

    area_sum = 0
    len_sum = 0

    ac_static = NetStatic6()
    static_brain = Brain(ac_static, args, acktr=True)
    static_brain.actor_critic.load_state_dict(torch.load(
        folderName + '/ACKTR_pretrain-2021.03.08-15_49_13-ent_0.01-pol_1.2-val_0.7-step_5-process_16-size_6-6_40k_pretrain.bak'))
    static_brain.actor_critic.eval()

    for i in range(iter):
        game = make_static_game(True)

        area, len = game.for_test(static_brain=static_brain.actor_critic)

        area_sum += area
        len_sum += len

        if area != 0:
            print(float(len) / float(area))
        else:
            print('area 0')

    win_ratio = float(len_sum) / float(area_sum)
    print("total win ratio : {}\n ".format(win_ratio))

    writer = SummaryWriter('runs/' + str(MAP_WIDTH) + '_pretrain')

    for i in range(60000):
        writer.add_scalar('Area_ratio', win_ratio, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', required=False, help='model structure number')
    parser.add_argument('-b', required=False, help='agent1 number')
    parser.add_argument('-c', required=False, help='agent2 number')
    parser.add_argument('-p', required=False, help='policy coefficient')
    parser.add_argument('-v', required=False, help='value coefficient')
    parser.add_argument('-a', required=False, help='get area instead of length')
    parser.add_argument('-e', required=False, help='True if end condition is separated')
    parser.add_argument('-u', required=False, help='unique string')

    args = parser.parse_args()

    main(args)

