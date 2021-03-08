from tron.util import *
from torch.utils.tensorboard import SummaryWriter

def main():
    iter=10000

    area_sum = 0
    len_sum = 0

    for i in range(iter):
        game = make_static_game(False)

        area, len = game.for_test()

        area_sum += area
        len_sum += len

        if area != 0:
            print(float(len) / float(area))
        else:
            print('area 0')

    win_ratio = float(len_sum) / float(area_sum)
    print("total win ratio : {}\n ".format(win_ratio))

    writer = SummaryWriter('runs/' + str(MAP_WIDTH) + '_greedy')

    for i in range(60000):
        writer.add_scalar('Area_ratio', win_ratio, i)

if __name__ == '__main__':
    main()
