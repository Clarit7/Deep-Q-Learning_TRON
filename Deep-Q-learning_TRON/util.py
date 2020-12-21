import numpy as np
import random

from config import *
from object.DDQN_game import *
from object.DDQN_player import *


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

    player1 = ACPlayer()
    player2 = ACPlayer()
    # player1 = MinimaxPlayer(2, "voronoi")
    # player2 = MinimaxPlayer(2, "voronoi")

    #
    game = Game(MAP_WIDTH, MAP_HEIGHT, [
        PositionPlayer(1, player1, [x1, y1]),
        PositionPlayer(2, player2, [x2, y2]), ])

    return game