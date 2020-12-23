import numpy as np
import random

from config import *
from tron.game import *
from tron.minimax import MinimaxPlayer
from tron.player import KeyboardPlayer,ACPlayer


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

def make_game(p1,p2):
    x1 = random.randint(0, MAP_WIDTH - 1)
    y1 = random.randint(0, MAP_HEIGHT - 1)
    x2 = random.randint(0, MAP_WIDTH - 1)
    y2 = random.randint(0, MAP_HEIGHT - 1)

    while x1 == x2 and y1 == y2:
        x1 = random.randint(0, MAP_WIDTH - 1)
        y1 = random.randint(0, MAP_HEIGHT - 1)
    # Initialize the game

    game = Game(MAP_WIDTH, MAP_HEIGHT, [
        PositionPlayer(1,  ACPlayer() if p1 else MinimaxPlayer(2, "voronoi"), [x1, y1]),
        PositionPlayer(2,  ACPlayer() if p2 else MinimaxPlayer(2, "voronoi"), [x2, y2]), ])

    return game
def get_reward(game,winner_len=0,loser_len=0):

    if game.winner is None:
        return 0,0

    elif game.winner == 1:

        if loser_len == 0 and winner_len == 0:
            return 10,-10
        else:
            return 10+ 150/loser_len, -10
    else:
        if loser_len == 0:
            return -10, 10

        else:
            return -10,10+ 150/loser_len
