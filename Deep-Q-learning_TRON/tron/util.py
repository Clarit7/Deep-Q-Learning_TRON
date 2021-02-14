import numpy as np
import random

from config import *
from tron.game import *
from tron.static_game import StaticGame
from tron.minimax import MinimaxPlayer
from tron.player import KeyboardPlayer,ACPlayer

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

def pop_up_static(map):
    my = np.zeros((map.shape[0],map.shape[1]))
    wall = np.zeros((map.shape[0],map.shape[1]))

    for i in range(len(map[0])):
        for j in range(len(map[1])):
            if map[i][j] == -1:
                wall[i][j]=1
            elif map[i][j] == -2:
                my[i][j] = 1
            elif map[i][j] == -3 or map[i][j] == -10:
                wall[i][j] = 1
                if i == 0 or i == len(map[0]) - 1 or j == 0 or j == len(map[1]):
                    wall[i][j] = 1
            elif map[i][j] == 10:
                my[i][j] = 10
                if i == 0 or i == len(map[0]) - 1 or j == 0 or j == len(map[1]):
                    wall[i][j] = 1

    wall = wall.reshape(1,wall.shape[0],wall.shape[1])
    my = my.reshape(1, my.shape[0], my.shape[1])

    wall = torch.from_numpy(wall)
    my = torch.from_numpy(my)

    return np.concatenate((wall,my),axis=0)

def make_game(p1,p2,mode=None):

    if mode == "fair":
        point_y=random.randint(0, MAP_HEIGHT - 1)
        point_x=random.randint(0, MAP_WIDTH - 1)


        low_bound1_x = max(0, point_x-1)
        upper_bound1_x = min(MAP_WIDTH - 1, point_x+1)
        low_bound1_y = max(0, point_y-1)
        upper_bound1_y = min(MAP_HEIGHT - 1, point_y+1)

        low_bound2_x = MAP_WIDTH - 1 - upper_bound1_x
        upper_bound2_x = MAP_WIDTH - 1 - low_bound1_x

        low_bound2_y = MAP_HEIGHT - 1 - upper_bound1_y
        upper_bound2_y = MAP_HEIGHT - 1 - low_bound1_y

    else:

        low_bound1_x,low_bound2_y,low_bound1_y,low_bound2_x = 0,0,0,0
        upper_bound1_x ,upper_bound2_x= MAP_WIDTH - 1,MAP_WIDTH - 1
        upper_bound1_y,upper_bound2_y = MAP_HEIGHT - 1,MAP_HEIGHT - 1

    x1 = random.randint(low_bound1_x, upper_bound1_x)
    y1 = random.randint(low_bound1_y, upper_bound1_y)

    x2 = random.randint(low_bound2_x, upper_bound2_x)
    y2 = random.randint(low_bound2_y, upper_bound2_y)

    while x1 == x2 and y1 == y2:
        x1 = random.randint(low_bound1_x, upper_bound1_x)
        y1 = random.randint(low_bound1_y, upper_bound1_y)
    # Initialize the game

    game = Game(MAP_WIDTH, MAP_HEIGHT, [
        PositionPlayer(1,  ACPlayer() if p1 else MinimaxPlayer(2, "voronoi"), [x1, y1]),
        PositionPlayer(2,  ACPlayer() if p2 else MinimaxPlayer(2, "voronoi"), [x2, y2]), ])
    return game


def make_static_game(is_AC, map=None, head_init=None):
    lower_bound_x, lower_bound_y = 0,0
    upper_bound_x = MAP_WIDTH - 1
    upper_bound_y = MAP_HEIGHT - 1

    x = random.randint(lower_bound_x, upper_bound_x)
    y = random.randint(lower_bound_y, upper_bound_y)

    wall_num = random.randint(0, 3)

    if wall_num == 0:
        game = StaticGame(is_AC, MAP_WIDTH, MAP_HEIGHT, [lower_bound_x, y], map, head_init)
    elif wall_num == 1:
        game = StaticGame(is_AC, MAP_WIDTH, MAP_HEIGHT, [x, upper_bound_y], map, head_init)
    elif wall_num == 2:
        game = StaticGame(is_AC, MAP_WIDTH, MAP_HEIGHT, [upper_bound_x, y], map, head_init)
    else:
        game = StaticGame(is_AC, MAP_WIDTH, MAP_HEIGHT, [x, lower_bound_y], map, head_init)

    return game


def get_reward(game, constants, winner_len=0, loser_len=0):
    if game.winner is None:
        return 0, 0
    elif game.winner == 1:
        if loser_len == 0 and winner_len == 0:
            return constants[0], constants[1]
        else:
            print("sep")
            return constants[2] + constants[3]/loser_len, constants[1]
    else:
        if loser_len == 0:
            return constants[1], constants[0]
        else:
            print("sep")
            return constants[1], constants[2] + constants[3]/loser_len


def get_mask(game_map, x, y, mask):
    mask[x, y] = 0

    if game_map[x + 1, y] == 0:
        game_map[x + 1, y] = 1
        mask = get_mask(game_map, x + 1, y, mask)

    if game_map[x - 1, y] == 0:
        game_map[x - 1, y] = 1
        mask = get_mask(game_map, x - 1, y, mask)

    if game_map[x, y + 1] == 0:
        game_map[x, y + 1] = 1
        mask = get_mask(game_map, x, y + 1, mask)

    if game_map[x, y - 1] == 0:
        game_map[x, y - 1] = 1
        mask = get_mask(game_map, x, y - 1, mask)

    return mask
