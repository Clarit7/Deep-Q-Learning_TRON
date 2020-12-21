import pygame

from object.DDQN_game import Game, PositionPlayer
from tron.window import Window
from object.DDQN_player import Direction, KeyboardPlayer, Mode, ACPlayer
from ais.ACKTR import ai as ACKTR
import torch
import torch.nn as nn

from tron.minimax import MinimaxPlayer

import random

folderName='ACKTR'

def randomPosition(width, height):
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)

    return [x, y]


def displayGameMenu(window, game):
    window.screen.fill([0, 0, 0])

    myimage = pygame.image.load("asset/TronTitle.png")
    myimage = pygame.transform.scale(myimage, pygame.display.get_surface().get_size())
    imagerect = myimage.get_rect(center=window.screen.get_rect().center)
    window.screen.blit(myimage, imagerect)

    pygame.display.flip()

    event = pygame.event.poll()
    while 1:
        event = pygame.event.poll()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                window = Window(game, 40)
                break


def printGameResults(game):
    if game.winner is None:
        print("It's a draw!")
    else:
        print('Player {} wins! Duration: {}'.format(game.winner, len(game.history)))


def main():
    pygame.init()

    width = 10
    height = 10

    model = ACKTR.Net().to('cuda')
    model.load_state_dict(torch.load('ais/' + folderName + '/ACKTR_player.bak'),strict=False)

    while (True):


        x1, y1 = randomPosition(width, height)
        x2, y2 = randomPosition(width, height)

        while x1 == x2 and y1 == y2:
            x1, y1 = randomPosition(width, height)

        # game = Game(width, height, [
        #     PositionPlayer(1, MinimaxPlayer(2, 'voronoi'), [x1, y1]),
        #     PositionPlayer(2, MinimaxPlayer(2, 'voronoi'), [x2, y2]),
        # ])
        game = Game(width, height, [
            PositionPlayer(1,ACPlayer(), [x1, y1]),
            PositionPlayer(2,ACPlayer(), [x2, y2]),
        ])


        pygame.mouse.set_visible(False)

        window = Window(game, 40)
        # displayGameMenu(window, game)

        game.main_loop(model,window)
        printGameResults(game)


if __name__ == '__main__':
    main()