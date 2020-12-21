import pygame
from tron.window import Window
from Net import Net
from util import *

import random

folderName='/save'

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

    model = Net().to('cuda')
    model.load_state_dict(torch.load(folderName + '/A2CPlayer.bak'),strict=False)

    while (True):

        game = make_game(True,True)
        pygame.mouse.set_visible(False)

        window = Window(game, 40)
        # displayGameMenu(window, game)

        game.main_loop(model,pop_up,window)
        printGameResults(game)


if __name__ == '__main__':
    main()