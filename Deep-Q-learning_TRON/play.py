import pygame
from tron.window import Window
from Net.ACNet import Net
from util import *
from games.ACKTR import Brain

import random

folderName='games/save'

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
    rating=True
    iter=30
    actor_critic = Net()  # 신경망 객체 생성
    global_brain = Brain(actor_critic, acktr=True)
    global_brain.actor_critic.load_state_dict(torch.load(folderName + '/ACKTR_player2.bak'))
    global_brain.actor_critic.eval()

    actor_critic2 = Net()  # 신경망 객체 생성
    global_brain2 = Brain(actor_critic2, acktr=True)
    global_brain2.actor_critic.load_state_dict(torch.load(folderName + '/ACKTR_player3.bak'))
    global_brain2.actor_critic.eval()

    if rating:
        nullgame=0
        p1_win=0
        p2_win=0

        for i in range(iter):

            game = make_game(True, False)
            pygame.mouse.set_visible(False)
            window = None

            game.main_loop(global_brain.actor_critic, pop_up, window, global_brain2.actor_critic)
            if(game.winner is None):
                nullgame+=1

            elif(game.winner ==1 ):
                p1_win+=1
            else:
                p2_win+=1

        print("Player 1:{} \n Player 2:{}\n ",format(p1_win,p2_win))
    else:

        while True:
            game = make_game(True, False)
            pygame.mouse.set_visible(False)

            window = Window(game, 40)
            # displayGameMenu(window, game)
            window=None

            game.main_loop(global_brain.actor_critic,pop_up,window,global_brain2.actor_critic)
            printGameResults(game)



if __name__ == '__main__':
    main()