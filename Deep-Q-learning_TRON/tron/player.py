
from enum import Enum
import random
import numpy as np

class Direction(Enum):

    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class Player(object):

    def __init__(self):
        pass

    def find_file(self, name):

        return '/'.join(self.__module__.split('.')[:-1]) + '/' + name

    def next_position(self, current_position, direction):

        if direction == Direction.UP:
            return (current_position[0] - 1, current_position[1])
        if direction == Direction.RIGHT:
            return (current_position[0], current_position[1] + 1)
        if direction == Direction.DOWN:
            return (current_position[0] + 1, current_position[1])
        if direction == Direction.LEFT:
            return (current_position[0], current_position[1] - 1)

    def next_position_and_direction(self, current_position, id, map):

        direction = self.action(map, id)
        return (self.next_position(current_position, direction), direction)

    def action(self, map, id):

        pass

    def manage_event(self, event):

        pass


class Mode(Enum):

    ARROWS = 1
    ZQSD = 2


class KeyboardPlayer(Player):

    def __init__(self, initial_direction, mode = Mode.ARROWS):

        super(KeyboardPlayer, self).__init__()
        self.direction = initial_direction
        self.mode = mode

    def left(self):
 
        import pygame
        return pygame.K_q if self.mode == Mode.ZQSD else pygame.K_LEFT

    def right(self):
  
        import pygame
        return pygame.K_d if self.mode == Mode.ZQSD else pygame.K_RIGHT

    def down(self):

        import pygame
        return pygame.K_s if self.mode == Mode.ZQSD else pygame.K_DOWN

    def up(self):

        import pygame
        return pygame.K_z if self.mode == Mode.ZQSD else pygame.K_UP

    def manage_event(self, event):

        import pygame
        if event.type == pygame.KEYDOWN:
            if event.key == self.left():
                self.direction = Direction.LEFT
            if event.key == self.up():
                self.direction = Direction.UP
            if event.key == self.right():
                self.direction = Direction.RIGHT
            if event.key == self.down():
                self.direction = Direction.DOWN

    def action(self, map, id):

        return self.direction


class RandomPlayer(Player):

    def __init__(self):
        super(RandomPlayer, self).__init__()
        self.direction = None

    def action(self, map, id):
        game_map = map.state_for_player(id)
        ind = np.unravel_index(np.argmax(game_map, axis=None), game_map.shape)
        blocked = np.zeros(4)

        if game_map[ind[0], ind[1] - 1] != 1:
            blocked[0] = 1
        if game_map[ind[0] + 1, ind[1]] != 1:
            blocked[1] = 1
        if game_map[ind[0], ind[1] + 1] != 1:
            blocked[2] = 1
        if game_map[ind[0] - 1, ind[1]] != 1:
            blocked[3] = 1

        all_blocked = True
        for element in blocked:
            if element == 0:
                all_blocked = False
                break

        next_action = random.randint(1, 4)
        if not all_blocked:
            while (blocked[next_action - 1] == 1):
                next_action = random.randint(1, 4)

        if next_action == 1:
            self.direction = Direction.UP
        if next_action == 2:
            self.direction = Direction.RIGHT
        if next_action == 3:
            self.direction = Direction.DOWN
        if next_action == 4:
            self.direction = Direction.LEFT

        return self.direction
