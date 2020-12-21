from enum import Enum


class Direction(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


class Player(object):

    def __init__(self):
        pass

    def find_file(self, name):
        pass

    def next_position(self, current_position, direction):
        pass

    def get_direction(self, current_position, direction):
        pass
    def next_position_and_direction(self, current_position,action):
        pass

    def action(self, map, id):

        pass

    def step(self, state, action, reward, next_step, done):
        pass

    def learn(self, experiences, gamma):
        pass

    def soft_update(self, local_model, target_model, tau):
        pass

    def manage_event(self, event):
        pass



class Mode(Enum):
    ARROWS = 1
    ZQSD = 2


class KeyboardPlayer(Player):

    def __init__(self, initial_direction, mode=Mode.ARROWS):

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


class ACPlayer(Player):
    def __init__(self):
        super(ACPlayer, self).__init__()

        """Initialize an Agent object.

               Params
               =======
                   state_size (int): dimension of each state
                   action_size (int): dimension of each action
                   seed (int): random seed
               """
    def get_direction(self,next_action):
        next_action=next_action+1

        if next_action == 1:
            next_direction = Direction.UP
        if next_action == 2:
            next_direction = Direction.RIGHT
        if next_action == 3:
            next_direction = Direction.DOWN
        if next_action == 4:
            next_direction = Direction.LEFT
        return next_direction

    def next_position_and_direction(self, current_position, action):
        direction = self.get_direction(action)
        return self.next_position(current_position, direction), direction

    def next_position(self, current_position, direction):
        if direction == Direction.UP:
            return current_position[0] - 1, current_position[1]
        elif direction == Direction.RIGHT:
            return current_position[0], current_position[1] + 1
        elif direction == Direction.DOWN:
            return current_position[0] + 1, current_position[1]
        elif direction == Direction.LEFT:
            return current_position[0], current_position[1] - 1