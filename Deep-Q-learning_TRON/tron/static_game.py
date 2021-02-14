from time import sleep
from enum import Enum

from tron.map import Map, Tile
from tron.player import ACPlayer
from tron.minimax import MinimaxPlayer
from orderedset import OrderedSet

import torch
import numpy as np
import queue
import random

class SetQueue(queue.Queue):
    def _init(self, maxsize):
        self.queue = OrderedSet()

    def _put(self, item):
        self.queue.add(item)

    def _get(self):
        head = self.queue.__getitem__(0)
        self.queue.remove(head)
        return head


class PositionPlayer:
    def __init__(self, player, position):
        self.player = player
        self.position = position
        self.alive = True

    def body(self):
        return Tile.PLAYER_ONE_BODY

    def head(self):
        return Tile.PLAYER_ONE_HEAD


class HistoryElement:
    def __init__(self, mmap, player_one_direction, player_two_direction):
        self.map = mmap
        self.player_one_direction = player_one_direction
        self.player_two_direction = player_two_direction


class StaticGame:
    def __init__(self, is_AC, width, height, wall, map_init=None, head_init=None):
        self.width = width
        self.height = height
        self.mmap = Map(width, height, Tile.EMPTY, Tile.WALL)
        self.next = []
        self.done = False

        if map_init is None:
            self.wall = wall
            body_list = self.generate_wall()
            start_position = self.generate_head(body_list)
            self.pp = PositionPlayer(ACPlayer() if is_AC else MinimaxPlayer(2, "voronoi"), start_position)

            self.history = [HistoryElement(self.mmap, None, None)]
            self.history[-1].map[self.pp.position[0], self.pp.position[1]] = self.pp.head()
        else:
            head_init = [head_init[0].item() - 1, head_init[1].item() - 1]
            self.pp = PositionPlayer(ACPlayer() if is_AC else MinimaxPlayer(2, "voronoi"), head_init)
            self.history = [HistoryElement(map_init, None, None)]

    def generate_head(self, body_list):
        directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        random.shuffle(directions)

        start_position = None
        while start_position is None:
            for direction in directions:
                start_wall = random.sample(body_list, 1)[0]
                if self.mmap.__getitem__([start_wall[0] + direction[0], start_wall[1] + direction[1]]) == Tile.EMPTY:
                    start_position = [start_wall[0] + direction[0], start_wall[1] + direction[1]]
                    break

        return start_position


    def generate_wall(self):
        if self.wall[0] == 0:
            prev_forward = [0, 1, 3]
            prev_left = [0, 1]
            prev_right = [0, 3]

            left = 1
            right = 3
        elif self.wall[1] == 0:
            prev_forward = [0, 1, 2]
            prev_left = [1, 2]
            prev_right = [0, 1]

            left = 2
            right = 0
        elif self.wall[0] == 9:
            prev_forward = [1, 2, 3]
            prev_left = [2, 3]
            prev_right = [1, 2]

            left = 3
            right = 1
        else:
            prev_forward = [0, 2, 3]
            prev_left = [0, 3]
            prev_right = [2, 3]

            left = 0
            right = 2

        prev_move = -1

        body_list = []

        same_move_combo = 0

        while 0 <= self.wall[0] <= 9 and 0 <= self.wall[1] <= 9:
            self.mmap.__setitem__(self.wall, Tile.PLAYER_TWO_BODY)
            body_list.append(self.wall.copy())

            if same_move_combo > 6:
                same_move_combo = 0
                while prev_move == current_move:
                    if prev_move == left:
                        current_move = random.sample(prev_left, 1)[0]
                    elif prev_move == right:
                        current_move = random.sample(prev_right, 1)[0]
                    else:
                        current_move = random.sample(prev_forward, 1)[0]
            else:
                if prev_move == left:
                    current_move = random.sample(prev_left, 1)[0]
                elif prev_move == right:
                    current_move = random.sample(prev_right, 1)[0]
                else:
                    current_move = random.sample(prev_forward, 1)[0]

            if current_move == 0:
                self.wall[0] += 1
            elif current_move == 1:
                self.wall[1] += 1
            elif current_move == 2:
                self.wall[0] -= 1
            else:
                self.wall[1] -= 1

            if prev_move == current_move:
                same_move_combo += 1

            prev_move = current_move

        return body_list


    def map(self):
        return self.history[-1].map.clone()

    def check_separated(self, map_clone, player):
        path_queue = SetQueue()
        dist_map = np.copy(map_clone.state_for_player(1))
        path_queue._put((player.position[0] + 1, player.position[1] + 1))

        while not path_queue.empty():
            queue_elem = path_queue._get()
            x = queue_elem[0]
            y = queue_elem[1]

            dist_map[x, y] = 5

            if dist_map[x, y - 1] == 1:
                path_queue._put((x, y - 1))
            elif dist_map[x, y - 1] == -10:
                return False
            if dist_map[x + 1, y] == 1:
                path_queue._put((x + 1, y))
            elif dist_map[x + 1, y] == -10:
                return False
            if dist_map[x, y + 1] == 1:
                path_queue._put((x, y + 1))
            elif dist_map[x, y + 1] == -10:
                return False
            if dist_map[x - 1, y] == 1:
                path_queue._put((x - 1, y))
            elif dist_map[x - 1, y] == -10:
                return False

        return True

    def get_longest_path(self, map_clone, player):
        player_length = self.get_length(np.copy(map_clone.state_for_player(1)), player.position[0] + 1, player.position[1] + 1, 0, None)

        return player_length

    def get_length(self, map_clone, x, y, length, prev_length):

        map_clone[x, y] = 5
        l1, l2, l3, l4 = -1, -1, -1, -1
        if map_clone[x, y - 1] == 1:
            l1 = self.get_length(map_clone, x, y - 1, length + 1, prev_length)
            if l1 == -10:
                return -10
        if map_clone[x + 1, y] == 1:
            l2 = self.get_length(map_clone, x + 1, y, length + 1, prev_length)
            if l2 == -10:
                return -10
        if map_clone[x, y + 1] == 1:
            l3 = self.get_length(map_clone, x, y + 1, length + 1, prev_length)
            if l3 == -10:
                return -10
        if map_clone[x - 1, y] == 1:
            l4 = self.get_length(map_clone, x - 1, y, length + 1, prev_length)
            if l4 == -10:
                return -10

        if prev_length is not None and max(l1, l2, l3, l4) > prev_length:
            return -10

        if l1 == -1 and l2 == -1 and l3 == -1 and l4 == -1:
            return length

        return max(l1, l2, l3, l4)

    def next_frame(self, action, window=None):

        map_clone = self.map()

        map_clone[self.pp.position[0], self.pp.position[1]] = self.pp.body()

        if type(self.pp.player) == type(ACPlayer()):
            (self.pp.position, self.pp.player.direction) = self.pp.player.next_position_and_direction(self.pp.position, action)
        else:
            (self.pp.position, self.pp.player.direction) = self.pp.player.next_position_and_direction(self.pp.position, self.map())

        self.history[-1].player_one_direction = self.pp.player.direction

        if self.pp.position[0] < 0 or self.pp.position[1] < 0 or \
                self.pp.position[0] >= self.width or self.pp.position[1] >= self.height:
            self.pp.alive = False
            map_clone[self.pp.position[0], self.pp.position[1]] = self.pp.head()
        elif map_clone[self.pp.position[0], self.pp.position[1]] is not Tile.EMPTY:
            self.pp.alive = False
            map_clone[self.pp.position[0], self.pp.position[1]] = self.pp.head()
        else:
            map_clone[self.pp.position[0], self.pp.position[1]] = self.pp.head()

        self.history.append(HistoryElement(map_clone, None, None))
        self.next = self.history[-1].map.state_for_player(1)

        if window:
            import pygame
            while True:
                event = pygame.event.poll()

                if event.type == pygame.NOEVENT:
                    break

                try:
                    self.pp.player.manage_event(event)
                except:
                    return False

        return True

    def step(self, action):

        if not self.next_frame(action):
            self.done = True

        if not self.pp.alive:
            self.done = True

        return self.next, self.done


    def main_loop(self,model, pop=None,window=None, condition=None):

        if window:
            window.render_map(self.map())

        while True:
            if window:
                sleep(0.1)

            map = self.map()

            if pop == None:
                with torch.no_grad():
                    action = model.act(torch.tensor(np.reshape(map.state_for_player(1),
                                                               (1, 1, map.state_for_player(1).shape[0],
                                                                map.state_for_player(1).shape[1]))).float())
            else:
                if condition:
                    with torch.no_grad():
                        action = model.act(torch.tensor(np.expand_dims(pop(map.state_for_player(1)), axis=0)).float())
                else:
                    action = model.act(torch.tensor(np.expand_dims(pop(map.state_for_player(1)), axis=0)).float())

            if not self.next_frame(action, window):
                break

            if window:
                window.render_map(self.map())
