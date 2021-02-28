from time import sleep
from enum import Enum
from orderedset import OrderedSet

from tron.map import Map, Tile
from tron.player import ACPlayer
from config import *

import torch
import numpy as np
import queue

class SetQueue(queue.Queue):
    def _init(self, maxsize):
        self.queue = OrderedSet()

    def _put(self, item):
        self.queue.add(item)

    def _get(self):
        head = self.queue.__getitem__(0)
        self.queue.remove(head)
        return head


class Winner(Enum):
    PLAYER_ONE = 1
    PLAYER_TWO = 2


class PositionPlayer:
    def __init__(self, id, player, position):
        self.id = id
        self.player = player
        self.position = position
        self.alive = True

    def body(self):
        if self.id == 1:
            return Tile.PLAYER_ONE_BODY
        elif self.id == 2:
            return Tile.PLAYER_TWO_BODY

    def head(self):
        if self.id == 1:
            return Tile.PLAYER_ONE_HEAD
        elif self.id == 2:
            return Tile.PLAYER_TWO_HEAD


class HistoryElement:
    def __init__(self, mmap, player_one_direction, player_two_direction):
        self.map = mmap
        self.player_one_direction = player_one_direction
        self.player_two_direction = player_two_direction


class Game:
    def __init__(self, width, height, pps):

        self.width = width
        self.height = height
        self.mmap = Map(width, height, Tile.EMPTY, Tile.WALL)
        self.history = [HistoryElement(self.mmap, None, None)]
        self.pps = pps
        self.winner = None
        self.loser_len=0
        self.winner_len = 0
        self.next_p1 = []
        self.next_p2 = []
        self.reword = 0
        self.done = False

        for pp in self.pps:
            self.history[-1].map[pp.position[0], pp.position[1]] = pp.head()

    def map(self, invert=False):
        return self.history[-1].map.clone(invert)

    def check_separated(self, map_clone, p1):
        path_queue = SetQueue()
        dist_map = np.copy(map_clone.state_for_player(1))
        path_queue._put((p1.position[0] + 1, p1.position[1] + 1))

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

    def get_longest_path(self, map_clone, p1, p2):
        p1_length = self.get_length(np.copy(map_clone.state_for_player(1)), p1.position[0] + 1, p1.position[1] + 1, 0, None)
        p2_length = self.get_length(np.copy(map_clone.state_for_player(2)), p2.position[0] + 1, p2.position[1] + 1, 0, None)

        # if p2_length == -10 or p1_length < p2_length:
        if p1_length < p2_length:
            self.loser_len=p1_length
            self.winner_len=p2_length
            return 2, p1_length - 1

        elif p1_length > p2_length:

            self.loser_len=p2_length
            self.winner_len=p1_length

            return 1, p1_length - 1
        else:
            return 0, p1_length - 1

        return 0, p1_length-1

    def get_longest_path_masking(self, static_brain, vs_minimax):
        p1_len = self.get_length_masking(1, static_brain)
        p2_len = self.get_length_masking(2, None if vs_minimax else static_brain)

        if p1_len > p2_len:
            return 1, p1_len
        elif p2_len > p1_len:
            return 2, p1_len
        else:
            return 0, p1_len

    def get_length(self, map_clone, x, y, length, prev_length):
        length += 1
        map_clone[x, y] = 5

        l1, l2, l3, l4 = -1, -1, -1, -1
        if map_clone[x, y - 1] == 1:
            l1 = self.get_length(map_clone, x, y - 1, length, prev_length)
            if l1 == -10:
                return -10
        if map_clone[x + 1, y] == 1:
            l2 = self.get_length(map_clone, x + 1, y, length, prev_length)
            if l2 == -10:
                return -10
        if map_clone[x, y + 1] == 1:
            l3 = self.get_length(map_clone, x, y + 1, length, prev_length)
            if l3 == -10:
                return -10
        if map_clone[x - 1, y] == 1:
            l4 = self.get_length(map_clone, x - 1, y, length, prev_length)
            if l4 == -10:
                return -10

        if prev_length is not None and max(l1, l2, l3, l4) > prev_length:
            return -10

        if l1 == -1 and l2 == -1 and l3 == -1 and l4 == -1:
            return length

        return max(l1, l2, l3, l4)

    def get_length_masking(self, player_num, static_brain):
        from tron.util import pop_up_static, make_static_game, get_mask, get_direction_area

        obs_np = self.map().state_for_player(player_num)
        obs = pop_up_static(obs_np)
        obs = torch.tensor(np.array(obs)).float()

        player_head = torch.nonzero(obs[1] == 10).squeeze(0)

        if player_num == 1:
            static_env = make_static_game(static_brain is not None, self.map(invert=False), player_head)
        else:
            static_env = make_static_game(static_brain is not None, self.map(invert=True), player_head)

        obs_uni = obs[0] + obs[1]
        masking = get_mask(obs_uni, player_head[0].item(), player_head[1].item(), torch.ones((MAP_WIDTH + 2, MAP_HEIGHT + 2)))
        masking = torch.where(obs[1] != 0, torch.zeros(1), masking)
        obs[0] = masking

        duration = 0
        done = 0

        while done == 0:
            duration += 1
            if static_brain is not None:
                with torch.no_grad():
                    act = static_brain.act(obs.unsqueeze(0))
            else:
                obs_uni = obs[0] + obs[1]
                player_head = torch.nonzero(obs[1] == 10).squeeze(0)
                act = get_direction_area(obs_uni, player_head[0].item(), player_head[1].item())

            obs_np, done = static_env.step(act, is_area=True)
            obs = pop_up_static(obs_np)
            obs = torch.tensor(np.array(obs)).float()
            obs[0] = masking

        return duration - 1

    def next_frame(self, action_p1, action_p2, window=None, static_brain=None, end_separated=False, vs_minimax=False):
        map_clone = self.map()

        action = [action_p1, action_p2]

        for pp in self.pps:
            map_clone[pp.position[0], pp.position[1]] = pp.body()

        for id, pp in enumerate(self.pps):
            if type(pp.player) == type(ACPlayer()):
                (pp.position, pp.player.direction) = pp.player.next_position_and_direction(pp.position, action[id])
            else:
                (pp.position, pp.player.direction) = pp.player.next_position_and_direction(pp.position, id + 1,self.map())

        self.history[-1].player_one_direction = self.pps[0].player.direction
        self.history[-1].player_two_direction = self.pps[1].player.direction

        done = False
        for (id, pp) in enumerate(self.pps):
            if pp.position[0] < 0 or pp.position[1] < 0 or \
                    pp.position[0] >= self.width or pp.position[1] >= self.height:
                pp.alive, done = False, True
                map_clone[pp.position[0], pp.position[1]] = pp.head()
            elif map_clone[pp.position[0], pp.position[1]] is not Tile.EMPTY:
                pp.alive, done = False, True
                map_clone[pp.position[0], pp.position[1]] = pp.head()
            else:
                map_clone[pp.position[0], pp.position[1]] = pp.head()

        self.history.append(HistoryElement(map_clone, None, None))
        self.next_p1 = self.history[-1].map.state_for_player(1)
        self.next_p2 = self.history[-1].map.state_for_player(2)

        p1_length = 0
        p1_area = 0

        sep = False
        """"
        if end_separated and not done and self.check_separated(map_clone, self.pps[0]):
            print("errrrrrrrrrrrrrrrrrrrrrrror")
            print(end_separated)
            if static_brain is None:
                from tron.util import get_area, pop_up
                obs_np1 = np.copy(map_clone.state_for_player(1))
                obs1 = pop_up(obs_np1)
                obs1 = torch.tensor(obs1).float()
                p1_a = get_area(obs1[0] + obs1[1] + obs1[2], self.pps[0].position[0] + 1, self.pps[0].position[1] + 1, -1, 0)

                obs_np2 = np.copy(map_clone.state_for_player(2))
                obs2 = pop_up(obs_np2)
                obs2 = torch.tensor(obs2).float()
                p2_a = get_area(obs2[0] + obs2[1] + obs2[2], self.pps[1].position[0] + 1, self.pps[1].position[1] + 1,-1, 0)

                if p1_a > p2_a:
                    winner = 1
                elif p2_a > p1_a:
                    winner = 2
                else:
                    winner = 0
                winner, p1_length = self.get_longest_path(map_clone, self.pps[0], self.pps[1])
            else:
                if not vs_minimax:
                    from tron.util import get_area, pop_up
                    obs_np = np.copy(map_clone.state_for_player(1))
                    obs = pop_up(obs_np)
                    obs = torch.tensor(obs).float()
                    p1_area = get_area(obs[0] + obs[1] + obs[2], self.pps[0].position[0] + 1, self.pps[0].position[1] + 1, -1, 0)
                winner, p1_length = self.get_longest_path_masking(static_brain, vs_minimax=vs_minimax)

            if winner == 1:
                self.pps[1].alive = False
            elif winner == 2:
                self.pps[0].alive = False
            else:
                self.pps[0].alive = False
                self.pps[1].alive = False

            sep = True
        """

        if window:
            import pygame
            while True:
                event = pygame.event.poll()

                if event.type == pygame.NOEVENT:
                    break

                for pp in self.pps:
                    try:
                        pp.player.manage_event(event)
                    except:
                        if id == 0:
                            self.winner = 2
                        elif id == 1:
                            self.winner = 1
                        return False

        return True, p1_length, p1_area, sep

    def step(self, action_p1, action_p2, static_brain=None, end_separated=False):
        alive_count = 0
        alive = None

        is_next_frame, p1_len, p1_area, sep = self.next_frame(action_p1, action_p2, static_brain=static_brain, end_separated=end_separated)

        if not is_next_frame:
            self.done = True

            print("is not next frame")
            return self.next_p1, self.next_p2, self.done, p1_len, p1_area, sep

        for pp in self.pps:
            if pp.alive:
                alive_count += 1
                alive = pp.id

        if alive_count <= 1:
            if alive_count == 1:
                if self.pps[0].position[0] != self.pps[1].position[0] or \
                        self.pps[0].position[1] != self.pps[1].position[1]:
                    self.winner = alive

            self.done = True

        return self.next_p1, self.next_p2, self.done, p1_len, p1_area, sep

    def main_loop(self,model, pop=None,window=None,model2=None,condition=None, static_brain=None, end_separated=False, vs_minimax=False):

        if window:
            window.render_map(self.map())

        if not model2:
            model2=model

        while True:
            alive_count = 0
            alive = None

            if window:
                sleep(0.1)

            map=self.map()

            if pop == None:
                with torch.no_grad():
                    action1 = model.act(torch.tensor(np.reshape(map.state_for_player(1), (1, 1, map.state_for_player(1).shape[0],
                                                                                          map.state_for_player(1).shape[1]))).float())
                    action2 = model2.act(torch.tensor(np.reshape(map.state_for_player(2), (1, 1, map.state_for_player(2).shape[0],
                                                                                          map.state_for_player(2).shape[1]))).float())

            else:
                if condition:
                    with torch.no_grad():
                        action1 = model.act(torch.tensor(np.expand_dims(pop(map.state_for_player(1)), axis=0)).float())

                        if condition[1]=="DQN":
                            action2 = model2.act(
                                torch.tensor(np.reshape(map.state_for_player(2), (1, 1, map.state_for_player(2).shape[0],
                                                                                  map.state_for_player(2).shape[1]))).float())
                        elif condition[1]=="AC":
                            action2 = model2.act(torch.tensor(np.expand_dims(pop(map.state_for_player(2)), axis=0)).float())

                else:
                    action1 = model.act(torch.tensor(np.expand_dims(pop(map.state_for_player(1)), axis=0)).float())
                    if vs_minimax:
                        action2 = 0
                    else:
                        action2 = model2.act(torch.tensor(np.expand_dims(pop(map.state_for_player(2)), axis=0)).float())

            is_next_frame, _, _, _ = self.next_frame(action1, action2, window, static_brain=static_brain, end_separated=end_separated,
                            vs_minimax=vs_minimax)

            if not is_next_frame:
                break

            for pp in self.pps:
                if pp.alive:
                    alive_count += 1
                    alive = pp.id

            if alive_count <= 1:
                if alive_count == 1:
                    if self.pps[0].position[0] != self.pps[1].position[0] or \
                            self.pps[0].position[1] != self.pps[1].position[1]:
                        self.winner = alive
                break

            if window:
                window.render_map(self.map())
