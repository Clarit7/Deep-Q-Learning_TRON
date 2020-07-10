from tron.player import Player, Direction
import numpy as np
import math
import random


class TreeNode(object):
    def __init__(self, parent, value, action):
        self._parent = parent
        self._children = []  # a map from action to TreeNode
        self._value = value
        self._action = action
        self._minimax_action = 0

    def is_leaf(self):
        return self._children == []

    def is_root(self):
        return self._parent is None

    def search(self):
        return self.search(child for child in self._children if not self.is_leaf())

    def expand(self, i):
        self._children.append(TreeNode(self, 0, i+1))

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def get_action(self):
        return self._action

    def set_action(self, action):
        self._action

    def get_minimax_action(self):
        return self._minimax_action

    def set_minimax_action(self, minimax_action):
        self._minimax_action = minimax_action


class Minimax(object):
    def __init__(self, depth):
        self.root = TreeNode(None, 0, 0)
        self.depth = depth

    def get_blocked(self, game_map, depth_even_odd):
        if depth_even_odd == 1:
            ind = np.unravel_index(np.argmax(game_map, axis=None), game_map.shape)
        else:
            ind = np.unravel_index(np.argmin(game_map, axis=None), game_map.shape)

        blocked = np.zeros(4)

        if game_map[ind[0], ind[1] - 1] != 1:
            if game_map[ind[0], ind[1] - 1] == 10:
                blocked[0] = 2
            else:
                blocked[0] = 1
        if game_map[ind[0] + 1, ind[1]] != 1:
            if game_map[ind[0] + 1, ind[1]] == 10:
                blocked[1] = 2
            else:
                blocked[1] = 1
        if game_map[ind[0], ind[1] + 1] != 1:
            if game_map[ind[0], ind[1] + 1] == 10:
                blocked[2] = 2
            else:
                blocked[2] = 1
        if game_map[ind[0] - 1, ind[1]] != 1:
            if game_map[ind[0] - 1, ind[1]] == 10:
                blocked[3] = 2
            else:
                blocked[3] = 1

        all_blocked = True
        for element in blocked:
            if element == 0:
                all_blocked = False
                break

        return blocked, all_blocked

    # game_map : numpy.array(12, 12)
    def distance_walls(self, game_map, depth_even_odd):
        if depth_even_odd == 1:
            ind = np.unravel_index(np.argmax(game_map, axis=None), game_map.shape)
        else:
            ind = np.unravel_index(np.argmin(game_map, axis=None), game_map.shape)

        head_crash = 0

        up = 1
        while game_map[ind[0], ind[1] - up] == 1:
            up += 1

        right = 1
        while game_map[ind[0] + right, ind[1]] == 1:
            right += 1

        down = 1
        while game_map[ind[0], ind[1] + down] == 1:
            down += 1

        left = 1
        while game_map[ind[0] - left, ind[1]] == 1:
            left += 1

        return up + right+ down + left


    def minimax_search(self, node, game_map, depth, crash = False):
        if crash:
            node.set_value(0)

        if depth == 0:
            cur_player_dist = self.distance_walls(game_map, 1)
            opp_player_dist = self.distance_walls(game_map, -1)
            node.set_value(cur_player_dist - opp_player_dist)  # To do: voronoi diagram implementation
            return 0

        depth_even_odd = 1 - 2 * (depth % 2)
        blocked, all_blocked = self.get_blocked(game_map, depth_even_odd)

        if all_blocked:
            return 0

        crash_act = 0
        if node.is_leaf():
            for i in range(4):
                if blocked[i] == 0:
                    node.expand(i)
                elif blocked[i] == 2:
                    node.expand(i)
                    crash_act = i + 1

        # To do: alpha-beta pruning
        for child in node._children:
            next_map = self.get_next_map(game_map, child.get_action(), depth_even_odd)
            if child.get_action() == crash_act:
                self.minimax_search(child, next_map, depth-1, crash = True)
            else:
                self.minimax_search(child, next_map, depth-1)

        if depth_even_odd == 1:
            minimax_value = max(child.get_value() for child in node._children)
        else:
            minimax_value = min(child.get_value() for child in node._children)

        node.set_value(minimax_value)
        minimax_acts = [child.get_action() for child in node._children if child.get_value() == minimax_value]
        node.set_minimax_action(random.choice(minimax_acts))

        if node.get_minimax_action() not in [1, 2, 3, 4]:
            print("error")

        return node.get_minimax_action()

    def get_next_map(self, game_map, action, depth_even_odd):
        game_map_copy = np.copy(game_map)

        if depth_even_odd == 1:
            ind = np.unravel_index(np.argmax(game_map, axis=None), game_map.shape)
        else:
            ind = np.unravel_index(np.argmin(game_map, axis=None), game_map.shape)

        if action == 1:
            game_map_copy[ind[0], ind[1] - 1] = 10 * depth_even_odd
        if action == 2:
            game_map_copy[ind[0] + 1, ind[1]] = 10 * depth_even_odd
        if action == 3:
            game_map_copy[ind[0], ind[1] + 1] = 10 * depth_even_odd
        if action == 4:
            game_map_copy[ind[0] - 1, ind[1]] = 10 * depth_even_odd

        game_map_copy[ind] = -1

        return game_map_copy

    def get_move(self, game_map):
        return self.minimax_search(self.root, game_map, self.depth)

    def update_with_move(self, last_move):
        if last_move in (child.get_action() for child in self.root._children):
            self.root = self.root._children[last_move]
            self.root._parent = None
        else:
            self.root = TreeNode(None, 0, 0)

    def __str__(self):
        return "Minimax"


class MinimaxPlayer(Player):

    def __init__(self, depth):
        super(MinimaxPlayer, self).__init__()
        self.minimax = Minimax(depth)
        self.direction = None
        self.depth = depth

    def initialize_minimax(self):
        self.minimax = Minimax(self.depth)

    def action(self, map, id):
        self.initialize_minimax()
        game_map = map.state_for_player(id)
        next_action = self.minimax.get_move(game_map)

        if next_action not in [1, 2, 3, 4]:
            next_action = random.randint(1, 4)
            
        if next_action == 1:
            next_direction = Direction.UP
        elif next_action == 2:
            next_direction = Direction.RIGHT
        elif next_action == 3:
            next_direction = Direction.DOWN
        elif next_action == 4:
            next_direction = Direction.LEFT

        return next_direction