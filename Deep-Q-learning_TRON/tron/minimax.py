from tron.player import Player, Direction
import numpy as np
import math
import random


class TreeNode(object):
    def __init__(self, parent, value, action):
        self.parent = parent
        self.children = []  # a map from action to TreeNode
        self.value = value
        self.action = action

    def is_leaf(self):
        return self.children == []

    def is_root(self):
        return self.parent is None

    def search(self):
        return self.search(child for child in self.children if not self.is_leaf())

    def expand(self, i):
        self.children.append(TreeNode(None, 0, i+1))

    def get_value(self):
        return self.value

class Minimax(object):
    def __init__(self, depth):
        self.root = TreeNode(None, 0, 0)
        self.depth = depth

    def get_blocked(self, game_map):
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

        return blocked, all_blocked

    def minimax_search(self, node, game_map, depth):
        if depth == 0:
            node.value = 1

            return 1

        blocked, all_blocked = self.get_blocked(game_map)

        if len(node.children) == 0:
            for i in range(4):
                if blocked[i] == 0:
                    node.expand(i)

        for child in node.children:
            next_map = self.get_next_map(game_map, child.action)
            self.minimax_search(child, next_map, depth-1)

        max_value = max(child.get_value() for child in node.children)
        max_acts = [child.action for child in node.children if child.get_value() == max_value]

        return random.choice(max_acts)


    def get_next_map(self, game_map, action):
        ind = np.unravel_index(np.argmax(game_map, axis=None), game_map.shape)

        if action == 1:
            game_map[ind[0], ind[1] - 1] = 10
        if action == 2:
            game_map[ind[0] + 1, ind[1]] = 10
        if action == 3:
            game_map[ind[0], ind[1] + 1] = 10
        if action == 4:
            game_map[ind[0] - 1, ind[1]] = 10

        game_map[ind[0], ind[1]] = -1

        return game_map

    def get_move(self, game_map):
        return self.minimax_search(self.root, game_map, self.depth)

    def update_with_move(self, last_move):
        if last_move in self._root.children:
            self._root = self._root.children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 0)

    def __str__(self):
        return "MCTS"


class MinimaxPlayer(Player):

    def __init__(self, depth):
        super(MinimaxPlayer, self).__init__()
        self.minimax = Minimax(depth)
        self.direction = None

    def action(self, map, id):
        game_map = map.state_for_player(id)
        next_action = self.minimax.get_move(game_map)

        if next_action == 1:
            next_direction = Direction.UP
        if next_action == 2:
            next_direction = Direction.RIGHT
        if next_action == 3:
            next_direction = Direction.DOWN
        if next_action == 4:
            next_direction = Direction.LEFT

        # self.update_with_move(next_direction)

        return next_direction