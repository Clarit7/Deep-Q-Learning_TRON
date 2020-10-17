from tron.player import Player, Direction
from orderedset import OrderedSet
from enum import Enum
import numpy as np
import queue
import random


class TreeNode(object):
    def __init__(self, parent, value, action):
        self._parent = parent
        self._children = []
        self._value = value
        self._action = action     # from parent which action played
        self._minimax_action = 0  # which is the best action from this state

    def is_leaf(self):
        return self._children == []

    def is_root(self):
        return self._parent is None

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


class SetQueue(queue.Queue):
    def _init(self, maxsize):
        self.queue = OrderedSet()

    def _put(self, item):
        self.queue.add(item)

    def _get(self):
        head = self.queue.__getitem__(0)
        self.queue.remove(head)
        return head


class Minimax(object):
    def __init__(self, depth, mode):
        self.root = TreeNode(None, 0, 0)
        self.depth = depth
        self.mode = mode

    def get_shortest_path(self, game_map, ind, pl_mi):
        path_queue = SetQueue()
        dist_map = np.copy(game_map)
        path_queue._put((ind[0], ind[1], pl_mi))

        while not path_queue.empty():
            queue_elem = path_queue._get()
            x = queue_elem[0]
            y = queue_elem[1]
            l = queue_elem[2]

            dist_map[x, y] = l+pl_mi

            if dist_map[x, y - 1] == 1:
                path_queue._put((x, y - 1, l + pl_mi))
            if dist_map[x + 1, y] == 1:
                path_queue._put((x + 1, y, l + pl_mi))
            if dist_map[x, y + 1] == 1:
                path_queue._put((x, y + 1, l + pl_mi))
            if dist_map[x - 1, y] == 1:
                path_queue._put((x - 1, y, l + pl_mi))

        return dist_map

    def get_voronoi_value(self, game_map, ind1, ind2):
        p1_map = self.get_shortest_path(game_map, ind1, 1)
        p2_map = self.get_shortest_path(game_map, ind2, -1)

        p1_area = 0
        p2_area = 0

        """ visual map (doesn't necessary)
        for i in range(p1_map.shape[0]):
            for j in range(p2_map.shape[1]):
                if p2_map[i, j] == -2:
                    p1_map[i, j] = -10
                elif p1_map[i, j] == 2:
                    p1_map[i, j] = 10
                elif p1_map[i, j] != -1:
                    if p1_map[i, j] + p2_map[i, j] == 0:
                        p1_map[i, j] = 0
                    elif p1_map[i, j] + p2_map[i, j] > 0:
                        p1_map[i, j] = -5
                    else:
                        p1_map[i, j] = 5
        """

        for i in range(p1_map.shape[0]):
            for j in range(p2_map.shape[1]):
                if not p1_map[i, j] == -1 and not p1_map[i, j] == 2 and not p2_map[i, j] == -2:
                    if p1_map[i, j] != 1 and p2_map[i, j] == 1:
                        p1_area += 1
                    elif p1_map[i, j] == 1 and p2_map[i, j] != 1:
                        p2_area += 1
                    elif p1_map[i, j] + p2_map[i, j] < 0:
                        p1_area += 1
                    elif p1_map[i, j] + p2_map[i, j] > 0:
                        p2_area += 1

        return p1_area - p2_area

    # game_map : numpy.array(12, 12)
    def distance_walls(self, game_map, ind):
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

    """
    def update_with_move(self, last_move):
        if last_move in (child.get_action() for child in self.root._children):
            self.root = self.root._children[last_move]
            self.root._parent = None
        else:
            self.root = TreeNode(None, 0, 0)
    """

    def minimax_search(self, node, game_map, depth, crash = False):
        if crash:  # head vs head crashing state
            node.set_value(0)

        if depth == 0:
            ind1 = np.unravel_index(np.argmax(game_map, axis=None), game_map.shape)
            ind2 = np.unravel_index(np.argmin(game_map, axis=None), game_map.shape)
            if self.mode == Mode.DISTWALL:
                cur_player_dist = self.distance_walls(game_map, ind1)
                opp_player_dist = self.distance_walls(game_map, ind2)
                node.set_value(cur_player_dist - opp_player_dist)
            else:  # Mode.VORONOI
                node.set_value(self.get_voronoi_value(game_map, ind1, ind2))

            return 0  # for exit 1 recursion step

        depth_even_odd = 1 - 2 * (depth % 2)  # even depth: 1, odd depth: -1
        blocked, all_blocked = self.get_blocked(game_map, depth_even_odd)

        if all_blocked:
            return random.randint(1, 4)

        crash_act = 0
        if node.is_leaf():
            for i in range(4):
                if blocked[i] == 0:
                    node.expand(i)
                elif blocked[i] == 2:
                    node.expand(i)
                    crash_act = i + 1

        for child in node._children:
            next_map = self.get_next_map(game_map, child.get_action(), depth_even_odd)
            if child.get_action() == crash_act:
                self.minimax_search(child, next_map, depth-1, crash = True)
            else:
                self.minimax_search(child, next_map, depth-1)

            # alpha-beta pruning
            if depth_even_odd == -1 and node._parent.get_minimax_action() != 0:
                if child.get_value() <= node._parent.get_value():
                    node.set_value(child.get_value())
                    node.set_minimax_action(child.get_action())

                    return 0  # for exit 1 recursion step

        if depth_even_odd == 1:
            minimax_value = max(child.get_value() for child in node._children)
        else:
            minimax_value = min(child.get_value() for child in node._children)

        node.set_value(minimax_value)
        minimax_acts = [child.get_action() for child in node._children if child.get_value() == minimax_value]
        node.set_minimax_action(random.choice(minimax_acts))

        return node.get_minimax_action()

    def get_move(self, game_map):
        return self.minimax_search(self.root, game_map, self.depth)

    def __str__(self):
        return "Minimax"


class Mode(Enum):

    DISTWALL = 1
    VORNOI = 2


class MinimaxPlayer(Player):

    def __init__(self, depth, mode = Mode.VORNOI):
        super(MinimaxPlayer, self).__init__()
        self.mode = mode
        self.minimax = Minimax(depth, mode)
        self.direction = None
        self.depth = depth

    def initialize_minimax(self):
        self.minimax = Minimax(self.depth, self.mode)

    def action(self, map, id):
        self.initialize_minimax()
        game_map = map.state_for_player(id).T
        next_action = self.minimax.get_move(game_map)

        if next_action == 1:
            next_direction = Direction.UP
        elif next_action == 2:
            next_direction = Direction.RIGHT
        elif next_action == 3:
            next_direction = Direction.DOWN
        elif next_action == 4:
            next_direction = Direction.LEFT

        return next_action-1