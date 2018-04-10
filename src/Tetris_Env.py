from copy import deepcopy
from src.configuration import pOrients, pWidth, pHeight, pTop, pBottom, Num_Types, Col, Row
from random import Random
import numpy as np


class ActionSpace(object):
    legal_moves = []

    def __init__(self):
        self.initialise_legal_moves()

    def initialise_legal_moves(self):
        for i in range(Num_Types):
            n = 0
            for j in range(pOrients[i]):
                n = n + Col + 1 - pWidth[i][j]
            type_i_actions = []
            for j in range(pOrients[i]):
                for k in range(Col + 1 - pWidth[i][j]):
                    action = (j, k)
                    type_i_actions.append(action)
            self.legal_moves.append(type_i_actions)


class TetrisEnv(object):
    randomness = Random()

    board = None
    top = None
    currentPiece = None
    total_score = 0
    action_space = ActionSpace()

    def __init__(self, seed = 0):
        self.seed(seed = seed)
        self.board = [[0] * Col for i in range(Row)]
        self.top = [0] * Col
        self.currentPiece = self.new_piece()
        self.total_score = 0

    def step(self, action):
        orient, slot = self.action_space.legal_moves[self.currentPiece][action]
        score, is_done = self.perform_action(self.board, self.top, orient, slot, self.currentPiece)
        self.total_score = self.total_score + score

        self.currentPiece = self.new_piece()
        observation = (deepcopy(self.board), deepcopy(self.top), self.currentPiece)
        return observation, self.total_score, is_done

    def reset(self, seed = 0):
        self.__init__(seed)
        observation = (deepcopy(self.board), deepcopy(self.top), self.currentPiece)
        return observation

    def seed(self, seed=None):
        if seed is not None:
            self.randomness.seed(seed)
        return self.randomness.seed

    def new_piece(self):
        return self.randomness.randrange(0, Num_Types)

    def perform_action(self, board, top, orient, slot, currentPiece):
        score = 0
        is_done = False
        height = top[slot] - pBottom[currentPiece][orient][0]
        for c in range(pWidth[currentPiece][orient]):
            height = max(height, top[slot + c] - pBottom[currentPiece][orient][c])

        if height + pHeight[currentPiece][orient] >= Row:
            is_done = True
            return 0, is_done

        for i in range(pWidth[currentPiece][orient]):
            for h in range(height + pBottom[currentPiece][orient][i], height + pTop[currentPiece][orient][i]):
                board[h][i + slot] = 1

        for c in range(pWidth[currentPiece][orient]):
            top[slot + c] = height + pTop[currentPiece][orient][c]

        for r in range(height + pHeight[currentPiece][orient] - 1, height - 1, -1):
            full = True
            for c in range(Col):
                if board[r][c] == 0:
                    full = False
                    break

            if full:
                score = score + 1
                for c in range(Col):
                    for i in range(r, top[c]):
                        board[i][c] = board[i + 1][c]
                    top[c] = top[c] - 1
                    while top[c] >= 1 and board[top[c] - 1][c] == 0:
                        top[c] = top[c] - 1

        return score, is_done

    def evaluate_board(self, board, top):
        average_height = sum(top) / 10.0
        max_height = max(top)
        diff_height = 0
        for i in range(Col - 1):
            diff_height = diff_height + abs(top[i] - top[i + 1])
        holes = 0
        depth_holes = 0
        for i in range(Row):
            for j in range(Col):
                if board[i][j] == 0 and i < top[j]:
                    holes = holes + 1
                    depth_holes = depth_holes + (top[j] - i)
        # arbitrary reward function from online source
        return average_height, max_height, diff_height, holes, depth_holes
