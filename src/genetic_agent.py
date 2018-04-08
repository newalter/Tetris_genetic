from copy import deepcopy

import numpy as np

from src.Tetris_Env import ActionSpace


class GeneticAgent(object):
    weight = None
    env = None
    action_space = ActionSpace()
    fitness = None

    def __init__(self, env, weight, fitness=None):
        self.weight = weight
        self.env = env
        if fitness == None:
            self.fitness = self.play()
        else:
            self.fitness = fitness
        print(self.fitness)

    def play(self, max_num_steps=500, seed=1):
        observation = self.env.reset(seed)
        is_done = False
        steps = 0
        reward = 0
        while (not is_done) and steps < max_num_steps:
            action = self.choose_action(observation)
            observation, reward, is_done = self.env.step(action)
            steps = steps + 1
        return reward

    def choose_action(self, observation):
        board, top, currentPiece, nextPiece = observation
        max_evaluation = -np.inf
        best_action = -1
        action_num = 0
        for orient, slot in self.action_space.legal_moves[currentPiece]:
            depth_1_board = deepcopy(board)
            depth_1_top = deepcopy(top)
            _, is_done = self.env.perform_action(depth_1_board, depth_1_top, orient, slot, currentPiece)
            if is_done:
                continue
            for orient2, slot2 in self.action_space.legal_moves[nextPiece]:
                depth_2_board = deepcopy(depth_1_board)
                depth_2_top = deepcopy(depth_1_top)
                self.env.perform_action(depth_2_board, depth_2_top, orient2, slot2, nextPiece)
                attributes = self.env.evaluate_board(depth_2_board, depth_2_top)
                evaluation = self.compute_evaluation(attributes)
                if max_evaluation < evaluation:
                    max_evaluation = evaluation
                    best_action = action_num
            action_num = action_num + 1
        return best_action

    def compute_evaluation(self, attributes):
        return sum(p * q for p, q in zip(self.weight, attributes))
