import random

import numpy

from src.Tetris_Env import TetrisEnv
from src.genetic_agent import GeneticAgent
from src.genetic_learner import GeneticLearner

env = TetrisEnv()
weight = numpy.array([-0.43048529747041575, -0.1950686893201384, -0.29151194284020493, -0.8015762242500877, -0.2216460218619617])
agent = GeneticAgent(env, weight, 329.5)
print(agent.play(1000,seed=5))
