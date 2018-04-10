import random

import numpy

from src.Tetris_Env import TetrisEnv
from src.genetic_agent import GeneticAgent
from src.genetic_learner import GeneticLearner

env = TetrisEnv()
weight = numpy.array([-0.5644820216419986, -0.01272622646578697, -0.3089446007668044, -0.7572691537270314, -0.11088170666746289])
agent = GeneticAgent(env, weight, 329.5)
print(agent.play(1000,seed=3))
