import random

import numpy

from src.Tetris_Env import TetrisEnv
from src.genetic_agent import GeneticAgent
from src.genetic_learner import GeneticLearner

env = TetrisEnv()
weight = numpy.array([-0.8066651877814951, -0.025211014142500453, -0.030428318607102118, -0.8589686185198616])
# agent = GeneticAgent(env, weight, None)
for i in GeneticLearner.normailize(GeneticLearner(),weight):
    print(i)
# print(agent.play(1000,seed=5))
