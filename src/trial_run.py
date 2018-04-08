import random

import numpy

from src.Tetris_Env import TetrisEnv
from src.genetic_agent import GeneticAgent
from src.genetic_learner import GeneticLearner

weight = numpy.array([-0.45259062694559726, -0.22343595532216368, -0.23456835034006537, -0.6637896140340586])



weight = weight / numpy.math.sqrt(sum(k ** 2 for k in weight))
print(sum(k**2 for k in weight))