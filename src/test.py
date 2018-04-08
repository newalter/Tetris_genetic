import random

from src.genetic_agent import GeneticAgent
from src.genetic_learner import GeneticLearner

learner = GeneticLearner()
learner.load_weight()
learner.learn()
learner.save_weight()