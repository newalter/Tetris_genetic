import random

from src.genetic_agent import GeneticAgent
from src.genetic_learner import GeneticLearner

learner = GeneticLearner()
learner.load_weight("weight.txt")
learner.learn(num_generations=1000)
#learner.sieve_similar()
#learner.replenish(47)
learner.save_weight()