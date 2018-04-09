import random

import numpy as np

from src.Tetris_Env import TetrisEnv
from src.genetic_agent import GeneticAgent
from scipy.spatial import distance

BATCH_SIZE = 100

# currently we have 5 attributes: average height, maximum height, height differences, number of holes, depth of holes,
NUM_ATTRIBUTE = 5

# Size of the subset of potential parents. A larger tournament_size indicates a larger selective pressure.
# A larger tournament_size might result in faster convergence but also lower diversity in the population
TOURNAMENT_SIZE = 5

# How close are two members to be considered as a
CLOSENESS = 0.01

# Probability and Degree of mutation
MUTATION_P = 0.05
MUTATION_RANGE = 0.2

# number of random members added to the population in each new generation; Helps in maintaining diversity
REPLENISH_SIZE = 15

# number of offsprings produced in each new generation
OFFSPRING_SIZE = 25

# number of rounds to be played to decide fitness
# consider decrease this if the learning is too slow
# PLEASE set this in genetic_agent.py directly
FITNESS_TEST = 10

# mutation operator
def mutate(weight):
    if random.random() < MUTATION_P:
        pos = random.randrange(0, NUM_ATTRIBUTE)
        weight[pos] = weight[pos] + random.random() * 2 * MUTATION_RANGE - MUTATION_RANGE
    return weight


class GeneticLearner():
    agents = []
    env = TetrisEnv()

    # replenishes the population with random members
    def replenish(self, num = BATCH_SIZE):
        for i in range(num):
            weight = self.normailize(np.random.rand(NUM_ATTRIBUTE)*-1)
            self.agents.append(GeneticAgent(env=self.env, weight=weight))

    # normalizes the weight
    def normailize(self, weight):
        square_sum_root = np.sqrt(sum(k ** 2 for k in weight))
        return weight / square_sum_root

    # saves weight to the designated filepath
    def save_weight(self, filepath="weight.txt"):
        f = open(filepath, "w+")
        for agent in self.agents:
            for value in agent.weight:
                f.write(str(value) + " ")
            f.write(str(agent.fitness))
            f.write("\n")
        f.close()

    # loads weight from the designated filepath
    def load_weight(self, filepath="weight.txt"):
        f = open(filepath, "r")
        for line in f:
            weight = np.array([float(k) for k in line.split()])
            fitness = None
            if len(weight) == NUM_ATTRIBUTE + 1:
                fitness = weight[NUM_ATTRIBUTE]
                weight.resize(NUM_ATTRIBUTE)
            self.agents.append(GeneticAgent(self.env, weight, fitness))
        f.close()

    # main learning method: in each generation, it replenishes the population, produces offsprings, sieves out similar
    # members and discard unfit members
    def learn(self, num_generations=1):
        for generation in range(num_generations):
            self.replenish(REPLENISH_SIZE)
            for offspring in range(OFFSPRING_SIZE):
                self.agents.append(self.crossover())
            self.agents.sort(key=lambda x: x.fitness, reverse=True)
            self.sieve_similar()
            del self.agents[BATCH_SIZE:]
            self.save_weight("{}th_generation.txt".format(generation + 1))
            print("{}th_generation, best two: {} {} \n".format(generation+1, self.agents[0].fitness, self.agents[1].fitness))

    # crossover operator
    # it employs the Tournament Selection Process
    def crossover(self):
        tournament = random.sample(self.agents, TOURNAMENT_SIZE)
        best = max(tournament, key=lambda x: x.fitness)
        tournament.remove(best)
        second_best = max(tournament, key=lambda x: x.fitness)
        weight = best.fitness * best.weight + second_best.fitness * second_best.weight
        weight = self.normailize(mutate(weight))
        agent = GeneticAgent(self.env, weight)
        return agent

    # delete away members that are too similar to maintain diversity
    def sieve_similar(self):
        i = 0
        while i < len(self.agents):
            j = i + 1
            while j < len(self.agents):
                if distance.euclidean(self.agents[i].weight, self.agents[j].weight) < CLOSENESS:
                    self.agents.remove(self.agents[j])
                else:
                    j = j + 1
            i = i + 1
        print("remaining number = {}".format(len(self.agents)))
        return
