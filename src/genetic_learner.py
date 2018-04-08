import random

import numpy as np

from src.Tetris_Env import TetrisEnv
from src.genetic_agent import GeneticAgent

BATCH_SIZE = 1000


def mutate(weight):
    if random.random() < 0.05:
        pos = random.randrange(0, 4)
        weight[pos] = weight[pos] + random.random() * 0.4 - 0.2
    return weight


class GeneticLearner():
    agents = []
    env = TetrisEnv()

    def initialization(self):
        for i in range(BATCH_SIZE):
            weight = self.normailize(np.random.rand(4) * 2 - 1)
            self.agents.append(GeneticAgent(env=self.env, weight=weight))

    def normailize(self, weight):
        square_sum = sum(k ** 2 for k in weight)
        return weight / square_sum

    def save_weight(self, filepath="weight.txt"):
        f = open(filepath, "w+")
        for agent in self.agents:
            for value in agent.weight:
                f.write(str(value) + " ")
            f.write(str(agent.fitness))
            f.write("\n")
        f.close()

    def load_weight(self, filepath="weight.txt"):
        f = open(filepath, "r")
        for line in f:
            weight = [float(k) for k in line.split()]
            fitness = None
            if len(weight) == 5:
                fitness = weight[4]
                del weight[4:]
            self.agents.append(GeneticAgent(self.env, weight, fitness))
        f.close()

    def learn(self, num_generations=1):
        for generation in range(num_generations):
            for offspring in range(int(0.3 * BATCH_SIZE)):
                self.agents.append(self.breeding())
            self.agents.sort(key=lambda x: x.fitness, reverse=True)
            del self.agents[BATCH_SIZE:]
            print("{} {} \n".format(self.agents[0].fitness, self.agents[1].fitness))

    def breeding(self):
        tournament = random.sample(self.agents, int(0.1 * BATCH_SIZE))
        best = max(tournament, key=lambda x: x.fitness)
        tournament.remove(best)
        second_best = max(tournament, key=lambda x: x.fitness)
        weight = best.fitness * best.weight + second_best.fitness * second_best.weight
        weight = self.normailize(mutate(weight))
        agent = GeneticAgent(self.env, weight)
        return agent
