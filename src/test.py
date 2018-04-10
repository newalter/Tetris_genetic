from src.genetic_learner import GeneticLearner

# Main Method for training

learner = GeneticLearner()
# fill up the first generation if you are starting with nothing
# learner.replenish()

learner.load_weight("weight.txt")

# main learning method
learner.learn(num_generations=1000)

learner.save_weight("weight.txt")
