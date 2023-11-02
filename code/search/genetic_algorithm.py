import random
import os
import pandas as pd

from data.dynamic import preprocess
from models.mlp import run_mlp_regression
from tqdm import tqdm, trange

from .utils import generate_random_combinations
from .fixed import search_run_fixed

POPULATION_SIZE = 10
NUM_GENERATIONS = 1000 # for testing
MUTATION_RATE = 0.1
MUTATION_N_CHANGES_RATE = 0.1
CROSSOVER_RATE = 0.8

def fill_random_to_length(individual, subtasks, target_len):
    # after mutation or crossover, we need to fill the selection to certain length
    subtasks_not_in_individual = set(subtasks) - set(individual)
    new_selected_tasks = random.sample(list(subtasks_not_in_individual), target_len - len(individual))

    individual = individual + new_selected_tasks
    assert len(individual) == target_len
    return individual

def mutation(individual, subtasks):
    n = len(individual) # = args.search_budget
    n_changes = max(int(MUTATION_N_CHANGES_RATE * n), 1)
    random.shuffle(individual)
    new_individual = individual[:-n_changes]

    new_individual = fill_random_to_length(new_individual, subtasks, n)

    return new_individual

def crossover(parent1, parent2, subtasks):
    n = len(parent1)
    crossover_point = random.randint(0, n - 1)
    random.shuffle(parent1)
    random.shuffle(parent2)

    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    child1 = fill_random_to_length(child1, subtasks, n)
    child2 = fill_random_to_length(child2, subtasks, n)

    return child1, child2

def search_genetic_algorithm(args, logger, run_func):

    df = pd.read_csv(args.full_file, index_col=False)
    subtasks = list(df["subtask"].unique())

    # generate initial population
    population = generate_random_combinations(subtasks, args.search_budget, POPULATION_SIZE)

    df = pd.DataFrame(columns=["id", "dev_r2", "n_tasks", "selected_tasks"])
    best_r2, best_task_list = -1e10, None

    d = {}

    for t in trange(NUM_GENERATIONS):

        # compute fitness of the current generation
        fitnesses = []
        for individual in population:
            key = frozenset(set(individual))
            if key in d:
                fitness = d[key]
            else:
                # fitness = random.random() # to check the overall workflow
                fitness = search_run_fixed(args, logger, run_func, individual) # dev_r2 used as "fitness"
                d[key] = fitness
                df.loc[len(df.index)] = [t, fitness, args.search_budget, individual]
                # add this to df too
            fitnesses.append(fitness)
        
            if fitness > best_r2:
                best_r2 = fitness
                best_task_list = individual
                
        # create the next generation
        new_population = []

        while len(new_population) < POPULATION_SIZE:
  
            parent1 = random.choices(population, weights=fitnesses, k=1)[0]
            parent2 = random.choices(population, weights=fitnesses, k=1)[0]

            # Perform crossover with probability CROSSOVER_RATE
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(parent1, parent2, subtasks)
            else:
                child1, child2 = parent1, parent2


            # Some chances for mutation
            if random.random() < MUTATION_RATE:
                child1 = mutation(child1, subtasks)
            
            if random.random() < MUTATION_RATE:
                child2 = mutation(child2, subtasks)

            new_population.extend([child1, child2])

        population = new_population

        if t % 10 == 0 and args.save_search_logs:
            df.to_csv(os.path.join(args.output_dir, "search.csv"), index=False)

    return best_r2, best_task_list, df
