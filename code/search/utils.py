import random

def generate_random_combinations(task_lists, budget, n_trials, seed=42):
        
    state = random.getstate() # save current random state
    random.seed(seed)

    samples = set()
    task_list = []

    while len(samples) < n_trials:
        sample = sorted(random.sample(task_lists, budget))
        if tuple(sample) not in samples:
            samples.add(tuple(sample))
            task_list.append(sample)

    return task_list