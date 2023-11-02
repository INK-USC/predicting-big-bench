import random
import torch

def generate_random_combinations(param_distributions, n_trials, seed=42):

    num_combinations = 1
    for values in param_distributions.values():
        num_combinations *= len(values)
    assert n_trials <= num_combinations
        
    state = random.getstate() # save current random state
    random.seed(seed)

    samples = set()
    hp_list = []

    while len(samples) < n_trials:
        sample = {}
        for key, values in param_distributions.items():
            sample[key] = random.choice(values)
        if tuple(sample.items()) not in samples:
            samples.add(tuple(sample.items()))
            hp_list.append(sample)

    return hp_list
            
def tensorize(*args):
    """if an input arg is a numpy array or pd series, turn it to a tensor"""
    new_args = []
    for arg in args:
        if arg is not None:
            new_args.append(torch.tensor(arg, dtype=torch.float32))
        else:
            new_args.append(arg)
    if len(new_args) == 1:
        return new_args[0]
    else:
        return tuple(new_args)

if __name__ == "__main__":
    param_distributions = {
        "n_estimators": [10, 20, 30],
        "learning_rate": [1.0, 0.9, 0.8]
    }
    hp_list = generate_random_combinations(param_distributions, n_samples=5)
    print(hp_list)