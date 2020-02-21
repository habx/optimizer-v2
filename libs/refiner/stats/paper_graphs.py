"""
Simple module to draw pretty graphs
"""

# get the data
import os
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

mpl.use("TkAgg")
def plot_generation():

    module_path = os.path.dirname(__file__)
    input_path = os.path.join(module_path, "fitness_history_elite_nsga.p")

    fitness_history = pickle.load(open(input_path, "rb"))

    color_map = cm.get_cmap(name="plasma")
    print("Pop of {} individuals • {} generations".format(len(fitness_history[0]), len(fitness_history)))
    i = 0

    fig = plt.figure()
    ax = plt.gca()

    for g in fitness_history:
        x = [-f[0] for f in g]
        y = [-f[3] for f in g]
        i += 1
        ax.scatter(x, y, c=[color_map(i / len(fitness_history))]*len(x))

    ax.set_yscale('log')
    ax.set_xlabel('Corners')
    ax.set_ylabel('Width Depth')
    ax.set_xscale('log')
    # ax.set_ylim(ymin=100, ymax=1000)
    plt.show()


def plot_fitness_convergence():
    """ plot the fitness convergence """
    slug = "ARCH005_blueprint"
    cx = 0.7
    el = 0.5
    m = 120
    n_time = 20
    module_path = os.path.dirname(__file__)
    input_path = os.path.join(module_path, "best_{}_mu_{}_cx_{}_elite_{}_n_{}.p".format(slug, m, cx, el, n_time))
    fitness_history = pickle.load(open(input_path, "rb"))

    x = list(range(max(len(f) for f in fitness_history)))
    n = len(fitness_history)

    # y = [sum(f[i] for f in fitness_history) / n for i in x]

    fig, ax = plt.subplots()
    for f in fitness_history:
        while len(f) < len(x):
            f.append(f[len(f)-1])

    y = [sum(f[i] for f in fitness_history)/n for i in range(1, 50)]
    y_min = [y[i - 1] - min(f[i] for f in fitness_history) for i in range(1, 50)]
    y_max = [max(f[i] for f in fitness_history) - y[i -1] for i in range(1, 50)]

    ax.step(x[1:50], y, where='post', label='post', c='black')
    ax.errorbar(x[1:50], y, yerr=[y_min, y_max], c='b', capsize=2, lw=1, fmt='.k')
    ax.set_xlabel('Generations')
    ax.set_ylabel('Average Fitness')
    # ax.set_title('Average, minimum and maximum fitness per generation for 25 runs')

    plt.show()


if __name__ == '__main__':

    plot_fitness_convergence()

# no crossover : cbx = 0
# time elapsed : 96.46 s / score: -2991.36 / # gen : 67
# time elapsed : 101s / score: --2518.36 / # gen : 53

# no crossover : cbx = 0.9
# time elapsed :  87s / score: -2450.9 / # gen : 61
# time elapsed : 112s / score: -2450.9  / # gen : 74
# time elapsed : 103s / score: -2450.9  / # gen : 65

