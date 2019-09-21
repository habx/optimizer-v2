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


if __name__ == '__main__':

    module_path = os.path.dirname(__file__)
    input_path = os.path.join(module_path, "fitness_history_elite_0_8_wd.p")

    fitness_history = pickle.load(open(input_path, "rb"))

    color_map = cm.get_cmap(name="plasma")
    print("Pop of {} individuals • {} generations".format(len(fitness_history[0]), len(fitness_history)))
    i = 0

    fig = plt.figure()
    ax = plt.gca()

    for g in fitness_history:
        x = [-f[0] for f in g]
        y = [-f[2] for f in g]
        i += 1
        ax.scatter(x, y, c=[color_map(i / len(fitness_history))]*len(x))

    ax.set_yscale('log')
    ax.set_xlabel('Corners')
    ax.set_ylabel('Width Depth')
    ax.set_xscale('log')
    # ax.set_xlim(xmin=100, xmax=1000)
    plt.show()

# no crossover : cbx = 0
# time elapsed : 96.46 s / score: -2991.36 / # gen : 67
# time elapsed : 101s / score: --2518.36 / # gen : 53

# no crossover : cbx = 0.9
# time elapsed :  87s / score: -2450.9 / # gen : 61
# time elapsed : 112s / score: -2450.9  / # gen : 74
# time elapsed : 103s / score: -2450.9  / # gen : 65

