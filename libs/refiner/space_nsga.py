"""
NSGA-II module
Taken from deap
Modified to apply the pareto front selection to the spaces components of the fitness rather
than the differents objectives
"""

import bisect
import math
import random

from itertools import chain
from operator import attrgetter, itemgetter
from collections import defaultdict
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from libs.refiner.core import Individual


######################################
# Non-Dominated Sorting   (NSGA-II)  #
######################################

def select_nsga(individuals: Sequence['Individual'], k: int, nd: str = 'standard'):
    """Apply NSGA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *k* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *k* will have no effect other
    than sorting the population according to their front rank. The
    list returned contains references to the input *individuals*. For more
    details on the NSGA-II operator see [Deb2002]_.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :returns: A list of selected individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """
    if nd == 'standard':
        pareto_fronts = sort_nondominated(individuals, k)
    elif nd == 'log':
        pareto_fronts = sort_log_nondominated(individuals, k)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))

    for front in pareto_fronts:
        assign_crowding_dist(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    k -= len(chosen)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"),
                              reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen


def sort_nondominated(individuals: Sequence['Individual'], k: int, first_front_only: bool = False):
    """Sort the first *k* *individuals* into different non domination levels
    using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
    see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
    where :math:`M` is the number of objectives and :math:`N` the number of
    individuals.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param first_front_only: If :obj:`True` sort only the first front and
                             exit.
    :returns: A list of Pareto fronts (lists), the first list includes
              nondominated individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """
    if k == 0:
        return []

    map_fit_ind = defaultdict(list)
    for ind in individuals:
        map_fit_ind[ind.fitness].append(ind)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i + 1:]:
            if fit_i.space_dominates(fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif fit_j.space_dominates(fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        n = min(len(individuals), k)
        while pareto_sorted < n:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts


def assign_crowding_dist(individuals: Sequence['Individual']):
    """Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.sp_wvalue, i) for i, ind in enumerate(individuals)]

    spaces_id = crowd[0][0].keys()
    n_spaces = len(spaces_id)

    for space_id in spaces_id:
        crowd.sort(key=lambda element: element[0][space_id])
        distances[crowd[0][1]] = math.inf
        distances[crowd[-1][1]] = math.inf
        if crowd[-1][0][space_id] == crowd[0][0][space_id]:
            continue
        norm = n_spaces * float(crowd[-1][0][space_id] - crowd[0][0][space_id])
        for prev, cur, _next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (_next[0][space_id] - prev[0][space_id]) / norm

    for space_id, dist in enumerate(distances):
        individuals[space_id].fitness.crowding_dist = dist


def select_tournament_dcd(individuals, k):
    """Tournament selection based on dominance (D) between two individuals, if
    the two individuals do not inter dominate the selection is made
    based on crowding distance (CD). The *individuals* sequence length has to
    be a multiple of 4. Starting from the beginning of the selected
    individuals, two consecutive individuals will be different (assuming all
    individuals in the input list are unique). Each individual from the input
    list won't be selected more than twice.

    This selection requires the individuals to have a :attr:`crowding_dist`
    attribute, which can be set by the :func:`assignCrowdingDist` function.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    """

    def _tourn(ind1, ind2):
        if ind1.fitness.space_dominates(ind2.fitness):
            return ind1
        elif ind2.fitness.space_dominates(ind1.fitness):
            return ind2

        if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
            return ind2
        elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
            return ind1

        if random.random() <= 0.5:
            return ind1
        return ind2

    individuals_1 = random.sample(individuals, len(individuals))
    individuals_2 = random.sample(individuals, len(individuals))

    chosen = []
    for i in range(0, k, 4):
        chosen.append(_tourn(individuals_1[i], individuals_1[i + 1]))
        chosen.append(_tourn(individuals_1[i + 2], individuals_1[i + 3]))
        chosen.append(_tourn(individuals_2[i], individuals_2[i + 1]))
        chosen.append(_tourn(individuals_2[i + 2], individuals_2[i + 3]))

    return chosen


#######################################
# Generalized Reduced runtime ND sort #
#######################################

def identity(obj):
    """Returns directly the argument *obj*.
    """
    return obj


def is_dominated(wvalues1, wvalues2):
    """Returns whether or not *wvalues1* dominates *wvalues2*.

    :param wvalues1: The weighted fitness values that would be dominated.
    :param wvalues2: The weighted fitness values of the dominant.
    :returns: :obj:`True` if wvalues2 dominates wvalues1, :obj:`False`
              otherwise.
    """
    not_equal = False
    for self_wvalue, other_wvalue in zip(wvalues1, wvalues2):
        if self_wvalue > other_wvalue:
            return False
        elif self_wvalue < other_wvalue:
            not_equal = True
    return not_equal


def median(seq: Sequence, key=identity):
    """Returns the median of *seq* - the numeric value separating the higher
    half of a sample from the lower half. If there is an even number of
    elements in *seq*, it returns the mean of the two middle values.
    """
    sseq = sorted(seq, key=key)
    length = len(seq)
    if length % 2 == 1:
        return key(sseq[(length - 1) // 2])
    else:
        return (key(sseq[(length - 1) // 2]) + key(sseq[length // 2])) / 2.0


def sort_log_nondominated(individuals: Sequence['Individual'],
                          k: int,
                          first_front_only: bool = False):
    """Sort *individuals* in pareto non-dominated fronts using the Generalized
    Reduced Run-Time Complexity Non-Dominated Sorting Algorithm presented by
    Fortin et al. (2013).

    :param individuals: A list of individuals to select from.
    :param k:
    :param first_front_only:
    :returns: A list of Pareto fronts (lists), with the first list being the
              true Pareto front.
    """
    if k == 0:
        return []

    # Separate individuals according to unique fitnesses
    unique_fits = defaultdict(list)
    for i, ind in enumerate(individuals):
        unique_fits[ind.fitness.sp_wvalue_t].append(ind)

    # Launch the sorting algorithm
    obj = len(individuals[0].fitness.sp_wvalue_t) - 1
    fitnesses = list(unique_fits.keys())
    front = dict.fromkeys(fitnesses, 0)

    # Sort the fitnesses lexicographically.
    fitnesses.sort(reverse=True)
    sort_helper_a(fitnesses, obj, front)

    # Extract individuals from front list here
    nb_fronts = max(front.values()) + 1
    pareto_fronts = [[] for _ in range(nb_fronts)]
    for fit in fitnesses:
        index = front[fit]
        pareto_fronts[index].extend(unique_fits[fit])

    # Keep only the fronts required to have k individuals.
    if not first_front_only:
        count = 0
        for i, front in enumerate(pareto_fronts):
            count += len(front)
            if count >= k:
                return pareto_fronts[:i + 1]
        return pareto_fronts
    else:
        return pareto_fronts[0]


def sort_helper_a(fitnesses, obj, front):
    """Create a non-dominated sorting of S on the first M objectives"""
    if len(fitnesses) < 2:
        return
    elif len(fitnesses) == 2:
        # Only two individuals, compare them and adjust front number
        s1, s2 = fitnesses[0], fitnesses[1]
        if is_dominated(s2[:obj + 1], s1[:obj + 1]):
            front[s2] = max(front[s2], front[s1] + 1)
    elif obj == 1:
        sweep_a(fitnesses, front)
    elif len(frozenset(list(map(itemgetter(obj), fitnesses)))) == 1:
        # All individuals for objective M are equal: go to objective M-1
        sort_helper_a(fitnesses, obj - 1, front)
    else:
        # More than two individuals, split list and then apply recursion
        best, worst = split_a(fitnesses, obj)
        sort_helper_a(best, obj, front)
        sort_helper_b(best, worst, obj - 1, front)
        sort_helper_a(worst, obj, front)


def split_a(fitnesses, obj):
    """Partition the set of fitnesses in two according to the median of
    the objective index *obj*. The values equal to the median are put in
    the set containing the least elements.
    """
    median_ = median(fitnesses, itemgetter(obj))
    best_a, worst_a = [], []
    best_b, worst_b = [], []

    for fit in fitnesses:
        if fit[obj] > median_:
            best_a.append(fit)
            best_b.append(fit)
        elif fit[obj] < median_:
            worst_a.append(fit)
            worst_b.append(fit)
        else:
            best_a.append(fit)
            worst_b.append(fit)

    balance_a = abs(len(best_a) - len(worst_a))
    balance_b = abs(len(best_b) - len(worst_b))

    if balance_a <= balance_b:
        return best_a, worst_a
    else:
        return best_b, worst_b


def sweep_a(fitnesses, front):
    """Update rank number associated to the fitnesses according
    to the first two objectives using a geometric sweep procedure.
    """
    stairs = [-fitnesses[0][1]]
    fstairs = [fitnesses[0]]
    for fit in fitnesses[1:]:
        idx = bisect.bisect_right(stairs, -fit[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[fit] = max(front[fit], front[fstair] + 1)
        for i, fstair in enumerate(fstairs[idx:], idx):
            if front[fstair] == front[fit]:
                del stairs[i]
                del fstairs[i]
                break
        stairs.insert(idx, -fit[1])
        fstairs.insert(idx, fit)


def sort_helper_b(best, worst, obj, front):
    """Assign front numbers to the solutions in H according to the solutions
    in L. The solutions in L are assumed to have correct front numbers and the
    solutions in H are not compared with each other, as this is supposed to
    happen after sortNDHelperB is called."""
    key = itemgetter(obj)
    if len(worst) == 0 or len(best) == 0:
        # One of the lists is empty: nothing to do
        return
    elif len(best) == 1 or len(worst) == 1:
        # One of the lists has one individual: compare directly
        for hi in worst:
            for li in best:
                if is_dominated(hi[:obj + 1], li[:obj + 1]) or hi[:obj + 1] == li[:obj + 1]:
                    front[hi] = max(front[hi], front[li] + 1)
    elif obj == 1:
        sweep_b(best, worst, front)
    elif key(min(best, key=key)) >= key(max(worst, key=key)):
        # All individuals from L dominate H for objective M:
        # Also supports the case where every individuals in L and H
        # has the same value for the current objective
        # Skip to objective M-1
        sort_helper_b(best, worst, obj - 1, front)
    elif key(max(best, key=key)) >= key(min(worst, key=key)):
        best1, best2, worst1, worst2 = split_b(best, worst, obj)
        sort_helper_b(best1, worst1, obj, front)
        sort_helper_b(best1, worst2, obj - 1, front)
        sort_helper_b(best2, worst2, obj, front)


def split_b(best, worst, obj):
    """Split both best individual and worst sets of fitnesses according
    to the median of objective *obj* computed on the set containing the
    most elements. The values equal to the median are attributed so as
    to balance the four resulting sets as much as possible.
    """
    median_ = median(best if len(best) > len(worst) else worst, itemgetter(obj))
    best1_a, best2_a, best1_b, best2_b = [], [], [], []
    for fit in best:
        if fit[obj] > median_:
            best1_a.append(fit)
            best1_b.append(fit)
        elif fit[obj] < median_:
            best2_a.append(fit)
            best2_b.append(fit)
        else:
            best1_a.append(fit)
            best2_b.append(fit)

    worst1_a, worst2_a, worst1_b, worst2_b = [], [], [], []
    for fit in worst:
        if fit[obj] > median_:
            worst1_a.append(fit)
            worst1_b.append(fit)
        elif fit[obj] < median_:
            worst2_a.append(fit)
            worst2_b.append(fit)
        else:
            worst1_a.append(fit)
            worst2_b.append(fit)

    balance_a = abs(len(best1_a) - len(best2_a) + len(worst1_a) - len(worst2_a))
    balance_b = abs(len(best1_b) - len(best2_b) + len(worst1_b) - len(worst2_b))

    if balance_a <= balance_b:
        return best1_a, best2_a, worst1_a, worst2_a
    else:
        return best1_b, best2_b, worst1_b, worst2_b


def sweep_b(best, worst, front):
    """Adjust the rank number of the worst fitnesses according to
    the best fitnesses on the first two objectives using a sweep
    procedure.
    """
    stairs, fstairs = [], []
    iter_best = iter(best)
    next_best = next(iter_best, False)
    for h in worst:
        while next_best and h[:2] <= next_best[:2]:
            insert = True
            for i, fstair in enumerate(fstairs):
                if front[fstair] == front[next_best]:
                    if fstair[1] > next_best[1]:
                        insert = False
                    else:
                        del stairs[i], fstairs[i]
                    break
            if insert:
                idx = bisect.bisect_right(stairs, -next_best[1])
                stairs.insert(idx, -next_best[1])
                fstairs.insert(idx, next_best)
            next_best = next(iter_best, False)

        idx = bisect.bisect_right(stairs, -h[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[h] = max(front[h], front[fstair] + 1)


######################################
# Strength Pareto         (SPEA-II)  #
######################################

def select_spea(individuals: Sequence['Individual'], k: int) -> Sequence['Individual']:
    """Apply SPEA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *n* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *n* will have no effect other
    than sorting the population according to a strength Pareto scheme. The
    list returned contains references to the input *individuals*. For more
    details on the SPEA-II operator see [Zitzler2001]_.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.

    .. [Zitzler2001] Zitzler, Laumanns and Thiele, "SPEA 2: Improving the
       strength Pareto evolutionary algorithm", 2001.
    """
    n = len(individuals)
    fit_len = len(individuals[0].fitness.sp_wvalue_t)
    sqrt_n = math.sqrt(n)
    strength_fits = [0] * n
    fits = [0] * n
    dominating_inds = [list() for _ in range(n)]

    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals[i + 1:], i + 1):
            if ind_i.fitness.space_dominates(ind_j.fitness):
                strength_fits[i] += 1
                dominating_inds[j].append(i)
            elif ind_j.fitness.space_dominates(ind_i.fitness):
                strength_fits[j] += 1
                dominating_inds[i].append(j)

    for i in range(n):
        for j in dominating_inds[i]:
            fits[i] += strength_fits[j]

    # Choose all non-dominated individuals
    chosen_indices = [i for i in range(n) if fits[i] < 1]

    if len(chosen_indices) < k:  # The archive is too small
        for i in range(n):
            distances = [0.0] * n
            for j in range(i + 1, n):
                dist = 0.0
                for m in range(fit_len):
                    val = (individuals[i].fitness.sp_wvalue_t[m] -
                           individuals[j].fitness.sp_wvalue_t[m])
                    dist += val * val
                distances[j] = dist
            kth_dist = _randomized_select(distances, 0, n - 1, sqrt_n)
            density = 1.0 / (kth_dist + 2.0)
            fits[i] += density

        next_indices = [(fits[i], i) for i in range(n)
                        if i not in chosen_indices]
        next_indices.sort()
        # print next_indices
        chosen_indices += [i for _, i in next_indices[:k - len(chosen_indices)]]

    elif len(chosen_indices) > k:  # The archive is too large
        n = len(chosen_indices)
        distances = [[0.0] * n for _ in range(n)]
        sorted_indices = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist = 0.0
                for m in range(fit_len):
                    val = (individuals[chosen_indices[i]].fitness.sp_wvalue_t[m] -
                           individuals[chosen_indices[j]].fitness.sp_wvalue_t[m])
                    dist += val * val
                distances[i][j] = dist
                distances[j][i] = dist
            distances[i][i] = -1

        # Insert sort is faster than quick sort for short arrays
        for i in range(n):
            for j in range(1, n):
                m = j
                while m > 0 and distances[i][j] < distances[i][sorted_indices[i][m - 1]]:
                    sorted_indices[i][m] = sorted_indices[i][m - 1]
                    m -= 1
                sorted_indices[i][m] = j

        size = n
        to_remove = []
        while size > k:
            # Search for minimal distance
            min_pos = 0
            for i in range(1, n):
                for j in range(1, size):
                    dist_i_sorted_j = distances[i][sorted_indices[i][j]]
                    dist_min_sorted_j = distances[min_pos][sorted_indices[min_pos][j]]

                    if dist_i_sorted_j < dist_min_sorted_j:
                        min_pos = i
                        break
                    elif dist_i_sorted_j > dist_min_sorted_j:
                        break

            # Remove minimal distance from sorted_indices
            for i in range(n):
                distances[i][min_pos] = math.inf
                distances[min_pos][i] = math.inf

                for j in range(1, size - 1):
                    if sorted_indices[i][j] == min_pos:
                        sorted_indices[i][j] = sorted_indices[i][j + 1]
                        sorted_indices[i][j + 1] = min_pos

            # Remove corresponding individual from chosen_indices
            to_remove.append(min_pos)
            size -= 1

        for index in reversed(sorted(to_remove)):
            del chosen_indices[index]

    return [individuals[i] for i in chosen_indices]


def _randomized_select(array, begin, end, i):
    """Allows to select the ith smallest element from array without sorting it.
    Runtime is expected to be O(n).
    """
    if begin == end:
        return array[begin]
    q = _randomized_partition(array, begin, end)
    k = q - begin + 1
    if i < k:
        return _randomized_select(array, begin, q, i)
    else:
        return _randomized_select(array, q + 1, end, i - k)


def _randomized_partition(array, begin, end):
    i = random.randint(begin, end)
    array[begin], array[i] = array[i], array[begin]
    return _partition(array, begin, end)


def _partition(array, begin, end):
    x = array[begin]
    i = begin - 1
    j = end + 1
    while True:
        j -= 1
        while array[j] > x:
            j -= 1
        i += 1
        while array[i] < x:
            i += 1
        if i < j:
            array[i], array[j] = array[j], array[i]
        else:
            return j


__all__ = ['select_nsga', 'select_spea', 'sort_nondominated', 'sort_log_nondominated',
           'select_tournament_dcd']
