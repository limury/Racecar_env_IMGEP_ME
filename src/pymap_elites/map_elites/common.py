#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
#
import os
import math
import numpy as np
import multiprocessing
from pathlib import Path
import sys
import random
from collections import defaultdict
from sklearn.cluster import KMeans

default_params = \
        {
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        # we evaluate in batches to paralleliez
        "batch_size": 100,
        # proportion of niches to be filled before starting
        "random_init": 0.1,
        # batch for random initialization
        "random_init_batch": 100,
        # when to write results (one generation = one batch)
        "dump_period": 10000,
        # do we use several cores?
        "parallel": True,
        # do we cache the result of CVT and reuse?
        "cvt_use_cache": True,
        # min/max of parameters
        "min": 0,
        "max": 1,
        # only useful if you use the 'iso_dd' variation operator
        "iso_sigma": 0.01,
        "line_sigma": 0.2
    }

class Species:
    def __init__(self, x, desc, fitness, centroid=None, parent=None, curiosity=0.001):
        # genotype
        self.x = x
        # BD
        self.desc = desc
        # fitness
        self.fitness = fitness
        # centroid it relates to in KDTree
        self.centroid = centroid
        # curiosity score
        self.curiosity = curiosity
        # Index of parent in archive (int)
        self.parent = parent
    def __getitem__(self, key):
        return None

# TODO: input to all mutations is now a np.array[Species]

def gaussian_mutation(parents, params):
    x = parents[0].x
    y = np.copy(parents[0].x)
    dev = params['std_dev']
    delta = np.random.normal(scale=dev, size=len(x))
    p = np.random.random(size=len(x))
    for i in range(0, len(x)):
        if (p[i] < params['mutation_prob']):
            y[i] += delta[i]
    y = np.clip(y, params['min'], params['max'])
    return y

def polynomial_mutation(parents, params):
    '''
    Cf Deb 2001, p 124 ; param: eta_m
    '''
    x = parents[0].x
    #print('parent: ', x)
    y = np.copy(parents[0].x)
    mutation_prob = params['mutation_prob']
    eta_m = params['eta_m'] 
    r = np.random.random(size=len(x))
    p = np.random.random(size=len(x))
    for i in range(0, len(y)):
        if (p[i] < mutation_prob):
            if r[i] < 0.5:
                delta_i = math.pow(2.0 * r[i], 1.0 / (eta_m + 1.0)) - 1.0
            else:
                delta_i = 1 - math.pow(2.0 * (1.0 - r[i]), 1.0 / (eta_m + 1.0))
            y[i] += delta_i

    y = np.clip(y, params['min'], params['max'])
    #print('child: ', y)
    #print('diff: ', y-x)
    #print('\n')
    return y


def sbx(parents, params):
    '''
    SBX (cf Deb 2001, p 113) Simulated Binary Crossover

    A large value ef eta gives a higher probablitity for
    creating a `near-parent' solutions and a small value allows
    distant solutions to be selected as offspring.
    '''
    eta = 10.0
    xl = params['min']
    xu = params['max']
    z = np.copy(parents[0].x)
    y = parents[1].x
    x = parents[0].x
    r1 = np.random.random(size=len(x))
    r2 = np.random.random(size=len(x))

    for i in range(0, len(x)):
        if abs(x[i] - y[i]) > 1e-15:
            x1 = min(x[i], y[i])
            x2 = max(x[i], y[i])

            beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            rand = r1[i]
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

            c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

            beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
            alpha = 2.0 - beta ** -(eta + 1)
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
            c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

            c1 = min(max(c1, xl), xu)
            c2 = min(max(c2, xl), xu)

            if r2[i] <= 0.5:
                z[i] = c2
            else:
                z[i] = c1
    
    z = np.clip(z, params['min'], params['max'])
    #print("diff1: ", z-x)
    #print("diff2: ", z-y)
    return z


def iso_dd(parents, params):
    '''
    Iso+Line
    Ref:
    Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation.
    GECCO 2018
    '''
    x = np.copy(parents[0].x)
    y = np.copy(parents[1].x)
    p_max = np.array(params["max"])
    p_min = np.array(params["min"])
    a = np.random.normal(0, params['iso_sigma'], size=len(x))
    b = np.random.normal(0, params['line_sigma'])
    norm = np.linalg.norm(x - y)
    z = np.copy(x) + a + b * (x - y)
    return np.clip(z, p_min, p_max)

def multi_mutation(parents, params):
    functions = []
    if ('sbx' in params['mutation']):
        functions.append(sbx)
    if ('iso_dd' in params['mutation']):
        functions.append(iso_dd)
    if ('polynomial' in params['mutation']):
        functions.append(polynomial_mutation)
    if ('gaussian' in params['mutation']):
        functions.append(gaussian_mutation)
    return functions[np.random.randint(low=0, high=len(functions))](parents, params)

def __centroids_filename(k, dim):
    return 'centroids_' + str(k) + '_' + str(dim) + '.dat'


def write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    #with open(filename, 'w') as f:
    #    for p in centroids:
    #        for item in p:
    #            f.write(str(item) + ' ')
    #        f.write('\n')


def cvt(k, dim, samples, cvt_use_cache=True):
    # check if we have cached values
    fname = __centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
    # otherwise, compute cvt
    print("Computing CVT (this can take a while...):", fname)

    x = np.random.rand(samples, dim)
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, n_jobs=-1, verbose=1)#,algorithm="full")
    k_means.fit(x)
    #__write_centroids(k_means.cluster_centers_)

    return k_means.cluster_centers_


def make_hashable(array):
    return tuple(map(float, array))

# to_evaluate is genotypes
# evaluate_function is eval()
# to_evaluate is a list of Species that we want to evaluate the genotype of
def parallel_eval(evaluate_function, to_evaluate, pool, params):
    if params['parallel'] == True:
        s_list = np.array(list(pool.map(evaluate_function, to_evaluate)), dtype=object)
    else:
        s_list = np.array(list(map(evaluate_function, to_evaluate)), dtype=object)
    return s_list 

def print_array( arr, log_dir, presets ):
    def write_array(a, f):
        for i in a:
            f.write(str(i) + ',')
    
    filename = log_dir + '/' + presets.txt + '.txt'
    with open(filename, 'w') as f:
        for line in arr:
            write_array(line, f)
            f.write('\n')

# format: fitness, centroid, desc, genome \n
# fitness, centroid, desc and x are vectors
def save_archive(archive, gen, log_dir="", presets=""):
    def write_array(a, f):
        for i in a:
            f.write(str(i) + ',')
    generation = str(gen)
    generation = '0'*(8-len(generation)) + generation
    filename = log_dir+'/' + presets + generation + '.dat'
    with open(filename, 'w') as f:
        for k in archive.values():
            #f.write(str(k.fitness) + ' ')
            write_array(k.centroid, f)
            #write_array(  k.desc  , f)
            #write_array(k.x, f)
            f.write("\n")