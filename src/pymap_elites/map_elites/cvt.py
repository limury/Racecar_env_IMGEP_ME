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
import os
import math
import numpy as np
import scipy
import multiprocessing
from typing import List

# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree

from pymap_elites.map_elites import common as cm

class MutationFunction:
    def __init__(self, func):
        self.func = func
        self.curiosity = 0.01
    def __call__(self, parents, params):
        return self.func(parents, params)

class MapElites:
        
    def curiosity_select(self):
        # get minimum and sum
        # get list of keys to archive
        keys = np.empty(len(list(self.archive.keys())), dtype=object)
        keys[:] = list(self.archive.keys())

        pop_size = len(keys)
        self.probs = np.empty(len(keys), dtype=np.float64)
        _min = 0
        _sum = 0

        for i in range(len(keys)):
            self.probs[i] = self.archive[keys[i]].curiosity

        _min = np.amin(self.probs)
        # this is done so that all probs are > 0 (so probabilities are never 0)
        self.probs += (-_min + 0.01)
        _sum = np.sum(self.probs)
        #print("avg: ", _sum / pop_size)
        # apply formula prob = (curiosity - min_curiosity + 0.1) / (sum_curiosity - (min_curiosity - 0.1)* len(keys))
        self.probs = (self.probs) / _sum
        
        # Get indeces of keys we want to select
        if (self.n_parents > 1):
            indeces = [ np.random.choice(pop_size, size=self.n_parents, p=self.probs, replace=False) for i in range(int(self.params['batch_size'])) ]
            indeces = np.concatenate(indeces)
        else:
            indeces = np.random.choice(pop_size, size=self.select_size, p=self.probs)
        # this will be used later to determine parents
        self.batch_keys = keys[indeces] 

        # Initialize output list of keys
        self.batch = np.empty(self.select_size, dtype=object)
        
        for i in range(self.select_size):
            self.batch[i] = self.archive[ self.batch_keys[i] ] 

        return self.batch

    def NN_select(self):
        key_tuples = np.empty(len(list(self.archive.keys())), dtype=object)
        key_tuples[:] = list(self.archive.keys())
        keys = np.array(list(self.archive.keys()))
        kd = scipy.spatial.cKDTree(keys)
        queries = np.random.uniform(size=(self.select_size, self.dim_map))
        qq, ii = kd.query(queries, k=1)
        self.batch_keys = key_tuples[ii]
        self.batch = np.empty(self.select_size, dtype=object)
        for i in range(self.select_size):
            self.batch[i] = self.archive[self.batch_keys[i]]

        return self.batch



    def selector(self):
        # if minimum amount of elements are not present, no selection is used
        if len(self.archive) <= self.params['random_init'] * self.n_niches or self.params['selector'] == "random_search":
            self.batch = None
            self.batch_keys = None
            return None 

        # Define how many parents we need for each mutation
        # select_size: total amount of parents we need to select
        if (self.params["mutation"] == "polynomial"):
            self.select_size = int(self.params['batch_size'])
            self.n_parents = 1
        else:
            self.select_size = int(self.params['batch_size']) * 2
            self.n_parents = 2
    
        # if using uniform selection create list of random indeces, select those to create the batch
        if (self.params["selector"] == "uniform"):
            # get list of keys to archive
            keys = np.empty(len(list(self.archive.keys())), dtype=object)
            keys[:] = list(self.archive.keys())
            # get list of indeces we want to substitute
            indeces = np.random.randint(len(keys), size=self.select_size)
            # get list of keys chosen for bach
            self.batch_keys = keys[indeces]

            self.batch = np.empty(self.select_size, dtype=object)
            for i in range(self.select_size):
                self.batch[i] = self.archive[self.batch_keys[i]]

        elif ('curiosity' in self.params["selector"]):
            self.curiosity_select()

        elif ('nn' in self.params["selector"]):
            self.NN_select()

        # reshape the array, so that new length is [batch_size, n_parents], this way we can apply mutation to each 
        # element and if the mutation requires 2 parents an array of 2 is passed to it
        self.batch = np.reshape(self.batch, [int(self.params['batch_size']), self.n_parents])
        self.batch_keys = np.reshape(self.batch_keys, [int(self.params['batch_size']), self.n_parents])
        return self.batch

        
    # batch = ndarray of Species objects (parents)
    def mutate(self, batch: np.ndarray):
        
        # if batch=None then we are below minimum population threshold
        # so no mutation, just create random genotypes
        if self.batch is None:
            if (self.params['selector'] == 'random_search'):
                self.mutated_batch = np.random.uniform(low=self.params['min'], high=self.params['max'],size=(self.params['batch_size'], self.dim_x))
            else:
                self.mutated_batch = np.random.uniform(low=self.params['min'], high=self.params['max'],size=(self.params['random_init_batch'], self.dim_x))
        else:   
            # batch = 2D np array of cm.Species. where the inner array has the correct number of parents depending on the mutation
            # species batch = for all elements in batch get the actual species object
            # map maintains order so we can reconstruct which genos are from which parents

            # curiosity score for mutation functions
            # self.mutators = set of mutation functions to use
            if self.params['mutation'] == 'curiosity_func':
                # mutation_functions is an array of MutationFunction objects
                probabilities = np.array([ f.curiosity for f in self.mutation_functions ])
                probabilities /= np.sum(probabilities)
                # give slight probability to pick any mutation function
                probabilities += 0.05
                probabilities /= np.sum(probabilities)
                self.printable_probs.append(probabilities)

                self.mutators = np.random.choice(self.mutation_functions, self.params['batch_size'], 
                                                p=probabilities)
            else:
                self.mutators = np.random.choice(self.mutation_functions, self.params['batch_size'])
                
            self.mutated_batch = np.empty((int(self.params['batch_size']), self.dim_x) )
            for i in range(int(self.params['batch_size'])):
                fn = self.mutators[i]
                self.mutated_batch[i] = fn(self.batch[i], self.params)

        #self.mutated_batch is an np_array of genotypes to be evaluated
        return self.mutated_batch


    # converts a genotype to a Species object
    def create_species(self, genos: np.ndarray): 
        # BDs are stored in the form of [ ([coord], index) ]
        # each input geno can return an array of BDs so we must sort through them
        BDs = cm.parallel_eval(self.eval_func, self.mutated_batch, self.pool, self.params)
        # curiosity( self, individual ) returns curiosity
        if ('curiosity-child' in self.params['selector'] and not (self.batch is None)):
            _curiosity = lambda self,parent: self.archive[parent[0]].curiosity
        else:
            _curiosity = lambda self,parent: 0.0

        if (self.batch is None):
            _parent = lambda par: None
            data = zip(self.mutated_batch, BDs, self.mutated_batch)
        else:
            _parent = lambda par: par
            data = zip(self.mutated_batch, BDs, self.batch_keys)

        
        # initialize the new generation
        self.new_generation = [0] * len(self.mutated_batch)

        # mutated batch needed for geno of new indiv
        # batch keys needed to define parent
        # BDs is a list of tuples, with fitness and list of BD relating to that individual
        j = -1
        
        for geno, BD_arr, parent in data:
            j += 1
            offspring = [0] * len(BD_arr[1])
            # for each BD relating to this fitness, genotype and parent
            for i in range(len(BD_arr[1])):
                # create a species object
                offspring[i] = cm.Species( x=geno, desc=BD_arr[1][i], fitness=BD_arr[0], parent=_parent(parent[0]), curiosity=_curiosity(self, parent))
            offspring = np.array(offspring)
            self.new_generation[j] = offspring

        self.new_generation= np.concatenate(self.new_generation)
            
        return self.new_generation

    # checks if individual should be added to the archive
    def inspect_archive(self, indiv: cm.Species):
        niche_index = self.kdt.query([indiv.desc], k=1)[1][0][0]
        niche = self.kdt.data[niche_index]
        n = cm.make_hashable(niche)
        indiv.centroid = n
        # if the space is occupied and the indivs fitness is not better, dont add it
        if ((n in self.archive) and (indiv.fitness <= self.archive[n].fitness)):
            return -1
        else:
            return n


    # map-elites algorithm (CVT variant)
    def compute(self):
        """CVT MAP-Elites
           Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.
    
           Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile
    
        """
        self.archive = {}   # init archive (empty)
        self.n_evals = 0    # number of evaluations since the beginning
        self.b_evals = 0    # number evaluation since the last dump
    
        # main loop
        # while we have not reached the desired number of evaluations
        while (self.n_evals < self.max_evals):
            # select batch we want to mutate
            # self.batch: np.array[cm.Species]
            # self.batch_keys: np.array[int] -> keys to where the batch items are in the archive
            # self.select_size: total number of individuals selected as parents
            # self.n_parents: number of parents needed given the mutation operator
            self.selector()
            
            # mutate the batch
            # self.mutated_batch: np.array[[]] array of genotypes (new individuals)
            self.mutate(self.batch)
            
            # convert the new genotypes into full sbatchs
            # self.new_generation: np.array[cm.Species] -> array of offspring species (with centroid still not set)
            self.create_species(self.mutated_batch)
           
            # self.target_locations: np.array[int] -> keys of where to insert indivs in the archive. if -1 means don't insert
            self.target_locations = list(map(self.inspect_archive, self.new_generation))

            # if using curiosity, update all the curiosity scores of the items
            #tmp = {}
            if ('curiosity' in self.params['selector'] and not (self.batch is None)):
                for indiv, loc in zip(self.new_generation, self.target_locations):
                    #if indiv.centroid not in tmp:
                    #    tmp[indiv.centroid] = 0
                    #tmp[indiv.centroid] += 1
                    if (loc == -1):
                        if (self.archive[indiv.parent].curiosity > -15):
                            self.archive[indiv.parent].curiosity -= 1
                    else:
                        self.archive[indiv.parent].curiosity += 2

            # for x in tmp:
            #     if tmp[x]>2:
            #         print("created " + str(tmp[x]) + " at " + str(x))
            # print("end_gen")
            
            # update curiosity of functions
            if ('curiosity_func' in self.params['mutation'] and not (self.batch is None)): 
                for indiv, loc in zip(self.mutators, self.target_locations):
                    if (loc == -1):
                        if (indiv.curiosity >= 1):
                            indiv.curiosity -= 1
                    else:
                        indiv.curiosity += 4

           
            # insert all individuals for which it is correct
            for indiv, loc in zip(self.new_generation, self.target_locations):
                if (loc != -1):
                    self.archive[loc] = indiv

            
            # count evals
            self.n_evals += len(self.mutated_batch)
            self.b_evals += len(self.mutated_batch)

            # write archive
            if self.b_evals >= self.params['dump_period'] and self.params['dump_period'] != -1:
                print("[{}/{}], population: {}".format(self.n_evals, int(self.max_evals), len(self.archive)), end=" ", flush=True)
                MetricUtils.save_archive(self.archive, self.n_evals,self.log_dir, presets=self.presets, params=self.params)
                #cm.save_archive(self.archive, self.n_evals,self.log_dir, presets=self.presets)
                self.b_evals %= self.params['dump_period']
            # write log
            if self.log_file != None:
                fit_list = np.array([x.fitness for x in self.archive.values()])
                self.log_file.write("{} {} {} {} {} {}\n".format(self.n_evals, len(self.archive.keys()),
                        fit_list.max(), np.mean(fit_list), np.median(fit_list),
                        np.percentile(fit_list, 5), np.percentile(fit_list, 95)))
                self.log_file.flush()
        MetricUtils.save_archive(self.archive, self.n_evals,self.log_dir, presets=self.presets, params=self.params)
        #cm.save_archive(self.archive, self.n_evals,self.log_dir, presets=self.presets)
        return self.archive


    def name_presets(self):
        self.presets = self.params["selector"] + '-' + self.params["mutation"] + '-batch_'+str(self.params["batch_size"])+'-'
        for x in self.params['variant']:
            self.presets += x
            self.presets += '-'
        return self.presets

    def custom_centroids(self, n_bins: List[int] ):
        centers = []
        for x in n_bins:
            diff = 0.5/x
            centers.append([ ((1/x)*i) - diff for i in range(1, x + 1) ])
    
        output = [ [x] for x in centers[0]]
        for x in centers[1:]:
            new_list = []
            for y in output:
                
                for i in x:
                    curr = y.copy()
                    curr.append(i)
                    new_list.append(curr)
                output=new_list.copy()
        
        return np.array(output)

    def cvt(self, dim_map = None, dim_x = None, bins: List[int] = None):
        # create the CVT
        c = []
        if (bins):
            c = self.custom_centroids(bins)
        else:
            c = cm.cvt(self.n_niches, dim_map,
                  self.params['cvt_samples'], self.params['cvt_use_cache'])
        #print("C is: ", c)
        self.kdt = KDTree(c, leaf_size=30, metric='euclidean')
        cm.write_centroids(c)

    # mutation func is an array of possible mutation functions
    def __init__(self, dim_map: int, dim_x: int, eval_func, n_niches=1000, max_evals=1e5,params=cm.default_params,log_file=None,bins=None,log_dir="", mutation_func=[cm.multi_mutation], random_search=False):

        # setup the parallel processing pool
        num_cores = 32  #multiprocessing.cpu_count()
        self.pool = multiprocessing.Pool(num_cores)

        self.dim_map    = dim_map
        self.dim_x      = dim_x
        self.eval_func  = eval_func
        self.n_niches   = n_niches
        self.max_evals  = max_evals
        self.params     = params
        self.log_file   = log_file
        self.bins       = bins
        self.log_dir    = log_dir
        self.printable_probs = []

        if self.params['mutation'] == 'curiosity_func':
            self.mutation_functions = [ MutationFunction(f) for f in mutation_func ]
        else:
            self.mutation_functions = mutation_func

        # stores the name of this variant in self.presets
        self.name_presets()
        # initialized the kdtree
        self.cvt(dim_map, dim_x, bins)
        # performs map_elites
        self.compute()




class MetricUtils:
    @staticmethod
    def write_array(a, f):
        for i in a:
            f.write(str(i) + ',')

    @staticmethod
    def save_archive(archive, gen, log_dir, presets, params):
        generation = str(gen)
        generation = '0'*(8-len(generation)) + generation
        filename = log_dir+'/' + presets + generation + '.dat'

        data = list(archive.values())
        metrics = []
        if ('genotype' in params['log_outputs']):
            genos = np.vectorize(lambda x: x.x)
            metrics.append( genos(data) )

        if ('BD' in params['log_outputs']):
            desc = np.vectorize(lambda x: x.desc)
            metrics.append( desc(data) )
            
        if ('fitness' in params['log_outputs']):
            fitness = np.vectorize(lambda x: x.fitness)
            metrics.append( fitness(data) )

        if ('centroid' in params['log_outputs']):
            centroid = np.vectorize(lambda x: x.centroid)
            metrics.append( np.stack(centroid(data), axis=1) )
        
        if ('curiosity' in params['log_outputs']):
            curiosity = np.vectorize(lambda x: x.curiosity)
            metrics.append( np.expand_dims(curiosity(data), axis=1) )
        
        metrics = np.concatenate(metrics, axis=1)

        #out = []
        #for x in archive.values():
            #out.append( [x.centroid[0], x.centroid[1], x.centroid[2], x.curiosity] )

        #val = True
        #for x, y in zip(metrics, out):
            #for a, b in zip(x, y):
                #if (a!=b):
                    #val = False
        #print("Are the arrays the same? ", val)


        with open(filename, 'w') as f:
            np.savetxt(filename, metrics, delimiter=',', fmt='%f')










