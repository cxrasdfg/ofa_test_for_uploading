# coding=utf-8

import numpy as np
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.model.individual import Individual
from pymoo.model.survival import Survival
from pymoo.model.evaluator import Evaluator
from pymoo.operators.crossover.point_crossover import PointCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.util.display import disp_multi_objective
from pymoo.util.dominator import Dominator
from pymoo.util.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.optimize import minimize
from pymop.problem import Problem

from ofa.ga.gene import MobileNetV3Gene

# =========================================================================================================
# Implementation
# based on nsga-net, plagiarized from https://github.com/ianwhale/nsga-net ...
# =========================================================================================================


class EvoOFA(GeneticAlgorithm):
    def __init__(self, problem, minimize_hook, **kwargs):
        kwargs['individial'] = Individual(rank=np.inf, crowding=-1)
        super(EvoOFA, self).__init__(**kwargs)

        self.tournament_type = 'comp_by_dom_and_crowding'
        self.func_display_attrs = disp_multi_objective
        self.problem = problem
        self.minimize_hook=minimize_hook
        self.n_gen=1

    @DeprecationWarning    
    def _solve(self, problem, termination):
        # generation counter
        self.n_gen = 1

        # initialize the first population and evaluate it
        self.pop = self._initialize()
        self._each_iteration(self, first=True)

        # while termination criterium not fulfilled
        while termination.do_continue(self):
            self.n_gen += 1

            # do the next iteration
            self.pop = self._next(self.pop)

            # execute the callback function in the end of each generation
            self._each_iteration(self)

        self._finalize()

        return self.pop

    def get_cur_pop(self):
        return self.cur_pop
    
    def next_stage(self):
        if self.n_gen == 1:
            self.history = []
            self.evaluator = Evaluator()
            # self.problem = problem
            # self.termination = termination
            self.pf = None

            self.disp = False
            self.callback = None
            self.save_history = False

            self.pop = self._initialize()
            self._each_iteration(self, first=True)

            self.cur_pop = self.pop
            self.n_gen += 1
        else:
            self.n_gen+=1
            self.pop = self._next(self.pop)
            self.cur_pop = self.pop 
            self._each_iteration(self)
    



# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(True)

    def _do(self, pop, n_survive, D=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F")

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = self.calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


    def calc_crowding_distance(self, F):
        infinity = 1e+14

        n_points = F.shape[0]
        n_obj = F.shape[1]

        if n_points <= 2:
            return np.full(n_points, infinity)
        else:

            # sort each column and get index
            I = np.argsort(F, axis=0, kind='mergesort')

            # now really sort the whole array
            F = F[I, np.arange(n_obj)]

            # get the distance to the last element in sorted list and replace zeros with actual values
            dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) \
                - np.concatenate([np.full((1, n_obj), -np.inf), F])

            index_dist_is_zero = np.where(dist == 0)

            dist_to_last = np.copy(dist)
            for i, j in zip(*index_dist_is_zero):
                dist_to_last[i, j] = dist_to_last[i - 1, j]

            dist_to_next = np.copy(dist)
            for i, j in reversed(list(zip(*index_dist_is_zero))):
                dist_to_next[i, j] = dist_to_next[i + 1, j]

            # normalize all the distances
            norm = np.max(F, axis=0) - np.min(F, axis=0)
            norm[norm == 0] = np.nan
            dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

            # if we divided by zero because all values in one columns are equal replace by none
            dist_to_last[np.isnan(dist_to_last)] = 0.0
            dist_to_next[np.isnan(dist_to_next)] = 0.0

            # sum up the distance to next and last and norm by objectives - also reorder from sorted list
            J = np.argsort(I, axis=0)
            crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # replace infinity with a large number
        crowding[np.isinf(crowding)] = infinity

        return crowding


class EvoNASProblem(Problem):
    def __init__(self, n_vars, n_obj, n_constr, lb, ub,
                 config, running_manager, gene_encoder):
        super(EvoNASProblem, self).__init__(n_var=n_vars, n_obj=n_obj, n_constr=n_constr, type_var=np.int32)
        self.xl = lb
        self.xu = ub
        self.config = config
        self._save_dir = self.config.path
        self._n_evaluated = 0
        self._cuda_device = 0
        # self.cur_pop = None
        self.running_manager = running_manager
        self.gene_encoder=gene_encoder
        

    def _evaluate(self, POP, F, *vargs, **kwargs):
        # self.running_manager.network.eval()

        # only need to sample this function, do not use this one for some reasons
        n_pop = len(POP)
        objs = np.full((n_pop, self.n_obj), np.nan)
        
        
        for idx, genome in enumerate(POP):
            arch_id = self._n_evaluated + 1
            self._n_evaluated += 1
            self.running_manager.write_log('Evaluate, Network id = {}'.format(arch_id))
            
            # decode the network and evaluate...
            # dummy test
            # performance = {'err':np.random.rand(),'flops':321.2 + np.random.normal()}
            
            ked_arch = self.gene_encoder.decode_arch(genome) # get the corresponding ks, expand, depth
            
            
            flops = self.running_manager.network.compute_flops(**ked_arch, input_size=224) # with gigabytes
            
            loss_, metrics = self.running_manager.validate(is_test=False)
            error_rate = 1- metrics[0] # top1
             
            objs[idx, 0] = error_rate
            objs[idx, 1] = flops

            # objs[idx, 0] = performance['err']
            # objs[idx, 1] = performance['flops']
            
            # time.sleep(1) # wait to release the gpu resources
        F['F'] = objs
        
        self.running_manager.network.train()
        
def binary_tournament(pop, P , algorithm, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(pop[a].F, pop[b].F)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, pop[a].rank, b, pop[b].rank,
                               method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, pop[a].get("crowding"), b, pop[b].get("crowding"),
                               method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(np.int32)

class NASSolver():
    def __init__(self, args, running_manager) -> None:
        self.args = args
        self.running_manager = running_manager
        self.logging = self.running_manager.write_log
        self.gene_encoder = MobileNetV3Gene(supernet=self.running_manager.net) # maybe other types such as resnet-50

        n_var = self.gene_encoder.get_n_var()
        lb = self.gene_encoder.lb
        ub = self.gene_encoder.ub

        self.problem = EvoNASProblem(n_vars=n_var, n_obj=2, n_constr=0, lb=lb, ub=ub, config=self.args, gene_encoder=self.gene_encoder, running_manager=self.running_manager)
        # please pay more attention for the duplication removal since the model at the early stage can be unmature.
        self.method = self.evoofa(pop_size=self.args.pop_size, n_offsprings=self.args.n_offsprings, eliminate_duplicates=True, problem=self.problem) 

    def step(self):
        self.method.next_stage()

    def get_cur_pop(self):
        return self.method.get_cur_pop()

    @DeprecationWarning
    def solve(self, **kwargs):
        if 'seed' not in kwargs:
            kwargs['seed'] = np.random.randint(1, 10000)

        # manually call the optmization step function.
        # res = minimize(problem, method, callback=self.minimize_hook, termination=('n_gen', 1e100), seed=self.args.manuel_seed) 
        res = self.method.solve(self.problem, termination=None, **kwargs) # this function will register the `problem`.


    def minimize_hook(self, algo):
        gen = algo.n_gen
        global GEN_ITER
        GEN_ITER = gen
        pop_var = algo.pop.get('X')
        pop_obj = algo.pop.get('F')
        
        # for indi in pop_var:
        #     for code in indi:
        #         print('%.2f'%(code), end=',')
        #     print()
        self.logging.info('generation = {}'.format(gen))
        self.logging.info('population error: best = {}, mean = {}, '
                    'median = {}, worst = {}'.format(np.min(pop_obj[:,0]), np.mean(pop_obj[:,0]),
                                                    np.median(pop_obj[:,0]), np.max(pop_obj[:,0])))

        self.logging.info('population complexity: best = {}, mean = {}, '
                    'median = {}, worst = {}'.format(np.min(pop_obj[:,1]), np.mean(pop_obj[:,1]),
                                                    np.median(pop_obj[:,1]), np.max(pop_obj[:,1])))
        
    
    def evoofa(self,
        pop_size,
        sampling=RandomSampling(var_type=np.int32),
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover=PointCrossover(n_points=2),
        mutation=PolynomialMutation(eta=3, var_type=np.int32),
        eliminate_duplicates=True,
        n_offsprings=None,
        **kwargs):
        """
        EvoPrune
        :param pop_size: size of the population
        :param sampling: sampling method
        :param selection: selection method
        :param crossover: crossover operator
        :param mutation: mutation operator
        :param eliminate_duplicates: enable duplicate removal
        :param n_offsprings: number of the offsprings
        :param kwargs: extra params
        :return: return the EvoPrune object
        """
        return EvoOFA(pop_size=pop_size, sampling=sampling,
                        selection=selection, crossover=crossover,
                        mutation=mutation, survival=RankAndCrowdingSurvival(),
                        eliminate_duplicates=eliminate_duplicates,
                        n_offsprings=n_offsprings, minimize_hook=self.minimize_hook,**kwargs)
