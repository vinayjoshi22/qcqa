import os
from typing import Union, Any
import numpy as np

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover as SBX
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling, BinaryRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.result import Result
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

class Constants:
    """
    Define constants for the QCQA problem.
    Attributes:
        ALGO1 (str): Algorithm 1
        ALGO2 (str): Algorithm 2
    """
    ALGO1 = "algo1"
    ALGO2 = "algo2"

class CustomOutput(Output):

    def __init__(self):
        """
        Define the custom output format for genetic algoritm run.
        Note: This assumes default multi-objective optimization problem with 2 objectives.
        """
        super().__init__()
        self.f1 = Column("KV (lowest)", width=20)
        self.f2 = Column("WSE (lowest)", width=20)
        self.columns += [self.f1, self.f2]

    def update(self, algorithm):
        super().update(algorithm)
        self.f1.set(np.amin(algorithm.pop.get("F")[:, 0]))
        self.f2.set(np.amin(algorithm.pop.get("F")[:, 1]))

class QCQAProblem(Problem):

    def __init__(self,  n_var:int, 
                        n_obj:int, 
                        xl:int, 
                        xu:int, 
                        fitness_kv_cache_callbk:Any, 
                        fitness_wse_callbk:Any,
                        instance:int=0
                ) -> None:
        """
        Initializes the problem class from Pymoo.
        Args:
            n_var (int): Number of variables
            n_obj (int): Number of objectives
            xl (int): Lower bound
            xu (int): Upper bound
            fitness_kv_cache_callbk (function): Fitness function for KV-cache
            fitness_wse_callbk (function): Fitness function for WSE
            instance (int): For algorithm 1, it is the layer index. For algorithm 2, it is the group info.
        Returns:
            None
        """
        self.fitness_kv_cache_callbk = fitness_kv_cache_callbk
        self.fitness_wse_callbk = fitness_wse_callbk
        self.instance = instance
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

    def fitness_kv_cache(self, x:np.ndarray) -> np.ndarray:
        """
        Obtain the fitness using the callback function for KV-cache.
        Args:
            x (np.ndarray): Population of candidates dim (pop_sz, num_heads)
        Returns:
            kv_cache (np.ndarray): KV-cache reduction ratio for the given population.
        """
        return self.fitness_kv_cache_callbk(x, self.instance)
    
    def fitness_wse(self, x:np.ndarray) -> np.ndarray:
        """
        Obtain the fitness using the callback function for WSE.
        Args:
            x (np.ndarray): Population of candidates dim (pop_sz, num_heads)
        Returns:
            wse (np.ndarray): Weight sharing error for the given population.
        """
        return self.fitness_wse_callbk(x, self.instance)

    def _evaluate(self, x:np.ndarray, out:Result, *args, **kwargs)  -> None:
        f1 = self.fitness_kv_cache(x)
        f2 = self.fitness_wse(x)
        out["F"] = np.column_stack([f1, f2])

class QCQA:
    def __init__(self,  num_heads:int, 
                        num_layers:int, 
                        num_groups:int, 
                        model_checkpoint:dict, 
                        KV_parse_strings:list, 
                        output_path:str, 
                        ga_config:dict
                ) -> None:
        """
        Initializes the QCQA object to run algorithm 1 from the paper.
        Args:
            num_heads (int): Number of attention heads
            num_layers (int): Number of layers
            num_groups (int): Number of groups
            model_checkpoint (dict): Model checkpoint dictionary
            KV_parse_strings (list): List of strings to parse the Key and Value weights
            output_path (str): Output path to save the KV weights
            ga_config (dict): Genetic algorithm configuration
        Returns:
            None
        """
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model_checkpoint = model_checkpoint
        self.KV_parse_strings = KV_parse_strings
        self.output_path = output_path
        self.kv_save_path, self.kv_weights = self.extract_KV_weights(self.num_heads, self.model_checkpoint, self.KV_parse_strings, self.output_path)
        self.num_groups = num_groups

        n_var = num_heads
        n_obj = 2
        xl = 0
        xu = num_groups - 1
        
        algo1_config = ga_config.get(Constants.ALGO1, {})
        pop_sz = algo1_config.get("pop_sz", 100)
        crossover_prob = algo1_config.get("crossover_prob", 0.9)
        mutation_prob = algo1_config.get("mutation_prob", 1.0/n_var)
        self.problems_algo1 = [self.get_problem(algorithm=Constants.ALGO1, n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, instance=i) for i in range(num_layers)]
        self.algorithm_1 = self.get_algorithm(algorithm=Constants.ALGO1, pop_sz=pop_sz, crossover_prob=crossover_prob, mutation_prob=mutation_prob)

        algo2_config = ga_config.get(Constants.ALGO2, {})
        pop_sz = algo2_config.get("pop_sz", 100)
        crossover_prob = algo2_config.get("crossover_prob", 0.9)
        mutation_prob = algo2_config.get("mutation_prob", 1.0/n_var)
        self.problems_algo2 = self.get_problem(algorithm=Constants.ALGO2, n_var=num_layers, n_obj=n_obj, xl=xl, xu=1, instance=0)
        self.algorithm_2 = self.get_algorithm(algorithm=Constants.ALGO2, pop_sz=pop_sz, crossover_prob=crossover_prob, mutation_prob=mutation_prob)

    def set_group_info(self, group_info:np.ndarray) -> np.ndarray:
        """
        Sets the group information for Algorithm 2 from the paper.
        Args:
            x (np.ndarray): Population of candidates dim (num_layers, num_heads)
        Returns:
            None
        """
        assert group_info.shape == (self.num_layers, self.num_heads), "Group info shape must be (num_layers, num_heads) but given {}.".format(group_info.shape)
        self.problems_algo2.instance = group_info

    def extract_KV_weights(self, num_heads:int, model_checkpoint:dict, KV_parse_strings:list, output_path:str) -> Union[str, np.ndarray]:
        """
        Extracts the Key and Value weights from the model checkpoint and saves them to a numpy file.
        Args:
            num_heads (int): Number of attention heads
            model_checkpoint (dict): Model checkpoint dictionary
            KV_parse_strings (list): List of strings to parse the Key and Value weights
            output_path (str): Output path to save the KV weights
        Returns:
            kv_save_path (str): Path to the saved KV weights
            KV (np.ndarray): KV weights
        """
        Ks, Vs = [], []
        for kk, kv in KV_parse_strings:
            assert kk in model_checkpoint.keys(), "Key {} not found in model checkpoint!".format(kk)
            assert kv in model_checkpoint.keys(), "Key {} not found in model checkpoint!".format(kv)
            vk = model_checkpoint[kk]
            vv = model_checkpoint[kv]
            model_dim = vk.shape[0]
            head_dim = model_dim // num_heads
            assert model_dim == num_heads*head_dim, "Model dim must be divisible by num_heads"
            Ks.append(vk.detach().cpu().numpy().reshape(num_heads, head_dim, -1))
            print("Recording Key weights for: {}".format(kk))
            Vs.append(vv.detach().cpu().numpy().reshape(num_heads, head_dim, -1))
            print("Recording Value weights for: {}".format(kv))
        assert (len(Ks) ==self.num_layers) and (len(Vs) ==self.num_layers) , "Number of Key:{} and Value:{} weights must be equal to number of layers:{}!".format(len(Ks), len(Vs), self.num_layers)
        Ks = np.stack(Ks, axis=0) # (num_layers, num_heads, head_dim, model_dim)
        Vs = np.stack(Vs, axis=0) # (num_layers, num_heads, head_dim, model_dim)
        KV = np.stack([Ks, Vs], axis=0) # (2, num_layers, num_heads, head_dim, model_dim)
        kv_save_path = os.path.join(output_path, "KV_weights.npy")
        np.save(kv_save_path, KV)
        print("KV weights saved to: {}".format(kv_save_path))
        return kv_save_path, KV

    def get_problem(self, algorithm:str, n_var:int, n_obj:int, xl:int, xu:int, instance:int=0) -> QCQAProblem:
        """ 
        Forms the QCQA problem.
        Args:
            n_var (int): Number of variables
            n_obj (int): Number of objectives
            xl (int): Lower bound
            xu (int): Upper bound
        Returns:
            problem (QCQAProblem): QCQA problem
        """
        if algorithm == Constants.ALGO1:
            problem = QCQAProblem(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, 
                              fitness_kv_cache_callbk=self.fitness_kv_cache, 
                              fitness_wse_callbk=self.fitness_wse, instance=instance)
        elif algorithm == Constants.ALGO2:
            problem = QCQAProblem(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, 
                              fitness_kv_cache_callbk=self.fitness_accum_kv_cache, 
                              fitness_wse_callbk=self.fitness_accum_wse, instance=instance)
        else:
            raise ValueError("Algorithm {} not supported".format(algorithm))
        return problem

    def get_algorithm(self, algorithm:str, pop_sz:int, crossover_prob:float, mutation_prob:float) -> NSGA2:
        """
        Forms the algorithm for the QCQA problem.
        Args:
            algorithm (str): Algorithm to use
            pop_sz (int): Population size
            n_gen (int): Number of generations
            n_offsprings (int): Number of offsprings
            n_var (int): Number of variables
            n_obj (int): Number of objectives
            xl (int): Lower bound
            xu (int): Upper bound
        Returns:
            algorithm (GeneticAlgorithm): Genetic algorithm for the QCQA problem
        """
        if algorithm == Constants.ALGO1:
            sampling = IntegerRandomSampling()
            crossover = SBX(prob=crossover_prob, vtype=int, repair=RoundingRepair())
            mutation = PM(prob=mutation_prob, vtype=int, repair=RoundingRepair())
            algorithm = NSGA2(pop_sz=pop_sz, crossover=crossover, mutation=mutation, sampling=sampling, eliminate_duplicates=True)
        elif algorithm == Constants.ALGO2:
            sampling = BinaryRandomSampling()
            crossover = TwoPointCrossover(prob=crossover_prob)
            mutation = BitflipMutation(prob=mutation_prob)
            algorithm = NSGA2(pop_sz=pop_sz, crossover=crossover, mutation=mutation, sampling=sampling, eliminate_duplicates=True)
        else:
            raise ValueError("Algorithm {} not supported".format(algorithm))
        return algorithm
    
    def get_num_groups(self, x:np.ndarray):
        """
        Computes the number of groups for the given population of candidates.
        Args:
            x (np.ndarray): Population of candidates dim (pop_sz, num_heads)
        Returns:
            num_groups (np.ndarray): Number of groups for the given population.
        """
        pop_sz = x.shape[0]
        cands = x.reshape([pop_sz, self.num_heads])
        cands = np.sort(cands, axis=-1)
        cands = np.diff(cands, axis=-1) 
        num_groups = np.sum(cands>0, axis=-1) + 1
        return num_groups
    
    def fitness_kv_cache(self, x:np.ndarray, instance:int) -> np.ndarray:
        """
        Computes KV-cache reduction ratio for the given population of candidates.
        Args:
            x (np.ndarray): Population of candidates dim (pop_sz, num_heads)
            instance (int): Layer index
        Returns:
            kv_cache (np.ndarray): KV-cache reduction ratio for the given population.
        """
        n_vars = x.shape[1]
        num_groups = self.get_num_groups(x)
        kv_cache = num_groups / (n_vars)
        return kv_cache

    def wse(self, x:np.ndarray, w:np.ndarray, group_idx:np.ndarray) -> np.ndarray:
        """
        Computes weight sharing error for the given population of candidates and weights.
        Args:
            x (np.ndarray): Population of candidates dim (pop_sz, num_heads)
            w (np.ndarray): Weights dim (num_heads, head_dim, model_dim)
            x_mask (np.ndarray): Mask for the given population of candidates indicating which head index along Head dimension belongs to the group.
        """
        x_mask = (x == group_idx)
        w_grouped = (x[..., None, None] * w[None]) * x_mask[..., None, None]
        norm_fac = x_mask.sum(axis=-1, keepdims=True)[..., None, None]
        norm_fac[norm_fac == 0] = 1
        w_mean = w_grouped.sum(axis=1, keepdims=True) / norm_fac
        wse = ((w_mean - w_grouped)**2).mean(axis=(1, 2, 3))
        return wse

    def fitness_wse(self, x:np.ndarray, instance:int) -> np.ndarray:
        """
        Computes weight sharing error for the given population of candidates.
        Args:
            x (np.ndarray): Population of candidates dim (pop_sz, 1, num_heads)
            instance (int): Layer index
        Returns:
            wse (np.ndarray): Weight sharing error for the given population.
        """
        k_weights, v_weights = self.kv_weights[0, instance, ...], self.kv_weights[1, instance, ...]
        wse = np.zeros(x.shape[0])
        for i in range(self.num_groups):
            wse += self.wse(x, k_weights, i) + self.wse(x, v_weights, i)
        return wse

    def fitness_accum_kv_cache(self, x:np.ndarray, group_info:np.ndarray) -> np.ndarray:
        """
        Computes accumulated KV-cache reduction ratio for the given population of candidates.
        Args:
            x (np.ndarray): Population of candidates dim (pop_sz, num_layers)
            group_info (np.ndarray): Group information for the given population of candidates dim (num_layers, num_heads)
        Returns:
            accum_kv_cache (np.ndarray): Accumulated KV-cache reduction ratio for the given population.
        """
        kv_cache = np.zeros([1, self.num_layers])
        for i in range(self.num_layers):
            kv_cache[:, i] = self.fitness_kv_cache(group_info[i:i+1, :], i)
        accum_kv_cache = ((kv_cache * x).sum(axis=-1) + (1-x).sum(axis=-1)) / self.num_layers
        return accum_kv_cache

    def fitness_accum_wse(self, x:np.ndarray, group_info:np.ndarray) -> np.ndarray:
        """
        Computes accumulated weight sharing error for the given population of candidates.
        Args:
            x (np.ndarray): Population of candidates dim (pop_sz, num_layers)
            group_info (np.ndarray): Group information for the given population of candidates dim (num_layers, num_heads)
        Returns:
            wse (np.ndarray): Accumulated weight sharing error for the given population.
        """
        wse = np.zeros([1, self.num_layers])
        for i in range(self.num_layers):
            wse[:, i] = self.fitness_wse(group_info[i:i+1, :], i)
        accum_wse = (wse * x).sum(axis=-1)
        return accum_wse

    def run(self, algorithm:str=Constants.ALGO1, n_gen:int=10, seed:int=1234) -> Any:
        """ 
        Runs multi-objective optimization problem to obtain pareto-optimal results.
        Args:
            n_gen (int): Number of generations
        Returns:
            A list of Pymoo result objects
        """
        if algorithm == Constants.ALGO1:
            results = []
            for i in range(self.num_layers):
                res = minimize(self.problems_algo1[i], self.algorithm_1, termination=("n_gen", n_gen), seed=seed, verbose=True, output=CustomOutput())
                results.append(res)
        elif algorithm == Constants.ALGO2:
            res = minimize(self.problems_algo2, self.algorithm_2, termination=("n_gen", n_gen), seed=seed, verbose=True, output=CustomOutput())
            results = [res]
        return results
