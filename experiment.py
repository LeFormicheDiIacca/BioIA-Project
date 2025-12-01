
from deap import gp, base, creator, tools, algorithms
import operator
import math
import random
from ACO.ACO_simulator import ACO_simulator
from sanity_check import weight_func, heuristic
from meshgraph import MeshGraph
import numpy as np
import functools


# FITNESS AND INDIVIDUAL SET UP

creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness = creator.FitnessMin)

# PRIMITIVE SET CREATION

# as common practice suggests, protectedDiv is introduced to avoid dividing a number by 0
def protectedDiv(num, den):
    if den == 0:
        return 1.0
    else:
        return num/den

# NB: evaluate whether to include "water" and "building" as arguments    
pset = gp.PrimitiveSet("main_elements", 4)
pset.renameArguments(ARG0 = "dist", ARG1 = "elev_diff", ARG2 = "elev_u", ARG3 = "elev_v")
pset.addPrimitive(operator.add, 2, "add")
pset.addPrimitive(operator.sub, 2, "sub")
pset.addPrimitive(operator.mul, 2, "mul")
pset.addPrimitive(protectedDiv, 2, "div")
pset.addPrimitive(operator.neg, 1, "neg")
pset.addPrimitive(math.log, 1, name ="log")
pset.addPrimitive(operator.pow, 2, "power")



#adding use of coefficients between -5 and 5, can change according to performance
pset.addEphemeralConstant("coefficients", functools.partial(random.uniform,-5.0, 5.0))

# BOOTSTRAPPING INITIALIZATION

toolbox = base.Toolbox()

# function to generate trees using the primitive set I set before
# choosing genHalfAndHalf (ramped half-and-half method) for balance between exploration (genFull) and exploitation (genGrow)
# potentially max_depth can be changed, I chose a small number to be cautious

toolbox.register("gen", gp.genHalfAndHalf, pset = pset, min_ = 1, max_ = 5)

#tools.initIterate invokes the "gen" function which createsn a tree and then stores the tree in the individual container
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.gen)

# repeats the "individual" function for N times and stores the trees in a list
toolbox.register("population", tools.initRepeat, container = list, func = toolbox.individual)

#names must be fixed or it doesn't work
#crossover #swaps two individuals in correspondence of random nodes (swaps from node onward)
toolbox.register("mate", gp.cxOnePoint)
#mutation #from a random node within tree the tree-generation function is called and mutated tree can even end up exceeding max depth
toolbox.register("mutate", gp.mutUniform, expr = toolbox.gen, pset = pset)
#selection #performs tournaments N tournaments where random 3 trees are sampled and best one survives
toolbox.register("select", tools.selTournament, tournsize = 3)

#it turns deap.function into a python function
toolbox.register("compile", gp.compile, pset = pset)

G = MeshGraph()

ant_colony_parameters = {"alpha": 1, "beta": 2, "rho": 0.1, "ant_number": 5, "max_iterations": 10, "max_no_updates": 50, "n_best_ants": 5, "average_cycle_length": 3600}




class WeightFunction:
    def __init__(self, cost_func, graph, heuristic_func):
        self.cost_func = cost_func
        self.graph = graph
        self.heuristic_func = heuristic_func
    
    def __call__(self, u, v):
        if self.graph.nodes[u].get('is_water') or self.graph.nodes[v].get('is_water'):
            return float('inf')
        
        dist = self.heuristic_func(u, v)
        elev_v = self.graph.nodes[v]["elevation"]
        elev_u = self.graph.nodes[u]["elevation"]
        elev_diff = abs(elev_u - elev_v)
        cost = self.cost_func(dist, elev_diff, elev_u, elev_v)
        return max(cost, 1.0)

def evaluate_aco_fitness(individual, graph, aco_params):
    og_weight_func = getattr(graph, 'weight_func', None)
    try:
        cost_func = toolbox.compile(expr=individual)
    except Exception:
        return (float('inf'),)  # Return tuple with inf instead of arbitrary number
    
    # Create picklable weight function
    graph.weight_func = WeightFunction(cost_func, graph, heuristic)
    
    path_cost = float("inf")
    try:
        aco = ACO_simulator(graph, **aco_params)
        best_paths = aco.simulation(retrieve_n_best_paths=1)
        if best_paths:
            path_cost = best_paths[0][1]
    except Exception as e:
        print(f"ACO simulation failed: {e}")
        pass
    finally:
        if og_weight_func is not None:
            graph.weight_func = og_weight_func
        else:
            delattr(graph, 'weight_func')
    
    return (path_cost,)


toolbox.register("evaluate", functools.partial(evaluate_aco_fitness, graph = G, aco_params = ant_colony_parameters))

def main():
    random.seed(31)
    try:
        pop = toolbox.population(n = 200)
    except:
        return 10101010
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    pop,log = algorithms.eaSimple(pop, toolbox, cxpb =0.8, mutpb = 0.05, ngen = 20, stats = stats, halloffame=hof, verbose = True) 
    return pop, log, hof

if __name__ == "__main__":
    main()

    


