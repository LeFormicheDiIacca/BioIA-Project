
from deap import gp, base, creator, tools
import operator
import math
import random
from ACO.ACO_simulator import ACO_simulator
from sanity_check import weight_func, heuristic


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

    
pset = gp.PrimitiveSet("main_elements", 4)
pset.renameArguments(ARG0 = "dist", ARG1 = "elev_diff", ARG2 = "elev_u", ARG3 = "elev_v")
pset.addPrimitive(operator.add, 2, "add")
pset.addPrimitive(operator.sub, 2, "sub")
pset.addPrimitive(operator.mul, 2, "mul")
pset.addPrimitive(protectedDiv, 2, "div")
pset.addPrimitive(operator.neg, 1, "neg")
pset.addPrimitive(math.log, 1, name ="log")

#adding use of coefficients between -5 and 5, can change according to performance
pset.addEphemeralConstant("coefficients", lambda:random.uniform(-5.0, 5.0))

# BOOTSTRAPPING INITIALIZATION

toolbox = base.Toolbox

# function to generate trees using the primitive set I set before
# choosing genHalfAndHalf (ramped half-and-half method) for balance between exploration (genFull) and exploitation (genGrow)
# potentially max_depth can be changed, I chose a small number to be cautious

toolbox.register("gen", gp.genHalfAndHalf, pset = pset, min_ = 1, max_ = 5)

#tools.initIterate invokes the "gen" function which creates a tree and then stores the tree in the individual container
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.gen)

# repeats the "individual" function for N times and stores the trees in a list
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# NB. NOTES SAY THAT CROSSOVER IS SET AS HIGHLY PROBABLE (P>0.8) AND MUTATION AS QUITE RARE (P < 0.05)

#crossover #swaps two individuals in correspondence of random nodes (swaps from node onward)
toolbox.register("crossover", gp.cxOnePoint)
#mutation #from a random node within tree the tree-generation function is called and mutated tree can even end up exceeding max depth
toolbox.register("mutation", gp.mutUniform, expr = toolbox.gen, pset = pset)
#selection #performs tournaments N tournaments where random 3 trees are sampled and best one survives
toolbox.register("selection", tools.selTournament, tournsize = 3)

#it turns deap.function into a python function
toolbox.register("compile", gp.compile, pset = pset)

def evaluate_aco_fitness(individual, graph, aco_params):
    try:
        cost_func = toolbox.compile(expr = individual)
    except:
        return float("inf") #worst fitness possible
    def updated_weight_func(graph, u,v):
        if graph.nodes[u].get('is_water') or graph.nodes[v].get('is_water'):
            return float('inf') # Impossible to cross water
        dist = heuristic(u,v)
        elev_v = graph.nodes[v]["elevation"]
        elev_u = graph.nodes[u]["elevation"]
        elev_diff = abs(elev_u - elev_v)
        # cost value computed with values of arch and specific individual tree
        cost = cost_func(dist, elev_diff,elev_u, elev_v)
        return max(cost, 1.0) #cannot return negative value, ACO could fail
    
    graph.weight_func = updated_weight_func
    


