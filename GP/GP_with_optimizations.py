import os
import sys
from time import sleep
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from deap import base, creator, gp, tools, algorithms
import operator
import random
import multiprocessing
from TerrainGraph.terraingraph import create_graph
from scenario import generate_scenarios
from edge_info import create_edge_dict
import time
from gp_logistics import protected_div, protected_log, protected_pow, tree_plotter, if_then_else, random_gen, save_run, compute_chebyshev, BASE
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from collections import defaultdict
from numba import njit
from deap import gp
import math

# shall we keep it?
PENALTY_MISSING_VALUES = 1e8

# new equation takes the edge parameters: distance and steepness (floats) and is_water (boolean)

pset = gp.PrimitiveSetTyped("MAIN", [float, float, bool], float) # output is also float
pset.renameArguments(ARG0="distance", ARG1="steepness", ARG2="is_water")
pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(protected_pow, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(protected_log, [float,float], float)
pset.addPrimitive(protected_div, [float,float], float)

# if then else
# input is boolean condition and outputs are floats

pset.addPrimitive(if_then_else, [bool, float, float], float)

# comparison operators: <, >, <=, >=, and, or

pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.le, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.ge, [float, float], bool)
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)

# add constant 
# # for now value is set between 0 and math.e
pset.addEphemeralConstant("constant", random_gen, ret_type=float)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

# define main functions

toolbox = base.Toolbox()

# to create only trees that have all the required inputs

def create_valid_individual():
    while True:
        expr = gp.genHalfAndHalf(pset=pset, min_=2, max_=5)
        ind = creator.Individual(expr)
        tree_str = str(ind)
        required_inputs = ["distance", "steepness", "is_water"]
        missing = any(inp not in tree_str for inp in required_inputs)
        if not missing:
            return ind


toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, create_valid_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# genetic operators

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mutate_unif", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.register("mutate_eph", gp.mutEphemeral, mode="all")


# to include both type of mutation # to be finetuned

def mutate_combined(individual):
    if random.random() < 0.7:
        return toolbox.mutate_unif(individual)
    else:
        return toolbox.mutate_eph(individual)


toolbox.register("mutate", mutate_combined)

# limit bloating

toolbox.decorate("mate", gp.staticLimit(operator.attrgetter("height"), max_value=5))
toolbox.decorate("mate", gp.staticLimit(len, max_value=15))
toolbox.decorate("mutate_unif", gp.staticLimit(operator.attrgetter("height"), max_value=5))
toolbox.decorate("mutate_unif", gp.staticLimit(len, max_value=15))
toolbox.decorate("mutate_eph", gp.staticLimit(operator.attrgetter("height"), max_value=5))
toolbox.decorate("mutate_eph", gp.staticLimit(len, max_value=15))

# variables for multiprocessing to avoid data duplication

_GLOBAL_EDGE_LOOKUP = None
_GLOBAL_EDGE_DATA = None
_GLOBAL_NODE_LIST = None
_GLOBAL_PSET = None

# Initialize Workers with shared data without need to copy variables

def init_worker(edge_lookup, edge_data, node_list, pset):
    global _GLOBAL_EDGE_LOOKUP, _GLOBAL_EDGE_DATA, _GLOBAL_NODE_LIST, _GLOBAL_PSET
    _GLOBAL_EDGE_LOOKUP = edge_lookup
    _GLOBAL_EDGE_DATA = edge_data
    _GLOBAL_NODE_LIST = node_list
    _GLOBAL_PSET = pset

# Turn graph into csr matrix for better performance

def create_edge_index_matrix(graph, node_to_idx):
    rows = []
    cols = []
    data = []
    for i, (u, v) in enumerate(graph.edges()):
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        rows.append(u_idx)
        cols.append(v_idx)
        data.append(i + 1)  

        rows.append(v_idx)
        cols.append(u_idx)
        data.append(i + 1)

    return csr_matrix((data, (rows, cols)), shape=(len(node_to_idx), len(node_to_idx)))

#Penalty function with numba for better calculations with JIT e vectorial operations
@njit
def compute_total_penalty_numba(predecessors, end_nodes, start_node_idx,
                                csr_indices, csr_indptr, csr_data, edge_data, water_count, res, chebyshev):
    total_penalty = 0.0

    for end_idx in end_nodes:
        curr = end_idx
        if predecessors[curr] == -9999 and curr != start_node_idx:
            total_penalty += 1_000_000  # Unreachable
            continue
        
        path_distance = 0.0
        path_steepness = 0.0
        path_water = 0
        tot_nodes = 0
        while curr != start_node_idx:
            prev = predecessors[curr]
            if prev == -9999: break

            # Matrix search in time (O(log degree))
            edge_idx = -1
            for i in range(csr_indptr[curr], csr_indptr[curr + 1]):
                if csr_indices[i] == prev:
                    edge_idx = csr_data[i] - 1
                    break

            if edge_idx == -1:
                curr = prev
                continue

            #Penalty calculations
            d = edge_data[edge_idx, 0]
            steepness = edge_data[edge_idx, 1]
            water = edge_data[edge_idx, 2]

            tot_nodes +=1

            path_distance += d
            path_steepness += steepness
            if water == True:
                path_water += 1

            curr = prev

    total_penalty = path_distance/tot_nodes * ((path_water/tot_nodes)*(BASE-1) + 1 + BASE**(path_steepness/tot_nodes)) 
    
    return total_penalty

# Turns graph data into simple arrays for better performance and use with numba&numpy

def precompute_edge_lookup_simple(graph, edge_dict, node_to_idx):
    edge_lookup_list = []
    edge_data_list = []

    for u, v in graph.edges():
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        u_ord, v_ord = (u, v) if u < v else (v, u)
        u_ord_idx, v_ord_idx = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)

        edge_key = f"{u_ord}-{v_ord}"
        d, steepness, water = edge_dict[edge_key]

        edge_lookup_list.append([u_ord_idx, v_ord_idx, len(edge_data_list)])
        edge_data_list.append([d, steepness, water])

    edge_lookup_arr = np.array(edge_lookup_list, dtype=np.int64)
    edge_data = np.array(edge_data_list, dtype=np.float32)

    return edge_lookup_arr, edge_data



def evaluate_fully_optimized(individual, scenarios, node_to_idx, edge_features_columns, csr_template, csr_components, water_nodes,res, chebyshev):
    global _GLOBAL_EDGE_DATA, _GLOBAL_PSET

    #Vectorial cost computation
    func = gp.compile(expr=individual, pset=_GLOBAL_PSET)
    try:
        costs = func(*edge_features_columns)

        if isinstance(costs, (int, float)):
            costs = np.full(len(edge_features_columns[0]), costs)

        costs = np.maximum(costs, 0.001)
    except:
        return (1e12,)

    #CSR matrix update
    csr_template.data = np.concatenate([costs, costs])

    #Dijkstra Batch (faster than loop)
    grouped = defaultdict(list)
    for s, e in scenarios: grouped[node_to_idx[s]].append(node_to_idx[e])

    sources = list(grouped.keys())
    dists, preds = dijkstra(csr_template, directed=False, indices=sources, return_predecessors=True)

    if len(sources) == 1:
        dists = dists.reshape(1, -1)
        preds = preds.reshape(1, -1)

    #Cost with numba
    total_penalty = 0.0
    csr_indices, csr_indptr, csr_data = csr_components

    for i, start_idx in enumerate(sources):
        total_penalty += compute_total_penalty_numba(
            preds[i], np.array(grouped[start_idx], dtype=np.int64), start_idx,
            csr_indices, csr_indptr, csr_data, _GLOBAL_EDGE_DATA, water_nodes, res, chebyshev
        )

    #Penalties for missing values
    tree_str = str(individual)
    for inp in ["distance", "steepness", "is_water"]:
        if inp not in tree_str: total_penalty += PENALTY_MISSING_VALUES

    final_fitness = total_penalty / len(scenarios)

    #NB: see if we can normalize wrt resolution (es. Chebyshev distance)

    return (final_fitness,)

# algorithm-running function
def run_EA(graph, scenarios, edge_dict, population, generations, base_folder, chebyshev):
    pop = toolbox.population(n=population)
    # for info about fitness of the evolved trees
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_fit.register("avg", np.mean)
    stats_fit.register("std", np.std)
    stats_fit.register("min", np.min)
    stats_fit.register("max", np.max)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    hof = tools.HallOfFame(5, similar=operator.eq)
    # vs overfitting: we update the scenarios every 10 generations
    node_list = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edge_features = []
    for u, v in graph.edges():
        u_ordered, v_ordered = min(u, v), max(u, v)
        edge_key = f"{u_ordered}-{v_ordered}"
        features = edge_dict[edge_key]
        edge_features.append(features)

    row_idx = [node_to_idx[u] for u, v in graph.edges()]
    col_idx = [node_to_idx[v] for u, v in graph.edges()]
    edge_lookup, edge_data = precompute_edge_lookup_simple(graph, edge_dict, node_to_idx)
    edge_features = np.array(edge_features)
    edge_index_matrix = create_edge_index_matrix(graph, node_to_idx)
    csr_components = (
        edge_index_matrix.indices,
        edge_index_matrix.indptr,
        edge_index_matrix.data
    )
    edge_features_columns = [np.array(c, dtype=np.float32) for c in zip(*edge_features)]
    water_count = sum(1 for features in edge_dict.values() if features[2] > 0)

    # creates fixed CSR template
    n_nodes = len(node_to_idx)
    row_idx_ext = np.concatenate([row_idx, col_idx])
    col_idx_ext = np.concatenate([col_idx, row_idx])
    dummy_data = np.zeros(len(row_idx_ext))
    csr_template = csr_matrix((dummy_data, (row_idx_ext, col_idx_ext)), shape=(n_nodes, n_nodes))

    pool = multiprocessing.Pool(
        processes=multiprocessing.cpu_count() - 1,
        initializer=init_worker,
        initargs=(edge_lookup, edge_data, node_list, pset)
    )
    toolbox.register("map", pool.map)
    start = time.time()
    toolbox.register("evaluate", evaluate_fully_optimized,
                        scenarios=scenarios,
                        node_to_idx=node_to_idx,
                        edge_features_columns=edge_features_columns, 
                        csr_template=csr_template,  
                        csr_components=csr_components, res = res,
                        water_nodes = water_count, chebyshev = chebyshev)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                       ngen=generations, stats=mstats, halloffame=hof, verbose=False)
    
    flattened_log = []
    gens = log.select("gen")
    nevals = log.select("nevals")
    fit_avg = log.chapters["fitness"].select("avg")
    fit_max = log.chapters["fitness"].select("max")
    fit_std = log.chapters["fitness"].select("std")
    fit_min = log.chapters["fitness"].select("min")
    
    # size_avg = log.chapters["size"].select("avg")

    # Reconstruct the list of dictionaries

    for i_gen in range(len(gens)):
        entry = {
            'gen': str(gens[i_gen]),
            'nevals': nevals[i_gen],
            'fit_avg': fit_avg[i_gen],
            'fit_max': fit_max[i_gen],
            'fit_min': fit_min[i_gen],
            'fit_std': fit_std[i_gen]
        }
        flattened_log.append(entry)

    print(log)
    end = time.time()
    diff = end - start
    hours, tmp = divmod(diff, 3600)
    minutes, seconds = divmod(tmp, 60)
    print(f"{generations} generations evolved in {hours} hours {minutes} minutes {seconds} seconds")
    save_run(population, hof, diff, len(scenarios), generations, res, pset=pset,path=base_folder)
    print(f"Generations log have been saved")


# main function to run, executes the code and saves logs

def main(population, scenarios_number, generations, graph, edge_dict, res, base_folder):
    scenarios = generate_scenarios(scenarios_number, graph)
    print(
        f"Evolving the cost function through {generations} generations with a population of {population}.")
    start = time.time()
    chebyshev = compute_chebyshev(res)
    run_EA(graph, scenarios, edge_dict, population, generations, base_folder=base_folder, chebyshev= chebyshev)
    print("The best individual has been saved")


if __name__ == "__main__":
    experiments = [
        # population, scenarios, generations
        [500,50,200],
        [500,50, 500]]
    res = 200
    today = datetime.now().strftime("%d_%m_%Y")
    runs_today_folder = f"GP/res/runs_{today}"
    if not os.path.exists(runs_today_folder):
        os.makedirs(runs_today_folder)
    trentino_graph = create_graph("TerrainGraph/trentino.tif",
                                  "TerrainGraph/trentino_alto_adige.pbf",
                                  resolution=res)
    edge_dict = create_edge_dict(trentino_graph)

    water_count = sum(1 for features in edge_dict.values() if features[2] > 0)
    if water_count == 0:
        print("No Water Node. Can't continue.")
        exit(-1)

    for experiment in experiments:
        population = experiment[0]
        scen_number = experiment[1]
        gens = experiment[2]

        base_folder = f"{runs_today_folder}/{population}pop_{gens}gen_{scen_number}scenarios_{res}res"
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        else:
            i=0
            while os.path.exists(base_folder):
                i += 1
                base_folder = f"{base_folder}_{i}"
            os.makedirs(base_folder)

        main(population=population, scenarios_number=scen_number, generations=gens, graph=trentino_graph, edge_dict=edge_dict, res=res, base_folder = base_folder)
        print("Small pause for CPU cooling")
        sleep(1*60)
