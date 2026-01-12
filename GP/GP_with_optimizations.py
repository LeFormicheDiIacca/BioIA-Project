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
from gp_logistics import protected_div, protected_log, protected_pow, tree_plotter, identity_water, if_then_else, \
    append_to_json, random_gen, save_run

# global variables

STEEPNESS_COEFFICIENT = 5.0
STEEPNESS_PENALTY = 50000.0
ELEVATION_COEFFICIENT = 10.0
WATER_COEFFICIENT = 5000.0
PENALTY_MISSING_VALUES = 1e8

# define primitive set

class WaterArg: pass


class OtherArgs: pass


# strongly typed # chosen to limit if_then function

pset = gp.PrimitiveSetTyped("MAIN", [OtherArgs, OtherArgs, OtherArgs, OtherArgs, WaterArg], OtherArgs)
pset.renameArguments(ARG0="distance", ARG1="steepness", ARG2="elevation_u", ARG3="elevation_v", ARG4="is_water")
pset.addPrimitive(operator.add, [OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(operator.mul, [OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(protected_pow, [OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(operator.sub, [OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(operator.neg, [OtherArgs], OtherArgs)
pset.addPrimitive(protected_log, [OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(protected_div, [OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(if_then_else, [WaterArg, OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(identity_water, [WaterArg], WaterArg)
pset.addEphemeralConstant("constant", random_gen, ret_type=OtherArgs)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

# define main functions

toolbox = base.Toolbox()


def create_valid_individual():
    while True:
        expr = gp.genHalfAndHalf(pset=pset, min_=2, max_=5)
        ind = creator.Individual(expr)
        tree_str = str(ind)
        required_inputs = ["distance", "steepness", "elevation_u", "elevation_v", "is_water"]
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


# to include both type of mutation

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


# fitness function
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from collections import defaultdict
from numba import njit
from deap import gp

_GLOBAL_EDGE_LOOKUP = None
_GLOBAL_EDGE_DATA = None
_GLOBAL_NODE_LIST = None
_GLOBAL_PSET = None

def init_worker(edge_lookup, edge_data, node_list, pset):
    """Initialize Workers with shared data without need to copy variables"""
    global _GLOBAL_EDGE_LOOKUP, _GLOBAL_EDGE_DATA, _GLOBAL_NODE_LIST, _GLOBAL_PSET
    _GLOBAL_EDGE_LOOKUP = edge_lookup
    _GLOBAL_EDGE_DATA = edge_data
    _GLOBAL_NODE_LIST = node_list
    _GLOBAL_PSET = pset


def create_edge_index_matrix(graph, node_to_idx):
    """Turn graph into csr matrix for better performance"""
    rows = []
    cols = []
    data = []
    for i, (u, v) in enumerate(graph.edges()):
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        # Memorizziamo l'indice i (0-based) dell'arco
        rows.append(u_idx)
        cols.append(v_idx)
        data.append(i + 1)  # +1 per distinguere dallo zero della matrice sparsa

        rows.append(v_idx)
        cols.append(u_idx)
        data.append(i + 1)

    return csr_matrix((data, (rows, cols)), shape=(len(node_to_idx), len(node_to_idx)))

#Penalty function with numba for better calculations with JIT e vectorial operations
@njit
def compute_total_penalty_numba(predecessors, end_nodes, start_node_idx,
                                csr_indices, csr_indptr, csr_data, edge_data, water_count, res):
    total_penalty = 0.0

    for end_idx in end_nodes:
        curr = end_idx
        if predecessors[curr] == -9999 and curr != start_node_idx:
            total_penalty += 1_000_000  # Unreachable
            continue

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
            incl = edge_data[edge_idx, 1]
            e_u = edge_data[edge_idx, 2]
            e_v = edge_data[edge_idx, 3]
            water = edge_data[edge_idx, 4]

            # Dynamic penalty

            dp = STEEPNESS_COEFFICIENT*incl + STEEPNESS_PENALTY * max(0, incl-1/3)

            elev_diff = max(e_v - e_u, 0.0)

            penalty = np.array([d, dp * d, ELEVATION_COEFFICIENT * elev_diff, WATER_COEFFICIENT * water])

            # Normalize total penalty w.r.t. to the Manhattan distance, aka the average length of the shortest path between two random
            # points in a resolution^2 grid
            
            manhattan_d = 2*(res+1)/3


            # In the average path, the total penalty will be equal to:
            # total_penalty = manhattan_d * d + manhattan_d * dp * d + 10.0 * manhattan_d/2 * elev_diff + water_count/res * 5000.0 * water
            # we assume that, in the average path, there will be on each edge at least one-unit dp increase, as well as a one-unit
            # increase in elevation difference in at least res/2 edges (in the remaining ones, it's assumed to be a decrease, which does 
            # not impact total penalty)
            # as the cost for crossing water is vary high, as some sort of likelihood to encounter water nodes, we consider the number of 
            # nodes with water that are present in the grid over the resolution

            manhattan_matrix = np.array([manhattan_d, manhattan_d, manhattan_d/2, water_count/res])
            normalized_penalty = penalty/manhattan_matrix
            normalized_penalty = np.sum(normalized_penalty)
            total_penalty += normalized_penalty
            curr = prev

    return total_penalty



def precompute_edge_lookup_simple(graph, edge_dict, node_to_idx):
    """
    Turns graph data into simple arrays for better performance and use with numba&numpy
    """
    edge_lookup_list = []
    edge_data_list = []

    for u, v in graph.edges():
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        u_ord, v_ord = (u, v) if u < v else (v, u)
        u_ord_idx, v_ord_idx = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)

        edge_key = f"{u_ord}-{v_ord}"
        d, incl, e_u, e_v, water = edge_dict[edge_key]

        edge_lookup_list.append([u_ord_idx, v_ord_idx, len(edge_data_list)])
        edge_data_list.append([d, incl, e_u, e_v, water])

    edge_lookup_arr = np.array(edge_lookup_list, dtype=np.int64)
    edge_data = np.array(edge_data_list, dtype=np.float32)

    return edge_lookup_arr, edge_data



def evaluate_fully_optimized(individual, scenarios, node_to_idx, edge_features_columns, csr_template, csr_components, water_nodes,res):
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
            csr_indices, csr_indptr, csr_data, _GLOBAL_EDGE_DATA, water_nodes, res
        )

    #Penalties for missing values
    tree_str = str(individual)
    for inp in ["distance", "steepness", "elevation_u", "elevation_v", "is_water"]:
        if inp not in tree_str: total_penalty += PENALTY_MISSING_VALUES

    return (total_penalty,)

# algorithm-running function
def run_EA(graph, scenarios, edge_dict, population, runs, scenario_dur):
    all_logs = []
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
    water_count = sum(1 for features in edge_dict.values() if features[4] > 0)

    # Crea un template CSR fisso
    n_nodes = len(node_to_idx)
    row_idx_ext = np.concatenate([row_idx, col_idx])
    col_idx_ext = np.concatenate([col_idx, row_idx])
    dummy_data = np.zeros(len(row_idx_ext))
    csr_template = csr_matrix((dummy_data, (row_idx_ext, col_idx_ext)), shape=(n_nodes, n_nodes))

    # Passa questi al toolbox

    pool = multiprocessing.Pool(
        processes=multiprocessing.cpu_count() - 1,
        initializer=init_worker,
        initargs=(edge_lookup, edge_data, node_list, pset)
    )
    toolbox.register("map", pool.map)
    for i in range(runs):
        print(f"Starting run {i + 1}")
        start = time.time()
        current_scenario = [el[i] for el in scenarios]
        toolbox.register("evaluate", evaluate_fully_optimized,
                         scenarios=current_scenario,
                         node_to_idx=node_to_idx,
                         edge_features_columns=edge_features_columns,  # Array di colonne
                         csr_template=csr_template,  # Matrice pre-allocata
                         csr_components=csr_components, res = res,
                         water_nodes = water_count)
        for ind in pop:
            del ind.fitness.values
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                       ngen=scenario_dur, stats=mstats, halloffame=hof, verbose=False)
        # archives runtime info
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
                'gen': str(i + 1) + "." + str(gens[i_gen]),
                'nevals': nevals[i_gen],
                'fit_avg': fit_avg[i_gen],
                'fit_max': fit_max[i_gen],
                'fit_min': fit_min[i_gen],
                'fit_std': fit_std[i_gen]
                # 'size_avg': size_avg[i_gen]
            }
            flattened_log.append(entry)

        all_logs.append(flattened_log)
        print(log)
        end = time.time()
        diff = end - start
        hours, tmp = divmod(diff, 3600)
        minutes, seconds = divmod(tmp, 60)

        print(f"{i + 1}° run completed in {hours} hours {minutes} minutes {seconds} seconds")
        timestamp = datetime.now().strftime("%d%m%Y%H%M")
        save_run(population, hof, diff, i+1, scenario_dur, res, pset=pset,path=f"GP/res/run_{timestamp}/GP_tree_{population}pop_{scenario_dur}gen_{runs}runs_{i+1}subrun.json")
        print(f"{i + 1}° run saved")

    return pop, hof, all_logs


# main function to run, executes the code and saves logs

def main(population, runs, graph, edge_dict, res,scenario_dur=15):
    scenarios = generate_scenarios(runs, graph, res)
    print(
        f"Evolving the cost function through {runs} runs of {scenario_dur} generations with a population of {population}.")
    start = time.time()
    ret = run_EA(graph, scenarios, edge_dict, population, runs, scenario_dur)
    end = time.time()
    diff = end - start
    hours, tmp = divmod(diff, 3600)
    minutes, seconds = divmod(tmp, 60)
    print(f"EA runtime: {hours} hours {minutes} minutes {seconds} seconds")
    pop = ret[0]
    logs = ret[2]
    hof = ret[1]
    if population >= 500:
        for i in range(len(hof)):
            try:
                tree_plotter(hof[i], f"pop{population}_run{runs}_res{res}_{i + 1}best_tree", pset=pset)
            except Exception as e:
                print(f"Could not plot tree: {e}")
    best = hof[0]
    hof_list = []
    for ind in hof:
        ind_diz = dict()
        ind_fit = ind.fitness.values[0]
        ind_str = str(ind)
        ind_diz[ind_str] = ind_fit
        hof_list.append(ind_diz)
    tree_diz = dict()
    tree_diz["hall_of_fame"] = hof_list
    tree_diz["best_tree_object"] = str(best)
    tree_diz["best_tree_fitness"] = best.fitness.values
    tree_diz["population"] = population
    tree_diz["resolution"] = res
    tree_diz["runs"] = runs
    tree_diz["scenario_duration"] = scenario_dur
    tree_diz["runtime"] = diff
    tree_diz["logs"] = logs
    timestamp = datetime.now().strftime("%d%m%Y%H%M")
    append_to_json(tree_diz, f"GP/res/run_{timestamp}/GP_tree_{population}pop_{scenario_dur}gen_{runs}runs.json")
    print("The best individual has been saved")


if __name__ == "__main__":
    population = 3000
    runs= 15
    generations = 20
    res = 200
    trentino_graph = create_graph("TerrainGraph/trentino.tif",
                                  "TerrainGraph/trentino_alto_adige.pbf",
                                  resolution=res)
    edge_dict = create_edge_dict(trentino_graph)

    water_count = sum(1 for features in edge_dict.values() if features[4] > 0)
    if water_count == 0:
        print("No Water Node. Can't continue.")
        exit(-1)


    main(population=population, runs=runs, graph=trentino_graph, edge_dict=edge_dict, res=res, scenario_dur=generations)
