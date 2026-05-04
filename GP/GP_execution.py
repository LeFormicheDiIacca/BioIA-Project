import csv
import json
import math
import multiprocessing
import operator
import os
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from time import sleep
import networkit as nk

from gp_logistics import protected_div, protected_log, protected_pow, if_then_else, random_gen, save_run, \
    step_penalty_adder, step_penalty_multiplier
import numpy as np
from deap import base, creator, gp, tools, algorithms
from numba import njit
from scipy.sparse import csr_matrix

#Ignore Scipy/Numpy math errors
np.seterr(all='ignore')
#Define project root for file management
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#Global variables for Hyperparameters
BASE = math.e
BASE = BASE**15
BASE_FOLDER = ""
STD_THRESHOLD = 0.001
EARLY_STOPPING = 10
MUT_RATE = 0.25
CROSS_RATE = 0.7
HOF_SIZE = 5
TOURNAMENT_SIZE = 3
PARSIMONY_VALUE = 1.3
MAX_HEIGHT = 5
MAX_NODES = 20
MAX_CORE_VALUE = 11
PENALTY_MISSING_VALUES = 1e3
PENALTY_ERROR_IN_CALCULATIONS =  1e9

#pretty logging for each generation
def print_gen_log(gen, nevals, record, num_dead, duration, is_header=False, scenarios=None, seed=None):
    header = f"{'Gen':>4} | {'Nevals':>6} | {'Avg Fit':>12} | {'Std Fit':>12} | {'Min Fit':>12} | {'Max Fit':>12} | {'Dead':>5} | {'Time':>7}"
    csv_path = os.path.join(BASE_FOLDER, "evolution_stats.csv")
    json_path = os.path.join(BASE_FOLDER, "experiment_config.json")

    if is_header:
        # --- LOGICA CSV ---
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        csv_headers = ["gen", "nevals", "avg", "std", "min", "max", "dead", "time"]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

        config_data = {
            "hyperparameters": {
                "BASE": BASE,
                "STD_THRESHOLD": STD_THRESHOLD,
                "EARLY_STOPPING": EARLY_STOPPING,
                "MUT_RATE": MUT_RATE,
                "CROSS_RATE": CROSS_RATE,
                "HOF_SIZE": HOF_SIZE,
                "TOURNAMENT_SIZE": TOURNAMENT_SIZE,
                "PARSIMONY_VALUE": PARSIMONY_VALUE,
                "MAX_HEIGHT": MAX_HEIGHT,
                "MAX_NODES": MAX_NODES,
                "MAX_CORE_VALUE": MAX_CORE_VALUE,
                "PENALTY_MISSING_VALUES": PENALTY_MISSING_VALUES,
                "PENALTY_ERROR_IN_CALCULATIONS": PENALTY_ERROR_IN_CALCULATIONS,
                "seed": seed
            },
            "scenarios_info": {
                "num_scenarios": len(scenarios) if scenarios is not None else 0,
                "scenarios": scenarios.tolist() if isinstance(scenarios, np.ndarray) else scenarios
            }
        }
        with open(json_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        return

    avg_str = f"{record['avg']:>12.2e}" if record['avg'] > 1e6 else f"{record['avg']:>12.2f}"
    min_str = f"{record['min']:>12.2f}"
    max_str = f"{record['max']:>12.2e}"
    std_str = f"{record['std']:>12.2e}" if record['avg'] > 1e6 else f"{record['std']:>12.2f}"

    print(f"{gen:>4} | {nevals:>6} | {avg_str} | {std_str} | {min_str} | {max_str} | {num_dead:>5} | {duration:>6.2f}s")
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow([gen, nevals, record['avg'], record['std'], record['min'], record['max'], num_dead, duration])
# --- DEAP SETUP ---
#DEAP primitives setup
pset = gp.PrimitiveSetTyped("MAIN", [float, float, bool], float)
pset.renameArguments(ARG0="distance", ARG1="steepness", ARG2="is_water")
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protected_pow, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(protected_log, [float, float], float)
pset.addPrimitive(protected_div, [float, float], float)
pset.addPrimitive(if_then_else, [bool, float, float], float)
pset.addPrimitive(np.less, [float, float], bool, name="lt")
pset.addPrimitive(np.less_equal, [float, float], bool, name="le")
pset.addPrimitive(np.greater, [float, float], bool, name="gt")
pset.addPrimitive(np.greater_equal, [float, float], bool, name="ge")
pset.addPrimitive(np.logical_and, [bool, bool], bool, name="and_")
pset.addPrimitive(np.logical_or, [bool, bool], bool, name="or_")
pset.addPrimitive(step_penalty_adder, [float, float, float], float)
pset.addPrimitive(step_penalty_multiplier, [float, float, float], float)

pset.addEphemeralConstant("constant", random_gen, ret_type=float)
#DEAP fitness and individuals
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()

def create_valid_individual():
    while True:
        expr = gp.genHalfAndHalf(pset=pset, min_=2, max_=5)
        ind = creator.Individual(expr)
        if len(ind) <= MAX_NODES:
            tree_str = str(ind)
            required_inputs = ["distance", "steepness", "is_water"]
            if all(inp in tree_str for inp in required_inputs):
                return ind

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, create_valid_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("select", tools.selDoubleTournament, fitness_size=TOURNAMENT_SIZE, parsimony_size=PARSIMONY_VALUE, fitness_first=True)
toolbox.register("mutate_unif", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.register("mutate_eph", gp.mutEphemeral, mode="all")
toolbox.register("mutNode", gp.mutNodeReplacement, pset=pset)
toolbox.register("mutShrink", gp.mutShrink)


def mutate_combined(individual):
    r = random.random()
    if r < 0.25:
        return toolbox.mutate_unif(individual)
    elif r < 0.5:
        return toolbox.mutNode(individual)
    elif r < 0.75:
        return toolbox.mutShrink(individual)
    else:
        return toolbox.mutate_eph(individual)

toolbox.register("mutate", mutate_combined)

toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), max_value=MAX_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(len, max_value=MAX_NODES))
toolbox.decorate("mate", gp.staticLimit(operator.attrgetter("height"), max_value=MAX_HEIGHT))
toolbox.decorate("mate", gp.staticLimit(len, max_value=MAX_NODES))
toolbox.decorate("mutate_unif", gp.staticLimit(operator.attrgetter("height"), max_value=MAX_HEIGHT))
toolbox.decorate("mutate_unif", gp.staticLimit(len, max_value=MAX_NODES))
toolbox.decorate("mutate_eph", gp.staticLimit(operator.attrgetter("height"), max_value=MAX_HEIGHT))
toolbox.decorate("mutate_eph", gp.staticLimit(len, max_value=MAX_NODES))

def similar_by_structure(ind1, ind2):
    str1 = re.sub(r'[-+]?\d*\.\d+|\d+', 'X', str(ind1))
    str2 = re.sub(r'[-+]?\d*\.\d+|\d+', 'X', str(ind2))
    str1 = re.sub(r'distance|steepness|is_water', 'V', str1)
    str2 = re.sub(r'distance|steepness|is_water', 'V', str2)
    return str1 == str2

#Variables for multiprocessing
_GLOBAL_PSET = None
_STATIC_DATA = {}

def init_worker(pset, static_data):
    global _GLOBAL_PSET, _STATIC_DATA
    _GLOBAL_PSET = pset
    _STATIC_DATA = static_data

#fast Numba compute path penalty
@njit(fastmath=True, cache=True)
def compute_penalty_from_path(path_nodes,
                              csr_indices, csr_indptr, csr_data,
                              edge_dist, edge_steep, edge_water):
    n = len(path_nodes)
    if n < 2:
        return PENALTY_ERROR_IN_CALCULATIONS

    path_distance = 0.0
    path_steepness = 0.0
    path_water = 0.0
    tot_nodes = n - 1
    #Calculate path stats
    for k in range(tot_nodes):
        curr = path_nodes[k]
        next_node = path_nodes[k + 1]

        edge_idx = -1
        for i in range(csr_indptr[curr], csr_indptr[curr + 1]):
            if csr_indices[i] == next_node:
                edge_idx = csr_data[i] - 1
                break

        if edge_idx == -1:
            return PENALTY_ERROR_IN_CALCULATIONS

        path_distance += edge_dist[edge_idx]
        path_steepness += edge_steep[edge_idx]
        if edge_water[edge_idx] > 0.5:
            path_water += 1

    path_steepness = path_steepness/tot_nodes
    path_water = path_water/tot_nodes

    path_steepness = ((BASE**path_steepness)-1)/100
    path_water = ((BASE**path_water)-1)/100
    #use fitness formula
    return path_distance * (1.0 + path_water + path_steepness)

#Function for individual evaluation
def evaluate_individual(individual, sources_list, targets_list, num_scenarios):
    global _GLOBAL_PSET, _STATIC_DATA
    #Try to compile the individual and calculate costs
    try:
        func = gp.compile(expr=individual, pset=_GLOBAL_PSET)
        raw_costs = func(*_STATIC_DATA['edge_features'])

        if np.isscalar(raw_costs) or getattr(raw_costs, 'ndim', 0) == 0:
            costs = np.full(len(_STATIC_DATA['edge_features'][0]), float(raw_costs), dtype=np.float64)
        else:
            costs = np.array(raw_costs, dtype=np.float64, copy=True)

        np.nan_to_num(costs, copy=False, nan=1e9, posinf=1e9, neginf=0.001)
        costs = np.logaddexp(0, costs)
        costs = costs + 0.001

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"\n[Worker {os.getpid()}] Crashed during cost calculation!")
        print(f"Individual: {str(individual)}")
        print(f"Error: {e}\n{err_msg}")
        return (PENALTY_MISSING_VALUES,)
    #Update weights
    weights_c = np.ascontiguousarray(costs[_STATIC_DATA['weight_mapping']], dtype=np.float64)
    rows_c = _STATIC_DATA['rows']
    cols_c = _STATIC_DATA['cols']

    nk_graph = nk.Graph(_STATIC_DATA['num_nodes'], weighted=True, directed=False)
    nk_graph.addEdges((weights_c, (rows_c, cols_c)))

    bd = nk.distance.BidirectionalDijkstra(nk_graph, 0, 1)

    total_penalty = 0.0
    #use bidirectional dijkstra to calculate path
    try:
        for source, targets in zip(sources_list, targets_list):
            s = int(source)
            for target in targets:
                t = int(target)

                bd.setSource(s)
                bd.setTarget(t)
                bd.run()

                dist = bd.getDistance()

                if dist == float('inf') or dist >= 1e15:
                    total_penalty += PENALTY_ERROR_IN_CALCULATIONS
                    continue

                path = np.array(bd.getPath(), dtype=np.int64)

                total_penalty += compute_penalty_from_path(
                    path,
                    _STATIC_DATA['csr_indices'], _STATIC_DATA['csr_indptr'], _STATIC_DATA['csr_data_ids'],
                    _STATIC_DATA['edge_features'][0],
                    _STATIC_DATA['edge_features'][1],
                    _STATIC_DATA['edge_features'][2]
                )
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"\n[Worker {os.getpid()}] Crashed during path calculation!")
        print(f"Individual: {str(individual)}")
        print(f"Error: {e}\n{err_msg}")
        return (PENALTY_ERROR_IN_CALCULATIONS,)
    #Avg fitness on paths
    final_fit = total_penalty / num_scenarios

    tree_str = str(individual)
    if "distance" not in tree_str or "steepness" not in tree_str or "is_water" not in tree_str:
        final_fit += PENALTY_MISSING_VALUES

    if math.isinf(final_fit) or math.isnan(final_fit):
        return (PENALTY_ERROR_IN_CALCULATIONS,)

    return (final_fit,)

#Main loop
def run_EA(population, generations, res, npz_path, scenario_path, mut_rate=MUT_RATE, cx_rate=CROSS_RATE,
           log: bool = False, seed=None):
    if seed is None:
        seed = random.getrandbits(32)

    random.seed(seed)
    np.random.seed(seed)

    start = time.time()
    if log:
        print(f"Evolving the cost function through {generations} generations with a population of {population}.")
        print(f"Evolving with SEED: {seed}")
        print("Loading from file")
    #Load precomputed data from file npz
    data = np.load(npz_path)
    edge_features_columns = [data['dist'], data['steep'], data['water']]
    csr_indices, csr_indptr, csr_data = data['csr_indices'], data['csr_indptr'], data['csr_data']
    num_nodes = int(data['num_nodes'])
    #load scenarios
    with open(scenario_path, 'r') as f:
        data = json.load(f)
        scenarios_indices = np.array(data['scenarios'], dtype=np.int64)
    actual_num_scenarios = len(scenarios_indices)

    if log:
        print(f"Loaded {actual_num_scenarios} scenarios from {scenario_path}")
    grouped = defaultdict(list)
    for s, e in scenarios_indices:
        grouped[s].append(e)

    sources_list = list(grouped.keys())
    targets_list = [np.array(grouped[src], dtype=np.int64) for src in sources_list]

    dummy_data = np.zeros(len(csr_data), dtype=np.float64)
    csr_template = csr_matrix((dummy_data, csr_indices, csr_indptr), shape=(num_nodes, num_nodes))
    coo = csr_template.tocoo()
    mask = coo.row <= coo.col

    #Load data for multiprocessing
    static_data = {
        'num_nodes': num_nodes,
        'edge_features': edge_features_columns,
        'csr_indices': csr_indices,
        'csr_indptr': csr_indptr,
        'csr_data_ids': csr_data,
        'rows': np.ascontiguousarray(coo.row[mask], dtype=np.uint64),
        'cols': np.ascontiguousarray(coo.col[mask], dtype=np.uint64),
        'weight_mapping': np.ascontiguousarray(csr_data[mask] - 1, dtype=np.int64)
    }

    #Initialize population
    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(HOF_SIZE, similar=similar_by_structure)
    #Initialize the stats
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_fit.register("avg", np.mean)
    stats_fit.register("min", np.min)
    stats_fit.register("max", np.max)
    stats_fit.register("std", np.std)
    mstats = tools.MultiStatistics(fitness=stats_fit)

    #Start the pool
    pool = multiprocessing.Pool(
        processes=min(MAX_CORE_VALUE, max(1, multiprocessing.cpu_count() - 1)),
        initializer=init_worker,
        initargs=(pset, static_data),
        maxtasksperchild=100
    )
    toolbox.register("map", pool.map)
    #Initialize the evaluation function
    toolbox.register("evaluate", evaluate_individual,
                     sources_list=sources_list, targets_list=targets_list, num_scenarios=actual_num_scenarios)

    if log:
        print_gen_log(0, 0, {}, 0, 0, is_header=True, scenarios=scenarios_indices, seed=seed)
    #Evaluate starting pop
    try:
        gen_start = time.time()
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop.sort(key=lambda ind: ind.fitness.values[0])
        hof.update(pop)

        record = mstats.compile(pop)['fitness']
        num_dead = sum(1 for ind in pop if ind.fitness.values[0] >= (PENALTY_ERROR_IN_CALCULATIONS * 0.9))

        if log:
            print_gen_log(0, len(invalid_ind), record, num_dead, time.time() - gen_start)
            save_run(population, hof, 0, 0, generations, res, pset=pset, path=BASE_FOLDER, generation=0)

        std = 0.0
        best = 0.0
        static = 0
        #Evaluate each generation
        for gen in range(1, generations + 1):
            gen_start = time.time()

            offspring = toolbox.select(pop, len(pop))
            offspring = algorithms.varAnd(offspring, toolbox, cxpb=cx_rate, mutpb=mut_rate)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            pop.sort(key=lambda ind: ind.fitness.values[0])
            hof.update(pop)

            for i in range(HOF_SIZE):
                pop[-(i + 1)] = toolbox.clone(hof[i])

            record = mstats.compile(pop)['fitness']
            num_dead = sum(1 for ind in pop if ind.fitness.values[0] >= (PENALTY_ERROR_IN_CALCULATIONS * 0.9))
            gen_end = time.time() - gen_start

            if log:
                print_gen_log(gen, len(invalid_ind), record, num_dead, gen_end)
                save_run(population, hof, gen_end, 0, generations, res, pset=pset, path=BASE_FOLDER,
                         generation=gen)

            curr_std = record["std"]
            curr_best = record["min"]

            if abs(std - curr_std) < STD_THRESHOLD and best == curr_best:
                static += 1
                if static >= EARLY_STOPPING:
                    break
            else:
                best = curr_best
                std = curr_std
                static = 0  # Reset counter

        end = time.time()
        diff = end - start
        hours, tmp = divmod(diff, 3600)
        minutes, seconds = divmod(tmp, 60)

        if log:
            print(f"{generations} generations evolved in {int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds")
            print(f"Generations log saved in {BASE_FOLDER}")
            save_run(population, hof, diff, 0, generations, res, pset=pset, path=BASE_FOLDER,
                     generation="_final")

        best_ind = hof[0]
        best_fitness = best_ind.fitness.values[0]

        return best_ind, best_fitness
    finally:
        pool.close()
        pool.join()


# --- MAIN ---
if __name__ == "__main__":
    #POP, Generations
    experiments = [
        [3500, 50],
        [3500, 50],
        [3500, 50],
        [3500, 50],
        [3500, 50],
        [3500, 50],
        [3500, 50],
        [3500, 50],
        [3500, 50]
    ]
    res = 200
    #Trentino
    #npz_path = f"Dataset/Trentino/precomputed_map_trentino_{res}.npz"
    #scenario_path = f"Dataset/Trentino/scenarios_trentino_{res}.json"
    #Naples
    npz_path = f"Dataset/Naples/precomputed_map_napoli_{res}.npz"
    scenario_path = f"Dataset/Naples/scenarios_napoli_{res}.json"

    if not os.path.exists(npz_path):
        print(f"Error: File {npz_path} not found!")
        sys.exit(-1)

    today = datetime.now().strftime("%d_%m_%Y")
    runs_today_folder = f"GP/res/runs_{today}"
    if not os.path.exists(runs_today_folder):
        os.makedirs(runs_today_folder)

    for experiment in experiments:
        population = experiment[0]
        gens = experiment[1]

        BASE_FOLDER = f"{runs_today_folder}/{population}pop_{gens}gen_{res}res"

        if os.path.exists(BASE_FOLDER):
            i = 1
            while os.path.exists(f"{BASE_FOLDER}_{i}"):
                i += 1
            BASE_FOLDER = f"{BASE_FOLDER}_{i}"
        os.makedirs(BASE_FOLDER)
        try:
            run_EA(population=population, generations=gens, res=res, npz_path=npz_path,
                   log=True, scenario_path = scenario_path)

        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"\nCrash in experiment: {BASE_FOLDER}")
            print(f"Error: {e}")
            print("Saving log and going to next\n")

            crash_log_path = os.path.join(BASE_FOLDER, "CRASH_LOG.txt")
            with open(crash_log_path, "w") as f:
                f.write(f"Failed at IL {datetime.now()}\n")
                f.write(
                    f"Parameters: Population Size={population}, Generations Number={gens}\n")
                f.write("-" * 50 + "\n")
                f.write(error_msg)

        finally:
            print("CPU cooldown for Davide PC")
            sleep(5 * 60)
