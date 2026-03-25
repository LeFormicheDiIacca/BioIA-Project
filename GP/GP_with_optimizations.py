import csv
import math
import multiprocessing
import operator
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from time import sleep
import numpy as np
from deap import base, creator, gp, tools, algorithms
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gp_logistics import protected_div, protected_log, protected_pow, if_then_else, random_gen, save_run

BASE = math.e
BASE_FOLDER = ""
STD_THRESHOLD = 0.001
EARLY_STOPPING = 10
MUT_RATE = 0.2
CROSS_RATE = 0.7


def print_gen_log(gen, nevals, record, num_dead, duration, is_header=False):

    header = f"{'Gen':>4} | {'Nevals':>6} | {'Avg Fit':>12} | {'Std Fit':>12} | {'Min Fit':>12} | {'Max Fit':>12} | {'Dead':>5} | {'Time':>7}"
    csv_path = os.path.join(BASE_FOLDER, "evolution_stats.csv")
    if is_header:
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        csv_headers = ["gen", "nevals", "avg", "std", "min", "max", "dead", "time"]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
        return

    avg_str = f"{record['avg']:>12.2e}" if record['avg'] > 1e6 else f"{record['avg']:>12.2f}"
    min_str = f"{record['min']:>12.2f}"
    max_str = f"{record['max']:>12.2e}"
    std_str = f"{record['std']:>12.2e}" if record['avg'] > 1e6 else f"{record['std']:>12.2f}"

    print(f"{gen:>4} | {nevals:>6} | {avg_str} | {std_str} | {min_str} | {max_str} | {num_dead:>5} | {duration:>6.2f}s")
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow([gen, nevals, avg_str, std_str, min_str, max_str, num_dead, duration])


PENALTY_MISSING_VALUES = 1e8

# --- DEAP SETUP ---
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
pset.addEphemeralConstant("constant", random_gen, ret_type=float)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()


def create_valid_individual():
    while True:
        expr = gp.genHalfAndHalf(pset=pset, min_=2, max_=5)
        ind = creator.Individual(expr)
        tree_str = str(ind)
        required_inputs = ["distance", "steepness", "is_water"]
        if not any(inp not in tree_str for inp in required_inputs):
            return ind


toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, create_valid_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mutate_unif", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.register("mutate_eph", gp.mutEphemeral, mode="all")


def mutate_combined(individual):
    if random.random() < 0.7:
        return toolbox.mutate_unif(individual)
    else:
        return toolbox.mutate_eph(individual)


toolbox.register("mutate", mutate_combined)
toolbox.decorate("mate", gp.staticLimit(operator.attrgetter("height"), max_value=5))
toolbox.decorate("mate", gp.staticLimit(len, max_value=15))
toolbox.decorate("mutate_unif", gp.staticLimit(operator.attrgetter("height"), max_value=5))
toolbox.decorate("mutate_unif", gp.staticLimit(len, max_value=15))
toolbox.decorate("mutate_eph", gp.staticLimit(operator.attrgetter("height"), max_value=5))
toolbox.decorate("mutate_eph", gp.staticLimit(len, max_value=15))

_GLOBAL_PSET = None


def init_worker(pset):
    global _GLOBAL_PSET
    _GLOBAL_PSET = pset

@njit(fastmath=True, cache=True)
def compute_total_penalty_numba(predecessors, end_nodes, start_node_idx,
                                csr_indices, csr_indptr, csr_data,
                                edge_dist, edge_steep, edge_water):
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

            edge_idx = -1
            for i in range(csr_indptr[curr], csr_indptr[curr + 1]):
                if csr_indices[i] == prev:
                    edge_idx = csr_data[i] - 1
                    break

            if edge_idx == -1:
                curr = prev
                continue

            path_distance += edge_dist[edge_idx]
            path_steepness += edge_steep[edge_idx]
            if edge_water[edge_idx] > 0.5:
                path_water += 1

            tot_nodes += 1
            curr = prev

        if tot_nodes > 0:
            total_penalty += (path_distance / tot_nodes) * (
                        (path_water / tot_nodes) * (BASE - 1) + 1 + BASE ** (path_steepness / tot_nodes))
        else:
            total_penalty += 1_000_000.0

    return total_penalty


def evaluate_individual(individual, sources_list, targets_list, num_scenarios, edge_features_columns, csr_template,
                        csr_components):
    global _GLOBAL_PSET

    #Calculate costs
    try:
        func = gp.compile(expr=individual, pset=_GLOBAL_PSET)
        costs = func(*edge_features_columns)

        if np.isscalar(costs) or getattr(costs, 'ndim', 0) == 0:
            costs = np.full(len(edge_features_columns[0]), costs)

        #Avoid negative costs
        costs = np.maximum(costs, 0.001)
    except Exception as e:
        #print(str(individual))
        #traceback.print_exc()
        return (1e12,)  #In case of error

    #Update CSR matrix
    csr_indices, csr_indptr, csr_data_ids = csr_components
    csr_template.data = costs[csr_data_ids - 1]

    #Dijkstra Batch
    try:
        dists, preds = dijkstra(csr_template, directed=False, indices=sources_list, return_predecessors=True)
    except:
        return (1e12,)

    if len(sources_list) == 1:
        preds = preds.reshape(1, -1)
    #Canculate penalty
    total_penalty = 0.0
    col_dist, col_steep, col_water = edge_features_columns

    for i, start_idx in enumerate(sources_list):
        total_penalty += compute_total_penalty_numba(
            preds[i], targets_list[i], start_idx,
            csr_indices, csr_indptr, csr_data_ids,
            col_dist, col_steep, col_water
        )
    final_fit = total_penalty / num_scenarios

    #Soft Penalty for missing parameters
    tree_str = str(individual)
    for inp in ["distance", "steepness", "is_water"]:
        if inp not in tree_str:
            final_fit += 1000.0
    return (final_fit,)

def run_EA(scenarios_number, population, generations, res, npz_path, mut_rate= MUT_RATE, cx_rate= CROSS_RATE, log: bool = False):
    if log:
        print(f"Evolving the cost function through {generations} generations with a population of {population}.")
    start = time.time()
    if log:
        print("Loading from file")
    data = np.load(npz_path)
    edge_features_columns = [data['dist'], data['steep'], data['water']]
    csr_indices, csr_indptr, csr_data = data['csr_indices'], data['csr_indptr'], data['csr_data']
    csr_components = (csr_indices, csr_indptr, csr_data)
    num_nodes = int(data['num_nodes'])

    #Generate random scenarios
    scenarios_indices = np.array([random.sample(range(num_nodes), 2) for _ in range(scenarios_number)], dtype=np.int64)
    grouped = defaultdict(list)
    for s, e in scenarios_indices: grouped[s].append(e)
    sources_list = list(grouped.keys())
    targets_list = [np.array(grouped[src], dtype=np.int64) for src in sources_list]

    # CSR Template
    dummy_data = np.zeros(len(csr_data), dtype=np.float64)
    csr_template = csr_matrix((dummy_data, csr_indices, csr_indptr), shape=(num_nodes, num_nodes))



    pop = toolbox.population(n=population)

    # Statistiche setup
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_fit.register("avg", np.mean)
    stats_fit.register("min", np.min)
    stats_fit.register("max", np.max)
    stats_fit.register("std", np.std)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    hof = tools.HallOfFame(5, similar=operator.eq)

    # Multiprocessing
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1, initializer=init_worker, initargs=(pset,), maxtasksperchild=100)
    toolbox.register("map", pool.map)

    toolbox.register("evaluate", evaluate_individual,
                     sources_list=sources_list, targets_list=targets_list, num_scenarios=scenarios_number,
                     edge_features_columns=edge_features_columns, csr_template=csr_template, csr_components=csr_components)
    #Header Tab
    if log:
        print_gen_log(0, 0, {}, 0, 0, is_header=True)
    try:
        #Inizial population
        gen_start = time.time()
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)
        record = mstats.compile(pop)['fitness']
        num_dead = sum(1 for ind in pop if ind.fitness.values[0] >= 1e11)

        if log:
            print_gen_log(0, len(invalid_ind), record, num_dead, time.time() - gen_start)
            save_run(population, hof, 0, scenarios_number, generations, res, pset=pset, path=BASE_FOLDER, generation=0)
        std = 0.0
        best = 0.0
        static = 0

        #Evolution loop
        for gen in range(1, generations + 1):
            gen_start = time.time()

            offspring = toolbox.select(pop, len(pop))
            offspring = algorithms.varAnd(offspring, toolbox, cxpb=cx_rate, mutpb=mut_rate)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)

            record = mstats.compile(pop)['fitness']
            num_dead = sum(1 for ind in pop if ind.fitness.values[0] >= 1e11)
            gen_end = time.time() - gen_start
            if log:
                print_gen_log(gen, len(invalid_ind), record, num_dead, gen_end)
                save_run(population, hof, gen_end, scenarios_number, generations, res, pset=pset, path=BASE_FOLDER, generation= gen)
            curr_std = record["std"]
            curr_best = record["min"]
            if abs(std - curr_std) < STD_THRESHOLD and best == curr_best:
                static +=1
                if static >= EARLY_STOPPING:
                    break
            else:
                best = curr_best
                std = curr_std


        end = time.time()
        diff = end - start
        hours, tmp = divmod(diff, 3600)
        minutes, seconds = divmod(tmp, 60)
        if log:
            print(f"{generations} generations evolved in {int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds")

        pool.close()
        pool.join()
        if log:
            print(f"Generations log saved in {BASE_FOLDER}")
            save_run(population, hof, diff, scenarios_number, generations, res, pset=pset, path=BASE_FOLDER,
                     generation="_final")

        best_ind = hof[0]
        best_fitness = best_ind.fitness.values[0]

        return best_ind, best_fitness
    finally:
        pool.close()
        pool.join()


# --- MAIN ---
if __name__ == "__main__":
    #Pop Size, Scenarios Number, Generations
    experiments = [
        [500, 10, 200],
        [750, 10, 200],
        [1000, 10, 200],
        [500, 10, 200],
        [750, 10, 200],
        [1000, 10, 200],
        [500, 10, 200],
        [750, 10, 200],
        [1000, 10, 200]
    ]
    res = 200

    npz_path = f"TerrainGraph/precomputed_map_trentino_{res}.npz"
    # 4 of 6 REPLACEME
    # npz_path = f"TerrainGraph/precomputed_map_napoli_{res}.npz"

    if not os.path.exists(npz_path):
        print(f"Error: File {npz_path} not found!")
        sys.exit(-1)

    today = datetime.now().strftime("%d_%m_%Y")
    runs_today_folder = f"GP/res/runs_{today}"
    if not os.path.exists(runs_today_folder):
        os.makedirs(runs_today_folder)

    for experiment in experiments:
        population = experiment[0]
        scen_number = experiment[1]
        gens = experiment[2]

        BASE_FOLDER = f"{runs_today_folder}/{population}pop_{gens}gen_{scen_number}scenarios_{res}res"

        if os.path.exists(BASE_FOLDER):
            i = 1
            while os.path.exists(f"{BASE_FOLDER}_{i}"):
                i += 1
            BASE_FOLDER = f"{BASE_FOLDER}_{i}"
        os.makedirs(BASE_FOLDER)
        try:
            run_EA(scenarios_number=scen_number, population=population, generations=gens, res=res, npz_path=npz_path, log= True)

        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"\nCrash in experiment: {BASE_FOLDER}")
            print(f"Error: {e}")
            print("Saving log and going to next\n")

            #Save crash log in folder
            crash_log_path = os.path.join(BASE_FOLDER, "CRASH_LOG.txt")
            with open(crash_log_path, "w") as f:
                f.write(f"Failed at IL {datetime.now()}\n")
                f.write(f"Paramters: Population Size={population}, Generations Number={gens}, Scenarios={scen_number}\n")
                f.write("-" * 50 + "\n")
                f.write(error_msg)

        finally:
            print("CPU cooldown for Davide PC")
            sleep(5 * 60)