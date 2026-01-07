from deap import base, creator, gp, tools, algorithms
import operator
import random
import networkx as nx
import numpy as np
import multiprocessing
from terraingraph import create_graph
from scenario import generate_scenarios
from edge_info import create_edge_dict
import json
import time
from gp_logistics import protected_div, protected_log, protected_pow, tree_plotter, identity_water, if_then_else, append_to_json, dynamic_penalty, random_gen
from collections import defaultdict
from scipy.sparse.csgraph import dijkstra

# define primitive set

class WaterArg: pass
class OtherArgs: pass

# strongly typed # chosen to limit if_then function

pset = gp.PrimitiveSetTyped("MAIN", [OtherArgs, OtherArgs, OtherArgs, OtherArgs, WaterArg], OtherArgs)
pset.renameArguments(ARG0 = "distance", ARG1 = "steepness", ARG2 = "elevation_u", ARG3 = "elevation_v", ARG4 = "is_water")
pset.addPrimitive(operator.add, [OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(operator.mul, [OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(protected_pow, [OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(operator.sub, [OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(operator.neg, [OtherArgs], OtherArgs)
pset.addPrimitive(protected_log, [OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(protected_div,[OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(if_then_else, [WaterArg, OtherArgs, OtherArgs], OtherArgs)
pset.addPrimitive(identity_water, [WaterArg], WaterArg)
pset.addEphemeralConstant("constant", random_gen, ret_type=OtherArgs)

creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness = creator.FitnessMin, pset = pset)

# define main functions

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset = pset, min_=2, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# if your pc allows it, parallelize! (mine doesn't lololol)
# pool = multiprocessing.Pool()
# toolbox.register("map", pool.map)

# genetic operators

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("select", tools.selTournament, tournsize = 3)
toolbox.register("mutate_unif", gp.mutUniform, expr = toolbox.expr, pset= pset) 
toolbox.register("mutate_eph", gp.mutEphemeral, mode="all")

# limit bloating

toolbox.decorate("mate", gp.staticLimit(operator.attrgetter("height"), max_value= 5))
toolbox.decorate("mate", gp.staticLimit(len, max_value= 15))
toolbox.decorate("mutate_unif", gp.staticLimit(operator.attrgetter("height"), max_value= 5))
toolbox.decorate("mutate_unif", gp.staticLimit(len, max_value= 15))

# to include both type of mutation

def mutate_combined(individual):
    if random.random() < 0.7:
        return toolbox.mutate_unif(individual)
    else:
        return toolbox.mutate_eph(individual)

toolbox.register("mutate", mutate_combined)

# fitness function

def evaluate(individual, graph, scenarios, edge_dict, node_list, node_to_idx):
    func = toolbox.compile(expr=individual)  
    # add edge cost
    for u, v in graph.edges():
        u_ordered, v_ordered = min(u,v), max(u,v)
        data = edge_dict[f"{u_ordered}-{v_ordered}"]
        result = func(*data)
        if isinstance(result, complex):
            result = result.real
        graph[u][v]['cost'] = max(float(result), 0.001)
    
    adj_matrix = nx.to_scipy_sparse_array(graph, weight='cost', nodelist=node_list)
    total_penalty = 0
    grouped_scenarios = defaultdict(list)
    for s, e in scenarios:
        grouped_scenarios[s].append(e)
    
    # check the formula for each scenario
    for start_node, end_nodes in grouped_scenarios.items():
        start_idx = node_to_idx[start_node]
        dist_matrix, predecessors = dijkstra(
            csgraph=adj_matrix, 
            directed=False, 
            indices=start_idx, 
            return_predecessors=True
        )

        for end_node in end_nodes:
            end_idx = node_to_idx[end_node]
            
            if dist_matrix[end_idx] == np.inf:
                total_penalty += 1000000
                continue
            
            # Reconstruct Path using the Predecessor Array (Much faster than nx)
            path_idxs = []
            curr = end_idx
            while curr != -9999: # -9999 is the default for 'no predecessor'
                path_idxs.append(curr)
                if curr == start_idx: break
                curr = predecessors[curr]
            path = [node_list[i] for i in reversed(path_idxs)]

            # 4. Path Evaluation (Vectorized if possible)
            # Use your existing penalty logic here...
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                u_ord, v_ord = (u,v) if u < v else (v,u)
                d, incl, e_u, e_v, water = edge_dict[f"{u_ord}-{v_ord}"]
                
                elev_diff = max(e_v - e_u, 0)
                coeff = np.array([1, dynamic_penalty(incl)*d, 10, 5000])
                current_scen = np.array([d, incl, elev_diff, water])
                total_penalty += np.vdot(coeff, current_scen) 
    
    string_tree = str(individual)
    required_inputs = ["distance", "steepness", "elevation_u", "elevation_v", "is_water"]
    
    # counts how many inputs are missing

    missing_count = sum(1 for inp in required_inputs if inp not in string_tree)
    if_counts = string_tree.count("if_then_else") 
    total_penalty += 100000*missing_count # 100km for each missing input
    # since it does not make sense to have multiple if_counts, the water evaluation must be made only once per node
    if if_counts > 1:
        total_penalty += 100000
    return total_penalty,


# algorithm-running function

def run_EA(graph, scenarios, edge_dict, population, runs, scenario_dur):
    all_logs = []
    pop = toolbox.population(n = population)
    # for info about fitness of the evolved trees
    stats_fit = tools.Statistics(key =lambda ind: ind.fitness.values)
    stats_fit.register("avg", np.mean)
    stats_fit.register("std", np.std)
    stats_fit.register("min", np.min)
    stats_fit.register("max", np.max)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    hof = tools.HallOfFame(5, similar=operator.eq)
    # vs overfitting: we update the scenarios every 10 generations
    all_logs = list()
    node_list = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    for i in range(runs):
        current_scenario = [el[i] for el in scenarios]
        toolbox.register("evaluate", evaluate, graph = graph, scenarios = current_scenario, edge_dict=edge_dict, node_list = node_list, node_to_idx = node_to_idx)
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
        #size_avg = log.chapters["size"].select("avg")

        # Reconstruct the list of dictionaries
        for i_gen in range(len(gens)):
            entry = {
                'gen': str(i+1) + "." + str(gens[i_gen]),
                'nevals': nevals[i_gen],
                'fit_avg': fit_avg[i_gen],
                'fit_max': fit_max[i_gen],
                'fit_min' : fit_min[i_gen],
                'fit_std' : fit_std[i_gen]
                #'size_avg': size_avg[i_gen]
            }
            flattened_log.append(entry)

        all_logs.append(flattened_log)
        print(log)
        print(f"{i+1}° run completed.")    
    return pop, hof, all_logs

# main function to run, executes the code and saves logs

def main(population, runs, graph, edge_dict, scenario_dur = 10, res = 80):
    scenarios = generate_scenarios(runs, graph, res)
    print(f"Evolving the cost function through {runs} runs of {scenario_dur} generations with a population of {population}.")
    start = time.time()
    ret = run_EA(graph, scenarios, edge_dict, population, runs, scenario_dur)
    end = time.time()
    diff = end-start
    print("EA runtime: ", round(diff/60, 2), " minutes")
    pop = ret[0]
    logs = ret[2]
    best = ret[1][0]
    try:
        tree_plotter(best, f"pop{population}_run{runs}_res{res}_best_tree")
    except Exception as e:
        print(f"Could not plot tree: {e}")
    tree_diz = dict()
    tree_diz["tree_object"] = str(best)
    tree_diz["tree_fitness"] = best.fitness.values
    tree_diz["population"] = population
    tree_diz["resolution"] = res
    tree_diz["runs"] = runs
    tree_diz["scenario_duration"] = scenario_dur
    tree_diz["runtime"] = diff
    tree_diz["logs"] = logs
    append_to_json(tree_diz)
    print("The best individual has been saved")
    

if __name__ == "__main__":
    to_try = [[500,3], [500, 5], [1000,3], [1000,5]]
    # for el in to_try:
    #     res = 160
    #     graph = create_graph("trentino.tif","trentino_alto_adige.pbf", res)
    #     edge_dict = create_edge_dict(graph)
    #     main(population=el[0], runs=el[1], graph=graph, edge_dict=edge_dict, res=res)
             
    res = 80
    graph = create_graph("trentino.tif","trentino_alto_adige.pbf", res)
    edge_dict = create_edge_dict(graph)
    main(10, 2, graph, edge_dict)

    # TODO: registrare total running time per resolution, population size, generations, etc.
    # TODO: finetuning


        





