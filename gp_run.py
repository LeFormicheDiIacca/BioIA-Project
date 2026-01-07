from deap import base, creator, gp, tools, algorithms
import operator
import math
import random
from functools import partial
import networkx as nx
import numpy as np
import multiprocessing
from terraingraph import create_graph
import pydot 
from scenario import generate_scenarios
from edge_info import create_edge_dict
import json
import time
import os

def protected_div(n1, n2):
    if isinstance(n1, complex):
        n1 = n2.real
    if isinstance(n2, complex):
        n2 = n2.real
    if n2 == 0:
        return 0
    else:
        return n1/n2
    
def protected_log(x, base):
    if isinstance(x, complex):
        x = x.real
    if isinstance(base, complex):
        base = base.real
    if x > 0 and base > 0 and base!=1:
        return math.log(x, base)
    else:
        return 1

def protected_pow(n1, n2):
    if isinstance(n1, complex):
        n1 = n2.real
    if isinstance(n2, complex):
        n2 = n2.real
    if n1 == 0:
        return 0
    try:
        base = float(np.abs(n1))
        exponent = np.clip(float(n2), -5, 5)
        return np.power(base, exponent)
    except (OverflowError, ValueError): 
        return 1e10

def if_then_else(condition, out1, out2):
    return np.where(condition > 0.5, out1, out2) 

def identity_water(x):
    return x

def dynamic_penalty(inclination):
    if inclination >= round(30/90, 4): # if normalized inclination is more than 1/3 (= aka inclination is >=30%)
        return 50000*inclination
    else:
        return 5*inclination

def round_random(a,b):
    return round(random.uniform(a,b), 3)
random_gen = partial(round_random, 0,10)

def mutate_combined(individual):
    if random.random() < 0.7:
        return toolbox.mutate_unif(individual)
    else:
        return toolbox.mutate_eph(individual)

def tree_plotter(tree, title):
    nodes, edges, labels = gp.graph(tree)
    f = "digraph G {\n"
    f += "    size=\"20,20\";\n"  
    f += "    dpi=300;\n"          
    f += "    labelloc=\"t\";\n"    
    f += f"    label=\"{title}\n\\n\";\n"
    f += "    labelloc = \"t\";\n"
    f += "    size = \"20,20\";\n"
    f += "    dpi = 300;\n"     
    for node in nodes:
        f += f'    {node} [label="{labels[node]}", shape=ellipse, style=filled, fillcolor=white, fontname="Arial", fixedsize=false, margin=0.2];\n'
    for edge in edges:
        f += f"    {edge[0]} -> {edge[1]};\n"
    f += "}"
    graphs = pydot.graph_from_dot_data(f)
    graph = graphs[0]
    graph.write_png(f"hof/{title}.png")

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
pool = multiprocessing.Pool()
toolbox.register("map", pool.map)

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

toolbox.register("mutate", mutate_combined)

# fitness function

def evaluate(individual, graph, scenarios, edge_dict):
    func = toolbox.compile(expr=individual)  
    # add edge cost
    max_d = 0
    max_incl = 90
    max_elev = 0
    for el in edge_dict.values():
        if el[0] > max_d: max_d = el[0]
        current_max_e = el[2] if el[2] > el[3] else el[3]
        if current_max_e > max_elev:
            max_elev = current_max_e
    for u, v in graph.edges():
        u_ordered, v_ordered = min(u,v), max(u,v)
        d, incl, e_u, e_v, water = edge_dict[f"{u_ordered}-{v_ordered}"]

        # max normalization of inputs # more robust in front of varying resolution
        incl = incl/max_incl
        d = d/max_d
        e_u = e_u/max_elev
        e_v = e_v/max_elev
        result = func(d, incl, e_u, e_v, water)
        if isinstance(result, complex):
            result = result.real
        graph[u][v]['cost'] = max(float(result), 0.001)
    
    total_penalty = 0
    
    # check the formula for each scenario
    for start_node, end_node in scenarios:
        try:
            # finds shortest path using dijkstra
            path = nx.shortest_path(graph, source=start_node, target=end_node, weight='cost')
            
            # computes the quality of the path in real world metrics
            real_dist = 0
            water_nodes = 0
            elev_gain = 0
            total_incl = 0
            
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                u_ordered, v_ordered = min(u,v), max(u,v)
                d, incl, e_u, e_v, water = edge_dict[f"{u_ordered}-{v_ordered}"]
                
                real_dist += d/max_d
                water_nodes += water #1.0 or 0.0
                total_incl += dynamic_penalty(incl/max_incl)*d/max_d # very high inclination (>= 30) is dramatically penalized; considers 
                                                      # distance as the longer the inclined edge, the worse it is
                if e_v > e_u:
                    elev_gain += (e_v - e_u)/max_elev            
            
            # penalty formula, coefficients just represent the relative weight of each feature 
 
            total_penalty += real_dist + (elev_gain * 10) + (water_nodes * 5000) + total_incl
            
        except nx.NetworkXNoPath:
            # no path
            total_penalty += 1000000 
    
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

def run_EA(population, runs, scenario_dur, res):
    graph = create_graph("trentino.tif","trentino_alto_adige.pbf", res)
    edge_dict = create_edge_dict(graph)
    scenarios = generate_scenarios(runs, graph, res)
    all_logs = []
    pop = toolbox.population(n = population)
    # for info about fitness of the evolved trees
    stats_fit = tools.Statistics(key =lambda ind: ind.fitness.values)
    stats_fit.register("avg", np.mean)
    stats_fit.register("std", np.std)
    stats_fit.register("min", np.min)
    stats_fit.register("max", np.max)
    # # for info about the complexity of the evolved trees
    mstats = tools.MultiStatistics(fitness=stats_fit)
    # stats_size = tools.Statistics(key = len)
    # stats_size.register("avg", np.mean)
    hof = tools.HallOfFame(5, similar=operator.eq)
    print(f"Evolving the cost function through {runs} runs of {scenario_dur} generations with a population of {population}.")
    # vs overfitting: we update the scenarios every 10 generations
    all_logs = list()
    start = time.time()
    for i in range(runs):
        current_scenario = [el[i] for el in scenarios]
        toolbox.register("evaluate", evaluate, graph = graph, scenarios = current_scenario, edge_dict=edge_dict)
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
    end = time.time()
    diff = end-start
    print("EA runtime: ", round(diff/60, 2), " minutes")    
    return pop, hof, all_logs, diff

# save runtime info

def main(population, runs, scenario_dur = 10, res = 80):
    ret = run_EA(population, runs, scenario_dur, res)
    pop = ret[0]
    logs = ret[2]
    diff = ret[3]
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
    pop_logs = list()
    for tree in pop:
        i = 1
        pop_diz = dict()
        pop_diz["id"] = i
        pop_diz["fitness"] = float(tree.fitness.values[0])
        pop_diz["size"] = len(tree)
        pop_logs.append(pop_diz)
        i +=1
    with open("pop_info.json", "w") as f:
        json.dump(pop_logs, f)
    print("The population has been stored")
    

# adds new data to a json file for finetuning

def append_to_json(new_data):
    # 1. Check if file exists and isn't empty
    if os.path.exists("tree_diz.json") and os.path.getsize("tree_diz.json") > 0:
        with open("tree_diz.json", 'r') as f:
            data = json.load(f)
    else:
        data = [] # Start with an empty list if file doesn't exist
    data.append(new_data)
    with open("tree_diz.json", 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    to_try = [[500,3], [500, 5], [1000,3], [1000,5]]
    for el in to_try:
        main(el[0], el[1], res=160)



    # TODO: registrare total running time per resolution, population size, generations, etc.
    # TODO: finetuning


        





