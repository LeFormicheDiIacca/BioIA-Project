from deap import base, creator, gp, tools, algorithms
import operator
import math
import random
from functools import partial
import networkx as nx
import numpy as np
import json
import multiprocessing
from terraingraph import create_graph
import pygraphviz as pgv
import pydot 


n = 80
tif_path = "trentino.tif"
osm_path = "trentino_alto_adige.pbf"
graph = create_graph(tif_path=tif_path, osm_pbf_path=osm_path, resolution=n)

with open(f"edge_dict_res{n}.json", "r") as f:
    edge_dict = json.load(f)

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
        base = np.abs(n1)
        exponent = np.clip(n2, -5, 5)
        return np.power(base, exponent)
    except: 
        return 1.0

def if_then_else(condition, out1, out2):
    return np.where(condition > 0.5, out1, out2) 

def identity_water(x):
    return x

def dynamic_penalty(inclination):
    if inclination >= 30: 
        return 50000*inclination
    else:
        return 5*inclination

def evaluate(individual, graph, scenarios):
    func = toolbox.compile(expr=individual)  
    # add edge cost
    for u, v in graph.edges():
        u_ordered, v_ordered = min(u,v), max(u,v)
        d, incl, e_u, e_v, water = edge_dict[f"{u_ordered}-{v_ordered}"]
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
                
                real_dist += d
                water_nodes += water #1.0 or 0.0
                total_incl += dynamic_penalty(incl)*d # very high inclination (>= 30) is dramatically penalized; considers 
                                                      # distance as the longer the inclined edge, the worse it is
                if e_v > e_u:
                    elev_gain += e_v - e_u            
            
            # penalty formula, coefficients just represent the relative weight of each feature 
 
            total_penalty += real_dist + (elev_gain * 10) + (water_nodes * 5000) + total_incl
            
        except nx.NetworkXNoPath:
            # no path
            total_penalty += 1000000 
            
    return total_penalty, # DEAP requires a tuple

def round_random(a,b):
    return round(random.uniform(a,b), 3)
random_gen = partial(round_random, 0,10)


# define primitive set

class WaterArg: pass
class OtherArgs: pass

# strongly typed # chosen to limit if_then function

pset = gp.PrimitiveSetTyped("MAIN", [OtherArgs, OtherArgs, OtherArgs, OtherArgs, WaterArg], OtherArgs)
pset.renameArguments(ARG0 = "dist", ARG1 = "inclination", ARG2 = "elev_u", ARG3 = "elev_v", ARG4 = "is_water")
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
toolbox.register("expr", gp.genHalfAndHalf, pset = pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# if your pc allows it, parallelize! (mine doesn't lololol)
# pool = multiprocessing.Pool()
# toolbox.register("map", pool.map)

# genetic operators

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("select", tools.selTournament, tournsize = 3)
toolbox.register("mutate", gp.mutUniform, expr = toolbox.expr, pset= pset) 

# limit bloating

toolbox.decorate("mate", gp.staticLimit(operator.attrgetter("height"), max_value= 15))
toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), max_value=15))

# scenarios # hand-picked based on the chosen resolution

nodes_with_water = [(15,36), (862, 1177), (2800, 3040)] #water in-between
nodes_different_altitude = [(110,912), (159, 868), (793, 6046)] #from mountains to plains or viceversa
nodes_through_obstacles = [(2996, 2214), (2586, 3615), (1837, 3821)] #some water and heights to go through

scenarios = [nodes_with_water, nodes_different_altitude, nodes_through_obstacles]

# algorithm run

def main():
    all_logs = []
    pop = toolbox.population(n = 10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(10, similar=operator.eq)
    ngen = 30
    scenario_run = 10
    # vs overfitting: we update the scenarios every 10 generations
    for i in range(0, ngen, scenario_run):
        current_scenario = [el[i//scenario_run] for el in scenarios]
        toolbox.register("evaluate", evaluate, graph = graph, scenarios = current_scenario)
        for ind in pop:
            del ind.fitness.values
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, 
                                    ngen=scenario_run, stats=stats, halloffame=hof, verbose=False)
        print(f"Scenario {i//scenario_run +1} complete.")
        all_logs.append(log)
        
    return pop, stats, hof

if __name__ == "__main__":
    ret = main()
    for i in range(len(ret[2])):
        f = ""
        nodes, edges, labels = gp.graph(ret[2][i])
        f += ("digraph G {\n")
        f += (f"label = best_{i+1}_tree\n")
        f += "size = 20\ndpi = 300\n"
        f += "ranksep = 0.5\nnodesep = 0.5\n"
        for node in nodes:
            label = str(labels[node]).replace('"', '')
            f += f'  {node} [label="{label}", fontsize = 10];\n'
        for edge in edges:
            f += f'  {edge[0]} -> {edge[1]};\n'
        f += "}"
        graphs = pydot.graph_from_dot_data(f)
        graph = graphs[0]
        graph.write_png(f"hof/best_{i+1}._tree.png")

        
        




