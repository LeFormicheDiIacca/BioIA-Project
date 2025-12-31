from deap import base, creator, gp, tools, algorithms
import operator
import math
import random
from functools import partial
import networkx as nx
from edge_info import graph, scenarios
import numpy as np
import json
# import multiprocessing

with open("edge_dict_res80.json", "r") as f:
    edge_dict = json.load(f)

# define the Primitive set

def protected_div(n1, n2):
    if n2 == 0:
        return 0
    else:
        return n1/n2
    
def protected_log(x, base):
    if x > 0 and base > 0 and base!=1:
        return math.log(x, base)
    else:
        return 1

def protected_pow(n1, n2):
    try:
        return n1**(min(n2,10))
    except OverflowError:
        return 1e10

def if_then_else(condition, out1, out2):
    # Se la condizione è > 0.5 (True), restituisci il primo ramo, altrimenti il secondo
    return out1 if condition > 0.5 else out2

def water_multiplier(is_water, penalty_value):
    # Se is_water è 1, restituisce la penalità, altrimenti 1 (neutro)
    if is_water > 0.5:
        return penalty_value # O un valore che il GP può evolvere
    return 1.0

def dynamic_penalty(inclination):
    if inclination >= 30:
        return 50000*inclination
    else:
        return 5*inclination

def evaluate(individual, graph, scenarios):
    # 1. Trasforma l'individuo in una funzione utilizzabile
    func = toolbox.compile(expr=individual)
    
    # 2. Applica la formula a tutti gli archi per aggiornare 'cost'
    for u, v in graph.edges():
        #m = get_edge_metadata(graph, u, v)
        u_ordered, v_ordered = min(u,v), max(u,v)
        m = edge_dict[f"{u_ordered}-{v_ordered}"]
        # Protezione: evita costi <= 0 che mandano in crash Dijkstra/ACO
        result = func(*m)

        if isinstance(result, complex):
            result = result.real

        graph[u][v]['cost'] = max(float(result), 0.001)
    
    total_penalty = 0
    
    # 3. Testa la formula su ogni scenario
    for start_node, end_node in scenarios:
        try:
            # Trova il percorso migliore secondo la formula del GP
            path = nx.shortest_path(graph, source=start_node, target=end_node, weight='cost')
            
            # Calcola quanto è brutto questo percorso nel mondo reale
            dist_reale = 0
            water_nodes = 0
            elev_gain = 0
            total_incl = 0
            
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                # Usa i metadati reali (non quelli del GP!)
                u_ordered, v_ordered = min(u,v), max(u,v)
                d, incl, e_u, e_v, water = edge_dict[f"{u_ordered}-{v_ordered}"]
                
                dist_reale += d
                water_nodes += water # water è 1.0 o 0.0
                total_incl += dynamic_penalty(incl)*d
                if e_v > e_u:
                    elev_gain += e_v - e_u
            
            # Formula della penalità per questo scenario
            # Esempio: Metri + (Salita * 10) + (Ogni nodo acqua vale 5km di penalità)
            total_penalty += dist_reale + (elev_gain * 10) + (water_nodes * 5000) + total_incl
            
        except nx.NetworkXNoPath:
            # Se la formula è così brutta da isolare i nodi
            total_penalty += 1000000 
            
    return total_penalty, # DEAP vuole una tupla

def round_random(a,b):
    return round(random.uniform(a,b), 3)
random_gen = partial(round_random, 0,10)


# define primitive set

pset = gp.PrimitiveSet("MAIN", arity = 5)
pset.renameArguments(ARG0 = "dist", ARG1 = "inclination", ARG2 = "elev_u", ARG3 = "elev_v", ARG4 = "is_water")
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_pow, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(protected_log, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(if_then_else, 3)
pset.addPrimitive(water_multiplier, 2)
pset.addEphemeralConstant("constant", (random_gen))

# define Fitness and Individual

creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness = creator.FitnessMin, pset = pset)

# define main functions

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset = pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
# pool = multiprocessing.Pool()
# toolbox.register("map", pool.map)

# genetic operators

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("select", tools.selTournament, tournsize = 3)
toolbox.register("mutate", gp.mutUniform, expr = toolbox.expr, pset= pset) 
toolbox.register("evaluate", evaluate, graph = graph, scenarios = scenarios)

# limit bloating

toolbox.decorate("mate", gp.staticLimit(operator.attrgetter("height"), max_value= 15))
toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), max_value=15))

# PREPARATION TO THE SIMULATION

def main():
    pop = toolbox.population(n = 10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=2, stats=stats)
    return pop, stats

if __name__ == "__main__":
    ret = main()
