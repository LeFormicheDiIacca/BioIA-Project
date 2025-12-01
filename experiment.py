
from deap import gp, base, creator, tools, algorithms
import operator
import math
import random
from ACO.ACO_simulator import ACO_simulator
from sanity_check import weight_func, heuristic
from terraingraph import create_graph
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


# data lista di archi con metadati, con costo predefinito da noi, la fitness vede quello che valuta individuo e quanto diciamo noi a priori
# comparazione per ogni caratteristica
# angolo in diagonale


# salvare individuo migliore in file

#extract graph

# E
def extract_edge_features(graph):
    edge_features = []
    
    for u, v, data in graph.edges(data=True):
        
        # Recupera i quattro argomenti della tua Primitive Set (pset)
        dist = data["dist"]
        elev_diff = data["elev_diff"]
        elev_u = data["elev_u"]
        elev_v = data["elev_v"]

        # Recupera l'etichetta booleana che userai per la "target cost"
        is_water = data["has_water"] 
        
        # Aggiungi il set di features alla lista
        edge_features.append((dist, elev_diff, elev_u, elev_v, is_water))
        
    return edge_features

def enrich_graph_edges(G):
    """
    Popola gli attributi degli archi richiesti per la funzione di costo del GP.
    
    :param G: Il grafo (MeshGraph) con tutti gli attributi dei nodi già popolati.
    :return: Il grafo modificato G.
    """
    
    for u, v in G.edges():
        
        # 1. Recupera i dati del nodo
        elev_u = G.nodes[u]["elevation"]
        elev_v = G.nodes[v]["elevation"]
        is_water_u = G.nodes[u]["is_water"]
        is_water_v = G.nodes[v]["is_water"]

        # 2. Calcola le caratteristiche
        dist = heuristic(G, u, v)  
        elev_diff = abs(elev_v - elev_u)

        # 3. Assegna gli attributi all'arco (u, v)
        G.edges[u, v]["dist"] = dist
        G.edges[u, v]["elev_diff"] = elev_diff
        G.edges[u, v]["elev_u"] = elev_u
        G.edges[u, v]["elev_v"] = elev_v
        
        # Aggiungi attributi booleani (per il training data)
        G.edges[u, v]["has_water"] = is_water_u or is_water_v

    return G

G = create_graph("trentino.tif", "trentino_alto_adige.pbf", 150)
G_complete = enrich_graph_edges(G)
all_edge_data = extract_edge_features(G_complete)

print(all_edge_data)

#GOAL NOW: EXTRACT GOOD FEATURES AND BAD FEATURES FOR EACH ATTRIBUTE

# Example: Create a dataset of paired contrasting edges (features)
# In a real scenario, these would come from your MeshGraph data.

# Features for a 'Good' edge (e.g., short, low elevation change, favorable start/end elevation)
GOOD_EDGE = (10.0, 0.5, 100.0, 100.5)

# Features for a 'Bad' edge (e.g., long, high elevation change, unfavorable start/end elevation)
BAD_EDGE = (50.0, 20.0, 100.0, 120.0)

# A list of edge examples you want the cost function to correctly differentiate
# In reality, you'd have many of these pairs.
TRAINING_DATA = [
    (GOOD_EDGE, 0.1),  # Expected Cost for Good Edge (Target: Low)
    (BAD_EDGE, 10.0)   # Expected Cost for Bad Edge (Target: High)
]


def evalSymbolicRegression(individual, toolbox, data):
    """
    Evaluates the fitness of an individual (a cost function) by 
    calculating the Mean Squared Error (MSE) against a training dataset.
    """
    # 1. Compile the tree into a callable function
    func = toolbox.compile(expr=individual)

    # 2. Calculate the squared error for each data point
    sq_errors = []
    
    for inputs, target_cost in data:
        dist, elev_diff, elev_u, elev_v = inputs
        
        # 3. Call the generated function (the cost function)
        # Handle potential runtime errors (e.g., math domain errors from 'log')
        try:
            predicted_cost = func(dist, elev_diff, elev_u, elev_v)
        except:
            # Assign a large error if the function fails (e.g., divide by zero, log of negative)
            return (10000.0,), 

        # 4. Calculate the squared error
        error = predicted_cost - target_cost
        sq_errors.append(error**2)

    # 5. Calculate the Mean Squared Error (MSE)
    mse = sum(sq_errors) / len(sq_errors)
    
    # Since fitness is minimizing (-1.0), the lower the MSE, the better the fitness.
    return (mse,)


# Assuming you have the TRAINING_DATA defined as shown in Step 1
# And the evalSymbolicRegression function defined as shown in Step 2

# Register the evaluation function, fixing the training data argument
toolbox.register("evaluate", 
                 evalSymbolicRegression, 
                 toolbox=toolbox, 
                 data=TRAINING_DATA)    








# class WeightFunction:
#     def __init__(self, cost_func, graph, heuristic_func):
#         self.cost_func = cost_func
#         self.graph = graph
#         self.heuristic_func = heuristic_func
    
#     def __call__(self, u, v):
#         if self.graph.nodes[u].get('is_water') or self.graph.nodes[v].get('is_water'):
#             return float('inf')
        
#         dist = self.heuristic_func(u, v)
#         elev_v = self.graph.nodes[v]["elevation"]
#         elev_u = self.graph.nodes[u]["elevation"]
#         elev_diff = abs(elev_u - elev_v)
#         cost = self.cost_func(dist, elev_diff, elev_u, elev_v)
#         return max(cost, 1.0)

# def evaluate_aco_fitness(individual, graph, aco_params):
#     og_weight_func = getattr(graph, 'weight_func', None)
#     try:
#         cost_func = toolbox.compile(expr=individual)
#     except Exception:
#         return (float('inf'),)  # Return tuple with inf instead of arbitrary number
    
#     # Create picklable weight function
#     graph.weight_func = WeightFunction(cost_func, graph, heuristic)
    
#     path_cost = float("inf")
#     try:
#         aco = ACO_simulator(graph, **aco_params)
#         best_paths = aco.simulation(retrieve_n_best_paths=1)
#         if best_paths:
#             path_cost = best_paths[0][1]
#     except Exception as e:
#         print(f"ACO simulation failed: {e}")
#         pass
#     finally:
#         if og_weight_func is not None:
#             graph.weight_func = og_weight_func
#         else:
#             delattr(graph, 'weight_func')
    
#     return (path_cost,)


# toolbox.register("evaluate", functools.partial(evaluate_aco_fitness, graph = G, aco_params = ant_colony_parameters))

# def main():
#     random.seed(31)
#     try:
#         pop = toolbox.population(n = 200)
#     except:
#         return 10101010
#     hof = tools.HallOfFame(1)
#     stats = tools.Statistics(lambda ind: ind.fitness.values)
#     stats.register("min", np.min)
#     # 1. Esegui l'algoritmo
#     pop, log, hof = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.05, ngen=20, stats=stats, halloffame=hof, verbose=True)
    
#     # 2. Estrai il miglior individuo dal Hall of Fame
#     best_individual = hof[0]
    
#     # 3. Converti l'albero in una stringa Python valida
#     individual_string = str(best_individual) 
    
#     # 4. Salva la stringa
#     with open("best_individual_expression.txt", "w") as f:
#         f.write(individual_string)
        
#     print(f"Miglior individuo salvato come stringa: {individual_string}")
    
#     return pop, log, hof


# if __name__ == "__main__":
#     main()


# try:
#     with open("best_individual_expression.txt", "r") as f:
#         individual_string = f.read()
# except FileNotFoundError:
#     print("Errore: file best_individual_expression.txt non trovato.")
#     exit()

# # 2. Parsa la stringa in un albero di espressione DEAP
# # Questo richiede la Primitive Set (pset) per risolvere i nomi
# best_tree = gp.PrimitiveTree.from_string(individual_string, pset)

# # 3. Compila l'albero nella funzione Python eseguibile
# # Utilizza la tua toolbox registrata: toolbox.register("compile", gp.compile, pset=pset)
# final_cost_func = toolbox.compile(expr=best_tree)

# # 4. Utilizza la funzione
# # La funzione attende 4 argomenti: dist, elev_diff, elev_u, elev_v
# test_cost = final_cost_func(dist=100.0, elev_diff=5.0, elev_u=50.0, elev_v=55.0)

# print(f"\nStringa Caricata: {individual_string}")
# print(f"Funzione Compilata: {final_cost_func}")
# print(f"Costo di Test (100.0, 5.0, 50.0, 55.0): {test_cost}")


