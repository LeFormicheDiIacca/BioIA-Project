import math
import random

from cost_functions import test_cost_assignment
from meshgraph import meshgraph_generator, cost_assignment, plot_graph

def weighted_random_choice(choices: dict):
    max = sum(choices.values())
    pick = random.uniform(0, max)
    current = 0
    for key, value in choices.items():
        current += value
        if current > pick:
            return key

class Ant:
    def __init__(self, starting_node, destination, graph, alpha, beta, exploration_threshold, rho):
        self.alpha = alpha
        self.beta = beta
        self.exploration_threshold = exploration_threshold
        self.path = [starting_node]
        self.visited_nodes = {starting_node}
        self.destination = destination
        self.graph = graph
        self.rho = rho

    def calculate_path(self, current_node):
        q = random.random()
        next_node = None
        if q <= self.exploration_threshold:
            pheromones = dict()
            for adj in self.graph[current_node]:
                if adj not in self.visited_nodes:
                    pheromones[(current_node, adj)] = self.graph[current_node][adj]["pheromone_level"]

            pheromones_sum = sum(pheromones.values())
            pheromones = {k: v/pheromones_sum for k, v in pheromones.items()}
            if len(pheromones) == 0:
                return self.path
            next_node = weighted_random_choice(pheromones)[1]
        else:
            probability = dict()
            for adj in self.graph[current_node]:
                if adj not in self.visited_nodes:
                    pheromones = self.graph[current_node][adj]["pheromone_level"]
                    pheromones = pheromones**self.alpha
                    cost = 1/self.graph[current_node][adj]["cost"]
                    cost = cost**self.beta
                    probability[(current_node,adj)] = pheromones*cost
            total = sum(list(probability.values()))
            probability = { k : v/total for k,v in probability.items()}
            if len(probability) == 0:
                return self.path
            next_node = weighted_random_choice(probability)[1]

        self.path.append(next_node)
        self.visited_nodes.add(next_node)
        if self.destination not in self.path:
            return self.calculate_path(next_node)
        else:
            return self.path


class ACO:
    def __init__(self, graph, node_to_pos, source, target, alpha, beta, exploration_threshold, rho, ant_number = 200, max_iterations = 1000, max_no_updates = 10):
        self.graph = graph
        self.node_to_pos = node_to_pos
        self.rho = rho
        self.source = source
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.exploration_threshold = exploration_threshold
        self.max_iterations = max_iterations
        self.ant_number = ant_number
        self.max_no_updates = max_no_updates

        for edge in graph.edges():
            self.graph[edge[0]][edge[1]]["pheromone_level"] = 1
            self.graph[edge[0]][edge[1]]["initial_pheromone_level"] = 1


    def calc_path_cost(self, path):
        path_cost = 0
        degree_45_penalty_factor = 1
        for i in range(len(path)-1):
            source, destination = path[i], path[i+1]
            x1, y1 = self.node_to_pos[source]
            x2, y2 = self.node_to_pos[destination]
            dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist != 1:
                path_cost += degree_45_penalty_factor
                print("cost sqrt(2)")
            path_cost += self.graph[source][destination]["cost"]

        return path_cost

    def global_pheromone_update(self, best_path):
        path_cost = self.calc_path_cost(best_path)
        for i in range(len(best_path)-1):
            source, destination = best_path[i], best_path[i+1]
            current_pheromone = self.graph[source][destination]["pheromone_level"]
            self.graph[source][destination]["pheromone_level"] = current_pheromone*(1-self.rho)
            self.graph[source][destination]["pheromone_level"] += self.rho/path_cost

    def local_pheromone_update(self, path):
        for i in range(len(path)-1):
            source, destination = path[i], path[i+1]
            current_pheromone = self.graph[source][destination]["pheromone_level"]
            self.graph[source][destination]["pheromone_level"] = current_pheromone*(1-self.rho)
            self.graph[source][destination]["pheromone_level"] += self.graph[source][destination]["initial_pheromone_level"] * self.rho


    def simulation(self):
        current_best_path = None
        current_best_path_cost = math.inf
        epoch = 0
        updated = False
        current_no_updates = 0
        while epoch < self.max_iterations and current_no_updates < self.max_no_updates:
            paths = []
            for i in range(self.ant_number):
                ant = Ant(self.source, self.target, self.graph, self.alpha, self.beta, self.exploration_threshold, self.rho)
                path = ant.calculate_path(self.source)
                if self.target not in path:
                    break
                path_cost = self.calc_path_cost(path)
                paths.append(path)
                if path_cost < current_best_path_cost:
                    current_best_path = path
                    current_best_path_cost = path_cost
                    updated = True
            if updated:
                current_no_updates = 0
            else:
                current_no_updates += 1
            for path in paths:
                self.local_pheromone_update(path)
            self.global_pheromone_update(current_best_path)
            epoch += 1

        return current_best_path, current_best_path_cost

mesh_graph, pos_to_node, node_to_pos = meshgraph_generator(8,5,5)
edges_metadata = dict()
cost_assignment(mesh_graph, edges_metadata, test_cost_assignment, print_assignment=False)

aco = ACO(mesh_graph,node_to_pos,1,22, 1,2,0.9, 0.1,20, 50)
path, path_cost = aco.simulation()
print(path)
print(path_cost)
#for edge in mesh_graph.edges():
#    print(f"{edge[0]}->{edge[1]}\nMetadata:{mesh_graph[edge[0]][edge[1]]}")
plot_graph(mesh_graph)

STE CAZZO DI FORMICHE FAVORISCONO LE CURVE DI 45° PERCHE' FIGLIE DI PUTTANA' cOME RISOLVO?