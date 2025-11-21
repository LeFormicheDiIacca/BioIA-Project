import math
import random

from ACO.Ant import Ant
from cost_functions import test_cost_assignment
from meshgraph import MeshGraph


class ACO_simulator:
    def __init__(self, graph, key_nodes, alpha, beta, rho, ant_number = 200, max_iterations = 1000, max_no_updates = 10, Q = 5000):
        self.graph = graph
        self.rho = rho
        self.key_nodes = key_nodes
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.ant_number = ant_number
        self.max_no_updates = max_no_updates

        initial_pheromone = 1.0 / Q
        self.graph.pheromone_initialization(initial_pheromone)

    def calc_path_cost(self, path):
        path_cost = 0
        degree_45_penalty_factor = 100
        for i in range(len(path)-1):
            source, destination = path[i], path[i+1]
            dist = self.graph.nodes_geometric_dist(source, destination)
            if dist != 1:
                path_cost += degree_45_penalty_factor
            path_cost += self.graph[source][destination]["cost"]

        return path_cost

    def global_pheromone_update(self, best_path):
        path_cost = self.calc_path_cost(best_path)
        for i in range(len(best_path)-1):
            source, destination = best_path[i], best_path[i+1]
            current_pheromone = self.graph[source][destination]["pheromone_level"]
            self.graph[source][destination]["pheromone_level"] = current_pheromone*(1-self.rho)
            self.graph[source][destination]["pheromone_level"] += self.rho/path_cost

    def simulation(self):
        current_best_path = None
        current_best_path_cost = math.inf
        epoch = 0
        updated = False
        current_no_updates = 0
        n_best_ants = 5
        while epoch < self.max_iterations and current_no_updates < self.max_no_updates:
            paths = []
            for i in range(self.ant_number):
                start_node = random.sample(tuple(self.graph.nodes()), 1)[0]
                ant = Ant(start_node, self.key_nodes, self.graph, self.alpha, self.beta, self.rho)
                path = ant.calculate_path()

                if path[-1] != start_node:
                    continue
                visited_set = set(path)
                if not self.key_nodes.issubset(visited_set):
                    continue
                path_cost = self.calc_path_cost(path)
                paths += [(path, path_cost)]
                if path_cost < current_best_path_cost:
                    current_best_path = path
                    current_best_path_cost = path_cost
                    updated = True
            if updated:
                current_no_updates = 0
            else:
                current_no_updates += 1

            paths.sort(key=lambda x: x[1])
            best_ants = paths[:n_best_ants]
            for (path, cost) in best_ants:
                self.global_pheromone_update(path)
            epoch += 1

        return current_best_path, current_best_path_cost