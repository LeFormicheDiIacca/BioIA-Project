import math
import random

from ACO.Ant import Ant
from cost_functions import test_cost_assignment
from meshgraph import MeshGraph


class ACO_simulator:
    def __init__(self, graph: MeshGraph, key_nodes, alpha, beta, rho, ant_number = 200, max_iterations = 1000, max_no_updates = 10, Q = 5000):
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

    def global_pheromone_update(self, best_path):
        path_cost = self.graph.calc_path_cost(best_path)
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
                path_cost = self.graph.calc_path_cost(path)
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


    def _TwoOptSwap(self, path, v1_index, v2_index):
        if v1_index >= v2_index:
            v1_index, v2_index = min(v1_index, v2_index), max(v1_index, v2_index)
        new_route = path[:v1_index + 1]
        segment_to_reverse = path[v1_index + 1: v2_index + 1]
        new_route.extend(segment_to_reverse[::-1])
        new_route.extend(path[v2_index + 1:])
        return new_route

    def TwoOptHeuristic(self, path):
        while True:
            improved = False
            best_dist = self.graph.calc_path_cost(path)
            num_nodes = len(path)
            for i in range(num_nodes - 1):
                for j in range(i + 1, num_nodes):
                    new_route = self._TwoOptSwap(path, i, j)
                    new_dist = self.graph.calc_path_cost(new_route)
                    if new_dist < best_dist:
                        path = new_route
                        best_dist = new_dist
                        improved = True
                        break

                if improved:
                    break
            if not improved:
                break

        return path, best_dist