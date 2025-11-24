import math
import multiprocessing
import random

from ACO.Ant import Ant
import ctypes
from functools import partial
from meshgraph import MeshGraph

shared_pheromones_array = None


def init_worker(shared_array):

    global shared_pheromones_array
    shared_pheromones_array = shared_array


def run_synchronized_ant(args):

    graph, key_nodes, alpha, beta, rho = args

    global shared_pheromones_array

    random.seed()

    try:
        start_node = random.sample(tuple(graph.nodes()), 1)[0]
        ant = Ant(start_node, key_nodes, graph, alpha, beta, rho, shared_pheromones=shared_pheromones_array)

        path = ant.calculate_path()

        if path[-1] != start_node: return None
        visited_set = set(path)
        if not key_nodes.issubset(visited_set): return None
        path_cost = graph.calc_path_cost(path)

        return (path, path_cost)
    except Exception as e:
        print(f"Error: {e}")
        return None

class ACO_simulator:
    def __init__(self, graph: MeshGraph, key_nodes, alpha, beta, rho, ant_number = 200, max_iterations = 1000, max_no_updates = 50, Q = 5000):
        self.graph = graph
        self.rho = rho
        self.key_nodes = key_nodes
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.ant_number = ant_number
        self.max_no_updates = max_no_updates

        self.initial_pheromone = 1.0 / Q

        self.edge_mapping = {}
        idx = 0
        for u, v in self.graph.edges():
            if "edge_id" not in self.graph[u][v]:
                self.graph[u][v]["edge_id"] = idx
                if self.graph.has_edge(v, u):
                    self.graph[v][u]["edge_id"] = idx
                idx += 1
        self.num_edges = idx

        self.graph.pheromone_initialization(self.initial_pheromone)

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
        shared_pheromones = multiprocessing.Array('d', self.num_edges, lock=False)
        for i in range(self.num_edges):
            shared_pheromones[i] = self.initial_pheromone
        updated = False
        current_no_updates = 0
        n_best_ants = 5

        with multiprocessing.Pool(
            processes=multiprocessing.cpu_count(),
            initializer=init_worker,
            initargs=(shared_pheromones,)
        ) as pool:
            while epoch < self.max_iterations and current_no_updates < self.max_no_updates:

                task_args = [
                    (self.graph, self.key_nodes, self.alpha, self.beta, self.rho)
                    for _ in range(self.ant_number)
                ]

                results = pool.map(run_synchronized_ant, task_args)

                paths = [res for res in results if res is not None]

                updated = False

                for path, path_cost in paths:
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
