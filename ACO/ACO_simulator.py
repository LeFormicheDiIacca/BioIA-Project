import math
import multiprocessing
import random

from ACO.Ant import Ant
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
        ant = Ant(start_node, graph, alpha, beta, rho, shared_pheromones=shared_pheromones_array)

        path = ant.calculate_path()

        if not graph.is_valid_path(path):
            #print(f"Ant found invalid path {path}")
            return None
        path_cost = graph.calc_path_cost(path)
        return (path, path_cost)
    except Exception as e:
        print(f"Error: {e}")
        return None

class ACO_simulator:
    def __init__(self, graph: MeshGraph, alpha, beta, rho, ant_number = 200, n_best_ants = 5, max_iterations = 1000, max_no_updates = 50, average_cycle_lenght = 5000):
        self.graph = graph
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.ant_number = ant_number
        self.max_no_updates = max_no_updates
        self.shared_pheromones = None
        self.initial_pheromone = 1.0 / average_cycle_lenght
        self.graph.tau0 = self.initial_pheromone
        self.n_best_ants = n_best_ants

    def global_pheromone_update(self, best_path):
        path_cost = self.graph.calc_path_cost(best_path)
        for i in range(len(best_path)-1):
            source, destination = best_path[i], best_path[i+1]
            edge_id = self.graph[source][destination]["edge_id"]
            old_val = self.shared_pheromones[edge_id]
            new_val = (1 - self.rho) * old_val + self.rho/path_cost
            self.shared_pheromones[edge_id] = new_val


    def simulation(self):
        current_best_path = None
        current_best_path_cost = math.inf
        epoch = 0
        self.shared_pheromones = multiprocessing.Array('d', self.graph.number_of_edges(), lock=False)
        for i in range(self.graph.number_of_edges()):
            self.shared_pheromones[i] = self.initial_pheromone
            s, d = self.graph.edge_mapping[i]
            self.graph[s][d]["pheromones"] = self.initial_pheromone
        updated = False
        current_no_updates = 0
        n_best_ants = 5

        with multiprocessing.Pool(
            processes=multiprocessing.cpu_count(),
            initializer=init_worker,
            initargs=(self.shared_pheromones,)
        ) as pool:
            while epoch < self.max_iterations and current_no_updates < self.max_no_updates:

                task_args = [
                    (self.graph, self.graph.key_nodes, self.alpha, self.beta, self.rho)
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
                best_ants = paths[:self.n_best_ants]

                for (path, cost) in best_ants:
                    self.global_pheromone_update(path)

                for edge in self.graph.edges():
                    edge_id = self.graph[edge[0]][edge[1]]["edge_id"]
                    pheromones = self.shared_pheromones[edge_id]
                    self.graph[edge[0]][edge[1]]["pheromones"] = pheromones
                self.graph.plot_graph_debug(draw_pheromones=True, figsize=(10,10))
                epoch += 1

        return current_best_path, current_best_path_cost
