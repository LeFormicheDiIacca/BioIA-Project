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
        """if path and shared_pheromones_array:
            for i in range(len(path) - 1):
                s, d = path[i], path[i + 1]
                edge_id = graph[s][d]["edge_id"]
                old_val = shared_pheromones_array[edge_id]
                new_val = (1 - rho) * old_val + rho * graph.tau0
                shared_pheromones_array[edge_id] = new_val"""
        if not graph.is_valid_path(path):
            #print(f"Ant found invalid path {path}")
            return None
        path_cost = graph.calc_path_cost(path)
        return (path, path_cost)
    except Exception as e:
        print(f"Error: {e}")
        return None

class ACO_simulator:
    def __init__(self, graph: MeshGraph, alpha, beta, rho, ant_number = 200, n_best_ants = 5, max_iterations = 1000, max_no_updates = 50, average_cycle_lenght = 4000):
        self.graph = graph
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.ant_number = ant_number
        self.max_no_updates = max_no_updates
        self.shared_pheromones = None
        self.average_cycle_lenght = average_cycle_lenght
        self.graph.tau0 = 1.0 / average_cycle_lenght
        self.n_best_ants = n_best_ants
        self.tau_min = 0
        self.tau_max = 1
        self._calculate_min_max_pheromones(average_cycle_lenght)

    def _calculate_min_max_pheromones(self, best_path_cost):
        self.tau_max = 1.0 / ((1 - self.rho) * best_path_cost)
        tau_min = self.tau_max / (self.graph.number_of_nodes()*self.average_cycle_lenght)
        self.tau_min = min(tau_min, self.tau_max)

    def path_pheromone_update(self, best_path):
        path_cost = self.graph.calc_path_cost(best_path)
        deposit = self.rho / path_cost
        for i in range(len(best_path)-1):
            source, destination = best_path[i], best_path[i+1]
            edge_id = self.graph[source][destination]["edge_id"]
            curr_val = self.shared_pheromones[edge_id]
            val = curr_val + deposit
            self.shared_pheromones[edge_id] = min(deposit, self.tau_max)

    def global_pheromone_evaporation(self):
        for edge in self.graph.edges():
            edge_id = self.graph[edge[0]][edge[1]]["edge_id"]
            old_val = self.shared_pheromones[edge_id]
            new_val = (1-self.rho) * old_val
            self.shared_pheromones[edge_id] = max(new_val, self.tau_min)

    def simulation(self):
        current_best_path = None
        current_best_path_cost = math.inf
        epoch = 0
        self.shared_pheromones = multiprocessing.Array('d', self.graph.number_of_edges(), lock=False)
        for i in range(self.graph.number_of_edges()):
            self.shared_pheromones[i] = self.graph.tau0
            s, d = self.graph.edge_mapping[i]
            self.graph[s][d]["pheromones"] = self.graph.tau0
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
                self.global_pheromone_evaporation()
                if len(best_ants) > 0:
                    sum_cost = 0
                    for (path, cost) in best_ants:
                        sum_cost += cost
                    sum_cost = sum_cost / len(best_ants)
                    self._calculate_min_max_pheromones(sum_cost)
                else:
                    print(f"Epoch {epoch} has no path")

                for (path, cost) in best_ants:
                    self.path_pheromone_update(path)

                for edge in self.graph.edges():
                    edge_id = self.graph[edge[0]][edge[1]]["edge_id"]
                    pheromones = self.shared_pheromones[edge_id]
                    self.graph[edge[0]][edge[1]]["pheromones"] = pheromones
                self.graph.plot_graph_debug(draw_pheromones=True, figsize=(10,10))
                epoch += 1

        return current_best_path, current_best_path_cost
