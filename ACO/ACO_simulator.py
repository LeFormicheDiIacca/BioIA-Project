import heapq
import math
import multiprocessing
import random
import time
from typing import List

import numpy as np

from ACO.Ant import Ant
from meshgraph import MeshGraph

shared_pheromones_arrays = None


def init_worker(shared_array):
    "So we can share the pheromones among the processes"
    global shared_pheromones_arrays
    shared_pheromones_arrays = shared_array


def run_synchronized_ant(args):
    "Used to actually operate the ant in the multiprocessing pool"
    graph, alpha, beta, q0, starting_in_key_nodes, colony_id, resilience_factor, TSP, log_print = args

    global shared_pheromones_arrays

    random.seed()

    try:
        if starting_in_key_nodes:
            start_node = random.sample(tuple(graph.key_nodes), 1)[0]
        else:
            start_node = random.sample(tuple(graph.nodes()), 1)[0]

        ant = Ant(
            graph= graph,
            shared_pheromones=shared_pheromones_arrays,
            alpha = alpha,
            beta = beta,
            q0 = q0,
            colony_id = colony_id,
            n_colonies = resilience_factor,
        )

        path = ant.calculate_path(start_node, log_print=log_print, TSP=TSP)
        if TSP and not graph.is_valid_path(path):
            if log_print:
                print(f"Ant found invalid path {path}")
            return None
        path_cost = graph.calc_path_cost(path)
        return (path, path_cost)
    except Exception as e:
        print(f"Error: {e}")
        return None

class ACO_simulator:
    """
    Main controller for the ant colony
    """
    def __init__(self,
                 graph: MeshGraph,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 rho: float = 0.1,
                 q0: float = 0.05,
                 ant_number:int = 200,
                 n_best_ants:int = 5,
                 max_iterations: int = 1000,
                 max_no_updates: int = 50,
                 average_cycle_length: int = 4000,
                 n_iterations_before_spawn_in_key_nodes: int = 25,
                 elitism_weight: float = 2.0,
                 early_stopping_threshold: float = 0.001,
    ):
        """
        :param graph: MeshGraph to explore
        :param alpha: Influence of pheromones in ants
        :param beta: Influence of edge costs in ants
        :param rho: Pheromones evaporation rates
        :param q0: Ant random exploration chance
        :param ant_number: Number of ants in the simulation
        :param n_best_ants: Number of best ats selected for the global pheromones update
        :param max_iterations: Maximum number of iterations before convergence
        :param max_no_updates: Maximum number of iterations with no update accepted before restarting
        :param average_cycle_length: Average length of optimal TSP cycles in the graph used to estimate tau
        :param n_iterations_before_spawn_in_key_nodes: After this number of iterations, ants will directly spawn in key_nodes in order to favour exploitation
        :param elitism_weight: Weight for best solution pheromone deposit
        :param early_stopping_threshold: Convergence threshold (% improvement)
        """
        self.graph = graph
        self.rho = rho
        self.q0 = q0
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.ant_number = ant_number
        self.max_no_updates = max_no_updates
        self.shared_pheromones_arrays = None
        self.average_cycle_length = average_cycle_length
        self.n_best_ants = n_best_ants
        self.tau_min = {0:0.0}
        self.tau_max = {0:1.0}
        self.elitism_weight = elitism_weight
        self.n_iterations_before_spawn_in_key_nodes = n_iterations_before_spawn_in_key_nodes
        self.early_stopping_threshold = early_stopping_threshold

    def _calculate_min_max_pheromones(self, best_path_cost, colony_id: int = 0):
        """
        Used to estimate the adpative values of tau_min and tau_max
        :param best_path_cost: Current best path cost
        """
        self.tau_max[colony_id] = 1.0 / ((1 - self.rho) * best_path_cost)
        tau_min = self.tau_max[colony_id] / (2 * self.graph.number_of_nodes())
        self.tau_min[colony_id] = max(tau_min, 1e-10)

    def _path_pheromone_update(self, path, colony_id:int = 0, elitism_weight: float = 1.0):
        "Given a path, we lay pheromones on the edges in the path"
        path_cost = self.graph.calc_path_cost(path)
        deposit = elitism_weight * self.rho / path_cost

        for i in range(len(path)-1):
            source, destination = path[i], path[i+1]
            edge_id = self.graph[source][destination]["edge_id"]
            curr_val = self.shared_pheromones_arrays[colony_id][edge_id]
            val = curr_val + deposit
            #Pheromones max level clamped using tau_max
            self.shared_pheromones_arrays[colony_id][edge_id] = min(val, self.tau_max[colony_id])

    def _global_pheromone_evaporation(self, resilience_factor: int = 2):
        "Each epoch we must evaporate the pheromones in the edges"
        for colony_id in range(resilience_factor):
            tau_min = self.tau_min[colony_id]
            arr = np.frombuffer(self.shared_pheromones_arrays[colony_id])
            arr[:] = np.maximum((1 - self.rho) * arr, tau_min)

    def _check_convergence(self, recent_costs: List[float]) -> bool:
        """Check if algorithm has converged"""
        if len(recent_costs) < 10:
            return False

        recent_avg = sum(recent_costs[-10:]) / 10
        older_avg = sum(recent_costs[-20:-10]) / 10 if len(recent_costs) >= 20 else recent_avg

        if older_avg == 0:
            return False

        improvement = abs(older_avg - recent_avg) / older_avg
        return improvement < self.early_stopping_threshold

    def simulation(self, retrieve_n_best_paths: int = 1, log_print: bool = False, draw_heatmap: bool = False, TSP: bool = False, resilience_factor: int = 2):
        """
        Actual Colony simulation
        :param retrieve_n_best_paths: Number of paths to return
        :param draw_heatmap: If true, each epoch will generate a heatmap of the pheromones
        :param TSP: If true, use TSP algorithm, else Steiner Tree
        :return: A list of paths
        """
        current_best_path_per_colony = {colony_id: None for colony_id in range(resilience_factor)}
        current_best_path_cost_per_colony = {colony_id: math.inf for colony_id in range(resilience_factor)}
        epoch = 0
        current_no_updates_per_colony = {colony_id: 0 for colony_id in range(resilience_factor)}
        best_paths_before_stagnation = []
        cost_history = {i: [] for i in range(resilience_factor)}

        #Initialize pheromones in graph and shared_array
        self.shared_pheromones_arrays = [
            multiprocessing.Array('d', self.graph.number_of_edges(), lock=False)
            for _ in range(resilience_factor)
        ]
        # Initialize pheromones
        for colony_id in range(resilience_factor):
            self._calculate_min_max_pheromones(self.average_cycle_length, colony_id)
            tau_max = self.tau_max[colony_id]
            for i in range(self.graph.number_of_edges()):
                self.shared_pheromones_arrays[colony_id][i] = tau_max
                s, d = self.graph.edge_mapping[i]
                self.graph[s][d][f"pheromones_{colony_id}"] = tau_max

        #MultiProcessing Pool used to get true parallelism. Python is pupù
        with multiprocessing.Pool(
            processes=multiprocessing.cpu_count(),
            initializer=init_worker,
            initargs=(self.shared_pheromones_arrays,)
        ) as pool:
            while epoch < self.max_iterations:
                best_ants_all_colonies = []
                if log_print:
                    print(f"\n=== Epoch {epoch} started ===")

                for colony_id in range(resilience_factor):

                    if log_print:
                        print(f"      Colony {colony_id} starting")
                        start_time = time.perf_counter()

                    #Parameters for ant creations
                    spawn_in_key_nodes = (current_no_updates_per_colony[colony_id] >= self.n_iterations_before_spawn_in_key_nodes)

                    task_args = [
                        (self.graph, self.alpha, self.beta, self.q0, spawn_in_key_nodes, colony_id, resilience_factor, TSP, log_print)
                        for _ in range(self.ant_number)
                    ]

                    #Create and run the ants. We remove all empty paths
                    results = pool.map(run_synchronized_ant, task_args)
                    paths = [(path, path_cost, colony_id)
                             for item in results
                             if item is not None
                             for (path, path_cost) in [item]
                             if path is not None]
                    if log_print:
                        res_time = time.perf_counter() - start_time
                        print(f"      Colony {colony_id} finished in {res_time} seconds")

                    if not paths:
                        if log_print:
                            print(f"  Colony {colony_id}: No valid paths found")

                        continue

                    #We find the best ants and use them to adapt tau_min and tau_max for the colony pheromones
                    paths.sort(key=lambda x: x[1])
                    best_ants = paths[:self.n_best_ants]
                    best_ants_all_colonies.extend(best_ants)

                    #Update pheromones bounds
                    avg_best_cost = sum(cost for _, cost, _ in best_ants) / len(best_ants)
                    self._calculate_min_max_pheromones(avg_best_cost, colony_id)

                    # Track improvements
                    best_path, best_cost, _ = paths[0]
                    cost_history[colony_id].append(best_cost)

                    if best_cost < current_best_path_cost_per_colony[colony_id]:
                        current_best_path_per_colony[colony_id] = best_path
                        current_best_path_cost_per_colony[colony_id] = best_cost
                        current_no_updates_per_colony[colony_id] = 0

                        if log_print:
                            print(f"  Colony {colony_id}: New best = {best_cost:.2f}")
                    else:
                        current_no_updates_per_colony[colony_id] += 1

                #Evaporate pheromones
                self._global_pheromone_evaporation(resilience_factor = resilience_factor)
                #Laying pheromones on best paths
                for path, cost, colony_id in best_ants_all_colonies:
                    weight = self.elitism_weight if path == current_best_path_per_colony[colony_id] else 1.0
                    self._path_pheromone_update(path, colony_id, weight)

                #Sincronize pheromones with real graph
                for colony_id in range(resilience_factor):
                    for edge in self.graph.edges():
                        edge_id = self.graph[edge[0]][edge[1]]["edge_id"]
                        pheromones = self.shared_pheromones_arrays[colony_id][edge_id]
                        self.graph[edge[0]][edge[1]][f"pheromones_{colony_id}"] = pheromones

                #Create a pheromone heatmap in order to check it
                if draw_heatmap:
                    self.graph.plot_graph_debug(draw_pheromones=True, draw_labels=True, figsize=(50,50), epoch=epoch)
                #If we detect stagnation the pheromones are restored to tau_max
                for colony_id in range(resilience_factor):
                    if current_no_updates_per_colony[colony_id] > self.max_no_updates:
                        if log_print:
                            print(f"  Colony {colony_id}: Restarting due to stagnation")

                        # Save current best
                        best_path = current_best_path_per_colony[colony_id]
                        best_cost = current_best_path_cost_per_colony[colony_id]
                        if (best_path, best_cost) not in best_paths_before_stagnation:
                            best_paths_before_stagnation.append((best_path, best_cost))

                        # Reset pheromones
                        tau_max = self.tau_max[colony_id]
                        for i in range(self.graph.number_of_edges()):
                            self.shared_pheromones_arrays[colony_id][i] = tau_max
                            s, d = self.graph.edge_mapping[i]
                            self.graph[s][d][f"pheromones_{colony_id}"] = tau_max

                        current_no_updates_per_colony[colony_id] = 0

                # Early stopping check
                if epoch > 100:
                    converged_colonies = sum(
                        1 for cid in range(resilience_factor)
                        if self._check_convergence(cost_history[cid])
                    )
                    if converged_colonies == resilience_factor:
                        if log_print:
                            print(f"\nEarly stopping at epoch {epoch}: All colonies converged")
                        break

                if log_print:
                    end_time = time.perf_counter() - start_time
                    print(f"\n=== Epoch {epoch} finished in {end_time} seconds ===")

                epoch += 1

        for colony_id in range(resilience_factor):
            best_path = current_best_path_per_colony[colony_id]
            best_cost = current_best_path_cost_per_colony[colony_id]
            if best_path and (best_path, best_cost) not in best_paths_before_stagnation:
                best_paths_before_stagnation.append((best_path, best_cost))

        # Return top paths
        n_paths = retrieve_n_best_paths * resilience_factor
        return heapq.nsmallest(n_paths, best_paths_before_stagnation, key=lambda x: x[1])