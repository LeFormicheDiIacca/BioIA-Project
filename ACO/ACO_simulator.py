import heapq
import math
import multiprocessing
import random
import time

from ACO.Ant import Ant
from meshgraph import MeshGraph

shared_pheromones_arrays = None


def init_worker(shared_array):
    "So we can share the pheromones among the processes"
    global shared_pheromones_arrays
    shared_pheromones_arrays = shared_array


def run_synchronized_ant(args):
    "Used to actually operate the ant in the multiprocessing pool"
    graph, alpha, beta, q0, starting_in_key_nodes, colony_id, resilience_factor = args

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

        path = ant.calculate_path(start_node)
        if not graph.is_valid_path(path):
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
        self.n_iterations_before_spawn_in_key_nodes = n_iterations_before_spawn_in_key_nodes

    def _calculate_min_max_pheromones(self, best_path_cost, colony_id: int = 0):
        """
        Used to estimate the adpative values of tau_min and tau_max
        :param best_path_cost: Current best path cost
        """
        self.tau_max[colony_id] = 1.0 / ((1 - self.rho) * best_path_cost)
        tau_min = self.tau_max[colony_id] / (self.graph.number_of_nodes()*self.average_cycle_length)
        self.tau_min[colony_id] = min(tau_min, self.tau_max[colony_id])

    def _path_pheromone_update(self, path, colony_id:int = 0):
        "Given a path, we lay pheromones on the edges in the path"
        path_cost = self.graph.calc_path_cost(path)
        deposit = self.rho / path_cost
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
            for edge in self.graph.edges():
                edge_id = self.graph[edge[0]][edge[1]]["edge_id"]
                old_val = self.shared_pheromones_arrays[colony_id][edge_id]
                new_val = (1-self.rho) * old_val
                #Pheromones min level clamped using tau_min
                self.shared_pheromones_arrays[colony_id][edge_id] = max(new_val, self.tau_min[colony_id])

    def simulation(self, retrieve_n_best_paths: int = 1, draw_heatmap: bool = False, resilience_factor: int = 2):
        """
        Actual Colony simulation
        :param retrieve_n_best_paths: Number of paths to return
        :param draw_heatmap: If true, each epoch will generate a heatmap of the pheromones
        :return: A list of paths
        """
        current_best_path_per_colony = {colony_id: None for colony_id in range(resilience_factor)}
        current_best_path_cost_per_colony = {colony_id: math.inf for colony_id in range(resilience_factor)}
        epoch = 0
        current_no_updates_per_colony = {colony_id: 0 for colony_id in range(resilience_factor)}
        best_paths_before_stagnation = []

        #Initialize pheromones in graph and shared_array
        self.shared_pheromones_arrays = [
            multiprocessing.Array('d', self.graph.number_of_edges(), lock=False)
            for _ in range(resilience_factor)
        ]
        for colony_id in range(resilience_factor):
            self._calculate_min_max_pheromones(self.average_cycle_length, colony_id)
            for i in range(self.graph.number_of_edges()):
                self.shared_pheromones_arrays[colony_id][i] = self.tau_max[colony_id]
                s, d = self.graph.edge_mapping[i]
                self.graph[s][d][f"pheromones_{colony_id}"] = self.tau_max[colony_id]
        res_all_colonies = []
        #MultiProcessing Pool used to get true parallelism. Python is pupù
        with multiprocessing.Pool(
            processes=multiprocessing.cpu_count(),
            initializer=init_worker,
            initargs=(self.shared_pheromones_arrays,)
        ) as pool:
            while epoch < self.max_iterations:
                best_ants_all_colonies = []

                print(f"\n=== Epoch {epoch} started ===")

                for colony_id in range(resilience_factor):
                    print(f"      Colony {colony_id} starting")
                    start_time = time.perf_counter()
                    if current_no_updates_per_colony[colony_id] >= self.n_iterations_before_spawn_in_key_nodes:
                        spawn_in_key_nodes = True
                    else:
                        spawn_in_key_nodes = False

                    #Parameters for ant creations
                    task_args = [
                        (self.graph, self.alpha, self.beta, self.q0, spawn_in_key_nodes, colony_id, resilience_factor)
                        for _ in range(self.ant_number)
                    ]

                    #Create and run the ants. We remove all empty paths
                    results = pool.map(run_synchronized_ant, task_args)
                    paths = [(path, path_cost, colony_id) for (path, path_cost) in results if path is not None]
                    res_all_colonies += paths
                    res_time = time.perf_counter() - start_time

                    print(f"      Colony {colony_id} finished in {res_time} seconds")

                    #We find the best ants and use them to adapt tau_min and tau_max for the colony pheromones
                    paths.sort(key=lambda x: x[1])
                    best_ants = paths[:self.n_best_ants]
                    best_ants_all_colonies += best_ants

                    if len(best_ants) > 0:
                        sum_cost = 0
                        for (path, cost, _) in best_ants:
                            sum_cost += cost
                        sum_cost = sum_cost / len(best_ants)
                        self._calculate_min_max_pheromones(sum_cost, colony_id)
                    else:
                        print(f"Epoch {epoch}: Colony {colony_id} has no path")

                    #We check if a better route has been found
                    updated = False
                    for path, path_cost, _ in paths:
                        if path_cost < current_best_path_cost_per_colony[colony_id]:
                            current_best_path_per_colony[colony_id] = path
                            current_best_path_cost_per_colony[colony_id] = path_cost
                            updated = True
                    if updated:
                        current_no_updates_per_colony[colony_id] = 0
                    else:
                        current_no_updates_per_colony[colony_id] += 1


                #Evaporate pheromones
                self._global_pheromone_evaporation()
                #Laying pheromones on best paths
                for (path, cost, colony_id) in best_ants_all_colonies:
                    self._path_pheromone_update(path, colony_id)

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
                        print(f"Stagnation. Restarting pheromones to tau_max for colony {colony_id}")
                        current_no_updates_per_colony[colony_id] = 0
                        for i in range(self.graph.number_of_edges()):
                            self.shared_pheromones_arrays[colony_id][i] = self.tau_max[colony_id]
                            s, d = self.graph.edge_mapping[i]
                            self.graph[s][d][f"pheromones_{colony_id}"] = self.tau_max[colony_id]
                        #Best path before stagnation is saved in order to have multiple best possible paths
                        if (current_best_path_per_colony[colony_id], current_best_path_cost_per_colony[colony_id]) not in best_paths_before_stagnation:
                            best_paths_before_stagnation.append((current_best_path_per_colony[colony_id], current_best_path_cost_per_colony[colony_id]))

                end_time = time.perf_counter() - start_time
                print(f"\n=== Epoch {epoch} finished in {end_time} seconds ===")
                epoch += 1

        for colony_id in range(resilience_factor):
            #Saving best path
            if (current_best_path_per_colony[colony_id], current_best_path_cost_per_colony[colony_id]) not in best_paths_before_stagnation:
                best_paths_before_stagnation.append((current_best_path_per_colony[colony_id], current_best_path_cost_per_colony[colony_id]))
        #Returning the n best paths
        return heapq.nsmallest(retrieve_n_best_paths*resilience_factor, best_paths_before_stagnation, key=lambda x: x[1])
