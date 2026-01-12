import heapq
import math
import multiprocessing
import random
import time
from typing import List

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix

from ACO.Ant import Ant
from TerrainGraph.meshgraph import MeshGraph

shared_pheromones_arrays = None
shared_graph_data = None


def init_worker(shared_array, graph_data):
    """Initialize worker with shared memory"""
    global shared_pheromones_arrays, shared_graph_data
    shared_pheromones_arrays = shared_array
    shared_graph_data = graph_data


def run_synchronized_ant(args):
    """Run a single ant using pre-computed structures"""
    (alpha, beta, q0, starting_in_key_nodes, colony_id, resilience_factor,
     TSP, log_print, start_node) = args

    global shared_pheromones_arrays, shared_graph_data

    random.seed()

    try:
        ant = Ant(
            graph_data=shared_graph_data,
            shared_pheromones=shared_pheromones_arrays,
            alpha=alpha,
            beta=beta,
            q0=q0,
            colony_id=colony_id,
            n_colonies=resilience_factor,
        )

        path = ant.calculate_path(start_node, log_print=log_print, TSP=TSP)

        if path is None or len(path) == 0:
            return None

        # Calculate path cost using pre-computed data
        path_cost = ant.calc_path_cost(path)

        return (path, path_cost)
    except Exception as e:
        if log_print:
            print(f"Error in ant: {e}")
        return None


class ACO_simulator:
    """Main controller for the ant colony with optimized sparse matrices"""

    def __init__(self,
                 graph: MeshGraph,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 rho: float = 0.1,
                 q0: float = 0.05,
                 ant_number: int = 200,
                 n_best_ants: int = 5,
                 max_iterations: int = 1000,
                 max_no_updates: int = 50,
                 average_cycle_length: int = 4000,
                 n_iterations_before_spawn_in_key_nodes: int = 25,
                 elitism_weight: float = 2.0,
                 early_stopping_threshold: float = 0.001,
                 ):
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
        self.tau_min = {0: 0.0}
        self.tau_max = {0: 1.0}
        self.elitism_weight = elitism_weight
        self.n_iterations_before_spawn_in_key_nodes = n_iterations_before_spawn_in_key_nodes
        self.early_stopping_threshold = early_stopping_threshold

        # PRE-COMPUTE ALL STRUCTURES
        self.edge_costs_csr = self._build_sparse_costs_matrix()
        self.adjacency_csr = self._build_sparse_adjacency_matrix()
        self.edge_id_csr = self._build_sparse_edge_id_matrix()
        self.key_nodes_array = self._build_key_nodes_array()
        self.key_nodes_list = list(graph.key_nodes)
        self.neighbors_list = self._build_neighbors_list()
        self.dist_to_key_nodes_matrix = self._build_dist_to_key_nodes()

        # Create shared graph data dictionary
        self.graph_data = {
            'n_nodes': graph.number_of_nodes(),
            'n_edges': graph.number_of_edges(),
            'key_nodes': self.key_nodes_list,
            'key_nodes_array': self.key_nodes_array,
            'neighbors_list': self.neighbors_list,
            'edge_costs_csr': self.edge_costs_csr,
            'adjacency_csr': self.adjacency_csr,
            'edge_id_csr': self.edge_id_csr,
            'dist_to_key_nodes': self.dist_to_key_nodes_matrix,
            'edge_mapping': graph.edge_mapping,  # For pheromone updates
        }

    def _build_sparse_costs_matrix(self):
        """Build sparse cost matrix (CSR format)"""
        n = self.graph.number_of_nodes()
        costs = lil_matrix((n, n), dtype=np.float32)

        for u, v, data in self.graph.edges(data=True):
            cost = data.get('cost', 1.0)
            costs[u, v] = cost
            costs[v, u] = cost

        return costs.tocsr()

    def _build_sparse_adjacency_matrix(self):
        """Build sparse adjacency matrix"""
        n = self.graph.number_of_nodes()
        adj = lil_matrix((n, n), dtype=np.bool_)

        for u, v in self.graph.edges():
            adj[u, v] = True
            adj[v, u] = True

        return adj.tocsr()

    def _build_sparse_edge_id_matrix(self):
        """Build sparse edge ID matrix (+1 offset, 0 = no edge)"""
        n = self.graph.number_of_nodes()
        edge_ids = lil_matrix((n, n), dtype=np.int32)

        for u, v, data in self.graph.edges(data=True):
            edge_id = data['edge_id']
            edge_ids[u, v] = edge_id + 1
            edge_ids[v, u] = edge_id + 1

        return edge_ids.tocsr()

    def _build_key_nodes_array(self):
        """Build boolean array for key nodes"""
        key_array = np.zeros(self.graph.number_of_nodes(), dtype=np.int32)
        for node in self.graph.key_nodes:
            key_array[node] = 1
        return key_array

    def _build_neighbors_list(self):
        """Pre-compute neighbors list for O(1) access"""
        neighbors = {}
        for node in self.graph.nodes():
            neighbors[node] = list(self.graph[node].keys())
        return neighbors

    def _build_dist_to_key_nodes(self):
        """
        Compute distance matrix only to key nodes
        Result: (n_nodes × n_key_nodes) instead of (n_nodes × n_nodes)
        Much smaller memory footprint!
        """
        n_nodes = self.graph.number_of_nodes()
        key_list = list(self.graph.key_nodes)
        n_keys = len(key_list)

        dist_matrix = np.full((n_nodes, n_keys), np.inf, dtype=np.float32)

        print(f"Computing distances to {n_keys} key nodes...")

        for i, key_node in enumerate(key_list):
            lengths = nx.single_source_dijkstra_path_length(
                self.graph, key_node, weight='cost'
            )
            for node, dist in lengths.items():
                dist_matrix[node, i] = dist

        return dist_matrix

    def _estimate_memory_saving(self):
        """Estimate memory saving vs dense matrices"""
        n = self.graph.number_of_nodes()
        n_keys = len(self.graph.key_nodes)

        # Dense would be: 3 full matrices (n×n) + full dist matrix (n×n)
        dense_size = n * n * 4 * 4  # 4 matrices, float32

        # Sparse: CSR data + our small dist matrix
        sparse_size = (
                self.edge_costs_csr.data.nbytes +
                self.edge_costs_csr.indices.nbytes +
                self.edge_costs_csr.indptr.nbytes +
                self.adjacency_csr.data.nbytes +
                self.adjacency_csr.indices.nbytes +
                self.adjacency_csr.indptr.nbytes +
                self.edge_id_csr.data.nbytes +
                self.edge_id_csr.indices.nbytes +
                self.edge_id_csr.indptr.nbytes +
                n * n_keys * 4  # dist_to_key_nodes
        )

        return (dense_size - sparse_size) / (1024 * 1024)

    def _calculate_min_max_pheromones(self, best_path_cost, colony_id: int = 0):
        """Calculate adaptive tau_min and tau_max"""
        self.tau_max[colony_id] = 1.0 / ((1 - self.rho) * best_path_cost)
        tau_min = self.tau_max[colony_id] / (2 * self.graph.number_of_nodes())
        self.tau_min[colony_id] = max(tau_min, 1e-10)

    def _path_pheromone_update(self, path, pheromone_view, colony_id: int = 0,
                               elitism_weight: float = 1.0):
        """Update pheromones along path using NumPy view"""
        path_cost = self._calc_path_cost_fast(path)
        deposit = elitism_weight * self.rho / path_cost
        tau_max = self.tau_max[colony_id]

        for i in range(len(path) - 1):
            source, destination = path[i], path[i + 1]
            edge_id = int(self.edge_id_csr[source, destination]) - 1

            if edge_id >= 0:
                curr_val = pheromone_view[edge_id]
                pheromone_view[edge_id] = min(curr_val + deposit, tau_max)

    def _calc_path_cost_fast(self, path):
        """Fast path cost calculation using CSR matrix"""
        cost = 0.0
        for i in range(len(path) - 1):
            cost += self.edge_costs_csr[path[i], path[i + 1]]
        return cost

    def _global_pheromone_evaporation(self, pheromone_views, resilience_factor: int = 2):
        """Evaporate pheromones using NumPy views"""
        for colony_id in range(resilience_factor):
            tau_min = self.tau_min[colony_id]
            pheromone_views[colony_id][:] = np.maximum(
                (1 - self.rho) * pheromone_views[colony_id],
                tau_min
            )

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

    def simulation(self, retrieve_n_best_paths: int = 1, log_print: bool = False,
                   draw_heatmap: bool = False, TSP: bool = False, resilience_factor: int = 2):
        """Run the colony simulation"""
        current_best_path_per_colony = {colony_id: None for colony_id in range(resilience_factor)}
        current_best_path_cost_per_colony = {colony_id: math.inf for colony_id in range(resilience_factor)}
        epoch = 0
        current_no_updates_per_colony = {colony_id: 0 for colony_id in range(resilience_factor)}
        best_paths_before_stagnation = []
        cost_history = {i: [] for i in range(resilience_factor)}

        # Initialize pheromones
        self.shared_pheromones_arrays = [
            multiprocessing.Array('d', self.graph.number_of_edges(), lock=False)
            for _ in range(resilience_factor)
        ]

        # Create NumPy views
        pheromone_views = [np.frombuffer(arr) for arr in self.shared_pheromones_arrays]

        for colony_id in range(resilience_factor):
            self._calculate_min_max_pheromones(self.average_cycle_length, colony_id)
            tau_max = self.tau_max[colony_id]
            pheromone_views[colony_id][:] = tau_max

        # MultiProcessing Pool
        with multiprocessing.Pool(
                processes=multiprocessing.cpu_count(),
                initializer=init_worker,
                initargs=(self.shared_pheromones_arrays, self.graph_data)
        ) as pool:
            while epoch < self.max_iterations:
                best_ants_all_colonies = []

                if log_print:
                    print(f"\n=== Epoch {epoch} started ===")
                    epoch_start = time.perf_counter()

                for colony_id in range(resilience_factor):
                    if log_print:
                        print(f"  Colony {colony_id} starting")
                        start_time = time.perf_counter()

                    # Determine starting nodes
                    spawn_in_key_nodes = (
                            current_no_updates_per_colony[colony_id] >=
                            self.n_iterations_before_spawn_in_key_nodes
                    )

                    if spawn_in_key_nodes:
                        start_nodes = [random.choice(self.key_nodes_list)
                                       for _ in range(self.ant_number)]
                    else:
                        start_nodes = [random.randint(0, self.graph.number_of_nodes() - 1)
                                       for _ in range(self.ant_number)]

                    # Create task arguments (no graph passed!)
                    task_args = [
                        (self.alpha, self.beta, self.q0, spawn_in_key_nodes,
                         colony_id, resilience_factor, TSP, log_print, start_node)
                        for start_node in start_nodes
                    ]

                    # Run ants in parallel
                    results = pool.map(run_synchronized_ant, task_args)
                    paths = [
                        (path, path_cost, colony_id)
                        for item in results
                        if item is not None
                        for (path, path_cost) in [item]
                        if path is not None
                    ]

                    if log_print:
                        res_time = time.perf_counter() - start_time
                        print(f"  Colony {colony_id} finished in {res_time:.2f}s, "
                              f"valid paths: {len(paths)}/{self.ant_number}")

                    if not paths:
                        if log_print:
                            print(f"  Colony {colony_id}: No valid paths found")
                        continue

                    # Select best ants
                    paths.sort(key=lambda x: x[1])
                    best_ants = paths[:self.n_best_ants]
                    best_ants_all_colonies.extend(best_ants)

                    # Update pheromone bounds
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

                # Evaporate pheromones
                self._global_pheromone_evaporation(pheromone_views, resilience_factor)

                # Lay pheromones on best paths
                for path, cost, colony_id in best_ants_all_colonies:
                    weight = (self.elitism_weight
                              if path == current_best_path_per_colony[colony_id]
                              else 1.0)
                    self._path_pheromone_update(path, pheromone_views[colony_id],
                                                colony_id, weight)

                # Handle stagnation
                for colony_id in range(resilience_factor):
                    if current_no_updates_per_colony[colony_id] > self.max_no_updates:
                        if log_print:
                            print(f"  Colony {colony_id}: Restarting due to stagnation")

                        best_path = current_best_path_per_colony[colony_id]
                        best_cost = current_best_path_cost_per_colony[colony_id]
                        if (best_path, best_cost) not in best_paths_before_stagnation:
                            best_paths_before_stagnation.append((best_path, best_cost))

                        # Reset pheromones
                        tau_max = self.tau_max[colony_id]
                        pheromone_views[colony_id][:] = tau_max
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
                    epoch_time = time.perf_counter() - epoch_start
                    print(f"=== Epoch {epoch} finished in {epoch_time:.2f}s ===")

                epoch += 1

        # Collect final best paths
        for colony_id in range(resilience_factor):
            best_path = current_best_path_per_colony[colony_id]
            best_cost = current_best_path_cost_per_colony[colony_id]
            if best_path and (best_path, best_cost) not in best_paths_before_stagnation:
                best_paths_before_stagnation.append((best_path, best_cost))

        # Return top paths
        n_paths = retrieve_n_best_paths * resilience_factor
        return heapq.nsmallest(n_paths, best_paths_before_stagnation, key=lambda x: x[1])