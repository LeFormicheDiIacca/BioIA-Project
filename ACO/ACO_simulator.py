import heapq
import math
import multiprocessing
import random
import ctypes
import numpy as np
from ACO.Ant import Ant

#Global variables for the workers
global_shared_data = {}


def create_shared_array(np_array):
    #Create RawArray for shared memory
    c_type = np.ctypeslib.as_ctypes_type(np_array.dtype)
    shared_arr = multiprocessing.RawArray(c_type, np_array.flatten())
    return shared_arr


def init_worker(shared_map, config):
    """
    Create worker without cloning data
    """
    global global_shared_data

    #Topology CSR
    indptr_arr = np.frombuffer(shared_map['indptr'], dtype=np.int32)
    indices_arr = np.frombuffer(shared_map['indices'], dtype=np.int32)

    #Data arrays
    costs_arr = np.frombuffer(shared_map['costs'], dtype=np.float32)
    edge_ids_arr = np.frombuffer(shared_map['edge_ids'], dtype=np.int32)
    key_nodes_arr = np.frombuffer(shared_map['key_nodes_mask'], dtype=np.int32)

    #Distances
    dist_arr = np.frombuffer(shared_map['dist_matrix'], dtype=np.float32)
    dist_matrix = dist_arr.reshape((config['n_nodes'], config['n_keys']))

    #Pheromones
    pheromones_views = [
        np.frombuffer(shared_map[f'pheromones_{i}'], dtype=np.float64)
        for i in range(config['n_colonies'])
    ]

    global_shared_data = {
        'indptr': indptr_arr,
        'indices': indices_arr,
        'costs': costs_arr,
        'edge_ids': edge_ids_arr,
        'key_nodes_mask': key_nodes_arr,
        'dist_matrix': dist_matrix,
        'pheromones': pheromones_views,
        'key_nodes_list': config['key_nodes_list'],
        'n_nodes': config['n_nodes']
    }


def run_synchronized_ant(args):
    """Run an Ant with Shared Memory"""
    (alpha, beta, q0, starting_in_key_nodes, colony_id, resilience_factor,
     TSP, log_print, start_node, ant_id) = args

    global global_shared_data
    random.seed()

    try:
        ant = Ant(
            shared_data=global_shared_data,
            alpha=alpha,
            beta=beta,
            q0=q0,
            colony_id=colony_id,
            n_colonies=resilience_factor,
            ant_id=ant_id
        )

        path = ant.calculate_path(start_node, log_print=log_print, TSP=TSP)

        if path is None or len(path) == 0:
            return None

        path_cost = ant.calc_path_cost(path)
        return (path, path_cost)
    except Exception as e:
        if log_print:
            print(f"Error in ant: {e}")
            import traceback
            traceback.print_exc()
        return None


class ACO_simulator:
    def __init__(self,
                 csr_indices,
                 csr_indptr,
                 csr_data,
                 cost_array,
                 num_nodes,
                 dist_matrix,
                 key_node_to_idx,
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
        self.num_nodes = num_nodes
        self.rho = rho
        self.q0 = q0
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.ant_number = ant_number
        self.max_no_updates = max_no_updates
        self.average_cycle_length = average_cycle_length
        self.n_best_ants = n_best_ants
        self.elitism_weight = elitism_weight
        self.n_iterations_before_spawn_in_key_nodes = n_iterations_before_spawn_in_key_nodes
        self.early_stopping_threshold = early_stopping_threshold
        self.tau_min = {0: 0.0}
        self.tau_max = {0: 1.0}

        #Create CSR matrices
        self.key_nodes_list = []
        self.key_nodes_mask = np.zeros(num_nodes, dtype=np.int32)
        self.dist_matrix = dist_matrix.astype(np.float32)
        self.key_node_to_idx = key_node_to_idx


        self.indptr = csr_indptr.astype(np.int32)
        self.indices = csr_indices.astype(np.int32)
        self.data_edge_ids = csr_data.astype(np.int32)  # edge IDs from npz
        self.data_costs = cost_array.astype(np.float32)
        #Initialize shared memory
        self.shared_map = {
            'indptr': create_shared_array(self.indptr),
            'indices': create_shared_array(self.indices),
            'costs': create_shared_array(self.data_costs),
            'edge_ids': create_shared_array(self.data_edge_ids),
            'key_nodes_mask': create_shared_array(self.key_nodes_mask),
            'dist_matrix': create_shared_array(self.dist_matrix),
        }

        #Configuration for workers
        self.worker_config = {
            'n_nodes': num_nodes,
            'n_keys': 0,
            'key_nodes_list': [],
            'n_colonies': 0
        }

    def construct_key_nodes_data(self, key_nodes: list):
        self.key_nodes_list = sorted(key_nodes)
        self.key_nodes_mask = self._build_key_nodes_array()
        # dist_matrix already computed externally and passed in constructor

        self.shared_map['key_nodes_mask'] = create_shared_array(self.key_nodes_mask)
        self.shared_map['dist_matrix']    = create_shared_array(self.dist_matrix)

        self.worker_config['key_nodes_list'] = self.key_nodes_list
        self.worker_config['n_keys']         = len(self.key_nodes_list)


    def _build_key_nodes_array(self):
        arr = np.zeros(self.num_nodes, dtype=np.int32)
        for node in self.key_nodes_list:
            arr[node] = 1
        return arr


    def _calculate_min_max_pheromones(self, best_path_cost, colony_id):
        self.tau_max[colony_id] = 1.0 / ((1 - self.rho) * best_path_cost)
        tau_min = self.tau_max[colony_id] / (2 * self.num_nodes)
        self.tau_min[colony_id] = max(tau_min, 1e-10)

    def _calc_path_cost_fast(self, path):
        cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            start_idx = self.indptr[u]
            end_idx = self.indptr[u + 1]
            for k in range(start_idx, end_idx):
                if self.indices[k] == v:
                    cost += self.data_costs[k]
                    break
        return cost

    def _path_pheromone_update(self, path, pheromone_view, colony_id=0, elitism_weight=1.0):
        path_cost = self._calc_path_cost_fast(path)
        deposit = elitism_weight * self.rho / path_cost
        tau_max = self.tau_max[colony_id]

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            #Find edge id
            start_idx = self.indptr[u]
            end_idx = self.indptr[u + 1]
            for k in range(start_idx, end_idx):
                if self.indices[k] == v:
                    eid = self.data_edge_ids[k] - 1
                    if eid >= 0:
                        curr = pheromone_view[eid]
                        pheromone_view[eid] = min(curr + deposit, tau_max)
                    break

    def _global_pheromone_evaporation(self, pheromone_views, resilience_factor):
        for colony_id in range(resilience_factor):
            tau_min = self.tau_min[colony_id]
            pheromone_views[colony_id][:] = np.maximum(
                (1 - self.rho) * pheromone_views[colony_id], tau_min
            )

    def _check_convergence(self, recent_costs):
        if len(recent_costs) < 10: return False
        recent = sum(recent_costs[-10:]) / 10
        older = sum(recent_costs[-20:-10]) / 10 if len(recent_costs) >= 20 else recent
        if older == 0: return False
        return (abs(older - recent) / older) < self.early_stopping_threshold

    def simulation(self, retrieve_n_best_paths=1, log_print=False, TSP=False, resilience_factor=2):
        #Create shared pheromones
        n_edges = len(self.data_costs)
        self.worker_config['n_colonies'] = resilience_factor
        for i in range(resilience_factor):
            arr = multiprocessing.RawArray(ctypes.c_double, n_edges)
            self.shared_map[f'pheromones_{i}'] = arr
            np_arr = np.frombuffer(arr, dtype=np.float64)
            self._calculate_min_max_pheromones(self.average_cycle_length, i)
            np_arr[:] = self.tau_max[i]

        pheromone_views = [
            np.frombuffer(self.shared_map[f'pheromones_{i}'], dtype=np.float64)
            for i in range(resilience_factor)
        ]

        current_best_path_per_colony = {i: None for i in range(resilience_factor)}
        current_best_path_cost_per_colony = {i: math.inf for i in range(resilience_factor)}
        current_no_updates = {i: 0 for i in range(resilience_factor)}
        cost_history = {i: [] for i in range(resilience_factor)}
        best_paths_before_stagnation = []
        epoch = 0

        # Pool Initialization
        with multiprocessing.Pool(
                processes=multiprocessing.cpu_count(),
                initializer=init_worker,
                initargs=(self.shared_map, self.worker_config)
        ) as pool:

            while epoch < self.max_iterations:
                if log_print:
                    print(f"\n=== Epoch {epoch} started ===")

                best_ants_epoch = []

                for colony_id in range(resilience_factor):
                    #Spawn in key nodes logic
                    spawn_key = (current_no_updates[colony_id] >= self.n_iterations_before_spawn_in_key_nodes)
                    if spawn_key:
                        starts = [random.choice(self.key_nodes_list) for _ in range(self.ant_number)]
                    else:
                        starts = [random.randint(0, self.num_nodes - 1) for _ in range(self.ant_number)]

                    task_args = [
                        (self.alpha, self.beta, self.q0, spawn_key, colony_id, resilience_factor,
                         TSP, log_print, s, ant_id) for ant_id,s in enumerate(starts)
                    ]

                    #Parallel execution
                    results = pool.map(run_synchronized_ant, task_args)

                    #Save only valid results
                    valid_paths = []
                    for res in results:
                        if res:
                            valid_paths.append((*res, colony_id))

                    if not valid_paths: continue

                    #get only best paths
                    valid_paths.sort(key=lambda x: x[1])
                    best_epoch = valid_paths[:self.n_best_ants]
                    best_ants_epoch.extend(best_epoch)

                    #Colony stats
                    best_p, best_c, _ = valid_paths[0]
                    cost_history[colony_id].append(best_c)

                    # Adaptive Min/Max
                    avg_cost = sum(c for _, c, _ in best_epoch) / len(best_epoch)
                    self._calculate_min_max_pheromones(avg_cost, colony_id)

                    if best_c < current_best_path_cost_per_colony[colony_id]:
                        current_best_path_per_colony[colony_id] = best_p
                        current_best_path_cost_per_colony[colony_id] = best_c
                        current_no_updates[colony_id] = 0
                        if log_print: print(f"  Colony {colony_id} New Best: {best_c:.2f}")
                    else:
                        current_no_updates[colony_id] += 1

                #Pheromones Evaporation and updates
                self._global_pheromone_evaporation(pheromone_views, resilience_factor)

                for path, _, c_id in best_ants_epoch:
                    w = self.elitism_weight if path == current_best_path_per_colony[c_id] else 1.0
                    self._path_pheromone_update(path, pheromone_views[c_id], c_id, w)

                # Stagnation check
                for c_id in range(resilience_factor):
                    if current_no_updates[c_id] > self.max_no_updates:
                        if log_print: print(f"  Colony {c_id} Stagnation Restart")
                        #Save current best
                        bp, bc = current_best_path_per_colony[c_id], current_best_path_cost_per_colony[c_id]
                        if bp and (bp, bc) not in best_paths_before_stagnation:
                            best_paths_before_stagnation.append((bp, bc))

                        #Reset
                        pheromone_views[c_id][:] = self.tau_max[c_id]
                        current_no_updates[c_id] = 0

                #Early Stopping
                if epoch > 100:
                    conv = sum(1 for i in range(resilience_factor) if self._check_convergence(cost_history[i]))
                    if conv == resilience_factor:
                        if log_print: print("Converged.")
                        break

                epoch += 1

        #Final Collection
        for i in range(resilience_factor):
            bp, bc = current_best_path_per_colony[i], current_best_path_cost_per_colony[i]
            if bp and (bp, bc) not in best_paths_before_stagnation:
                best_paths_before_stagnation.append((bp, bc))

        return heapq.nsmallest(retrieve_n_best_paths * resilience_factor, best_paths_before_stagnation,
                               key=lambda x: x[1])