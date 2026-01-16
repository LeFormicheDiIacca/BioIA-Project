import heapq
import math
import multiprocessing
import random
import time
import ctypes
from typing import List, Tuple

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix

from ACO.Ant import Ant
from TerrainGraph.meshgraph import MeshGraph

# Variabili globali per i worker (tengono le views numpy sulla memoria condivisa)
global_shared_data = {}


def create_shared_array(np_array):
    """Crea un RawArray compatibile con multiprocessing dai dati numpy"""
    c_type = np.ctypeslib.as_ctypes_type(np_array.dtype)
    # RawArray non ha lock, è più veloce e leggero
    shared_arr = multiprocessing.RawArray(c_type, np_array.flatten())
    return shared_arr


def init_worker(shared_map, config):
    """
    Inizializza il worker ricostruendo le views numpy dai buffer condivisi.
    Non alloca memoria extra per i dati del grafo.
    """
    global global_shared_data

    # Ricostruiamo le views numpy dai buffer condivisi
    # Topology (CSR standard: indptr punta ai blocchi in indices)
    indptr_arr = np.frombuffer(shared_map['indptr'], dtype=np.int32)
    indices_arr = np.frombuffer(shared_map['indices'], dtype=np.int32)

    # Data arrays
    costs_arr = np.frombuffer(shared_map['costs'], dtype=np.float32)
    edge_ids_arr = np.frombuffer(shared_map['edge_ids'], dtype=np.int32)
    key_nodes_arr = np.frombuffer(shared_map['key_nodes_mask'], dtype=np.int32)

    # Distanze (Flattened -> Reshaped)
    dist_arr = np.frombuffer(shared_map['dist_matrix'], dtype=np.float32)
    dist_matrix = dist_arr.reshape((config['n_nodes'], config['n_keys']))

    # Feromoni (Lista di views)
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
    """Esegue una formica usando i dati globali condivisi"""
    (alpha, beta, q0, starting_in_key_nodes, colony_id, resilience_factor,
     TSP, log_print, start_node, ant_id) = args

    global global_shared_data
    random.seed()

    try:
        # Passiamo direttamente il dizionario globale che contiene le views
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
        self.average_cycle_length = average_cycle_length
        self.n_best_ants = n_best_ants
        self.elitism_weight = elitism_weight
        self.n_iterations_before_spawn_in_key_nodes = n_iterations_before_spawn_in_key_nodes
        self.early_stopping_threshold = early_stopping_threshold
        self.tau_min = {0: 0.0}
        self.tau_max = {0: 1.0}

        # --- 1. COSTRUZIONE MATRICI SPARSE ---
        # Usiamo cost matrix come riferimento topologico
        costs_csr = self._build_sparse_costs_matrix()
        self.key_nodes_list = list(self.graph.key_nodes)
        self.key_nodes_mask = self._build_key_nodes_array()
        self.dist_matrix = self._build_dist_to_key_nodes()

        # Estraiamo i dati grezzi numpy
        self.indptr = costs_csr.indptr.astype(np.int32)
        self.indices = costs_csr.indices.astype(np.int32)
        self.data_costs = costs_csr.data.astype(np.float32)

        # Costruiamo array paralleli per Edge IDs usando la STESSA topologia
        self.data_edge_ids = self._build_aligned_edge_ids(costs_csr)

        # --- 2. MEMORIA CONDIVISA ---
        # Creiamo dizionario di RawArrays da passare a init_worker
        self.shared_map = {
            'indptr': create_shared_array(self.indptr),
            'indices': create_shared_array(self.indices),
            'costs': create_shared_array(self.data_costs),
            'edge_ids': create_shared_array(self.data_edge_ids),
            'key_nodes_mask': create_shared_array(self.key_nodes_mask),
            'dist_matrix': create_shared_array(self.dist_matrix),
        }

        # Configurazione leggera per worker
        self.worker_config = {
            'n_nodes': graph.number_of_nodes(),
            'n_keys': len(self.key_nodes_list),
            'key_nodes_list': self.key_nodes_list,
            'n_colonies': 0  # Sarà aggiornato in simulation
        }

    def construct_key_nodes_data(self, key_nodes):
        self.graph.assign_key_nodes(key_nodes)
        # Distanze e Key Nodes
        self.key_nodes_list = list(self.graph.key_nodes)
        self.key_nodes_mask = self._build_key_nodes_array()
        self.dist_matrix = self._build_dist_to_key_nodes()

    def _build_sparse_costs_matrix(self):
        n = self.graph.number_of_nodes()
        costs = lil_matrix((n, n), dtype=np.float32)
        for u, v, data in self.graph.edges(data=True):
            cost = data.get('cost', 1.0)
            costs[u, v] = cost
            costs[v, u] = cost
        return costs.tocsr()

    def _build_aligned_edge_ids(self, ref_csr):
        """Costruisce array di Edge ID allineato con indices del CSR dei costi"""
        n_edges = len(ref_csr.data)
        edge_ids = np.zeros(n_edges, dtype=np.int32)

        # Iteriamo sugli stessi indici del CSR
        rows, cols = ref_csr.nonzero()
        for i, (u, v) in enumerate(zip(rows, cols)):
            eid = self.graph[u][v]['edge_id'] + 1
            edge_ids[i] = eid

        return edge_ids

    def _build_key_nodes_array(self):
        arr = np.zeros(self.graph.number_of_nodes(), dtype=np.int32)
        for node in self.graph.key_nodes:
            arr[node] = 1
        return arr

    def _build_dist_to_key_nodes(self):
        n_nodes = self.graph.number_of_nodes()
        key_list = self.key_nodes_list
        n_keys = len(key_list)
        dist_matrix = np.full((n_nodes, n_keys), np.inf, dtype=np.float32)
        for i, key_node in enumerate(key_list):
            lengths = nx.single_source_dijkstra_path_length(
                self.graph, key_node, weight='cost'
            )
            for node, dist in lengths.items():
                dist_matrix[node, i] = dist
        return dist_matrix

    def _calculate_min_max_pheromones(self, best_path_cost, colony_id):
        self.tau_max[colony_id] = 1.0 / ((1 - self.rho) * best_path_cost)
        tau_min = self.tau_max[colony_id] / (2 * self.graph.number_of_nodes())
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
            # Trova edge ID
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

    def simulation(self, retrieve_n_best_paths=1, log_print=False, draw_heatmap=False, TSP=False, resilience_factor=2):
        # Setup Feromoni Shared
        n_edges = self.graph.number_of_edges()

        # Aggiungiamo i feromoni alla shared map
        self.worker_config['n_colonies'] = resilience_factor
        for i in range(resilience_factor):
            # Allocazione memoria condivisa per feromoni
            arr = multiprocessing.RawArray(ctypes.c_double, n_edges)
            self.shared_map[f'pheromones_{i}'] = arr

            # Inizializzazione
            np_arr = np.frombuffer(arr, dtype=np.float64)
            self._calculate_min_max_pheromones(self.average_cycle_length, i)
            np_arr[:] = self.tau_max[i]

        # Viste locali per l'aggiornamento nel processo padre
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
                    t_start = time.perf_counter()

                best_ants_epoch = []

                for colony_id in range(resilience_factor):
                    # Logica spawn
                    spawn_key = (current_no_updates[colony_id] >= self.n_iterations_before_spawn_in_key_nodes)
                    if spawn_key:
                        starts = [random.choice(self.key_nodes_list) for _ in range(self.ant_number)]
                    else:
                        starts = [random.randint(0, self.graph.number_of_nodes() - 1) for _ in range(self.ant_number)]

                    task_args = [
                        (self.alpha, self.beta, self.q0, spawn_key, colony_id, resilience_factor,
                         TSP, log_print, s, ant_id) for ant_id,s in enumerate(starts)
                    ]

                    # Esecuzione parallela
                    results = pool.map(run_synchronized_ant, task_args)

                    # Filtra risultati validi
                    valid_paths = []
                    for res in results:
                        if res:
                            valid_paths.append((*res, colony_id))

                    if not valid_paths: continue

                    # Ordina e prendi i migliori
                    valid_paths.sort(key=lambda x: x[1])
                    best_epoch = valid_paths[:self.n_best_ants]
                    best_ants_epoch.extend(best_epoch)

                    # Statistiche Colony
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

                # Evaporazione e Aggiornamento
                self._global_pheromone_evaporation(pheromone_views, resilience_factor)

                for path, _, c_id in best_ants_epoch:
                    w = self.elitism_weight if path == current_best_path_per_colony[c_id] else 1.0
                    self._path_pheromone_update(path, pheromone_views[c_id], c_id, w)

                # Stagnation check
                for c_id in range(resilience_factor):
                    if current_no_updates[c_id] > self.max_no_updates:
                        if log_print: print(f"  Colony {c_id} Stagnation Restart")
                        # Salva best corrente
                        bp, bc = current_best_path_per_colony[c_id], current_best_path_cost_per_colony[c_id]
                        if bp and (bp, bc) not in best_paths_before_stagnation:
                            best_paths_before_stagnation.append((bp, bc))

                        # Reset
                        pheromone_views[c_id][:] = self.tau_max[c_id]
                        current_no_updates[c_id] = 0

                # Early Stopping
                if epoch > 100:
                    conv = sum(1 for i in range(resilience_factor) if self._check_convergence(cost_history[i]))
                    if conv == resilience_factor:
                        if log_print: print("Converged.")
                        break

                epoch += 1

        # Final Collection
        for i in range(resilience_factor):
            bp, bc = current_best_path_per_colony[i], current_best_path_cost_per_colony[i]
            if bp and (bp, bc) not in best_paths_before_stagnation:
                best_paths_before_stagnation.append((bp, bc))

        return heapq.nsmallest(retrieve_n_best_paths * resilience_factor, best_paths_before_stagnation,
                               key=lambda x: x[1])