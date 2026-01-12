import random
import time
import numpy as np
from numba import jit


class Ant:
    """
    Optimized Ant - uses pre-computed sparse structures, no graph reference
    """
    __slots__ = ('alpha', 'beta', 'q0', 'path', 'visited_nodes', 'visited_array',
                 'shared_pheromones', 'colony_id', 'n_colonies',
                 'n_nodes', 'n_edges', 'key_nodes', 'key_nodes_array',
                 'neighbors_list', 'edge_costs_csr', 'adjacency_csr',
                 'edge_id_csr', 'dist_to_key_nodes', 'edge_mapping',
                 'ant_id')

    def __init__(self,
                 graph_data: dict,
                 shared_pheromones,
                 colony_id: int = 0,
                 n_colonies: int = 1,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 q0: float = 0.05,
                 ant_id: int = 1
                 ):
        """
        :param graph_data: Dictionary with pre-computed structures
        :param shared_pheromones: Shared Memory C Array
        :param alpha: Influence of pheromones
        :param beta: Influence of edge cost
        :param q0: Exploration threshold
        """
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.path = []
        self.visited_nodes = set()
        self.colony_id = colony_id
        self.n_colonies = n_colonies
        self.shared_pheromones = shared_pheromones
        self.ant_id = ant_id

        # Extract pre-computed structures from graph_data
        self.n_nodes = graph_data['n_nodes']
        self.n_edges = graph_data['n_edges']
        self.key_nodes = set(graph_data['key_nodes'])
        self.key_nodes_array = graph_data['key_nodes_array']
        self.neighbors_list = graph_data['neighbors_list']
        self.edge_costs_csr = graph_data['edge_costs_csr']
        self.adjacency_csr = graph_data['adjacency_csr']
        self.edge_id_csr = graph_data['edge_id_csr']
        self.dist_to_key_nodes = graph_data['dist_to_key_nodes']
        self.edge_mapping = graph_data['edge_mapping']

        # Fast visited tracking
        self.visited_array = np.zeros(self.n_nodes, dtype=np.bool_)

    def select_next_node(self, current_node, nodes_to_visit=None):
        """
        Optimized node selection using sparse matrices and pre-computed data
        """
        # Get neighbors from pre-computed list
        all_neighbors = self.neighbors_list[current_node]

        # Filter visited using array (faster than set lookup)
        neighbors = [n for n in all_neighbors if not self.visited_array[n]]

        if not neighbors:
            # Fallback: accept visited nodes if stuck
            if all_neighbors:
                return random.choice(all_neighbors)
            return None

        n_neighbors = len(neighbors)
        neighbors_arr = np.array(neighbors, dtype=np.int32)

        # Extract edge costs using CSR (very fast!)
        edge_costs = np.array([
            self.edge_costs_csr[current_node, n] for n in neighbors
        ], dtype=np.float32)

        # Calculate distances to target key nodes
        if nodes_to_visit is None:
            nodes_to_visit = self.key_nodes - self.visited_nodes

        if nodes_to_visit:
            # Get indices of target key nodes in our distance matrix
            key_nodes_list = list(self.key_nodes)
            target_indices = np.array([
                key_nodes_list.index(t) for t in nodes_to_visit
            ], dtype=np.int32)

            # Extract distances using our compact matrix
            dist_to_targets = self.dist_to_key_nodes[neighbors_arr][:, target_indices].min(axis=1)
        else:
            dist_to_targets = np.zeros(n_neighbors, dtype=np.float32)

        # Calculate total effort
        total_effort = edge_costs + dist_to_targets
        total_effort = 1/(total_effort+0.001)
        # Apply key node bias
        key_nodes_bias = 20.0
        is_key_node = self.key_nodes_array[neighbors_arr].astype(bool)
        total_effort[is_key_node] *= key_nodes_bias

        # Get pheromones with dominance factor
        pheromones = np.zeros(n_neighbors, dtype=np.float32)
        for i, neighbor in enumerate(neighbors):
            edge_id = int(self.edge_id_csr[current_node, neighbor]) - 1

            if edge_id >= 0:
                colony_pher = self.shared_pheromones[self.colony_id][edge_id]

                # Calculate pheromone dominance
                sum_pher = sum(
                    self.shared_pheromones[c][edge_id]
                    for c in range(self.n_colonies)
                )

                if sum_pher > 0:
                    dominance = colony_pher / sum_pher
                    pheromones[i] = colony_pher * (dominance ** 2)

        # Calculate probabilities
        probs = (pheromones ** self.alpha) * (total_effort ** self.beta)

        # Normalize
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            probs = np.ones(n_neighbors) / n_neighbors

        # Selection: exploitation vs exploration
        if random.random() <= self.q0:
            # Exploitation: choose best
            return neighbors[np.argmax(probs)]
        else:
            # Exploration: roulette wheel
            return np.random.choice(neighbors, p=probs)

    def calculate_path(self, starting_node, log_print: bool = False, TSP: bool = False):
        """Calculate path visiting all key nodes"""
        if log_print:
            t0 = time.time()

        self.path.append(starting_node)
        self.visited_array[starting_node] = True
        self.visited_nodes.add(starting_node)

        current_node = starting_node
        nodes_to_visit = self.key_nodes.copy()

        if current_node in nodes_to_visit:
            nodes_to_visit.remove(current_node)

        # Visit all key nodes
        while nodes_to_visit:
            next_node = self.select_next_node(current_node, nodes_to_visit)

            if next_node is None:
                if log_print:
                    print("Ant got stuck, returning partial path")
                return self.path

            self.path.append(next_node)
            self.visited_nodes.add(next_node)
            self.visited_array[next_node] = True

            if next_node in nodes_to_visit:
                nodes_to_visit.remove(next_node)

            current_node = next_node

        # TSP: return to start
        if TSP:
            nodes_to_visit = {starting_node}
            while current_node != starting_node:
                next_node = self.select_next_node(current_node, nodes_to_visit)

                if next_node is None:
                    if log_print:
                        print("Ant couldn't return to start")
                    return self.path

                self.path.append(next_node)
                self.visited_nodes.add(next_node)
                self.visited_array[next_node] = True

                current_node = next_node

        t1 = time.time()
        # Optimization heuristics
        self.path_pruning_optimization()
        t2 = time.time()
        self.two_opt_heuristic()
        t3 = time.time()

        if log_print:
            print(f"Ant {self.ant_id} from Colony {self.colony_id}: Path Cons {t1-t0:.2f}s - Pruning {t2-t1:.2f}s - 2-Opt {t3-t2:.2f}")

        return self.path

    def path_pruning_optimization(self):
        """Remove redundant nodes using shortcuts"""
        if len(self.path) < 3:
            return

        path_array = np.array(self.path, dtype=np.int32)

        # Convert CSR to dense for numba (only adjacency, small cost)
        adjacency_dense = self.adjacency_csr.toarray().astype(np.int32)

        shortcuts = find_shortcuts_numba(path_array, adjacency_dense, self.key_nodes_array)

        # Apply shortcuts in reverse order
        for i, shortcut_idx in reversed(shortcuts):
            del self.path[i + 1:shortcut_idx]

    def two_opt_heuristic(self):
        """2-opt optimization"""
        if len(self.path) < 4:
            return

        path_array = np.array(self.path, dtype=np.int32)

        # Convert sparse to dense for numba
        cost_dense = self.edge_costs_csr.toarray()
        adjacency_dense = self.adjacency_csr.toarray()

        optimized_path = two_opt_numba(path_array, cost_dense, adjacency_dense)
        self.path = optimized_path.tolist()

    def calc_path_cost(self, path):
        """Calculate total path cost"""
        cost = 0.0
        for i in range(len(path) - 1):
            cost += self.edge_costs_csr[path[i], path[i + 1]]
        return cost


@jit(nopython=True)
def find_shortcuts_numba(path_array, adjacency_matrix, key_nodes_array):
    """
    Find shortcuts in path without skipping key nodes
    """
    n = len(path_array)
    shortcuts = []

    i = 0
    while i < n - 1:
        curr = path_array[i]
        best_shortcut_idx = -1

        # Look ahead for shortcut opportunities
        max_look = min(i + 50, n)  # Limit search window

        for j in range(i + 2, max_look):
            neighbor = path_array[j]

            # Check if direct edge exists
            if adjacency_matrix[curr, neighbor] > 0:
                # Check if any key nodes would be skipped
                has_key_node = False
                for k in range(i + 1, j):
                    if key_nodes_array[path_array[k]] > 0:
                        has_key_node = True
                        break

                if not has_key_node:
                    best_shortcut_idx = j
                    break

        if best_shortcut_idx != -1:
            shortcuts.append((i, best_shortcut_idx))
            i = best_shortcut_idx
        else:
            i += 1

    return shortcuts


@jit(nopython=True)
def two_opt_numba(path_array, cost_matrix, adjacency_matrix, window_size=30):
    """
    2-opt optimization with limited window
    """
    n = len(path_array)
    improved = True
    iterations = 0

    while improved and iterations < 3:
        iterations += 1
        improved = False

        for i in range(n - 2):
            a = path_array[i]
            b = path_array[i + 1]

            if adjacency_matrix[a, b] == 0:
                continue

            cost_ab = cost_matrix[a, b]
            max_j = min(i + window_size, n - 1)

            for j in range(i + 2, max_j):
                c = path_array[j]
                d = path_array[j + 1]

                if adjacency_matrix[a, c] == 0 or adjacency_matrix[b, d] == 0:
                    continue

                current_cost = cost_ab + cost_matrix[c, d]
                new_cost = cost_matrix[a, c] + cost_matrix[b, d]

                if new_cost < current_cost:
                    # Reverse segment
                    path_array[i + 1:j + 1] = path_array[i + 1:j + 1][::-1]
                    improved = True
                    break

            if improved:
                break

    return path_array