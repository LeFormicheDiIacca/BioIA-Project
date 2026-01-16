import random
import time
import numpy as np
from numba import jit


class Ant:
    """
    Optimized Ant - Uses shared memory CSR structures (Numpy views).
    Drastically reduces RAM usage by avoiding object duplication and dictionary lookups.
    """
    __slots__ = ('alpha', 'beta', 'q0', 'path', 'visited_nodes', 'visited_array',
                 'colony_id', 'n_colonies', 'ant_id',
                 # Shared Data Arrays (Views)
                 'indptr', 'indices', 'costs', 'edge_ids', 'key_nodes_mask',
                 'dist_matrix', 'pheromones_views', 'key_nodes_list', 'key_node_to_index_map')

    def __init__(self,
                 shared_data: dict,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 q0: float = 0.05,
                 colony_id: int = 0,
                 n_colonies: int = 1,
                 ant_id: int = 1
                 ):
        """
        Initialize Ant with references to shared memory arrays.
        """
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.colony_id = colony_id
        self.n_colonies = n_colonies
        self.ant_id = ant_id

        # Unpack shared numpy views (No memory copy, just pointers)
        self.indptr = shared_data['indptr']  # [N+1]
        self.indices = shared_data['indices']  # [E]
        self.costs = shared_data['costs']  # [E]
        self.edge_ids = shared_data['edge_ids']  # [E]
        self.key_nodes_mask = shared_data['key_nodes_mask']  # [N]
        self.dist_matrix = shared_data['dist_matrix']  # [N x K]
        self.pheromones_views = shared_data['pheromones']  # List of [E]
        self.key_nodes_list = shared_data['key_nodes_list']

        # Fast lookup for distance matrix columns
        self.key_node_to_index_map = {node: i for i, node in enumerate(self.key_nodes_list)}

        self.path = []
        self.visited_nodes = set()
        # Boolean array for O(1) visited check
        self.visited_array = np.zeros(shared_data['n_nodes'], dtype=np.bool_)

    def select_next_node(self, current_node, nodes_to_visit=None):
        """
        Selects next node using CSR arithmetic and vectorized numpy operations.
        Includes fallback for when ant is stuck (surrounded by visited nodes).
        """
        # 1. Get neighbors range from CSR indptr
        start_idx = self.indptr[current_node]
        end_idx = self.indptr[current_node + 1]

        # No neighbors at all (isolated node)
        if start_idx == end_idx:
            return None

        # 2. Slice arrays to get neighbors data (Views)
        neighbor_indices = self.indices[start_idx:end_idx]

        # Indices relative to the global arrays (costs, edge_ids)
        # This range corresponds to the edges connecting current_node
        global_edge_idxs = np.arange(start_idx, end_idx)

        # 3. Filter visited nodes
        valid_mask = ~self.visited_array[neighbor_indices]

        # --- FIX: FALLBACK LOGIC ---
        # If all neighbors are visited, we are stuck.
        # We MUST revert to allowing visited nodes to escape the dead end.
        if not np.any(valid_mask):
            # STUCK MODE: Consider ALL neighbors
            candidates = neighbor_indices
            candidates_edge_idxs = global_edge_idxs
        else:
            # NORMAL MODE: Only unvisited neighbors
            candidates = neighbor_indices[valid_mask]
            candidates_edge_idxs = global_edge_idxs[valid_mask]

        n_candidates = len(candidates)

        # --- HEURISTIC CALCULATION (eta) ---
        # Edge costs
        costs = self.costs[candidates_edge_idxs]

        # Distance to remaining key nodes
        if nodes_to_visit:
            # Slicing: [Candidates, Targets] -> Min distance per candidate
            target_indices = [self.key_node_to_index_map[k] for k in nodes_to_visit if k in self.key_node_to_index_map]

            if target_indices:
                # Use numpy advanced indexing to find min distance to any target
                dists_to_targets = self.dist_matrix[candidates][:, target_indices].min(axis=1)
            else:
                dists_to_targets = np.zeros(n_candidates, dtype=np.float32)
        else:
            dists_to_targets = np.zeros(n_candidates, dtype=np.float32)

        # Total Effort (Heuristic)
        # Avoid division by zero
        total_effort = 1.0 / (costs + dists_to_targets + 0.001)

        # Key Node Bias
        # Using mask array for fast lookup
        is_key_node = self.key_nodes_mask[candidates] > 0
        total_effort[is_key_node] *= 20.0

        # --- PHEROMONE CALCULATION (tau) ---
        # Get Edge IDs (-1 because stored IDs are 1-based from graph generation)
        e_ids = self.edge_ids[candidates_edge_idxs] - 1

        # Current colony pheromone
        colony_pher = self.pheromones_views[self.colony_id][e_ids]

        # Sum of all colonies (Resilience)
        sum_pher = np.zeros(n_candidates, dtype=np.float64)
        for i in range(self.n_colonies):
            sum_pher += self.pheromones_views[i][e_ids]

        # Calculate dominance
        with np.errstate(divide='ignore', invalid='ignore'):
            # Safe division
            dominance = np.where(sum_pher > 0, colony_pher / sum_pher, 0)
            pheromones = colony_pher * (dominance ** 2)
            # Ensure strictly positive for probability calculation
            pheromones = np.maximum(pheromones, 1e-10)

        # --- PROBABILITIES ---
        probs = (pheromones ** self.alpha) * (total_effort ** self.beta)
        prob_sum = probs.sum()

        if prob_sum > 0:
            probs /= prob_sum
        else:
            probs = np.ones(n_candidates) / n_candidates

        # --- SELECTION ---
        if random.random() <= self.q0:
            best_idx = np.argmax(probs)
            return candidates[best_idx]
        else:
            # Roulette Wheel
            cum_probs = np.cumsum(probs)
            r = random.random()
            idx = np.searchsorted(cum_probs, r)
            # Clip index just in case of float precision issues
            return candidates[min(idx, n_candidates - 1)]
    def calculate_path(self, starting_node, log_print=False, TSP=False):
        """Calculate path visiting all key nodes"""
        if log_print:
            t0 = time.time()

        self.path = [starting_node]
        self.visited_array[starting_node] = True
        self.visited_nodes.add(starting_node)

        current_node = starting_node
        # Create set of keys to visit
        nodes_to_visit = set(self.key_nodes_list)

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
            nodes_to_visit_tsp = {starting_node}
            while current_node != starting_node:
                # Pass None as target keys, just trying to reach start (which is in neighbors check)
                # Or treat start node as a temporary key node target
                # Here logic simplifies to just moving until start is reached.
                # NOTE: The heuristic might need to know we want to go to starting_node.
                # But select_next_node expects key nodes.
                # Simple fix: just run greedy towards start.

                # We temporarily treat start_node as a target for calculation
                # But since it's not in key_nodes_list, distance calculation might fail if we pass it.
                # We rely on pheromones or standard movement.

                next_node = self.select_next_node(current_node, None)

                if next_node is None:
                    break

                self.path.append(next_node)
                self.visited_nodes.add(next_node)
                self.visited_array[next_node] = True

                if next_node == starting_node:
                    break

                current_node = next_node

        t1 = time.time()

        # --- OPTIMIZATIONS ---
        # Convert path to numpy array for Numba
        path_array = np.array(self.path, dtype=np.int32)

        # 1. Pruning
        shortcuts = find_shortcuts_csr(path_array, self.indptr, self.indices, self.key_nodes_mask)

        # Apply shortcuts (Python list manipulation is easier here)
        if len(shortcuts) > 0:
            for i, shortcut_idx in reversed(shortcuts):
                del self.path[i + 1:shortcut_idx]
            # Re-sync array for next step
            path_array = np.array(self.path, dtype=np.int32)

        t2 = time.time()

        # 2. 2-Opt
        # Passes CSR structures to avoid dense matrix creation
        optimized_path = two_opt_csr(path_array, self.indptr, self.indices, self.costs)
        self.path = optimized_path.tolist()

        t3 = time.time()

        if log_print:
            print(f"Ant {self.ant_id}: Path {t1 - t0:.2f}s - Prun {t2 - t1:.2f}s - 2Opt {t3 - t2:.2f}s")

        return self.path

    def calc_path_cost(self, path):
        """Calculate total path cost using CSR lookup"""
        # Convert to array for potential numba usage, or keep simple loop
        # Simple loop is usually fast enough for path evaluation
        cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            c = get_edge_cost_numba(u, v, self.indptr, self.indices, self.costs)
            cost += c
        return cost


# --- NUMBA OPTIMIZED HELPER FUNCTIONS ---

@jit(nopython=True)
def get_edge_cost_numba(u, v, indptr, indices, costs):
    """
    Retrieve edge cost in O(degree) using CSR.
    Returns infinity if edge not found.
    """
    start = indptr[u]
    end = indptr[u + 1]

    for i in range(start, end):
        if indices[i] == v:
            return costs[i]
    return 1e9  # Large number for infinity


@jit(nopython=True)
def check_edge_exists(u, v, indptr, indices):
    """
    Check if edge u->v exists in O(degree).
    """
    start = indptr[u]
    end = indptr[u + 1]
    for i in range(start, end):
        if indices[i] == v:
            return True
    return False


@jit(nopython=True)
def find_shortcuts_csr(path_array, indptr, indices, key_nodes_mask):
    """
    Find shortcuts in path respecting key nodes using CSR structure.
    """
    n = len(path_array)
    shortcuts = []
    i = 0

    while i < n - 1:
        curr = path_array[i]
        best_shortcut_idx = -1

        # Limit lookahead to avoid O(N^2) on long paths
        max_look = min(i + 50, n)

        for j in range(i + 2, max_look):
            neighbor = path_array[j]

            # Check if direct edge exists using CSR
            if check_edge_exists(curr, neighbor, indptr, indices):
                # Check for skipped key nodes
                has_key_node = False
                for k in range(i + 1, j):
                    if key_nodes_mask[path_array[k]] > 0:
                        has_key_node = True
                        break

                if not has_key_node:
                    best_shortcut_idx = j
                    # Keep looking for a longer shortcut

        if best_shortcut_idx != -1:
            shortcuts.append((i, best_shortcut_idx))
            i = best_shortcut_idx
        else:
            i += 1

    return shortcuts


@jit(nopython=True)
def two_opt_csr(path_array, indptr, indices, costs, max_iterations=3, window_size=30):
    """
    2-opt optimization using CSR cost lookups.
    Avoids dense matrix usage.
    """
    n = len(path_array)
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        iterations += 1
        improved = False

        # Limit the window to keep it fast
        for i in range(n - 2):
            a = path_array[i]
            b = path_array[i + 1]

            cost_ab = get_edge_cost_numba(a, b, indptr, indices, costs)
            if cost_ab >= 1e9: continue  # Should not happen on valid path

            max_j = min(i + window_size, n - 1)

            for j in range(i + 2, max_j):
                c = path_array[j]
                d = path_array[j + 1]

                cost_cd = get_edge_cost_numba(c, d, indptr, indices, costs)
                if cost_cd >= 1e9: continue

                # Check existence of cross edges (a->c) and (b->d)
                # Since path is undirected for 2-opt, we check swap
                # New edges: a->c and b->d (segment b...c is reversed)

                cost_ac = get_edge_cost_numba(a, c, indptr, indices, costs)
                if cost_ac >= 1e9: continue

                cost_bd = get_edge_cost_numba(b, d, indptr, indices, costs)
                if cost_bd >= 1e9: continue

                current_cost = cost_ab + cost_cd
                new_cost = cost_ac + cost_bd

                if new_cost < current_cost:
                    # Reverse segment [i+1 : j+1]
                    # Numba supports slice assignment
                    path_array[i + 1:j + 1] = path_array[i + 1:j + 1][::-1]
                    improved = True
                    break  # Restart inner loop or continue? Standard 2-opt usually continues

            if improved:
                break  # Restart outer loop

    return path_array