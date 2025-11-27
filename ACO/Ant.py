import random
import time
import networkx as nx
import numpy as np
from numba import jit
from meshgraph import MeshGraph

class Ant:
    """
    Main villain in this story. I've lost too many hairs due to it.
    It follows the normal Ant Path Calculations with a bias toward avoiding diagonals.
    The bias will be removed in the future because it should be already considered in the edge cost.
    """
    __slots__ = ('alpha', 'beta', 'rho', 'q0', 'path', 'visited_nodes', 'graph','shared_pheromones','key_nodes_array')
    def __init__(self,
                 graph: MeshGraph,
                 shared_pheromones,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 q0: float = 0.05
    ):
        """
        :param graph: MeshGraph to explore
        :param shared_pheromones: Shared Memory C Array used for multiprocessing
        :param alpha: Influence of pheromones
        :param beta: Influence of edge cost
        :param q0: Exploration threshold
        """

        self.alpha = alpha
        self.beta = beta
        self.path = []
        self.visited_nodes = set()
        self.graph = graph
        self.q0 = q0
        self.shared_pheromones = shared_pheromones
        self.key_nodes_array = self._build_key_nodes_array()


    def _build_key_nodes_array(self):
        key_array = np.zeros(self.graph.number_of_nodes(), dtype=np.int32)
        for node in self.graph.key_nodes:
            key_array[node] = 1
        return key_array

    def select_next_node(self, current_node, nodes_to_visit = None):
        """
        May God's light shine on this fucking ant and force it to make a good choice. Amen
        """
        neighbors = [n for n in self.graph[current_node] if n not in self.visited_nodes]
        candidates = dict()
        degree_45_penalty_factor = 0.5
        key_nodes_bias = 2.0
        #We initialize a list of node to reach. They'll guide the ant like a compass
        active_targets = []
        if nodes_to_visit is None:
            #If no target nodes are provided we aim for the key nodes not visited by the ant
            nodes_to_visit = self.graph.key_nodes - self.visited_nodes
        if nodes_to_visit:
            active_targets = list(nodes_to_visit)

        for neighbor in neighbors:
            if neighbor in self.visited_nodes:
                continue

            edge_cost = self.graph[current_node][neighbor]["cost"]
            #Ant Compass. We decide if by going in a node we are getting closer to some key node
            if active_targets:
                min_distances = {
                    neighbor: min(self.graph.dist_matrix[neighbor, t] for t in active_targets)
                    for neighbor in neighbors
                }
            else:
                min_distances = {neighbor: 0.0 for neighbor in neighbors}

            dist_to_target = min_distances[neighbor]
            #Total effort will be the edge cost + how close we are to a key node
            total_estimated_effort = edge_cost + dist_to_target
            heuristic = 1.0 / (total_estimated_effort + 0.1)
            #Key nodes have a better heuristic chance to be chosen
            if neighbor in self.graph.key_nodes and neighbor not in self.visited_nodes:
                heuristic *= key_nodes_bias

            #Pheromone retrieval
            edge_id = self.graph[current_node][neighbor]["edge_id"]
            pheromone =  self.shared_pheromones[edge_id]

            prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
            #Diagonal movements have less probability
            if self.graph.dist_matrix[current_node, neighbor] > 1.01:
                prob *= degree_45_penalty_factor
            candidates[neighbor] = prob
        #If the ant is stuck it can go in an already visited node. We don't care about the only 1 visit rule because we'll prune the path later
        if not candidates:
            neighbors = [n for n in self.graph[current_node]]
            return random.choice(neighbors)

        if random.random() <= self.q0:
            #Random chance to exploit instead of exploring
            best_node = max(candidates, key=candidates.get)
            return best_node
        else:
            #Roulette wheel selection
            keys = list(candidates.keys())
            weights = list(candidates.values())
            selected_node = random.choices(keys, weights=weights, k=1)[0]
            return selected_node

    def calculate_path(self, starting_node):
        self.path.append(starting_node)
        current_node = starting_node
        nodes_to_visit = self.graph.key_nodes.copy()
        if current_node in nodes_to_visit:
            nodes_to_visit.remove(current_node)
        #Cycle used to search all key nodes
        t0 = time.time()
        while nodes_to_visit:
            next_node = self.select_next_node(current_node)
            #If we are stuck with no way out we dump the invalid path
            if next_node is None:
                return self.path
            #Add the new node to all the important data structures
            self.path.append(next_node)
            self.visited_nodes.add(next_node)
            if next_node in nodes_to_visit:
                nodes_to_visit.remove(next_node)

            current_node = next_node

        #In this way the ant's compass will guide it toward the starting node
        nodes_to_visit = {starting_node}
        while current_node != starting_node:
            next_node = self.select_next_node(current_node, nodes_to_visit)
            #If we are stuck with no way out we dump the invalid path
            if next_node is None:
                return self.path

            #Add the new node to all the important data structures
            self.path.append(next_node)
            self.visited_nodes.add(next_node)

            current_node = next_node
        t1 = time.time()
        #Heuristic Used to refine the path and avoid redundancy
        self.path_pruning_optimization()
        t2 = time.time()
        self.TwoOptHeuristic()
        t3 = time.time()
        print(f"Path construction: {t1 - t0:.2f}s, Pruning: {t2 - t1:.2f}s, 2-opt: {t3 - t2:.2f}s")
        return self.path

    def path_pruning_optimization(self):
        path_array = np.array(self.path, dtype=np.int32)
        adjacency_matrix = nx.to_numpy_array(self.graph, weight=None).astype(np.int32)

        shortcuts = find_shortcuts_numba(path_array, adjacency_matrix, self.key_nodes_array)
        for i, shortcut_idx in reversed(shortcuts):
            del self.path[i + 1:shortcut_idx]

    def TwoOptHeuristic(self):
        path_array = np.array(self.path, dtype=np.int32)
        cost_matrix = nx.to_numpy_array(self.graph, weight='cost')
        adjacency_matrix = nx.to_numpy_array(self.graph, weight=None)

        optimized_path = two_opt_numba(path_array, cost_matrix, adjacency_matrix)
        self.path = optimized_path.tolist()

@jit(nopython=True)
def find_shortcuts_numba(path_array, adjacency_matrix, key_nodes_array):
    """
    Trova shortcuts nel path senza ricostruire dict Python
    """
    n = len(path_array)
    shortcuts = []

    i = 0
    while i < n - 1:
        curr = path_array[i]
        best_shortcut_idx = -1

        for j in range(i + 2, n):
            neighbor = path_array[j]

            if adjacency_matrix[curr, neighbor] > 0:
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
    2-opt with numba
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