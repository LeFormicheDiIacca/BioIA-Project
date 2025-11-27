import math
import random
from meshgraph import MeshGraph

class Ant:
    __slots__ = ('alpha', 'beta', 'rho', 'path', 'visited_nodes', 'starting_node', 'graph','shared_pheromones')
    def __init__(self, starting_node, graph: MeshGraph, alpha, beta, rho, shared_pheromones):
        self.alpha = alpha
        self.beta = beta
        self.path = [starting_node]
        self.visited_nodes = set()
        self.starting_node = starting_node
        self.graph = graph
        self.rho = rho
        self.shared_pheromones = shared_pheromones

    def _get_pheromone(self, u, v):
        if self.shared_pheromones is not None:
            edge_id = self.graph[u][v]["edge_id"]
            return self.shared_pheromones[edge_id]
        else:
            return self.graph[u][v]["pheromone_level"]


    def select_next_node(self, current_node, nodes_to_visit = None):
        """
        May God's light shine on this fucking ant and force it to make a good choice. Amen
        """
        neighbors = [n for n in self.graph[current_node] if n not in self.visited_nodes]
        candidates = dict()
        degree_45_penalty_factor = 0.5

        active_targets = []
        if nodes_to_visit is None:
            nodes_to_visit = self.graph.key_nodes - self.visited_nodes
        if nodes_to_visit:
            active_targets = list(nodes_to_visit)
        for neighbor in neighbors:
            if neighbor in self.visited_nodes:
                continue

            edge_cost = self.graph[current_node][neighbor]["cost"]
            step_dist = self.graph.dist_matrix[current_node, neighbor]
            is_diagonal = not math.isclose(step_dist, 1.0, rel_tol=1e-5)

            dist_to_target = 0.0
            if active_targets:
                min_dist = math.inf
                for t in active_targets:
                    d = self.graph.dist_matrix[neighbor, t]
                    if d < min_dist:
                        min_dist = d
                dist_to_target = min_dist

            total_estimated_effort = edge_cost + dist_to_target

            pheromone = self._get_pheromone(neighbor, current_node)
            heuristic = 1.0 / (total_estimated_effort + 0.1)
            if neighbor in self.graph.key_nodes and neighbor not in self.visited_nodes:
                heuristic *= 2.0

            prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
            if is_diagonal:
                prob *= degree_45_penalty_factor

            candidates[neighbor] = prob

        if not candidates:
            return None

        keys = list(candidates.keys())
        weights = list(candidates.values())
        selected_node = random.choices(keys, weights=weights, k=1)[0]
        return selected_node

    def calculate_path(self):
        next_node = None
        current_node = self.starting_node
        nodes_to_visit = self.graph.key_nodes.copy()
        if current_node in nodes_to_visit:
            nodes_to_visit.remove(current_node)
        while nodes_to_visit:
            next_node = self.select_next_node(current_node)

            if next_node is None:
                return self.path

            self.path.append(next_node)
            self.visited_nodes.add(next_node)

            if next_node in nodes_to_visit:
                nodes_to_visit.remove(next_node)
            current_node = next_node
        nodes_to_visit = {self.starting_node}
        while current_node != self.starting_node:
            next_node = self.select_next_node(current_node, nodes_to_visit)

            if next_node is None:
                return self.path

            self.path.append(next_node)
            self.visited_nodes.add(next_node)
            current_node = next_node

        self.TwoOptHeuristic()
        self.path_pruning_optimization()
        self.TwoOptHeuristic()
        return self.path

    def TwoOptHeuristic(self):

        n = len(self.path)
        improved = True

        while improved:
            improved = False
            for i in range(n - 2):
                a = self.path[i]
                b = self.path[i + 1]
                cost_ab = self.graph[a][b]["cost"]
                for j in range(i + 2, n - 1):
                    c = self.path[j]
                    d = self.path[j + 1]
                    if c not in self.graph[a] or d not in self.graph[b]:
                        continue

                    cost_cd = self.graph[c][d]["cost"]
                    current_cost = cost_ab + cost_cd
                    new_cost = self.graph[a][c]["cost"] + self.graph[b][d]["cost"]

                    if new_cost < current_cost:
                        self.path[i + 1:j + 1] = self.path[i + 1:j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break

    def path_pruning_optimization(self):
        node_indices = {node: i for i, node in enumerate(self.path)}
        i = 0
        while i < len(self.path) - 1:
            curr = self.path[i]
            best_shortcut_idx = -1

            for neighbor in self.graph[curr]:
                if neighbor in node_indices:
                    idx_neighbor = node_indices[neighbor]
                    if idx_neighbor > i + 1:
                        skipped_segment = self.path[i + 1: idx_neighbor]

                        contains_key_node = False
                        for skipped in skipped_segment:
                            if skipped in self.graph.key_nodes:
                                contains_key_node = True
                                break

                        if not contains_key_node:
                            if idx_neighbor > best_shortcut_idx:
                                best_shortcut_idx = idx_neighbor

            if best_shortcut_idx != -1:
                del self.path[i + 1: best_shortcut_idx]
                node_indices = {node: k for k, node in enumerate(self.path)}
            else:
                i += 1