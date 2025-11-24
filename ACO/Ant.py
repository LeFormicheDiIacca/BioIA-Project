import math
import random
from meshgraph import MeshGraph

class Ant:
    def __init__(self, starting_node, key_nodes, graph: MeshGraph, alpha, beta, rho, shared_pheromones):
        self.alpha = alpha
        self.beta = beta
        self.path = [starting_node]
        self.visited_nodes = set()
        self.starting_node = starting_node
        self.key_nodes = key_nodes
        self.graph = graph
        self.rho = rho
        self.shared_pheromones = shared_pheromones

    def _get_pheromone(self, u, v):
        if self.shared_pheromones is not None:
            edge_id = self.graph[u][v]["edge_id"]
            return self.shared_pheromones[edge_id]
        else:
            return self.graph[u][v]["pheromone_level"]

    def _update_local_shared_pheromone(self, u, v):
        if self.shared_pheromones is not None:
            edge_id = self.graph[u][v]["edge_id"]
            old_val = self.shared_pheromones[edge_id]
            initial_pheromone_level = self.graph[u][v]["initial_pheromone_level"]
            new_val = (1 - self.rho) * old_val + self.rho * initial_pheromone_level

            self.shared_pheromones[edge_id] = new_val
    def select_next_node(self, current_node):
        """
        May God's light shine on this fucking ant and force it to make a good choice. Amen
        """
        neighbors = self.graph[current_node]
        candidates = dict()
        degree_45_penalty_factor = 2

        active_targets = []
        nodes_to_visit = self.key_nodes - self.visited_nodes
        if nodes_to_visit:
            active_targets = list(nodes_to_visit)
        for neighbor in neighbors:
            if neighbor in self.visited_nodes:
                continue

            edge_cost = self.graph[current_node][neighbor]["cost"]
            step_dist = self.graph.nodes_geometric_dist(current_node, neighbor)
            is_diagonal = not math.isclose(step_dist, 1.0, rel_tol=1e-5)

            dist_to_target = 0.0
            if active_targets:
                min_dist = math.inf
                for t in active_targets:
                    d = self.graph.nodes_geometric_dist(current_node, t)
                    if d < min_dist:
                        min_dist = d
                dist_to_target = min_dist

            total_estimated_effort = edge_cost + dist_to_target

            pheromone = self._get_pheromone(neighbor, current_node)
            heuristic = 1.0 / (total_estimated_effort + 0.1)
            if neighbor in self.key_nodes and neighbor not in self.visited_nodes:
                heuristic *= 2.0

            prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
            if is_diagonal:
                prob /= degree_45_penalty_factor

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
        nodes_to_visit = self.key_nodes.copy()
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
            self._update_local_shared_pheromone(next_node, current_node)
            current_node = next_node

        while current_node != self.starting_node:
            next_node = self.select_next_node(current_node)

            if next_node is None:
                return self.path

            self.path.append(next_node)
            self.visited_nodes.add(next_node)
            self._update_local_shared_pheromone(next_node, current_node)
            current_node = next_node

        self.TwoOptHeuristic()
        self.path_pruning_optimization()
        self.TwoOptHeuristic()
        return self.path

    def _TwoOptSwap(self, v1_index, v2_index):
        if v1_index >= v2_index:
            v1_index, v2_index = min(v1_index, v2_index), max(v1_index, v2_index)
        new_route = self.path[:v1_index + 1]
        segment_to_reverse = self.path[v1_index + 1: v2_index + 1]
        new_route.extend(segment_to_reverse[::-1])
        new_route.extend(self.path[v2_index + 1:])
        return new_route

    def TwoOptHeuristic(self):
        while True:
            improved = False
            best_dist = self.graph.calc_path_cost(self.path)
            num_nodes = len(self.path)
            for i in range(num_nodes - 1):
                for j in range(i + 1, num_nodes):
                    new_route = self._TwoOptSwap(i, j)
                    new_dist = self.graph.calc_path_cost(new_route)
                    if new_dist < best_dist:
                        self.path = new_route
                        best_dist = new_dist
                        improved = True
                        break

                if improved:
                    break
            if not improved:
                break

        return self.path, best_dist

    def path_pruning_optimization(self):
        current_best_cost = self.graph.calc_path_cost(self.path)
        for current_node in self.path:
            for neighbor in self.graph[current_node]:
                if neighbor in self.path:
                    new_path = self.list_slicing(current_node, neighbor)
                    if self.key_nodes.issubset(set(new_path)):
                        new_path_cost = self.graph.calc_path_cost(new_path)
                        if new_path_cost < current_best_cost:
                            self.path = new_path
                            self.path_pruning_optimization()
                            return

    def list_slicing(self, val1, val2):
        try:
            idx1 = self.path.index(val1)
            idx2 = self.path.index(val2)
            start = min(idx1, idx2)
            end = max(idx1, idx2)
            return self.path[:start + 1] + self.path[end:]

        except ValueError:
            print(f"Error: at least one node is in the path.\nNodes: {val1}, {val2}\nPath: {self.path}")
            return self.path