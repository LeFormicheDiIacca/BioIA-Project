import math
import random

from meshgraph import MeshGraph


class Ant:
    def __init__(self, starting_node, key_nodes, graph: MeshGraph, alpha, beta, rho):
        self.alpha = alpha
        self.beta = beta
        self.path = [starting_node]
        self.visited_nodes = set()
        self.starting_node = starting_node
        self.key_nodes = key_nodes
        self.graph = graph
        self.rho = rho

    def local_pheromone_update(self, source, destination):
        current_pheromone = self.graph[source][destination]["pheromone_level"]
        self.graph[source][destination]["pheromone_level"] = current_pheromone * (1 - self.rho)
        self.graph[source][destination]["pheromone_level"] += self.graph[source][destination]["initial_pheromone_level"] * self.rho

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

            pheromone = self.graph[current_node][neighbor]["pheromone_level"]
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

            self.local_pheromone_update(current_node, next_node)
            current_node = next_node

        while current_node != self.starting_node:
            next_node = self.select_next_node(current_node)

            if next_node is None:
                return self.path

            self.path.append(next_node)
            self.visited_nodes.add(next_node)

            self.local_pheromone_update(current_node, next_node)

            current_node = next_node

        return self.path
