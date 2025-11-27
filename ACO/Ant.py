import math
import random
from meshgraph import MeshGraph

class Ant:
    """
    Main villain in this story. I've lost too many hairs due to it.
    It follows the normal Ant Path Calculations with a bias toward avoiding diagonals.
    The bias will be removed in the future because it should be already considered in the edge cost.
    """
    __slots__ = ('alpha', 'beta', 'rho', 'q0', 'path', 'visited_nodes', 'graph','shared_pheromones')
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
            dist_to_target = 0.0
            if active_targets:
                min_dist = math.inf
                for t in active_targets:
                    d = self.graph.dist_matrix[neighbor, t]
                    if d < min_dist:
                        min_dist = d
                dist_to_target = min_dist
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
        #Heuristic Used to refine the path and avoid redundancy
        self.path_pruning_optimization()
        self.TwoOptHeuristic()
        return self.path

    def TwoOptHeuristic(self):
        """
        Local search that verify if it's possible to swap the edges in order to reduce the total cost.
        """
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
        """
        Don't ask why, accept the code and embrace it.
        (Ideally it cuts the useless parts in the path).
        """
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