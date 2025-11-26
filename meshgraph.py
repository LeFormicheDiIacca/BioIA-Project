import math
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class MeshGraph(nx.Graph):
    def __init__(self, key_nodes: set = None, n_neighbours: int = 8, n_row: int = 10, n_col: int = 10):
        super().__init__()
        if n_neighbours not in [4, 8]:
            raise ValueError('Invalid number of neighbours. Must be 4 or 8')
        if n_row < 3 or n_col < 3:
            raise ValueError('Minimum mesh size must be 3x3')


        self.key_nodes = key_nodes
        self._construct_graph(n_row, n_col, n_neighbours)

        nodes_pos = np.array([[self.node_to_pos[node][0], self.node_to_pos[node][1]] for node in self.nodes()])
        compress_dist = pdist(nodes_pos, metric='euclidean')
        self.dist_matrix = squareform(compress_dist)
        self.assign_edge_indexes()

    def assign_key_nodes(self, key_nodes: set):
        self.key_nodes = key_nodes

    def assign_edge_indexes(self):
        idx = 0
        for u, v in self.edges():
            self[u][v]["edge_id"] = idx
            if self.has_edge(v, u):
                self[v][u]["edge_id"] = idx
            idx += 1

    def _construct_graph(self, n_row, n_col, n_neighbours):
        n_nodes = n_row * n_col
        self.pos_to_node = {}
        self.node_to_pos = {}
        x, y = 0, 0
        for i in range(n_nodes):
            label = f"N{i}"
            self.add_node(i)
            self.nodes[i]["label"] = label
            # Saving grid pos in a separate dictionary for faster retrieval
            self.pos_to_node[x, y] = i
            self.node_to_pos[i] = (x, y)
            y += 1
            if y == n_row:
                x += 1
                y = 0

        # Connecting nodes
        for i in range(1, n_row - 1):
            for j in range(1, n_col - 1):
                # By default a 4 neighbors grid
                self.add_edge(self.pos_to_node[(i, j)], self.pos_to_node[(i - 1, j)])
                self.add_edge(self.pos_to_node[(i, j)], self.pos_to_node[(i + 1, j)])
                self.add_edge(self.pos_to_node[(i, j)], self.pos_to_node[(i, j - 1)])
                self.add_edge(self.pos_to_node[(i, j)], self.pos_to_node[(i, j + 1)])

                # Expand to 8 if needed
                if n_neighbours >= 8:
                    self.add_edge(self.pos_to_node[(i, j)], self.pos_to_node[(i - 1, j - 1)])
                    self.add_edge(self.pos_to_node[(i, j)], self.pos_to_node[(i - 1, j + 1)])
                    self.add_edge(self.pos_to_node[(i, j)], self.pos_to_node[(i + 1, j - 1)])
                    self.add_edge(self.pos_to_node[(i, j)], self.pos_to_node[(i + 1, j + 1)])

        # Edge Cases
        for i in range(n_col - 1):
            self.add_edge(self.pos_to_node[(0, i)], self.pos_to_node[(0, i + 1)])
        for i in range(n_col - 1):
            self.add_edge(self.pos_to_node[(n_row - 1, i)], self.pos_to_node[(n_row - 1, i + 1)])
        for i in range(n_row - 1):
            self.add_edge(self.pos_to_node[(i, 0)], self.pos_to_node[(i + 1, 0)])
        for i in range(n_row - 1):
            self.add_edge(self.pos_to_node[(i, n_col - 1)], self.pos_to_node[(i + 1, n_col - 1)])
        # Expand edge cases
        if n_neighbours >= 8:
            self.add_edge(self.pos_to_node[(0, 1)], self.pos_to_node[(1, 0)])
            self.add_edge(self.pos_to_node[(0, n_col - 2)], self.pos_to_node[(1, n_col - 1)])
            self.add_edge(self.pos_to_node[(n_row - 2, 0)], self.pos_to_node[(n_row - 1, 1)])
            self.add_edge(self.pos_to_node[(n_row - 2, n_col - 1)], self.pos_to_node[(n_row - 1, n_col - 2)])

    def plot_graph(self, paths=None, draw_labels = False, figsize= (100,100), dpi=100):
        plt.figure(figsize=figsize, dpi=dpi)
        labels = nx.get_node_attributes(self, 'label')
        if self.key_nodes is not None:
            node_color = ["#1f78b4" if x not in self.key_nodes else "red" for x in self.nodes()]
        else:
            node_color = "#1f78b4"

        pos = self.node_to_pos

        # nx.draw_networkx_nodes(self, self.node_to_pos, node_color=node_color)
        nx.draw_networkx_edges(self, pos, edge_color="gray")

        node_costs = [self.nodes[node].get('elevation', 0) for node in self.nodes()]
        nx.draw_networkx_nodes(
            self, pos,
            node_color=node_costs,
            cmap='magma', 
            node_size=10, 
        )

        if paths is not None:
            for (path, color) in paths:
                path_graph = nx.path_graph(path)
                nx.draw_networkx_edges(self, self.node_to_pos, width=2, edgelist=list(path_graph.edges()), edge_color=color)
        if draw_labels:
            nx.draw_networkx_labels(self, self.node_to_pos, labels=labels)


        # water_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_water')]
        # nx.draw_networkx_nodes(
        #     graph, pos,
        #     nodelist=water_nodes,
        #     node_color='lightblue',
        #     node_size=10,
        # )

        plt.axis('off')
        plt.show()

    def cost_assignment(self, edges_metadata, assignment_function, print_assignment=False):
        for edge in self.edges():
            # If no metadata is available, a big value is assigned as cost
            try:
                metadata = edges_metadata[edge]
                cost = assignment_function(metadata)
            except KeyError:
                cost = 100
                metadata = None
                if print_assignment:
                    print(f"Error: edge {edge} has no metadata. Cost set to {cost}.")
            self[edge[0]][edge[1]]['cost'] = cost
            if print_assignment:
                print(f"Assignment for edge {edge[0]}->{edge[1]} cost: {cost}")
                print(f"Metadata of edge {edge[0]}->{edge[1]}:\n{metadata}")

    def calc_path_cost(self, path, degree_45_penalty_factor = 100):
        path_cost = 0
        for i in range(len(path)-1):
            source, destination = path[i], path[i+1]
            dist = self.dist_matrix[source, destination]
            if not math.isclose(dist, 1.0, rel_tol=1e-5):
                path_cost += degree_45_penalty_factor
            try:
                path_cost += self[source][destination]["cost"]
            except KeyError:
                path_cost = math.inf

        return path_cost

    def is_valid_path(self, path):
        if not self.key_nodes.issubset(set(path)):
            return False
        if path[0] != path[-1]:
            return False
        return True
