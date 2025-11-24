import math

import networkx as nx
import matplotlib.pyplot as plt

class MeshGraph(nx.Graph):
    def __init__(self, n_neighbours: int = 8, n_row: int = 10, n_col: int = 10):
        super().__init__()
        if n_neighbours not in [4, 8]:
            raise ValueError('Invalid number of neighbours. Must be 4 or 8')
        if n_row < 3 or n_col < 3:
            raise ValueError('Minimum mesh size must be 3x3')

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


    def plot_graph(self, paths=None, key_nodes=None):
        plt.figure(figsize=(10, 10))
        labels = nx.get_node_attributes(self, 'label')
        if key_nodes is not None:
            node_color = ["#1f78b4" if x not in key_nodes else "red" for x in self.nodes()]
        else:
            node_color = "#1f78b4"
        pos = nx.spring_layout(self, iterations=10000)
        nx.draw_networkx_nodes(self, pos, node_color=node_color)
        nx.draw_networkx_edges(self, pos, edge_color="black")
        if paths is not None:
            for (path, color) in paths:
                path_graph = nx.path_graph(path)
                nx.draw_networkx_edges(self, pos, width=2, edgelist=list(path_graph.edges()), edge_color=color)

        nx.draw_networkx_labels(self, pos, labels=labels)
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

    def pheromone_initialization(self, initial_pheromone_level):
        for edge in self.edges():
            self[edge[0]][edge[1]]["pheromone_level"] = initial_pheromone_level
            self[edge[0]][edge[1]]["initial_pheromone_level"] = initial_pheromone_level

    def nodes_geometric_dist(self, source, target):
        x_s, y_s = self.node_to_pos[source]
        x_t, y_t = self.node_to_pos[target]
        d = math.sqrt((x_s - x_t) ** 2 + (y_s - y_t) ** 2)
        return d

    def calc_path_cost(self, path, degree_45_penalty_factor = 100):
        path_cost = 0
        for i in range(len(path)-1):
            source, destination = path[i], path[i+1]
            dist = self.nodes_geometric_dist(source, destination)
            if dist != 1:
                path_cost += degree_45_penalty_factor
            try:
                path_cost += self[source][destination]["cost"]
            except KeyError:
                path_cost = math.inf

        return path_cost