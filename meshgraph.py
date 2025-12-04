import math
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class MeshGraph(nx.Graph):
    """
    Extension of networkx graph used to store some terrain extra information
    """
    def __init__(
            self,
            key_nodes: set = None,
            n_neighbours: int = 8,
            resolution: int = 10
    ):
        super().__init__()
        if n_neighbours not in [4, 8]:
            raise ValueError('Invalid number of neighbours. Must be 4 or 8')
        if resolution < 3:
            raise ValueError('Minimum mesh size must be 3x3')

        self.key_nodes = key_nodes
        #Used to get the geometric position of a node and a node from (x,y) coords
        self.pos_to_node = {}
        self.node_to_pos = {}
        self._construct_graph(resolution, n_neighbours)

        #Create a geometric distance matrix used by the Ants' compass
        nodes_pos = np.array([[self.node_to_pos[node][0], self.node_to_pos[node][1]] for node in self.nodes()])
        compress_dist = pdist(nodes_pos, metric='euclidean')
        self.dist_matrix = squareform(compress_dist)

        #Creates a mapping from edges to ids and vice versa. Used in order to get pheromones info in the ants
        self.edge_mapping = dict()
        idx = 0
        for u, v in self.edges():
            self[u][v]["edge_id"] = idx
            self.edge_mapping[idx] = (u, v)
            if self.has_edge(v, u):
                self[v][u]["edge_id"] = idx
            idx += 1

    def assign_key_nodes(self, key_nodes: set):
        self.key_nodes = key_nodes


    def _construct_graph(self, resolution, n_neighbours):
        n_nodes = resolution * resolution
        x, y = 0, 0
        for i in range(n_nodes):
            label = f"N{i}"
            self.add_node(i)
            self.nodes[i]["label"] = label
            # Saving grid pos in a separate dictionary for faster retrieval
            self.pos_to_node[x, y] = i
            self.node_to_pos[i] = (x, y)
            y += 1
            if y == resolution:
                x += 1
                y = 0
        diagonals = [(1, 1), (1, -1)]
        right_top = [(1,0),(0,1)]
        # Connecting nodes
        for i in range(0, resolution):
            for j in range(0, resolution):
                # By default a 4 neighbors grid
                for di, dj in right_top:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < resolution and 0 <= nj < resolution:
                        self.add_edge(self.pos_to_node[(i, j)], self.pos_to_node[(ni,nj)])
                # Expand to 8 if needed
                if n_neighbours >= 8:
                    for di, dj in diagonals:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < resolution and 0 <= nj < resolution:
                            self.add_edge(self.pos_to_node[(i, j)], self.pos_to_node[(ni, nj)])

    def plot_graph(
            self,
            paths=None,
            paths_colors = None,
            draw_labels = False,
            figsize= (100,100),
            dpi=100
    ):
        plt.figure(figsize=figsize, dpi=dpi)
        if paths is not None and paths_colors is not None:
            if len(paths) > len(paths_colors):
                raise ValueError('More paths than colors')

        labels = nx.get_node_attributes(self, 'label')
        pos = self.node_to_pos
        nx.draw_networkx_edges(self, pos, edge_color="gray")
        node_costs = [self.nodes[node].get('elevation', 0) for node in self.nodes()]
        nx.draw_networkx_nodes(
            self, pos,
            node_color=node_costs,
            cmap='magma', 
            node_size=10,
        )

        if self.key_nodes is not None:
            nx.draw_networkx_nodes(
                self, pos,
                nodelist=self.key_nodes,
                node_color="green",
                node_size=300,
            )
        if paths is not None:
            for i in range(len(paths)):
                path_graph = nx.path_graph(paths[i])
                nx.draw_networkx_edges(self, self.node_to_pos, width=4, edgelist=list(path_graph.edges()), edge_color=paths_colors[i])
        if draw_labels:
            nx.draw_networkx_labels(self, self.node_to_pos, labels=labels)


        water_nodes = [n for n, d in self.nodes(data=True) if d.get('is_water')]
        nx.draw_networkx_nodes(
            self, pos,
            nodelist=water_nodes,
            node_color='lightblue',
            node_size=10,
        )

        plt.axis('off')
        plt.show()

    def plot_graph_debug(
            self,
            epoch: int = -1,
            draw_pheromones = False,
            paths=None,
            paths_colors=None,
            draw_labels = False,
            figsize = (100,100),
            dpi=100
    ):
        plt.figure(figsize=figsize, dpi=dpi)
        if self.key_nodes is not None:
            node_color = ["#1f78b4" if x not in self.key_nodes else "red" for x in self.nodes()]
        else:
            node_color = "#1f78b4"
        nx.draw_networkx_nodes(self, self.node_to_pos, node_color=node_color)
        nx.draw_networkx_edges(self, self.node_to_pos, edge_color="gray")
        if paths is not None:
            for i in range(len(paths)):
                path_graph = nx.path_graph(paths[i])
                nx.draw_networkx_edges(self, self.node_to_pos, width=2, edgelist=list(path_graph.edges()), edge_color=paths_colors[i])
        if draw_pheromones:
            cmap = plt.cm.Reds
            edge_values = [self[edge[0]][edge[1]]["pheromones"] for edge in self.edges()]
            edges = nx.draw_networkx_edges(self, self.node_to_pos, width=2, edge_cmap=cmap, edge_color=edge_values)
            plt.colorbar(edges, label="Pheromone Level")

        if draw_labels:
            labels = nx.get_node_attributes(self, 'label')
            nx.draw_networkx_labels(self, self.node_to_pos, labels=labels)
        if epoch >= 0:
            plt.xlabel(f"Current epoch: {epoch}")
        plt.axis('off')
        plt.show()

    def cost_assignment(
            self,
            edges_metadata,
            assignment_function,
            default_cost: int = 100,
            print_assignment=False
    ):
        """
        Used to assign a weight/cost to each edge
        :param edges_metadata: edges informations such as elevation difference, terrain type and so on
        :param assignment_function: Evolved combinatorial function
        :param default_cost: default cost for edges. Should be bigger than the biggest cost in the edges
        :param print_assignment: Debug value used to show the assigned value
        """
        for edge in self.edges():
            # If no metadata is available, a big value is assigned as cost
            try:
                metadata = edges_metadata[edge]
                cost = assignment_function(metadata)
            except KeyError:
                cost = default_cost
                metadata = None
                if print_assignment:
                    print(f"Error: edge {edge} has no metadata. Cost set to {cost}.")
            self[edge[0]][edge[1]]['cost'] = cost
            if print_assignment:
                print(f"Assignment for edge {edge[0]}->{edge[1]} cost: {cost}")
                print(f"Metadata of edge {edge[0]}->{edge[1]}:\n{metadata}")

    def calc_path_cost(
            self,
            path,
            degree_45_penalty_factor = 100
    ):
        """
        Used to calculate path cost
        :param path: Path to evaluate
        :param degree_45_penalty_factor: Tmp factor that will be removed when the evolved GP function will take care of costs
        :return:
        """
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
        """
        Verify if the path is a valid TSP circuit
        :param path: path to evaluate
        """
        if not self.key_nodes.issubset(set(path)):
            return False
        if path[0] != path[-1]:
            return False
        return True

    def cost_normalization(self):
        all_costs = [data["cost"] for u, v, data in self.edges(data=True)]
        for v in self.nodes():
            for u in self[v]:
                cost = self[v][u]['cost']
                min_cost = min(all_costs)
                max_cost = max(all_costs)
                cost_range = max_cost - min_cost
                normalized_cost = 1 + 9 * (cost - min_cost) / (cost_range + 1e-6)
                self[v][u]['cost'] = normalized_cost
