from TerrainGraph.terraingraph import create_graph
import random
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl
import numpy as np

import random


def generate_scenarios(scenarios_number, graph):
    # Filter for ground nodes only
    ground_nodes = [node for node, data in graph.nodes(data=True) if not data.get("is_water")]

    # Requirement: we need 2 * scenarios_number unique nodes
    if len(ground_nodes) < scenarios_number * 2:
        raise ValueError("Not enough ground nodes to create unique pairs.")

    # Shuffle once to ensure randomness
    random.shuffle(ground_nodes)

    scenarios = []
    for i in range(0, scenarios_number * 2, 2):
        start = ground_nodes[i]
        end = ground_nodes[i + 1]
        scenarios.append((start, end))

    return scenarios

def visualize_scenarios(graph,scenario, runs,
            draw_labels = False,
            figsize= (100,100),
            dpi=100
    ):
        plt.figure(figsize=figsize, dpi=dpi)
        labels = nx.get_node_attributes(graph, 'label')
        pos = graph.node_to_pos
        nx.draw_networkx_edges(graph, pos, edge_color="gray")
        node_costs = [graph.nodes[node].get('elevation', 0) for node in graph.nodes()]
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_costs,
            cmap='Greys', 
            node_size=10,
        )
        if graph.key_nodes is not None:
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=graph.key_nodes,
                node_color="green",
                node_size=300,
            )
        if draw_labels:
            nx.draw_networkx_labels(graph, graph.node_to_pos, labels=labels)
        water_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_water')]
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=water_nodes,
            node_color='lightblue',
            node_size=10,
        )
        colors = mpl.colormaps["Reds"].resampled(len(scenario)*runs)(range(len(scenario)*runs))
        k = 0
        for i in range(len(scenario)):
            for j in range(len(scenario[i])):
                nx.draw_networkx_nodes(
                    graph, pos,
                    nodelist = list(scenario[i][j]),
                    node_color=[colors[k]],
                    node_size=10,
                )
                k +=1
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    res = 100
    tif_path = "../TerrainGraph/trentino.tif"
    osm_path = "../TerrainGraph/trentino_alto_adige.pbf"
    # 5 of 6 REPLACEME
    # tif_path = "../TerrainGraph/napoli.tif"
    # osm_path = "../TerrainGraph/sud-260324.osm.pbf"

    graph = create_graph(tif_path=tif_path, osm_pbf_path=osm_path, resolution=res)
    runs = 5
    scenarios = generate_scenarios(runs,graph,res= res)
    visualize_scenarios(graph, scenarios, dpi = 200, runs = runs)

     
        
            





