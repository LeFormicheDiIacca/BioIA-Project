import json
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from TerrainGraph.terraingraph import create_graph


def generate_and_save_scenarios(num_scenarios, res, npz_path, output_json="scenarios.json"):
    """
    Genera x scenari basandosi sui dati NumPy dell'NPZ.
    Gestisce numeri non divisibili per 3 e mancanze di dati.
    """
    data = np.load(npz_path)

    is_water = data.get('water', np.zeros(res * res))
    elevations = data.get('elevation', data.get('steep', np.zeros(res * res)))

    num_nodes = res * res
    indices = np.arange(num_nodes)
    rows = indices // res
    cols = indices % res
    half = res // 2

    #Define quadrants
    q1 = indices[(rows < half) & (cols < half)]
    q2 = indices[(rows < half) & (cols >= half)]
    q3 = indices[(rows >= half) & (cols >= half)]
    q4 = indices[(rows >= half) & (cols < half)]

    #Categorize
    water_nodes = np.where(is_water > 0.5)[0]
    low_nodes = np.where(elevations < np.percentile(elevations, 30))[0]
    high_nodes = np.where(elevations > np.percentile(elevations, 70))[0]

    if len(water_nodes) == 0: water_nodes = indices
    if len(low_nodes) == 0: low_nodes = indices
    if len(high_nodes) == 0: high_nodes = indices

    final_scenarios = []

    #Divide by 3
    base_count = num_scenarios // 3
    remainder = num_scenarios % 3
    counts = [base_count + (1 if i < remainder else 0) for i in range(3)]

    #Water
    for _ in range(counts[0]):
        w_node = random.choice(water_nodes)
        r_w, c_w = w_node // res, w_node % res
        start = random.choice(q1 if r_w >= half else q3)
        finish = random.choice(q2 if c_w < half else q4)
        final_scenarios.append([int(start), int(finish)])

    #Elevation
    for _ in range(counts[1]):
        start = random.choice(low_nodes)
        finish = random.choice(high_nodes)
        if random.random() > 0.5: start, finish = finish, start
        final_scenarios.append([int(start), int(finish)])

    #Distant
    for _ in range(counts[2]):
        if random.random() > 0.5:
            start, finish = random.choice(q1), random.choice(q3)
        else:
            start, finish = random.choice(q2), random.choice(q4)
        final_scenarios.append([int(start), int(finish)])

    random.shuffle(final_scenarios)

    #Save
    output_data = {
        "metadata": {
            "num_scenarios": num_scenarios,
            "distribution": {"water": counts[0], "elevation": counts[1], "distant": counts[2]}
        },
        "scenarios": final_scenarios
    }

    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=4)

    return final_scenarios

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
    res = 200
    #tif_path = "../TerrainGraph/trentino.tif"
    #osm_path = "../TerrainGraph/trentino_alto_adige.pbf"
    # 5 of 6 REPLACEME
    # tif_path = "../TerrainGraph/napoli.tif"
    # osm_path = "../TerrainGraph/sud-260324.osm.pbf"

    #graph = create_graph(tif_path=tif_path, osm_pbf_path=osm_path, resolution=res)
    #runs = 5
    scenarios = generate_and_save_scenarios(50,res,f"../TerrainGraph/precomputed_map_trentino_{res}.npz", output_json=f"scenarios_trentino{res}.json")
    #visualize_scenarios(graph, scenarios, dpi = 200, runs = runs)

     
        
            





