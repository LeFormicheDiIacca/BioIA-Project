import json
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from TerrainGraph.terraingraph import create_graph

import numpy as np
import random
import json


def generate_scenarios_from_npz(runs, npz_path, output_json=None):
    """
    Genera scenari caricando i dati direttamente dal file .npz precomputato.
    """
    # 1. Caricamento dati
    data = np.load(npz_path)

    # Recuperiamo le feature. Nota: 'steep' qui rappresenta l'elevazione/pendenza per arco.
    # Per categorizzare i nodi come "high/low" in assenza di un array 'nodes_elevation',
    # usiamo una media della pendenza degli archi collegati o i dati disponibili.
    dist = data['dist']
    steep = data['steep']
    water_edges = data['water']
    num_nodes = int(data['num_nodes'])
    res = int(np.sqrt(num_nodes))
    half = res // 2

    # 2. Definizione Quadranti (Logica basata sugli indici della griglia)
    indices = np.arange(num_nodes)
    rows = indices // res
    cols = indices % res

    quad1 = indices[(rows < half) & (cols < half)]
    quad2 = indices[(rows < half) & (cols >= half)]
    quad3 = indices[(rows >= half) & (cols >= half)]
    quad4 = indices[(rows >= half) & (cols < half)]

    # 3. Categorizzazione Nodi
    # Troviamo i nodi che toccano l'acqua (almeno un arco incidente è acqua)
    # Usiamo csr_indices e csr_indptr per navigare il grafo senza NetworkX
    csr_indices = data['csr_indices']
    csr_indptr = data['csr_indptr']
    csr_data = data['csr_data']  # Contiene l'indice dell'arco + 1

    water_nodes = set()
    high_nodes = set()
    low_nodes = set()

    # Percentili per elevazione (usiamo la pendenza degli archi come proxy se non hai l'elevazione nodo)
    # Nota: Se hai l'elevazione specifica dei nodi nell'NPZ, usa quella.
    threshold_low = np.percentile(steep, 30)
    threshold_high = np.percentile(steep, 70)

    for i in range(num_nodes):
        # Esaminiamo gli archi del nodo i
        start_ptr = csr_indptr[i]
        end_ptr = csr_indptr[i + 1]

        node_steepness = []
        is_near_water = False

        for ptr in range(start_ptr, end_ptr):
            edge_idx = csr_data[ptr] - 1
            if water_edges[edge_idx]:
                is_near_water = True
            node_steepness.append(steep[edge_idx])

        if is_near_water:
            water_nodes.add(i)

        avg_steep = np.mean(node_steepness) if node_steepness else 0
        if avg_steep > threshold_high:
            high_nodes.add(i)
        elif avg_steep < threshold_low:
            low_nodes.add(i)

    # Convertiamo in liste per il campionamento
    water_list = list(water_nodes)
    high_list = list(high_nodes)
    low_list = list(low_nodes)

    # Fallback se le liste sono vuote
    if not water_list: water_list = indices.tolist()
    if not high_list: high_list = indices.tolist()
    if not low_list: low_list = indices.tolist()

    final_scenarios = []
    taken = set()

    # 4. Generazione Coppie
    for _ in range(runs):
        # Categoria 1: WATER (Attraversamento acqua)
        w_node = random.choice(water_list)
        # Cerchiamo un punto lontano dal nodo acqua per forzare il percorso
        r_w, c_w = w_node // res, w_node % res
        start_w = random.choice(quad1 if r_w >= half else quad3)
        finish_w = random.choice(quad2 if c_w < half else quad4)
        final_scenarios.append([int(start_w), int(finish_w)])

        # Categoria 2: ELEVATION (Low to High)
        s_e = random.choice(low_list)
        f_e = random.choice(high_list)
        final_scenarios.append([int(s_e), int(f_e)])

        # Categoria 3: DISTANT (Estremi opposti)
        r = random.random()
        if r < 0.5:
            s_d, f_d = random.choice(quad1), random.choice(quad3)
        else:
            s_d, f_d = random.choice(quad2), random.choice(quad4)
        final_scenarios.append([int(s_d), int(f_d)])

    # 5. Salvataggio (Opzionale)
    output_data = {
        "metadata": {
            "num_scenarios": len(final_scenarios),
            "npz_source": npz_path
        },
        "scenarios": final_scenarios
    }

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Scenari salvati in {output_json}")

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
    scenarios = generate_scenarios_from_npz(20,f"../TerrainGraph/precomputed_map_trentino_{res}.npz", output_json=f"scenarios_trentino{res}.json")
    #visualize_scenarios(graph, scenarios, dpi = 200, runs = runs)

     
        
            





