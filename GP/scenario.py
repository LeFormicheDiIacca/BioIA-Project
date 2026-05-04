import numpy as np
import random
import json
import os


def generate_scenarios_from_npz(runs, npz_path, output_json=None, seed=None):

    if seed is None:
        seed = random.getrandbits(32)

    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(npz_path):
        print(f"Error: File {npz_path} not found!")
        return []

    data = np.load(npz_path)
    dist = data['dist']
    steep = data['steep']
    water_edges = data['water']
    num_nodes = int(data['num_nodes'])
    res = int(np.sqrt(num_nodes))
    half = res // 2

    indices = np.arange(num_nodes)
    rows = indices // res
    cols = indices % res

    quad1 = indices[(rows < half) & (cols < half)]
    quad2 = indices[(rows < half) & (cols >= half)]
    quad3 = indices[(rows >= half) & (cols >= half)]
    quad4 = indices[(rows >= half) & (cols < half)]

    csr_indices = data['csr_indices']
    csr_indptr = data['csr_indptr']
    csr_data = data['csr_data']

    water_nodes = set()
    high_nodes = set()
    low_nodes = set()

    threshold_low = np.percentile(steep, 30)
    threshold_high = np.percentile(steep, 70)

    for i in range(num_nodes):
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

    water_list = list(water_nodes) or indices.tolist()
    high_list = list(high_nodes) or indices.tolist()
    low_list = list(low_nodes) or indices.tolist()

    final_scenarios = []


    for _ in range(runs):
        # WATER
        w_node = random.choice(water_list)
        r_w, c_w = w_node // res, w_node % res
        start_w = random.choice(quad1 if r_w >= half else quad3)
        finish_w = random.choice(quad2 if c_w < half else quad4)
        final_scenarios.append([int(start_w), int(finish_w)])

        # ELEVATION
        s_e = random.choice(low_list)
        f_e = random.choice(high_list)
        final_scenarios.append([int(s_e), int(f_e)])

        # DISTANT
        if random.random() < 0.5:
            s_d, f_d = random.choice(quad1), random.choice(quad3)
        else:
            s_d, f_d = random.choice(quad2), random.choice(quad4)
        final_scenarios.append([int(s_d), int(f_d)])

    output_data = {
        "metadata": {
            "num_scenarios": len(final_scenarios),
            "npz_source": npz_path,
            "seed": seed,
            "distribution": {"runs_per_type": runs}
        },
        "scenarios": final_scenarios
    }

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Scenarios saved in {output_json} (Seed: {seed})")

    return final_scenarios

if __name__ == "__main__":
    res = 200
    npz_path = f"../Dataset/Naples/precomputed_map_napoli_{res}.npz"
    #npz_path = f"../Dataset/Trentino/precomputed_map_trentino_{res}.npz"

    scenarios = generate_scenarios_from_npz(
        runs=20,
        npz_path=npz_path,
        output_json=f"../Dataset/Naples/scenarios_napoli_{res}.json",
        seed=None
    )

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

     
        
            





