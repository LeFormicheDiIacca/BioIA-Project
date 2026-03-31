import numpy as np

from TerrainGraph.edge_info import create_edge_dict
from TerrainGraph.terraingraph import create_graph
from scipy.sparse import csr_matrix

REGION = "trentino"

def create_edge_index_matrix(graph, node_to_idx):
    rows = []
    cols = []
    data = []
    for i, (u, v) in enumerate(graph.edges()):
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        rows.append(u_idx)
        cols.append(v_idx)
        data.append(i + 1)

        rows.append(v_idx)
        cols.append(u_idx)
        data.append(i + 1)

    return csr_matrix((data, (rows, cols)), shape=(len(node_to_idx), len(node_to_idx)))


def main():
    res = 200
    print("-Creating graph from TIF and PBF...")
    graph = create_graph(f"{REGION}.tif", f"{REGION}.pbf", resolution=res)

    print("-Creating edge dictionary")
    edge_dict = create_edge_dict(graph)

    print("-Conversion into matrices")
    node_list = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    #Feature Extraction
    distances = []
    steepness = []
    waters = []

    for u, v in graph.edges():
        u_ordered, v_ordered = min(u, v), max(u, v)
        d, s, w = edge_dict[f"{u_ordered}-{v_ordered}"]
        distances.append(d)
        steepness.append(s)
        waters.append(w)

    #Slicing into 3 different arrays
    dist_array = np.array(distances, dtype=np.float32)
    steep_array = np.array(steepness, dtype=np.float32)
    water_array = np.array(waters, dtype=np.bool_)

    #Turnign graph into Matrix CSR
    edge_index_matrix = create_edge_index_matrix(graph, node_to_idx)

    #Saving on disk
    save_path = f"precomputed_map_{REGION}_{res}.npz"
    np.savez_compressed(
        save_path,
        dist=dist_array,
        steep=steep_array,
        water=water_array,
        csr_indices=edge_index_matrix.indices,
        csr_indptr=edge_index_matrix.indptr,
        csr_data=edge_index_matrix.data,
        num_nodes=len(node_list)
    )
    print(f"-Finished. Operation saved in {save_path}")

if __name__ == "__main__":
    main()
