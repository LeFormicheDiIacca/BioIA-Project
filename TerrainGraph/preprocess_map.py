import numpy as np
import os

from GP.edge_info import create_edge_dict
from TerrainGraph.terraingraph import create_graph
from scipy.sparse import csr_matrix


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
    graph = create_graph("trentino.tif", "trentino_alto_adige.pbf", resolution=res)

    print("-Creating edge dictionary")
    edge_dict = create_edge_dict(graph)

    print("-Conversion into matrices")
    node_list = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    #Feature Extraction
    edge_features = []
    for u, v in graph.edges():
        u_ordered, v_ordered = min(u, v), max(u, v)
        features = edge_dict[f"{u_ordered}-{v_ordered}"]
        edge_features.append(features)

    #Slicing into 3 different arrays
    edge_features_columns = [np.array(c, dtype=np.float32) for c in zip(*edge_features)]

    #Turnign graph into Matrix CSR
    edge_index_matrix = create_edge_index_matrix(graph, node_to_idx)

    #Saving on disk
    save_path = f"precomputed_map_trentino_{res}.npz"
    np.savez_compressed(
        save_path,
        dist=edge_features_columns[0],
        steep=edge_features_columns[1],
        water=edge_features_columns[2],
        csr_indices=edge_index_matrix.indices,
        csr_indptr=edge_index_matrix.indptr,
        csr_data=edge_index_matrix.data,
        num_nodes=len(node_list)
    )
    print(f"-Finished. Operation saved in {save_path}")

if __name__ == "__main__":
    main()