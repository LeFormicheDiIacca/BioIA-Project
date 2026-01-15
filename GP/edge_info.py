import math
import json
import numpy as np

def get_edge_metadata(G, u, v):
    
    node_u = G.nodes[u]
    node_v = G.nodes[v]

    dx = (node_u['x'] - node_v['x']) * 77000 # per riportare a km (larghezza meridiano a TN)
    dy = (node_u['y'] - node_v['y']) * 111320 # per riportare a km
    dist = math.sqrt(dx**2 + dy**2) # in meters
    
    # inclination
    elev_diff = node_v['elevation'] - node_u['elevation']
    inclination = (elev_diff / dist) * 100 if dist != 0 else 0 # in %

    # elevation    
    elev_u = node_u['elevation']
    elev_v = node_v['elevation']
    
    is_water = 1.0 if (node_u['is_water'] or node_v['is_water']) else 0.0
    return np.array([dist, abs(inclination), elev_u, elev_v, is_water]).astype(float)

def create_edge_dict(graph):
    edge_dict = {}
    max_elev = 0
    max_dist = 0
    for u,v in graph.edges():
        u_ordered, v_ordered = min(u, v), max(u, v)
        key = f"{u_ordered}-{v_ordered}"
        ret = get_edge_metadata(graph, u,v)
        edge_dict[key] = ret
        max_elev = max(max_elev, ret[2], ret[3])
        max_dist = max(max_dist, ret[0])
    norm_matrix = np.array([max_dist, 90, max_elev, max_elev, 1])
    # normalization gives more robustness in case of varying resolution
    for el in edge_dict:
        edge_dict[el] /= norm_matrix
    return edge_dict

