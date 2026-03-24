import math
import json
import numpy as np

# adjusted with Haversine formula


def get_edge_metadata(G, u, v):
    
    # get edges coordinates
    node_u = G.nodes[u] 
    node_v = G.nodes[v]

    # Earth's ray in meters
    r = 6731000

    lat1 = math.radians(node_u["x"])
    lat2 = math.radians(node_v["x"])
    delta_lat = math.radians(node_v["x"] - node_u["x"])
    delta_lon = math.radians(node_v["y"] - node_u["y"])

    a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2)**2

    distance = 2 * r * math.asin(math.sqrt(a))

    # steepness
    elev_diff = node_v['elevation'] - node_u['elevation']
    steepness = (elev_diff / distance) if distance != 0 else 0 # in %
    
    is_water = True if (node_u['is_water'] or node_v['is_water']) else False
    return np.array([distance, abs(steepness), is_water]).astype(float)

# AS GRAPH IS NOT DIRECTED

def create_edge_dict(graph):
    edge_dict = {}    
    for u,v in graph.edges():
        u_ordered, v_ordered = min(u, v), max(u, v)
        key = f"{u_ordered}-{v_ordered}"
        ret = get_edge_metadata(graph, u,v)
        edge_dict[key] = ret
    return edge_dict

