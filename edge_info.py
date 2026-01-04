import math
import json
from terraingraph import create_graph


n = 80
tif_path = "trentino.tif"
osm_path = "trentino_alto_adige.pbf"
graph = create_graph(tif_path=tif_path, osm_pbf_path=osm_path, resolution=n)

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
    return (dist, abs(float(inclination)), float(elev_u), float(elev_v), is_water)

def create_edge_dict(graph):
    edge_dict = {}
    for u,v in graph.edges():
        u_ordered, v_ordered = min(u, v), max(u, v)
        key = f"{u_ordered}-{v_ordered}"
        edge_dict[key] =get_edge_metadata(graph, u,v)
    return edge_dict


# with open(f"edge_dict_res{n}.json", "w") as f:
#     json.dump(edge_dict, f)


# TODO: normalizzo i valori nella mia funzione evaluate (max_normalization, dove max_distance è resolution-specific)


