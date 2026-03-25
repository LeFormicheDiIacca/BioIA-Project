import math
import numpy as np

def get_edge_metadata(G, u, v):

    #Retrieve edges coordinates
    node_u = G.nodes[u]
    node_v = G.nodes[v]

    #Earth's ray in meters
    r = 6371000

    lon1 = math.radians(node_u["x"])
    lon2 = math.radians(node_v["x"])
    lat1 = math.radians(node_u["y"])
    lat2 = math.radians(node_v["y"])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    #Haversine Distance
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    distance = 2 * r * math.asin(math.sqrt(a))

    #Steepness Calculation
    elev_diff = node_v['elevation'] - node_u['elevation']
    ratio = abs(elev_diff / distance)
    angular_steepness = math.atan(ratio)
    steepness_norm = angular_steepness/(math.pi / 2) #In range [0,1]

    is_water = True if (node_u['is_water'] or node_v['is_water']) else False

    return np.array([distance, abs(steepness_norm), is_water]).astype(float)

# AS GRAPH IS NOT DIRECTED

def create_edge_dict(graph):
    edge_dict = {}
    d_max = 0
    for u,v in graph.edges():
        u_ordered, v_ordered = min(u, v), max(u, v)
        key = f"{u_ordered}-{v_ordered}"
        ret = get_edge_metadata(graph, u,v)
        edge_dict[key] = ret
        dist = ret[0]
        if dist > d_max:
            d_max = dist
    if d_max == 0:
        return edge_dict
    for ret in edge_dict.values():
        ret[0] = ret[0]/d_max
    return edge_dict

