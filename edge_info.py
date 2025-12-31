from terraingraph import create_graph
import math
import json

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
    return (dist, float(inclination), float(elev_u), float(elev_v), is_water)


#graph.plot_graph(figsize= (400,400), dpi = 200, draw_labels=True)



# TROVARE ALMENO DUE RANDOM NODI INIZIALE E FINALE PER CUI:
#   - IN MEZZO PASSA ACQUA
#   - UNO è IN MONTAGNA E ALTRO IN PIANURA; DA UNA PARTE PENDENZA ALTA, DALL'ALTRA PIù DOLCE
#   - PERCORSO MISTO CON PICCOLI OSTACOLI (COLLINA E LAGHETTI)

# OGNI 10 GENERAZIONI CAMBIO NODI SPECIFICI MA TENGO STRUTTURA VS OVERFITTING


nodes_with_water = [(15,36), (862, 1177), (2800, 3040)]

nodes_different_altitude = [(110,912), (159, 868), (793, 6046)]

nodes_through_obstacles = [(2996, 2214), (2586, 3615), (1837, 3821)]

scenarios = [nodes_with_water[0], nodes_different_altitude[0], nodes_through_obstacles[0]]

edge_dict = {}

for u,v in graph.edges():
    u_ordered, v_ordered = min(u, v), max(u, v)
    key = f"{u_ordered}-{v_ordered}"
    edge_dict[key] =get_edge_metadata(graph, u,v)



with open(f"edge_dict_res{n}.json", "w") as f:
    json.dump(edge_dict, f)
    
