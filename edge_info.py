import math
import json


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


#graph.plot_graph(figsize= (400,400), dpi = 200, draw_labels=True)



# TROVARE ALMENO DUE RANDOM NODI INIZIALE E FINALE PER CUI:
#   - IN MEZZO PASSA ACQUA
#   - UNO è IN MONTAGNA E ALTRO IN PIANURA; DA UNA PARTE PENDENZA ALTA, DALL'ALTRA PIù DOLCE
#   - PERCORSO MISTO CON PICCOLI OSTACOLI (COLLINA E LAGHETTI)

# OGNI 10 GENERAZIONI CAMBIO NODI SPECIFICI MA TENGO STRUTTURA VS OVERFITTING




edge_dict = {}

for u,v in graph.edges():
    u_ordered, v_ordered = min(u, v), max(u, v)
    key = f"{u_ordered}-{v_ordered}"
    edge_dict[key] =get_edge_metadata(graph, u,v)



with open(f"edge_dict_res{n}.json", "w") as f:
    json.dump(edge_dict, f)
    
