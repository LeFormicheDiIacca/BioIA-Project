import csv
from datetime import datetime
import json
import time
import numpy as np
from pathlib import Path

from rasterio.coords import BoundingBox
from ACO.ACO_simulator import ACO_simulator
from GP.edge_info import create_edge_dict
from TerrainGraph.meshgraph import MeshGraph
from TerrainGraph.terraingraph import create_graph
from TerrainGraph.path_render import visualize_paths
from cost_functions import second_best_CF

today = datetime.now().strftime("%d_%m_%Y")
OUTPUT_FOLDER = f"Results/day_{today}"
FILENAME = "PathOutputs"

def get_closest_indices(key_coords, bounds, resolution):
    pts = np.array(key_coords)
    lats, lons = pts[:, 0], pts[:, 1]
    
    x_idxs = np.round(((lons - bounds.left) / (bounds.right - bounds.left)) * (resolution - 1)).astype(int)
    y_idxs = np.round(((lats - bounds.bottom) / (bounds.top - bounds.bottom)) * (resolution - 1)).astype(int)
    
    x_idxs = np.clip(x_idxs, 0, resolution - 1)
    y_idxs = np.clip(y_idxs, 0, resolution - 1)
    
    return (y_idxs +  x_idxs * resolution).tolist()

def create_file_path(extension):
    file_path = Path(OUTPUT_FOLDER) / f"{FILENAME}.{extension}"
    counter = 1
    while file_path.exists():
        new_filename = f"{FILENAME}_{counter}.{extension}"
        file_path = Path(OUTPUT_FOLDER) / new_filename
        counter += 1
    return file_path

if __name__ == '__main__':
    #Config values for the entire simulation
    mesh_graph_parameters = {
        "n_neighbours": 8,
        "resolution": 200,
        "area" : BoundingBox( left=11.014309, bottom=45.990134, right=11.348362, top=46.118939)
    }

    key_coords = [
        #Local benchmark 1
        [
            (46.060883,11.236782), # Pergine
            (46.066461,11.126490), # Trento
            (46.072764,11.058383), # Sopramonte
            (46.038994,11.057160), # Vason
        ],
        # Local benchmark 2
        [
            (46.066461,11.126490), # Trento
            (46.140279, 11.112309),  # Lavis
            (46.171863, 11.223905),  # Cembra
            (46.229869, 11.303687),  # Grauno
            (46.278656, 11.418387), # Fiemme

        ],
        # Local benchmark 3
        [
            (46.066461,11.126490), # Trento
            (46.084793, 11.060668),  # Cadine
            (46.050281, 10.949068),  # Sarche
        ],
        # Local benchmark 4
        [
            (46.318805, 11.067134),  # Taio
            (46.413274, 11.145174),  # Cavareno
            (46.416922, 11.238757),  # Caldaro
            (46.420113, 11.334323),  # Laives
        ],
        # Local benchmark 5
        [
            (46.066461,11.126490), # Trento
            (46.125632, 11.244739), # Pinè
            (46.225367, 11.314955),  # Sover
            (46.295118, 11.459256),  # Cavalese
        ],
        # Provincial benchmark 1
        [
            (46.066461,11.126490), # Trento
            (46.2145, 11.1206),  # Mezzocorona
            (46.420113, 11.334323),  # Laives
        ],
        # Provincial benchmark 2
        [
            (46.2145, 11.1206),  # Mezzocorona
            (46.344509, 11.289412),  # Ora
            (46.288044, 11.534489),  # Fiemme
        ],
        # Provincial benchmark 3
        [
            (46.066461,11.126490),  # Trento
            (46.21061, 11.09327),  # Mezzolombardo
            (46.29435, 11.07199),  # Mollaro
            (46.3647, 11.0316),  # Cles
        ],
    ]
    n_test_cases = len(key_coords)

    ant_colony_parameters = {
        "alpha": 1, # influence of pheromones 
        "beta": 2, # influence of edge costs
        "rho": 0.1, # pheromone evaporation rate
        "q0": 0.1, # greedy choice probability
        "ant_number": 50,
        "max_iterations": 100, # number of epochs
        "max_no_updates": 15, # how many times sim accepts not having updates before resetting
        "n_best_ants": 5, # how much elitism do you want
        "average_cycle_length": 5000, # serve per inizializzare i feromoni con dei valori sensati 'cit davide'
        "n_iterations_before_spawn_in_key_nodes": 8
    }
    config_data = {
        "MeshGraph": mesh_graph_parameters,
        "AntColony": ant_colony_parameters,
        "KeyNodes": list(key_coords),
    }
    n_iterations = 18 # how many times to run the entire simulation from scratch (used for performance evaluation)
    resilience_factor = 1 # how many independent paths to find

    # Debug configs
    log_data = True
    print_res = True
    print_graph = False
    save_rendered_paths = True 
    writer = None
    synthetic_data = False

    if log_data:
        #creates JSON file with all config of the iteration
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        file_path_json = create_file_path("json")
        with open(file_path_json, 'w') as f:
            json.dump(config_data, f, indent=4)

        #Creates a CSV used to store the best paths found by the algorithm
        file_path_csv = create_file_path("csv")
        fields_csv = ["iteration_time", "path_cost", "path"]
        try:
            csvfile = open(file_path_csv, 'w', newline='')
            writer = csv.DictWriter(csvfile, fieldnames=fields_csv)
            writer.writeheader()
        finally:
            pass

    #Creates a random mesh graph for testing
    if synthetic_data:
        edges_metadata = dict()
        #mesh_graph.cost_assignment(edges_metadata, test_cost_assignment, print_assignment=False)
    else:
        mesh_graph = create_graph("TerrainGraph/trentino.tif", "TerrainGraph/trentino_alto_adige.pbf", mesh_graph_parameters["resolution"], mesh_graph_parameters["area"])
        key_nodes = get_closest_indices(key_coords[0], mesh_graph_parameters["area"],
                                        mesh_graph_parameters["resolution"])
        mesh_graph.assign_key_nodes(key_nodes)
        edge_dict = create_edge_dict(mesh_graph)
        for v in mesh_graph.nodes():
            for u in mesh_graph[v]:
                u_ordered, v_ordered = min(u, v), max(u, v)
                key = f"{u_ordered}-{v_ordered}"
                metadata = edge_dict[key]
                cost = second_best_CF(metadata[0],metadata[1],metadata[2],metadata[3],metadata[4])
                mesh_graph[v][u]['cost'] = cost
        mesh_graph.cost_normalization()

    #Create the simulators
    aco = ACO_simulator(mesh_graph, **ant_colony_parameters)

    print("Running ACO simulation...")
    #Simulate n_iterations times
    res_paths = []
    res_paths_alls = []
    color =["green", "cyan", "blue", "yellow", "red", "magenta"]
    try:
        for i in range(n_iterations):
            key_nodes_idx = i%n_test_cases
            key_nodes = get_closest_indices(key_coords[key_nodes_idx], mesh_graph_parameters["area"], mesh_graph_parameters["resolution"])
            aco.construct_key_nodes_data(key_nodes)
            #Simulate a colony
            start_time = time.perf_counter()
            paths = aco.simulation(retrieve_n_best_paths = 1, log_print = True, TSP = False, resilience_factor = resilience_factor)
            end_time = time.perf_counter() - start_time
            for (path, path_cost) in paths:
                #Write path info in CSV
                if log_data and writer:
                    csv_row = {
                        "iteration_time": end_time,
                        "path_cost": path_cost,
                        "path": ", ".join(map(str, path))
                    }
                    writer.writerow(csv_row)
                    csvfile.flush()
                #Print in console
                if print_res:
                    print(f"Time: {end_time} - Path_cost: {path_cost} - Path: {path}\n")
                if (print_graph or save_rendered_paths) and path is not None:
                    res_paths.append(path)
            if print_graph:
                print("Plotting mesh graph...")
                mesh_graph.plot_graph(figsize=(35, 35), paths=res_paths, paths_colors=color)
            if save_rendered_paths:
                file_path_html = create_file_path("html")
                print("Generating road visualization...")
                visualize_paths(
                    mesh_graph=mesh_graph,
                    paths=res_paths,
                    key_nodes=key_nodes,
                    output_file=file_path_html,
                )
            res_paths_alls.append(res_paths)
            res_paths = []
            print("Small CPU sleep of 1 minute for cooling")
            time.sleep(1 * 60)

    except Exception as e:
        print(f"Eh kaput :(")
        print(f"Exception: {e}")



#Ants no longer stupids as fuck. Now just a little bit stupid. Maybe it was my fault :(. Sorry ants
#Glory to C and the Ants🫡