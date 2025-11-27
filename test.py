import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from ACO.ACO_simulator import ACO_simulator
from cost_functions import test_cost_assignment
from meshgraph import MeshGraph
from terraingraph import create_graph

OUTPUT_FOLDER = "Results"
FILENAME = "PathOutputs"

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
        "resolution": 10
    }
    ant_colony_parameters = {
        "alpha": 1,
        "beta": 2,
        "rho": 0.1,
        "q0": 0.05,
        "ant_number": 25,
        "max_iterations": 50,
        "max_no_updates": 10,
        "n_best_ants": 5,
        "average_cycle_length": 4000,
        "n_iterations_before_spawn_in_key_nodes": 5
    }
    key_nodes = {1, 973, 546, 345, 871, 675}
    config_data = {
        "MeshGraph": mesh_graph_parameters,
        "AntColony": ant_colony_parameters,
        "KeyNodes": list(key_nodes)
    }
    #Debug configs
    log_data = False
    print_res = True
    print_graph = True
    writer = None

    """
    n_paths = 8
    consider_key_nodes_neighborhood = False
    if n_paths > mesh_graph_parameters["n_neighbours"]:
        if consider_key_nodes_neighborhood:
            print("WARNING: More paths than node connections. We are considering the key nodes neighborhood but could be not enough.")
        else:
            raise Exception("Error: n_paths must be equal or lower to n_neighbours")
    """

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
    synthetic_data = False
    if synthetic_data:
        mesh_graph = MeshGraph(key_nodes=key_nodes,**mesh_graph_parameters)
        edges_metadata = dict()
        mesh_graph.cost_assignment(edges_metadata, test_cost_assignment, print_assignment=False)
    else:
        mesh_graph = create_graph("trentino.tif", "trentino_alto_adige.pbf", resolution=100)
    #Create the simulators
    aco = ACO_simulator(mesh_graph, **ant_colony_parameters)

    #Simulate n_iterations times
    res_paths = []
    color =["red", "green", "blue", "yellow", "cyan", "magenta"]
    n_iterations = 1
    try:
        for i in range(n_iterations):
            #Simulate a colony
            start_time = time.perf_counter()
            paths = aco.simulation(retrieve_n_best_paths = 1, draw_heatmap = True)
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
                #Print in console
                if print_res:
                    print(f"Time: {end_time} - Path_cost: {path_cost} - Path: {path}\n")
                if print_graph:
                    if path is not None:
                        res_paths.append(path)

    finally:
        if print_graph:
            mesh_graph.plot_graph(figsize=(10, 10), paths = res_paths, paths_colors = color)

#Ants no longer stupids as fuck. Now just a little bit stupid. Maybe it was my fault :(. Sorry ants
#Glory to C and the Ants🫡
"""            
TODO:          
    1. Need to fine tune the ACO hyperparameters and we are done
    2. Convert to find k as different as possible routes
"""