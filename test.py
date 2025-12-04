import csv
import json
import time
from pathlib import Path
from ACO.ACO_simulator import ACO_simulator
from cost_functions import test_cost_assignment
from meshgraph import MeshGraph
from sanity_check import weight_func
from terraingraph import create_graph
from path_render import visualize_paths

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
        "resolution": 20
    }
    ant_colony_parameters = {
        "alpha": 1,
        "beta": 3,
        "rho": 0.2,
        "q0": 0.1,
        "ant_number": 50,
        "max_iterations": 100,
        "max_no_updates": 15,
        "n_best_ants": 5,
        "average_cycle_length": 9000,
        "n_iterations_before_spawn_in_key_nodes": 8
    }
    key_nodes = {15, 381, 99, 210, 5, 294, 142, 337, 78, 266}
    config_data = {
        "MeshGraph": mesh_graph_parameters,
        "AntColony": ant_colony_parameters,
        "KeyNodes": list(key_nodes)
    }
    #Debug configs
    log_data = True
    print_res = True
    print_graph = False
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
    synthetic_data = True
    if synthetic_data:
        mesh_graph = MeshGraph(key_nodes=key_nodes,**mesh_graph_parameters)
        edges_metadata = dict()
        mesh_graph.cost_assignment(edges_metadata, test_cost_assignment, print_assignment=False)
    else:
        mesh_graph = create_graph("trentino.tif", "trentino_alto_adige.pbf", resolution=100)
        mesh_graph.assign_key_nodes(key_nodes)
        for v in mesh_graph.nodes():
            for u in mesh_graph[v]:
                cost = weight_func(mesh_graph, v,u, None)
                mesh_graph[v][u]['cost'] = cost
        mesh_graph.cost_normalization()

    #Create the simulators
    aco = ACO_simulator(mesh_graph, **ant_colony_parameters)

    #Simulate n_iterations times
    res_paths = []
    color =["green", "cyan", "blue", "yellow", "red", "magenta"]
    n_iterations = 100
    try:
        for i in range(n_iterations):
            #Simulate a colony
            start_time = time.perf_counter()
            paths = aco.simulation(retrieve_n_best_paths = 1, log_print = False, TSP = False, resilience_factor = 1)
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
                if print_graph:
                    if path is not None:
                        res_paths.append(path)

    finally:
        if print_graph:
            if synthetic_data:
                mesh_graph.plot_graph(figsize=(35, 35), paths = res_paths, paths_colors = color)
            else:
                visualize_paths(
                    mesh_graph=mesh_graph,
                    paths=res_paths,
                    key_nodes=key_nodes,
                    output_file="my_geo_paths.html",
                )



#Ants no longer stupids as fuck. Now just a little bit stupid. Maybe it was my fault :(. Sorry ants
#Glory to C and the Ants🫡
"""            
TODO:          
    1. Need to fine tune the ACO hyperparameters and we are done
    2. Need to improve 2-opt and path optimization
    3. Sometimes tsp problem ignored and some nodes are repeated? Ignore
    4. Formalize better the steiner tree mode?
"""
