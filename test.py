import csv
import json
import time
from copy import deepcopy
from pathlib import Path

from networkx.classes import neighbors

from ACO.ACO_simulator import ACO_simulator
from cost_functions import test_cost_assignment
from meshgraph import MeshGraph

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
    mesh_graph_parameters = {"n_neighbours": 8, "n_row": 10, "n_col": 10}
    n_paths = 8
    consider_key_nodes_neighborhood = False
    if n_paths > mesh_graph_parameters["n_neighbours"]:
        if consider_key_nodes_neighborhood:
            print("WARNING: More paths than node connections. We are considering the key nodes neighborhood but could be not enough.")
        else:
            raise Exception("Error: n_paths must be equal or lower to n_neighbours")

    ant_colony_parameters = {"alpha": 1, "beta": 2, "rho": 0.1, "ant_number": 25, "max_iterations": 10, "max_no_updates": 50, "n_best_ants": 5, "average_cycle_lenght": 3600}
    key_nodes = {1, 44, 59, 81}
    config_data = {
        "MeshGraph": mesh_graph_parameters,
        "AntColony": ant_colony_parameters,
        "KeyNodes": list(key_nodes)
    }
    log_data = False
    print_res = True
    print_graph = True
    n_iterations = 1
    writer = None
    if log_data:
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        file_path_json = create_file_path("json")
        with open(file_path_json, 'w') as f:
            json.dump(config_data, f, indent=4)
        file_path_csv = create_file_path("csv")
        fields_csv = ["iteration_time", "path_cost", "path"]
        try:
            csvfile = open(file_path_csv, 'w', newline='')
            writer = csv.DictWriter(csvfile, fieldnames=fields_csv)
            writer.writeheader()
        finally:
            pass

    mesh_graph = MeshGraph(key_nodes=key_nodes,**mesh_graph_parameters)
    mesh_graph.plot_graph_debug(draw_labels = True, figsize=(10,10))
    mesh_graph_to_draw = deepcopy(mesh_graph)
    edges_metadata = dict()
    mesh_graph.cost_assignment(edges_metadata, test_cost_assignment, print_assignment=False)
    aco = ACO_simulator(mesh_graph, **ant_colony_parameters)
    paths = []
    color =["red", "green", "blue", "yellow", "cyan", "magenta"]
    try:
        for i in range(n_iterations):

            start_time = time.perf_counter()
            path, path_cost = aco.simulation()
            end_time = time.perf_counter() - start_time

            if log_data and writer:
                csv_row = {
                    "iteration_time": end_time,
                    "path_cost": path_cost,
                    "path": ", ".join(map(str, path))
                }
                writer.writerow(csv_row)
            if print_res:
                print(f"Time: {end_time} - Path_cost: {path_cost} - Path: {path}\n")
            if print_graph:
                if path is not None:
                    paths.append((path,color[i]))
            if path:
                path = [node for node in path if node not in key_nodes]
                #mesh_graph.remove_nodes_from(path)
                #mesh_graph.assign_edge_indexes()
    finally:
        if print_graph:
            mesh_graph.plot_graph_debug(figsize=(10, 10), paths = paths)

#Ants still stupids as fuck
"""
TODO:
    1. Add possibility to see what each ant is doing
    2. Need to fine tune the ACO hyperparameters and we are done
    3. Convert to find k as different as possible routes
    4. Dopo qualche generazioen partono dalle città siccome cis ono giàò i feromoni in giro
    5.HeatMap dei feromoni (Davide)
    6.formiche più intelligenti (Davide)

    Should local update be done when the ants finish with their path calculation?
        Shared memory and let's go for now. Glory to C and the Ants🫡
"""