import csv
import json
import time
from pathlib import Path

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
    ant_colony_parameters = {"alpha": 1,"beta": 2, "rho": 0.1, "ant_number": 25, "max_iterations": 1000}
    key_nodes = {1, 44, 59, 81}
    config_data = {
        "MeshGraph": mesh_graph_parameters,
        "AntColony": ant_colony_parameters,
        "KeyNodes": list(key_nodes)
    }
    log_data = False
    print_res = True
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
    try:
        for i in range(n_iterations):
            mesh_graph = MeshGraph(**mesh_graph_parameters)
            edges_metadata = dict()
            mesh_graph.cost_assignment(edges_metadata, test_cost_assignment, print_assignment=False)
            aco = ACO_simulator(mesh_graph, key_nodes, **ant_colony_parameters)

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
            #for edge in mesh_graph.edges():
            #    print(f"{edge[0]}->{edge[1]}\nMetadata:{mesh_graph[edge[0]][edge[1]]}")
            #mesh_graph.plot_graph([(path, "red")], key_nodes = key_nodes)
    finally:
        pass

#Ants still stupids as fuck
"""
TODO:
    1. Add possibility to see what each ant is doing
    2. Need to fine tune the ACO hyperparameters and we are done
    3. Should local update be done when the ants finish with their path calculation?
        Shared memory and let's go for now. Glory to C and the Ants🫡
    4. Convert to k-indpaths
"""