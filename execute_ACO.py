
import csv
from datetime import datetime
import json
import time
import numpy as np
from pathlib import Path
from rasterio.coords import BoundingBox
from ACO.ACO_simulator import ACO_simulator
from TerrainGraph.edge_info import create_edge_dict
from TerrainGraph.terraingraph import create_graph
from TerrainGraph.path_render import visualize_paths
from cost_functions import trentino_first_cost_function,trentino_second_cost_function,naples_first_cost_function,naples_second_cost_function

REGION = "trentino"
TIF_PATH = f"Dataset/Trentino/{REGION}.tif"
PBS_PATH = f"Dataset/Trentino/{REGION}.pbf"

# REGION = "napoli"  
#TIF_PATH = f"Dataset/Naples/{REGION}.tif"
#PBS_PATH = f"Dataset/Naples/{REGION}.pbf"

today = datetime.now().strftime("%d_%m_%Y")
OUTPUT_FOLDER = f"Results/{today}_{REGION}"
FILENAME = f"PathOutputs_{REGION}"

def get_closest_indices(key_coords, bounds, resolution):
    pts = np.array(key_coords)
    lats, lons = pts[:, 0], pts[:, 1]
    x_idxs = np.round(((lons - bounds.left) / (bounds.right - bounds.left)) * (resolution - 1)).astype(int)
    y_idxs = np.round(((lats - bounds.bottom) / (bounds.top - bounds.bottom)) * (resolution - 1)).astype(int)

    x_idxs = np.clip(x_idxs, 0, resolution - 1)
    y_idxs = np.clip(y_idxs, 0, resolution - 1)

    return (y_idxs + x_idxs * resolution).tolist()

def create_file_path(folder, extension):
    file_path = folder / f"{FILENAME}.{extension}"
    counter = 1
    while file_path.exists():
        new_filename = f"{FILENAME}_{counter}.{extension}"
        file_path = folder / new_filename
        counter += 1
    return file_path

def create_bbox_with_margin(points, margin_ratio=0.2):
    # Note: Input is (lat, lon), but BoundingBox needs (lon, lat) logic (x, y)
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Calculate span (difference)
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon

    # Calculate margin based on the span
    lat_margin = lat_span * margin_ratio
    lon_margin = lon_span * margin_ratio

    # Apply margin to get new bounds
    # BoundingBox signature is (left, bottom, right, top) -> (min_lon, min_lat, max_lon, max_lat)
    return BoundingBox(
        left   = min_lon - lon_margin,
        bottom = min_lat - lat_margin,
        right  = max_lon + lon_margin,
        top    = max_lat + lat_margin
    )

if __name__ == '__main__':
    mesh_graph_parameters = {
        "n_neighbours": 8,
        "resolution": 200,
    }

    # coords for Napoli
    key_coords_list = [
        [
            (41.235685, 13.932124), #Sessa Aurunca
            (41.289622, 13.982943), #Roccamonfina
            (41.333720, 13.991184), #Conca della Campania
            (41.325470, 13.939679), #Sipicciano
            (41.252413, 14.067275), #Teano
        ],

        [
            (41.235582, 13.932468), #Sessa Aurunca
            (41.331709, 13.988714), #Conca della Campania
            (41.271147, 14.189791), #Pietramelara
            (41.246100, 14.335379), #Alvignano
            (41.191324, 14.252971), #Pontelatone
            (41.105761, 14.209431), #Capua
        ],

        [
            (41.372686, 14.074325), #Presenzano
            (41.388658, 14.199998), #Ailano
            (41.357177, 14.266955), #Sant'Angelo d'Alife
            (41.271212, 14.311113), #Dragoni
            (41.167190, 14.337277), #Piana di Monte Verna
            (41.200176, 14.192719), #Pozzillo
            (41.154053, 14.026391), #Ciamprisco
            (41.235988, 13.932308), #Sessa Aurunca
        ],

        [
            (41.0356, 13.9420), #Castel Volturno
            (41.2355, 13.9321), #Sessa Aurunca
            (41.2510, 14.0701), #Teano
            (41.1083, 14.2075), #Capua
            (41.1125, 13.8943), #Mondragone
        ]

    ]

    # coords for Trentino
    # key_coords_list = [
    #     [
    #         (46.060883,11.236782), #Pergine Valsugana
    #         (46.072728,11.162217), #oltrecastello
    #         (46.066461,11.126490), #trento
    #         (46.06306,11.09665), #sardagna
    #         (46.072764,11.058383), #Sopramonte
    #         (46.05325,11.07190), #vaneze
    #         (46.038994,11.057160), #Vason
    #     ],
    #     [
    #         (46.066461,11.126490), #trento
    #         (46.140279, 11.112309),#lavis
    #         (46.15623,11.15260), #verla
    #         (46.171863, 11.223905), #Cembra
    #         (46.229869, 11.303687), #Grauno
    #         (46.26267,11.33824), #capriana
    #         (46.278656, 11.418387), #Castello di Fiemme
    #
    #     ],
    #     [
    #         (46.318805, 11.067134), #Taio
    #         (46.37992,11.09076),#malgolo
    #         (46.413274, 11.145174), #Sarnonico
    #         (46.41762,11.20653), #mendola
    #         (46.416922, 11.238757), #Caldaro
    #         (46.420113, 11.334323), #Laives
    #     ],
    #     [
    #         (46.066461,11.126490), rento
    #         (46.08919,11.17992),#civezzano
    #         (46.125632, 11.244739),#Miola
    #         (46.225367, 11.314955), #Sover
    #         (46.24973,11.34365),#casatta
    #         (46.295118, 11.459256), #Cavalese
    #     ],
    # ]

    #key_coords_list = key_coords_list[:1]

    ant_colony_parameters = {
        "alpha": 1,
        "beta": 3,
        "rho": 0.1,
        "q0": 0.1,
        "ant_number": 50,
        "max_iterations": 100,
        "max_no_updates": 10,
        "n_best_ants": 5,
        "average_cycle_length": 4000,
        "n_iterations_before_spawn_in_key_nodes": 10
    }

    n_iterations = 3
    resilience_factor = 1

    log_data = True
    print_res = True
    print_graph = False
    save_rendered_paths = True
    synthetic_data = False

    cost_functions_list = [trentino_first_cost_function, trentino_second_cost_function, naples_first_cost_function, naples_second_cost_function]
    fields_csv = ["iteration_time", "path_cost", "path", "cost_function"]

    print("Running ACO simulation...")
    res_paths = []
    res_paths_alls = []
    color =["green", "cyan", "blue", "yellow", "red", "magenta"]

    # try:

    # iterate first through key coords to avoid rebuilding graph more than needed
    for key_coords in key_coords_list:
        area = create_bbox_with_margin(key_coords)
        mesh_graph = create_graph(TIF_PATH, PBS_PATH, mesh_graph_parameters["resolution"], area)
        edge_dict = create_edge_dict(mesh_graph)

        key_nodes = get_closest_indices(key_coords, area, mesh_graph_parameters["resolution"])
        mesh_graph.assign_key_nodes(key_nodes)

        # then iterate through the cost functions
        for f, current_cf in enumerate(cost_functions_list):
            print(f"Running cost function {f}")
            # Define specific folder for this cost function
            cf_folder = Path(OUTPUT_FOLDER) / str(f)
            cf_folder.mkdir(parents=True, exist_ok=True)

            # Update Graph Costs
            for v in mesh_graph.nodes():
                for u in mesh_graph[v]:
                    u_ordered, v_ordered = min(u, v), max(u, v)
                    key = f"{u_ordered}-{v_ordered}"
                    metadata = edge_dict[key]
                    cost = current_cf(metadata[0], metadata[1], metadata[2])
                    mesh_graph[v][u]['cost'] = cost
            mesh_graph.cost_normalization()



            aco = ACO_simulator(mesh_graph, **ant_colony_parameters)
            aco.construct_key_nodes_data(key_nodes)

            for i in range(n_iterations):
                # Config data needs to include current CF context, saving it per iteration
                config_data = {
                    "MeshGraph": mesh_graph_parameters,
                    "AntColony": ant_colony_parameters,
                    "KeyNodes": list(key_nodes),
                    "CurrentCostFunction": f
                }
                if log_data:
                    # Save JSON in the CF folder
                    file_path_json = create_file_path(cf_folder, "json")
                    with open(file_path_json, 'w') as file:
                        json.dump(config_data, file, indent=4)
                start_time = time.perf_counter()
                paths = aco.simulation(retrieve_n_best_paths = 1, log_print = True, TSP = False, resilience_factor = resilience_factor)
                end_time = time.perf_counter() - start_time

                for (path, path_cost) in paths:
                    if log_data:
                        # Append to CSV specific to this CF folder
                        file_path_csv = cf_folder / "PathOutputs.csv"
                        file_exists = file_path_csv.exists()

                        with open(file_path_csv, 'a', newline='') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=fields_csv)
                            if not file_exists:
                                writer.writeheader()

                            csv_row = {
                                "iteration_time": end_time,
                                "path_cost": path_cost,
                                "path": ", ".join(map(str, path)),
                                "cost_function": str(i)
                            }
                            writer.writerow(csv_row)

                    if print_res:
                        print(f"Time: {end_time} - Path_cost: {path_cost} - Path: {path}\n")
                    if (print_graph or save_rendered_paths) and path is not None:
                        res_paths.append(path)

                if print_graph:
                    print("Plotting mesh graph...")
                    mesh_graph.plot_graph(figsize=(35, 35), paths=res_paths, paths_colors=color)

                if save_rendered_paths:
                    file_path_html = create_file_path(cf_folder, "html")
                    print("Generating road visualization...")
                    visualize_paths(
                        mesh_graph=mesh_graph,
                        paths=res_paths,
                        key_nodes=key_nodes,
                        bbox = area,
                        output_file=file_path_html,
                    )

                res_paths_alls.append(res_paths)
                res_paths = []
        print("Small CPU sleep of 5 minutes for cooling")
        time.sleep(5 * 60)
