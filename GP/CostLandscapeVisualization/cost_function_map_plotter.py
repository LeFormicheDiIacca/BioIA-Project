from rasterio.coords import BoundingBox
from GP.evolved_cost_functions import naples_second_cost_function
from TerrainGraph.edge_info import create_edge_dict
from TerrainGraph.terraingraph import create_graph

res = 200

mesh_graph = create_graph("../../Dataset/Naples/napoli.tif", "../../Dataset/Naples/napoli.pbf", resolution=res,
                          area=BoundingBox( left=13.83059, bottom=41.03715, right=14.32309, top=41.40308)
                          )

edge_dict = create_edge_dict(mesh_graph)
current_cf = naples_second_cost_function
for v in mesh_graph.nodes():
    for u in mesh_graph[v]:
        u_ordered, v_ordered = min(u, v), max(u, v)
        key = f"{u_ordered}-{v_ordered}"
        metadata = edge_dict[key]
        cost = current_cf(metadata[0], metadata[1], metadata[2])

        mesh_graph[v][u]['cost'] = cost

mesh_graph.cost_normalization()
mesh_graph.plot_edge_heatmap(output_name= "naples_second_cf.png")
