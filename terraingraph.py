import rasterio
from rasterio.warp import transform
import geopandas as gpd 
from shapely.geometry import Point
import numpy as np
import fiona

from meshgraph import MeshGraph

fiona.drvsupport.supported_drivers['OSM'] = 'r'

def create_graph(tif_path, osm_pbf_path, resolution):
    G = MeshGraph(key_nodes=None, n_neighbours=8, resolution=resolution)

    src = rasterio.open(tif_path)
    bounds = src.bounds
    # print(f"File CRS: {src.crs}")

    xs_native = np.linspace(bounds.left + 1, bounds.right - 1, resolution)
    ys_native = np.linspace(bounds.bottom + 1, bounds.top - 1, resolution)

    native_coords = [(x, y) for y in ys_native for x in xs_native]
    
    native_xs = [c[0] for c in native_coords]
    native_ys = [c[1] for c in native_coords]

    wgs_lons, wgs_lats = transform(src.crs, 'EPSG:4326', native_xs, native_ys)
    elevations = [val[0] for val in src.sample(native_coords)]

    node_ids = []
    node_geoms_native = []

    for idx, (lon_84, lat_84, elev, x_nat, y_nat) in enumerate(zip(wgs_lons, wgs_lats, elevations, native_xs, native_ys)):
        i = idx // resolution
        j = idx % resolution

        node_key = G.pos_to_node[(j, i)]

        G.nodes[node_key]["x"] = float(lon_84)
        G.nodes[node_key]["y"] = float(lat_84)
        G.nodes[node_key]["elevation"] = float(elev)
        G.nodes[node_key]["is_water"] = False
        
        node_ids.append(node_key)
        node_geoms_native.append(Point(x_nat, y_nat))

    bbox = (min(wgs_lons), min(wgs_lats), max(wgs_lons), max(wgs_lats))

    osm_gdf = gpd.read_file(osm_pbf_path, layer='multipolygons', bbox=bbox)

    if not osm_gdf.empty:
        mask = np.zeros(len(osm_gdf), dtype=bool)
        for col, kw in [('natural', 'water'), ('landuse', 'reservoir'), ('landuse', 'basin')]:
            if col in osm_gdf.columns:
                mask |= (osm_gdf[col] == kw)
        
        if 'water' in osm_gdf.columns:
            mask |= osm_gdf['water'].notna()
            
        water_gdf = osm_gdf[mask]

        if not water_gdf.empty:
            water_gdf = water_gdf.to_crs(src.crs)
            nodes_gdf = gpd.GeoDataFrame({'id': node_ids}, geometry=node_geoms_native, crs=src.crs)
            water_nodes = gpd.sjoin(nodes_gdf, water_gdf, how='inner', predicate='intersects')

            for node_id in water_nodes['id']:
                G.nodes[node_id]["is_water"] = True

    return G   
