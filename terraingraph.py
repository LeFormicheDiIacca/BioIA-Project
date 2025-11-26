import rasterio
from rasterio.warp import transform
import geopandas as gpd 
from shapely.geometry import Point
import networkx as nx
import numpy as np
    
import fiona

from meshgraph import MeshGraph

fiona.drvsupport.supported_drivers['OSM'] = 'r'

def create_graph(tif_path, osm_pbf_path, resolution):
    G = MeshGraph(key_nodes=None,n_neighbours = 8, n_row = resolution, n_col = resolution)

    src = rasterio.open(tif_path)

    bounds = src.bounds
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top

    # print(f"File CRS: {src.crs}")
    # print(bounds)

    lons = np.linspace(bounds.left + 1, bounds.right - 1, resolution)
    lats = np.linspace(bounds.bottom + 1, bounds.top - 1, resolution)

    coords = [(lon, lat) for lat in lats for lon in lons]
    samples = src.sample(coords)

    node_ids = []
    node_geoms = []

    # ============================ elevation info ========================================

    for idx, val in enumerate(samples):
        i = idx // resolution  
        j = idx % resolution   
        
        x = float(lons[j])
        y = float(lats[i])

        node_key = G.pos_to_node[(j,i)]

        G.nodes[node_key]["elevation"] = float(val[0])
        G.nodes[node_key]["x"] = x
        G.nodes[node_key]["y"] = y
        G.nodes[node_key]["is_water"] = False
        
        node_ids.append(node_key)
        node_geoms.append(Point(x, y))

    # ============================ coordinate conversions ========================================

    src_crs = src.crs
    dst_crs = 'EPSG:4326'
    
    utm_xs = [left, right]
    utm_ys = [bottom, top]
    wgs84_lons, wgs84_lats = transform(src_crs, dst_crs, utm_xs, utm_ys)
    
    bbox = (
        min(wgs84_lons), 
        min(wgs84_lats), 
        max(wgs84_lons), 
        max(wgs84_lats)
    )
    
    # ============================ query local file to get osm info ===================================

    osm_gdf = gpd.read_file(osm_pbf_path, layer='multipolygons', bbox=bbox)

    if not osm_gdf.empty:
        mask = np.zeros(len(osm_gdf), dtype=bool)
        
        if 'natural' in osm_gdf.columns:
            mask |= (osm_gdf['natural'] == 'water')
        if 'water' in osm_gdf.columns:
            mask |= osm_gdf['water'].notna()
        if 'landuse' in osm_gdf.columns:
            mask |= osm_gdf['landuse'].isin(['reservoir', 'basin'])
        
        if 'other_tags' in osm_gdf.columns:
            mask |= osm_gdf['other_tags'].astype(str).str.contains('water|reservoir|basin', case=False, na=False)

        water_gdf = osm_gdf[mask]

        if not water_gdf.empty:
            water_gdf = water_gdf.to_crs(src.crs)
            nodes_gdf = gpd.GeoDataFrame({'id': node_ids}, geometry=node_geoms, crs=src.crs)
            
            water_nodes = gpd.sjoin(nodes_gdf, water_gdf, how='inner', predicate='intersects')

            for node_id in water_nodes['id']:
                G.nodes[node_id]["is_water"] = True


    return G
