import rasterio
from rasterio.warp import transform
import geopandas as gpd 
from shapely.geometry import Point
import numpy as np
import fiona

from meshgraph import MeshGraph

fiona.drvsupport.supported_drivers['OSM'] = 'r'

def create_graph(tif_path, osm_pbf_path, resolution):
    print(f"Creating graph with resolution {resolution}:")
    print("- Running MeshGraph empty constructor...")
    G = MeshGraph(n_neighbours=8, resolution=resolution)

    print("- Creating coordinate lists...")
    src = rasterio.open(tif_path)
    bounds = src.bounds

    eastings = np.linspace(bounds.left + 1, bounds.right - 1, resolution)
    northings = np.linspace(bounds.bottom + 1, bounds.top - 1, resolution)
    utm_coordinates = [(x, y) for y in northings for x in eastings]

    longitudes, latitudes = transform(src.crs, 'EPSG:4326', eastings, northings)
    geo_coordinates = [(x, y) for y in latitudes for x in longitudes]

    print("- Reading elevation data...")
    elevations = [val[0] for val in src.sample(utm_coordinates)]

    node_ids = []
    node_geoms_native = []

    for idx, ((lon, lat), elev) in enumerate(zip(geo_coordinates, elevations)):
        i = idx // resolution
        j = idx % resolution

        node_key = G.pos_to_node[(j, i)]

        G.nodes[node_key].update({
            "x": lon,
            "y": lat,
            "elevation": elev,
            "is_water": False
        })
        
        node_ids.append(node_key)
        node_geoms_native.append(Point(lon,lat))

    print("- Reading osm data...")
    bbox = (min(longitudes), min(latitudes), max(longitudes), max(latitudes))
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

    if not osm_gdf.empty:
        mask = np.zeros(len(osm_gdf), dtype=bool)
        
        for col, kw in [('natural', 'water'), ('landuse', 'reservoir'), ('landuse', 'basin')]:
            if col in osm_gdf.columns:
                mask |= (osm_gdf[col] == kw)
        if 'water' in osm_gdf.columns:
            mask |= osm_gdf['water'].notna()
        
        water_gdf = osm_gdf[mask]
        
        if not water_gdf.empty:
            # if water_gdf.crs != src.crs:
            #     print("different")
            #     water_gdf = water_gdf.to_crs(src.crs)
            
            nodes_gdf = gpd.GeoDataFrame(
                {'id': node_ids}, 
                geometry=node_geoms_native, 
                crs='EPSG:4326'
            )
            
            water_nodes = gpd.sjoin(
                nodes_gdf, 
                water_gdf[['geometry']], 
                how='inner', 
                predicate='intersects'
            )
            
            water_node_ids = water_nodes['id'].values
            for node_id in water_node_ids:
                G.nodes[node_id]["is_water"] = True

    return G   
