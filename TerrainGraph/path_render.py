import math
import numpy as np
import folium
import os
from scipy.interpolate import splprep, splev

# def smooth_path(lats, lons, smoothing_points=20):
#     clean_lats, clean_lons = [], []
#     for i in range(len(lats)):
#         if i == 0 or not (lats[i] == lats[i-1] and lons[i] == lons[i-1]):
#             clean_lats.append(lats[i])
#             clean_lons.append(lons[i])
#
#     if len(clean_lats) < 3:
#         return clean_lats, clean_lons
#
#     tck, u = splprep([clean_lats, clean_lons], s=0, k=3)
#     u_new = np.linspace(0, 1, smoothing_points)
#     s_lats, s_lons = splev(u_new, tck)
#     return s_lats, s_lons

def calculate_path_statistics(mesh_graph, paths):
    total_distance_2d = 0.0
    total_length_3d = 0.0
    slopes = []

    # 1. Get a reference latitude to calculate metric conversion factors
    # We just need one valid node to determine where in the world we are
    ref_node = next(iter(mesh_graph.nodes(data=True)))
    center_lat = ref_node[1].get('y', 0)

    # 2. Calculate constant conversion factors for this specific latitude
    # 1 deg lat is approx 111,132 meters everywhere
    # 1 deg lon depends on latitude: 111,132 * cos(lat)
    DEG_TO_RAD = math.pi / 180
    lat_scale = 111132.0
    lon_scale = 111132.0 * math.cos(center_lat * DEG_TO_RAD)

    for path in paths:
        if not path or len(path) < 2:
            continue
            
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            if u not in mesh_graph.nodes or v not in mesh_graph.nodes:
                continue

            node_u = mesh_graph.nodes[u]
            node_v = mesh_graph.nodes[v]

            # Get differences in degrees
            d_lat = node_v.get('y', 0) - node_u.get('y', 0)
            d_lon = node_v.get('x', 0) - node_u.get('x', 0)
            
            # Convert to meters using pre-calculated scales
            dy_meters = d_lat * lat_scale
            dx_meters = d_lon * lon_scale
            
            # 2D Euclidean distance in meters
            dist_segment = math.sqrt(dx_meters**2 + dy_meters**2)
            
            # Elevation stuff
            elev1 = node_u.get('elevation', 0)
            elev2 = node_v.get('elevation', 0)
            elev_diff = abs(elev2 - elev1)
            
            # 3D Length
            length_segment = math.sqrt(dist_segment**2 + elev_diff**2)
            
            if dist_segment > 0.1: # avoid division by zero or tiny noise
                slope = (elev_diff / dist_segment) * 100
                slopes.append(slope)
            
            total_distance_2d += dist_segment
            total_length_3d += length_segment

    avg_inclination = np.mean(slopes) if slopes else 0.0
    
    return {
        "total_distance_2d": total_distance_2d,
        "total_length_3d": total_length_3d,
        "avg_inclination": avg_inclination
    }

def visualize_paths(
    mesh_graph, 
    paths, 
    key_nodes=None,
    bbox = None,
    output_file="map.html"
):
    # 1. Calculate Center
    all_lats = [d['y'] for n, d in mesh_graph.nodes(data=True) if 'y' in d]
    all_lons = [d['x'] for n, d in mesh_graph.nodes(data=True) if 'x' in d]
    
    if not all_lats:
        raise ValueError("Graph nodes do not contain 'x' and 'y' coordinates.")

    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']

    if bbox is not None:
        bounds = [[bbox.bottom, bbox.left], [bbox.top, bbox.right]]
        
        folium.Rectangle(
            bounds=bounds,
            color="black",
            weight=3,    
            fill=False, 
            opacity=1,
            tooltip="Area Bounding Box"
        ).add_to(m)

    # 2. Draw Paths
    for i, path in enumerate(paths):
        if not path: continue
        
        raw_lats = []
        raw_lons = []
        
        for node_id in path:
            if node_id in mesh_graph.nodes:
                raw_lats.append(mesh_graph.nodes[node_id]['y'])
                raw_lons.append(mesh_graph.nodes[node_id]['x'])
        
        if not raw_lats: continue

        # smoothing
        # s_lats, s_lons = smooth_path(raw_lats, raw_lons)
        # points = list(zip(s_lats, s_lons))
        
        # Use raw coordinates
        points = list(zip(raw_lats, raw_lons))
        
        color = colors[i % len(colors)]
        folium.PolyLine(
            points, 
            color=color, 
            weight=5, 
            opacity=0.8, 
            tooltip=f"Path {i}"
        ).add_to(m)

    # 3. Draw Key Nodes (Highlighters)
    if key_nodes:
        for node_id in key_nodes:
            if node_id in mesh_graph.nodes:
                lat = mesh_graph.nodes[node_id]['y']
                lon = mesh_graph.nodes[node_id]['x']
                
                folium.CircleMarker(
                    [lat, lon],
                    radius=8,          # Big dot
                    color='black',     # Border color
                    weight=2,
                    fill=True,
                    fill_color='white',# Center color
                    fill_opacity=1.0,
                    popup=f"Key Node: {node_id}"
                ).add_to(m)

    stats = calculate_path_statistics(mesh_graph, paths)
    
    stats_html = f"""
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; 
        width: 320px; height: auto; 
        z-index:9999; 
        font-family: sans-serif;
        font-size: 14px;
        background-color: white;
        border: 2px solid rgba(0,0,0,0.2);
        border-radius: 6px;
        padding: 15px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
    ">
        <h4 style="margin-top:0; margin-bottom:10px;">Path Statistics</h4>
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
            <b>Total Distance (2D):</b> 
            <span>{stats['total_distance_2d'] / 1000:.2f} km</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
            <b>Total Length (3D):</b> 
            <span>{stats['total_length_3d'] / 1000:.2f} km</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <b>Avg. Inclination:</b> 
            <span>{stats['avg_inclination']:.2f} %</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    m.save(output_file)
    print(f"Map saved to {output_file}")
