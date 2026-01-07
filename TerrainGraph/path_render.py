import numpy as np
import folium
import os
from scipy.interpolate import splprep, splev

def smooth_path(lats, lons, smoothing_points=200):
    """
    (Currently unused) Smooths the path using B-Spline interpolation.
    """
    clean_lats, clean_lons = [], []
    for i in range(len(lats)):
        if i == 0 or not (lats[i] == lats[i-1] and lons[i] == lons[i-1]):
            clean_lats.append(lats[i])
            clean_lons.append(lons[i])

    if len(clean_lats) < 3:
        return clean_lats, clean_lons

    try:
        tck, u = splprep([clean_lats, clean_lons], s=0, k=3)
        u_new = np.linspace(0, 1, smoothing_points)
        s_lats, s_lons = splev(u_new, tck)
        return s_lats, s_lons
    except:
        return clean_lats, clean_lons

def visualize_paths(
    mesh_graph, 
    paths, 
    key_nodes=None,
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

        # --- Smoothing Disabled for now ---
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

    m.save(output_file)
    print(f"Map saved to {output_file}")
