import json
import re
import matplotlib.pyplot as plt
import contextily as cx

#AI Generated Function for fast relative high screenshots of the paths

with open('31_03_2026_trentino/3/PathOutputs_trentino_10.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# Extract Polyline
polyline_match = re.search(r'L\.polyline\(\s*(\[\[.*?\]\])', html_content, re.DOTALL)
if polyline_match:
    coords = json.loads(polyline_match.group(1))
else:
    raise ValueError("Error")


lats = [c[0] for c in coords]
lons = [c[1] for c in coords]

#Plot with matplot
fig, ax = plt.subplots(figsize=(12, 12), dpi=600)
ax.plot(lons, lats, color='red', linewidth=3, label='ACO Path')

#Add map as background
cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.OpenStreetMap.Mapnik, zoom=14)

ax.set_xlim(min(lons) - 0.01, max(lons) + 0.01)
ax.set_ylim(min(lats) - 0.01, max(lats) + 0.01)

ax.set_axis_off()

plt.savefig('road_output_trentino_1.png', bbox_inches='tight', dpi=100)
