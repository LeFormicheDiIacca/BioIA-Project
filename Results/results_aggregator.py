import csv
import json
from bs4 import BeautifulSoup
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Extract stats from the html files
def extract_path_statistics(html_file_path):
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    results = {}

    labels = {
        "Total Distance (2D):": "total_distance_2d",
        "Total Length (3D):": "total_length_3d",
        "Avg. Inclination:": "avg_inclination"
    }

    for b_tag in soup.find_all('b'):
        label_text = b_tag.get_text(strip=True)
        if label_text in labels:
            value_tag = b_tag.find_next('span')
            if value_tag:
                raw_value = value_tag.get_text(strip=True)
                clean_value = re.findall(r"[-+]?\d*\.\d+|\d+", raw_value)[0]
                results[labels[label_text]] = float(clean_value)
    return results

date = "17_01_2026"
#Nodes Mapping to paths
key_nodes = dict()
key_nodes[(5628,16507,18371,7359,34253)] = "1-Path North Campania 1"
key_nodes[(5710,9771,23932,34317,28282,25228)] = "2-Path North Campania 2"
key_nodes[(15761,24571,29351,32299,34236,24056,12228,5678)] = "3-Path North Campania 3"
key_nodes[(10028,9360,21771,34276,5679)] = "4-Path North Campania 4"
key_nodes[(34321,22570,16744,12130,5971,8088,5628)] = "5-Path Trentino 1"
key_nodes[(7028,5678,9489,16099,23538,26760,34371)] = "6-Path Trentino 2"
key_nodes[(5628,8314,14161,20767,24166,34371)] = "7-Path Trentino 3"
key_nodes[(5628, 10243, 15865, 21927, 24342, 34371)] = "8-Path Trentino 4"

data_records = []

for i in range(5):
    results_count = dict()
    results = dict()
    path_2d_length = dict()
    path_3d_length = dict()
    path_avg_inclination = dict()

    try:
        print(i)
        with open(f"day_{date}/{i}/PathOutputs.csv", newline="", encoding="ISO-8859-1") as file_csv:
            open_csv = csv.reader(file_csv)
            next(open_csv)  #Avoid Header
            for j, lines in enumerate(open_csv):
                # JSON Parsing
                json_path = f"day_{date}/{i}/PathOutputs_{j}.json"
                try:
                    with open(json_path) as json_file:
                        json_data = json.load(json_file)
                        path_id = key_nodes.get(tuple(json_data["KeyNodes"]))
                        results.setdefault(path_id, 0)
                        results[path_id] += float(lines[1])
                        results_count.setdefault(path_id, 0)
                        results_count[path_id] += 1
                except FileNotFoundError:
                    print(f"File not found: {json_path}")
                    continue

                # HTML Parsing
                file_path = f"day_{date}/{i}/PathOutputs_{j}.html"
                try:
                    data = extract_path_statistics(file_path)

                    path_2d_length.setdefault(path_id, 0)
                    path_2d_length[path_id] += data.get('total_distance_2d', 0)

                    path_3d_length.setdefault(path_id, 0)
                    path_3d_length[path_id] += data.get('total_length_3d', 0)

                    path_avg_inclination.setdefault(path_id, 0)
                    path_avg_inclination[path_id] += data.get('avg_inclination', 0)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    except FileNotFoundError:
        print(f"Directory or CSV not found for index {i}")
        continue

    #Avg and saving in
    for pid in results.keys():
        count = results_count[pid]
        if count > 0:
            data_records.append({
                "Cost Function": f"CF {i+1}",
                "Path ID": pid,
                "Average Cost": results[pid] / count,
                "2D Length": path_2d_length.get(pid, 0) / count,
                "3D Length": path_3d_length.get(pid, 0) / count,
                "Avg Inclination": path_avg_inclination.get(pid, 0) / count
            })

    print(f"Processed Cost Function {i}")

#Plotting with seaborn
df = pd.DataFrame(data_records)
sns.set_theme(style="whitegrid")
metrics = ["Average Cost", "2D Length", "3D Length", "Avg Inclination"]
titles = ["Average Cost", "Average Planar Length (km)", "Average Real Length (km)", "Average Steepness (%)"]

if not df.empty:
    output_filename = f"path_data{date}.csv"
    df.to_csv(output_filename, index=False, encoding='utf-8-sig', sep=';')
"""
for idx, metric in enumerate(metrics):
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    sns.barplot(
        data=df,
        x="Path ID",
        y=metric,
        hue="Cost Function",
        ax=ax,
        palette="viridis",
        edgecolor="black"
    )

    ax.set_title(titles[idx], fontsize=14, fontweight='bold')
    ax.set_xlabel("Path", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    ax.legend(title="Cost Function", loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{titles[idx]}.svg", dpi=300)
    plt.show()"""