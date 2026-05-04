import csv
import json
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# --- CONFIGURAZIONE ---
date = "31_03_2026"
# Mappa il nome della cartella alla "parola chiave" usata nei nomi dei file
folders_config = {
    f"{date}_napoli": "napoli",
    f"{date}_trentino": "trentino"
}

metrics = ["Average Cost", "2D Length", "3D Length", "Avg Inclination"]
titles = ["Average Cost", "Average Planar Length (km)", "Average Real Length (km)", "Average Steepness (%)"]

key_nodes = {
    (5628, 16507, 18371, 7359, 34253): "Sessa Aurunca, Roccamonfina,\nConca dellaCampania,\nSipicciano, Teano",
    (5710, 9771, 23932, 34317, 28282, 25228): "Sessa Aurunca, Conca della\nCampania,Pietramelara, Alvignano,\nPontelatone, Capua",
    (15761, 24571, 29351, 32299, 34236, 24056, 12228, 5678): "Presenzano, Ailano,\nSant'Angelo d'Alife, Dragoni,\nPiana di Monte Verna, Pozzillo,\nCiamprisco, Sessa Aurunca",
    (10028, 9360, 21771, 34276, 5679): "Castel Volturno, Sessa Aurunca,\nTeano, Capua, Mondragone",
    (34321, 22570, 16744, 12130, 5971, 8088, 5628): "Pergine Valsugana, Oltrecastello,\nTrento, Sardagna, Sopramonte,\nVaneze, Vason",
    (7028, 5678, 9489, 16099, 23538, 26760, 34371): "Trento, Lavis, Verla, Cembra,\nGrauno, Capriana, C. di Fiemme",
    (5628, 8314, 14161, 20767, 24166, 34371): "Taio, Malgolo, Sarnonico,\nMondola, Caldaro, Laives",
    (5628, 10243, 15865, 21927, 24342, 34371): "Trento, Civezzano, Miola,\nSover, Casatta, Cavalese"
}


def extract_path_statistics(html_file_path):
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        results = {}
        labels = {"Total Distance (2D):": "total_distance_2d", "Total Length (3D):": "total_length_3d",
                  "Avg. Inclination:": "avg_inclination"}
        for b_tag in soup.find_all('b'):
            label_text = b_tag.get_text(strip=True)
            if label_text in labels:
                value_tag = b_tag.find_next('span')
                if value_tag:
                    raw_value = value_tag.get_text(strip=True)
                    match = re.search(r"[-+]?\d*\.\d+|\d+", raw_value)
                    if match: results[labels[label_text]] = float(match.group())
        return results
    except:
        return {}


data_records = []

for folder, region_key in folders_config.items():
    print(f"--- Processing Folder: {folder} (Key: {region_key}) ---")
    for i in range(5):
        results_count, results = {}, {}
        path_2d_length, path_3d_length, path_avg_inclination = {}, {}, {}

        base_path = f"{folder}/{i}"
        csv_path = f"{base_path}/PathOutputs.csv"

        if not os.path.exists(csv_path):
            continue

        with open(csv_path, newline="", encoding="ISO-8859-1") as file_csv:
            open_csv = csv.reader(file_csv)
            next(open_csv)

            for j, lines in enumerate(open_csv):
                # LOGICA NOME FILE:
                # Se j=0 -> PathOutputs_napoli.json
                # Se j>0 -> PathOutputs_napoli_j.json
                suffix = f"_{j}" if j > 0 else ""
                file_base_name = f"PathOutputs_{region_key}{suffix}"

                json_path = f"{base_path}/{file_base_name}.json"
                html_path = f"{base_path}/{file_base_name}.html"

                path_id = None
                if os.path.exists(json_path):
                    with open(json_path) as json_file:
                        json_data = json.load(json_file)
                        path_id = key_nodes.get(tuple(json_data["KeyNodes"]))

                if path_id:
                    results.setdefault(path_id, 0)
                    results[path_id] += float(lines[1])
                    results_count.setdefault(path_id, 0)
                    results_count[path_id] += 1

                    data = extract_path_statistics(html_path)
                    path_2d_length.setdefault(path_id, 0)
                    path_2d_length[path_id] += data.get('total_distance_2d', 0)
                    path_3d_length.setdefault(path_id, 0)
                    path_3d_length[path_id] += data.get('total_length_3d', 0)
                    path_avg_inclination.setdefault(path_id, 0)
                    path_avg_inclination[path_id] += data.get('avg_inclination', 0)

        for pid in results.keys():
            count = results_count[pid]
            if count > 0:
                data_records.append({
                    "Cost Function": f"CF {i + 1}",
                    "Path ID": pid,
                    "Average Cost": results[pid] / count,
                    "2D Length": path_2d_length.get(pid, 0) / count,
                    "3D Length": path_3d_length.get(pid, 0) / count,
                    "Avg Inclination": path_avg_inclination.get(pid, 0) / count
                })
        print(f"  > CF {i + 1} completata ({len(results)} percorsi validi)")

# --- PLOTTING ---
df = pd.DataFrame(data_records)
if not df.empty:
    df = df.sort_values(by=["Path ID", "Cost Function"])
    df.to_csv(f"path_data_final_{date}.csv", index=False, encoding='utf-8-sig', sep=';')

    sns.set_theme(style="whitegrid")
    for idx, metric in enumerate(metrics):
        plt.figure(figsize=(16, 8))
        sns.barplot(data=df, x="Path ID", y=metric, hue="Cost Function", palette="magma", edgecolor="black")
        plt.title(titles[idx], fontsize=15, fontweight='bold')
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        plt.show()
else:
    print("ERRORE: Nessun dato trovato. Controlla se i file JSON esistono con i nomi cercati.")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# Paper requirements for font embedding
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Load the generated data
df = pd.read_csv(f'path_data_final_{date}.csv', sep=';')

# Plotting with seaborn
sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "black"})
metrics = ["3D Length", "Avg Inclination"]
y_labels = ["Real Length (km)", "Steepness (%)"]

# Create a 4x1 grid of subplots that share the X-axis
fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

for idx, metric in enumerate(metrics):
    sns.barplot(
        data=df,
        x="Path ID",
        y=metric,
        hue="Cost Function",
        ax=axes[idx],
        palette="viridis",
        edgecolor="black"
    )

    axes[idx].set_ylabel(y_labels[idx], fontsize=9, fontweight='bold')
    axes[idx].set_xlabel("")  # Remove individual x-labels
    axes[idx].tick_params(axis='y', labelsize=8)

    # Handle the legend: only keep it on the top plot, spread it horizontally to save vertical space
    if idx == 0:
        axes[idx].legend(title="Cost Function", loc='upper center', bbox_to_anchor=(0.5, 1.45),
                         fontsize=8, title_fontsize=9, ncol=4)
    else:
        axes[idx].get_legend().remove()

# Format the shared X-axis (only visible on the bottom plot)
axes[-1].set_xlabel("Path", fontsize=10, fontweight='bold')
axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=55, ha='right', fontsize=8)
plt.subplots_adjust(bottom=0.45)
plt.savefig("Combined_Metrics.pdf", format='pdf', dpi=300)
plt.close()