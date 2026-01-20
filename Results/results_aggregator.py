import csv
import json
from bs4 import BeautifulSoup
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
                # Estrae numeri float anche se ci sono % o km
                clean_value = re.findall(r"[-+]?\d*\.\d+|\d+", raw_value)[0]
                results[labels[label_text]] = float(clean_value)
    return results


# Mappatura nodi
key_nodes = dict()
key_nodes[(34321, 16744, 5971, 5628)] = "Pergine Valsugana, Trento\nSopramonte, Vason"
key_nodes[(7028, 5678, 16099, 23538, 34371)] = "Trento, Lavis\nCembra, Capriana"
key_nodes[(34295, 23771, 5628)] = "Trento, Sopramonte\nSarche"
key_nodes[(5628, 14161, 24166, 34371)] = "Fondo\nPasso della Mendola\nAppiano, Laives"
key_nodes[(5628, 15865, 21927, 34371)] = "Trento, Baselga di Piné\nValfloriana, Moena"
key_nodes[(6428, 5688, 34371)] = "Trento, Mezzocorona\nLaives"
key_nodes[(5628, 17371, 34309)] = "Mezzocorona, Cavalese\nCanal San Bovo"
key_nodes[(34228, 24297, 17937, 5771)] = "Trento, Mezzolombardo\nDenno, Cles"

# Lista per raccogliere tutti i dati per il DataFrame finale
data_records = []

for i in range(5):
    results_count = dict()
    results = dict()
    path_2d_length = dict()
    path_3d_length = dict()
    path_avg_inclination = dict()

    # Lettura e processamento dati (identico al tuo script originale)
    try:
        print(i)
        with open(f"day_17_01_2026/{i}/PathOutputs.csv", newline="", encoding="ISO-8859-1") as file_csv:
            open_csv = csv.reader(file_csv)
            next(open_csv)  # Salta header
            for j, lines in enumerate(open_csv):
                # JSON Parsing
                json_path = f"day_17_01_2026/{i}/PathOutputs_{j}.json"
                try:
                    with open(json_path) as json_file:
                        json_data = json.load(json_file)
                        path_id = key_nodes.get(tuple(json_data["KeyNodes"]))

                        # Se il path_id non è mappato, saltiamo o gestiamo l'errore
                        if path_id is None: continue

                        results.setdefault(path_id, 0)
                        results[path_id] += float(lines[1])
                        results_count.setdefault(path_id, 0)
                        results_count[path_id] += 1
                except FileNotFoundError:
                    print(f"File not found: {json_path}")
                    continue

                # HTML Parsing
                file_path = f"day_17_01_2026/{i}/PathOutputs_{j}.html"
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

    # Calcolo medie e salvataggio nella lista globale
    for pid in results.keys():
        count = results_count[pid]
        if count > 0:
            # Aggiungiamo un record (riga) per ogni Path ID trovato in questa Cost Function
            data_records.append({
                "Cost Function": f"CF {i+1}",  # Etichetta per la funzione di costo
                "Path ID": pid,
                "Average Cost": results[pid] / count,
                "2D Length": path_2d_length.get(pid, 0) / count,
                "3D Length": path_3d_length.get(pid, 0) / count,
                "Avg Inclination": path_avg_inclination.get(pid, 0) / count
            })

    print(f"Processed Cost Function {i}")

# ---------------------------------------------------------
# CREAZIONE DEI GRAFICI CON SEABORN
# ---------------------------------------------------------

# Creiamo il DataFrame
df = pd.DataFrame(data_records)

# Impostiamo lo stile di Seaborn
sns.set_theme(style="whitegrid")

# Definiamo le 4 metriche che vogliamo plottare (nomi delle colonne del DF)
metrics = ["Average Cost", "2D Length", "3D Length", "Avg Inclination"]
titles = ["Average Cost", "Average Planar Length (km)", "Average Real Length (km)", "Average Steepness (%)"]

# Creiamo una figura con 4 sottografici (2 righe, 2 colonne)
fig, axes = plt.subplots(1,1, figsize=(16, 12))

for idx, metric in enumerate(metrics):
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Creazione del Barplot Raggruppato
    sns.barplot(
        data=df,
        x="Path ID",
        y=metric,
        hue="Cost Function",  # Questo crea le colonne colorate diverse per ogni CF
        ax=ax,
        palette="viridis",  # Palette di colori
        edgecolor="black"  # Bordo nero per le barre
    )

    ax.set_title(titles[idx], fontsize=14, fontweight='bold')
    ax.set_xlabel("Path", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    ax.legend(title="Cost Function", loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{titles[idx]}.svg", dpi=300)
    plt.show()