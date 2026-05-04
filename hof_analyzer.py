import json
import re
from pathlib import Path


def analyze_hof_stagnation(input_folder, stagnation_threshold=3):
    """
    Analyzes JSON files to detect when the Hall of Fame stops evolving.
    """
    folder_path = Path(input_folder)
    json_files = list(folder_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in: {input_folder}")
        return

    def extract_generation(filepath):
        match = re.search(r"_gen(\d+)\.json$", filepath.name)
        return int(match.group(1)) if match else -1

    # Sort files chronologically by generation
    json_files.sort(key=extract_generation)

    prev_hof = set()
    stag_counter = 0
    stagnation_start = None
    prev_gen = None

    print(f"Analyzing {len(json_files)} files...\n")

    for file_path in json_files:
        current_gen = extract_generation(file_path)
        if current_gen == -1:
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                content = data[0] if isinstance(data, list) else data

                hof_list = content.get("hall_of_fame", [])
                current_hof = frozenset([ind["individual"] for ind in hof_list])
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            continue

        if current_hof == prev_hof and len(current_hof) > 0:
            stag_counter += 1
        else:
            if stag_counter >= stagnation_threshold:
                print(f"Stagnation detected:")
                print(f"   From gen {stagnation_start} to gen {prev_gen} ({stag_counter} generations)")
                print("-" * 50)

            stag_counter = 1
            stagnation_start = current_gen
            prev_hof = current_hof

        prev_gen = current_gen

    if stag_counter >= stagnation_threshold:
        print(f"Final stagnation detected:")
        print(f"   From gen {stagnation_start} to gen {prev_gen} ({stag_counter} generations)")
        print("-" * 50)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    PATH = "/home/davide/Downloads/Napoli/Downloads/runs_31_03_2026/3500pop_50gen_200res/"
    THRESHOLD = 5
    analyze_hof_stagnation(PATH, THRESHOLD)