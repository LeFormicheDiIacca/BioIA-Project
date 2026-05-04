# Learning Terrain-Aware Edge Costs for Ant-Colony Route Planning via Genetic Programming

[![DOI](https://img.shields.io/badge/DOI-10.1145/3795101.3814665-blue.svg)](https://doi.org/10.1145/3795101.3814665)
[![Conference](https://img.shields.io/badge/Conference-GECCO_Companion_'26-success.svg)](https://gecco-2026.sigevo.org/)

This repository contains the code for the paper **"Learning Terrain-Aware Edge Costs for Ant-Colony Route Planning via Genetic Programming"**, presented at the Genetic and Evolutionary Computation Conference (GECCO Companion '26), San Jose, Costa Rica.

## Overview

Pathfinding in complex geographical environments remains a challenge for standard **Ant Colony Optimization (ACO)** algorithms, as their performance is heavily dependent on the design of the edge cost function. 
Traditional manual tuning often fails to capture the intricate trade-offs between distance, terrain morphology, and environmental constraints.

This work introduces an evolutionary framework that utilizes **Genetic Programming (GP)** to automatically evolve terrain-aware cost functions. By penalizing suboptimal features—such as excessive length, steep gradients, and water crossings—the GP-evolved functions enable the **Min-Max Ant System (MMAS)** to discover more viable and realistic routes tailored to specific regional morphologies.

Key Contributions:
* **Hybrid Evolutionary Approach**: Combining GP for high-level function synthesis with ACO for low-level path optimization.
* **Morphology-Aware Planning**: Integration of topographic data (slope, terrain type, obstacles) directly into the cost calculation.
* **Performance Optimization**: Automatic discovery of cost functions that outperform standard heuristics in rugged or constrained environments.

<table style="border: none; border-collapse: collapse; width: 100%;">
  <tr style="border: none;">
    <td style="border: none; width: 48%; text-align: center; vertical-align: top;">
      <img src="https://github.com/user-attachments/assets/1261048c-1a0f-4a6e-8801-02f11a3e3205" width="100%" />
      <br>
      <em>Figure 1: Example of a GP-generated cost function.</em>
    </td>
    <td style="border: none; width: 4%;">&nbsp;</td> <!-- Spazio tra le colonne -->
    <td style="border: none; width: 48%; text-align: center; vertical-align: top;">
      <img src="https://github.com/user-attachments/assets/4aae335b-a8e7-400d-98ac-fa3f3d5f393e" width="100%" />
      <br>
      <em>Figure 2: A path generated through the ACO algorithm.</em>
    </td>
  </tr>
</table>

## Repository Structure

The project is organized into modular components:
* **ACO/**: Core implementation of the Ant Colony Optimization algorithm. Contains the classes responsible for agent navigation, pheromone management, and path construction.
* **GP/**: Tools for the Genetic Programming framework. This includes the evolutionary logic used to synthesize cost functions and a collection of the best-performing evolved functions generated during our experiments.
* **TerrainGraph/**: Geometry and terrain data processing module. Contains scripts to transform raw topographic data into a searchable mesh graph and utilities to export/visualize the final paths.
* **Results/**: Comprehensive logs and output data from the ACO runs, including performance metrics and path coordinates used for the paper's benchmarks.
* **Dataset/**: Raw and processed geographic data. This folder includes:
  * Topographic Data (.tif): High-resolution GeoTIFF files representing Digital Elevation Models (DEM) used to extract terrain slope and elevation.
  * Geospatial Features (.pbf): OpenStreetMap data in Protocolbuffer Binary Format, used to identify hydrology (water bodies) and land-cover constraints.
  * Evaluation Scenarios: A list of start-end node pairs used to test the algorithm's performance across different morphologies.

## Getting Started

### Prerequisites
To run the code, you will need a standard deep learning environment. We recommend installing the dependencies in the file "requirements.txt".

### Running the Code
1. Clone this repository to your local machine.
2. Run GP/GP_execution.py.
3. Observe your outputs and transcribe the resulting individuals into functions.
4. Run execute_ACO.py using the evolved cost functions.

Remember to select the desired region when running any file.

### Advanced Use
1. Download the TIF and PBF files from https://tinitaly.pi.ingv.it/Download_Area1_1.html (TIF file) and https://www.geofabrik.de/data/download.html (PBF) for the desired region.
2. Run TerrainGraph/preprocess_map.py with the chosen files and indicate the correct bounding box.
3. Execute GP/scenario.py specifying the area to be considered.
4. Run GP/GP_execution.py selecting the various parameters, specifying the npz file path (step 2) as well as the chosen scenario file (step 3).
5. Analyze the output; it is possible to use hof_analyzer to understand when the Hall of Fame starts stagnating.
6. Transcribe the individuals into functions.
7. Run execute_ACO.py with the resulting functions, providing the geographic coordinates for the desired origin and destination.

## Authors
* Asia Panizza, University of Trento, Italy
* Davide Colosimo, University of Trento, Italy
* Filippo Marcon, University of Trento, Italy
* Erik Nielsen, University of Trento, Italy
* Stefano Genetti, University of Trento, Italy
* Giovanni Iacca, University of Trento, Italy

## Citation

If you use this code or find our work helpful in your research, please cite our paper:

```bibtex
@inproceedings{kalaj2026bio,
  author = {Panizza, Asia and Colosimo, Davide and Marcon, Filippo and Nielsen, Erik and Genetti, Stefano and Iacca, Giovanni},
  title = {Learning Terrain-Aware Edge Costs for Ant-Colony Route Planning via Genetic Programming},
  year = {2026},
  url = {[https://doi.org/10.1145/3795101.3814665](https://doi.org/10.1145/3795101.3814665)},
  doi = {10.1145/3795101.3814665},
  booktitle = {Genetic and Evolutionary Computation Conference (GECCO Companion '26)},
  location = {San Jose, Costa Rica},
  series = {GECCO '26}
}
