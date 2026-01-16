import random
import math
import numpy as np
from deap import gp
import pydot
import json
from functools import partial 
from collections import deque
import re

def protected_div(n1, n2):
    if isinstance(n1, complex):
        n1 = n2.real
    if isinstance(n2, complex):
        n2 = n2.real
    if n2 == 0:
        return 0
    else:
        return n1/n2
    
def protected_log(x, base):
    if isinstance(x, complex):
        x = x.real
    if isinstance(base, complex):
        base = base.real
    if x > 0 and base > 0 and base!=1:
        return math.log(x, base)
    else:
        return 1

def protected_pow(n1, n2):
    if isinstance(n1, complex):
        n1 = n2.real
    if isinstance(n2, complex):
        n2 = n2.real
    if n1 == 0:
        return 0
    try:
        base = float(np.abs(n1))
        exponent = np.clip(float(n2), -5, 5)
        return np.power(base, exponent)
    except (OverflowError, ValueError): 
        return 1e10

def if_then_else(condition, out1, out2):
    return np.where(condition > 0.5, out1, out2) 

def identity_water(x):
    return x

def dynamic_penalty(inclination):
    if inclination >= round(30/90, 4): # if normalized inclination is more than 1/3 (= aka inclination is >=30%)
        return 50000*inclination
    else:
        return 5*inclination

def round_random(a,b):
    return round(random.uniform(a,b), 3)
random_gen = partial(round_random, 0,10)


import os
import platform
import pydot

def tree_plotter(tree, title, pset, destination = "GP/hof"):
    if platform.system() == "Windows":
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            graphviz_bin_path = os.path.join(conda_prefix, 'Library', 'bin')
            if graphviz_bin_path not in os.environ['PATH']:
                os.environ['PATH'] = graphviz_bin_path + os.pathsep + os.environ['PATH']
    if not isinstance(tree, gp.PrimitiveTree):
        try:
            tree = gp.PrimitiveTree.from_string(tree, pset)
        except TypeError:    
            tree = from_tree_to_string(tree, pset)         
    nodes, edges, labels = gp.graph(tree)
    f = "digraph G {\n"
    f += "    margin=0.5;\n"
    f += "    center=1;\n"
    f += "    rankdir=TB;\n"
    f += "    labelloc=\"t\";\n"    
    f += f"    label=\"{title}\\n\\n\";\n"
    for node in nodes:
        f += f'    {node} [label="{labels[node]}", shape=ellipse, style=filled, fillcolor=white, fontname="Arial", margin=0.1];\n' 
    for edge in edges:
        f += f"    {edge[0]} -> {edge[1]};\n"
    f += "}"
    graphs = pydot.graph_from_dot_data(f)
    graph = graphs[0]
    if not os.path.exists(destination):
        os.makedirs(destination)
    output_path = f"{destination}/{title}"
    try:
        graph.write(f"{output_path}.svg", format="svg")
    except Exception as e:
        print(f"SVG printing error: {e}")

# to solve TypeError problem
def from_tree_to_string(string, pset):
    tokens = re.split("[ \t\n\r\f\v(),]", string)
    expr = []
    ret_types = deque()
    for token in tokens:
        if token == '':
            continue
        type_ = ret_types.popleft() if len(ret_types) != 0 else None
        if token in pset.mapping:
            item = pset.mapping[token]
            if type_ is not None and not issubclass(item.ret, type_):
                raise TypeError(f"Type mismatch for {token}")
            expr.append(item)
            if hasattr(item, 'args'): 
                ret_types.extendleft(reversed(item.args))
        else:
            try:
                #to solve TypeError
                val = eval(token) 
                name = "OtherArgs"
                new_terminal_cls = gp.MetaEphemeral.__new__(
                    gp.MetaEphemeral, 
                    name, 
                    (gp.Terminal,), 
                    {'value': val}
                )
                gp.MetaEphemeral.cache[id(new_terminal_cls)] = new_terminal_cls
                terminal_instance = gp.Terminal(token, False, type_ if type_ else type(val))
                terminal_instance.value = val
                expr.append(terminal_instance)

            except Exception as e:
                raise TypeError(f"Unable to evaluate terminal: {token}. Error: {e}")
    return gp.PrimitiveTree(expr)

# adds new data to a json file for finetuning
def save_run(population, hof, diff, run,scenario_dur, res, pset, path: str = "GP/res", sub_run_idx: int = -1, logs = None, plot_tree = False):
    title = f"{population}pop_{scenario_dur}gen_run{run}_res{res}"
    if sub_run_idx != -1:
        title = f"{title}_{sub_run_idx}subrun"

    if population >=500 and plot_tree:
        path_hof = f"{path}/hof/{title}"
        if not os.path.exists(path_hof):
            os.makedirs(path_hof)
        for i in range(len(hof)):
            try:
                tree_plotter(hof[i], f"{title}_{i+1}best_tree",pset = pset, destination = path_hof)
            except Exception as e:
                print(f"Could not plot tree: {e}")
    hof_list = []
    best = hof[0]
    for ind in hof:
        ind_diz = dict()
        ind_diz["individual"] = str(ind)
        ind_diz["fitness"] = ind.fitness.values[0]
        hof_list.append(ind_diz)
    tree_diz = dict()
    tree_diz["run"] = run
    tree_diz["resolution"] = res
    tree_diz["population"] = population
    tree_diz["scenario_duration"] = scenario_dur
    tree_diz["best_individual"] = str(best)
    tree_diz["best_individual_fitness"] = best.fitness.values
    tree_diz["hall_of_fame"] = hof_list
    tree_diz["runtime_in_seconds"] = diff
    if logs is not None:
        tree_diz["logs"] = logs
    path = f"{path}/{title}.json"
    with open(path, 'w') as f:
        json.dump([tree_diz], f, indent=4)


if __name__ == "__main__":
    
    #from GP_with_optimizations import pset
    pop_size = [500, 1000, 2000, 2500, 5000]
    runs = [15,20,25]
    all_candidates = []

    for size in pop_size:
        for run in runs:
            for i in range(1, run + 1):
                file_path = f"GP/res/runs_15_01_2026/{size}pop_15gen_{run}run_200res/{size}pop_15gen_run{i}_res200_{i-1}subrun.json"
                try:
                    with open(file_path) as f:
                        data = json.load(f)[0]
                        hof = data["hall_of_fame"]
                    
                    # If i < run, we only look at the top individual in HOF
                    # Otherwise, we look at the whole HOF list
                    candidates_to_check = [hof[0]] if i < run else hof
                    
                    for entry in candidates_to_check:
                        all_candidates.append({
                            "fitness": float(entry["fitness"]),
                            "tree_string": entry["individual"],
                            "individual_id": f"size{size}, run{run}.{i}"
                        })
                except (FileNotFoundError, IndexError, KeyError):
                    print("NOT WORKING") # Skip files that are missing or formatted incorrectly

    # 1. Sort by fitness (lowest is best)
    all_candidates.sort(key=lambda x: x["fitness"])

    # 2. Filter for unique fitness values
    unique_best = []
    seen_fitness = set()

    for cand in all_candidates:
        if cand["fitness"] not in seen_fitness:
            seen_fitness.add(cand["fitness"])
            unique_best.append(cand)
        if len(unique_best) == 5: # Stop once we have top 5
            break

    # 3. Format for your final output
    best = []
    for idx, item in enumerate(unique_best):
        best.append({
            "place": idx + 1,
            "tree_string": item["tree_string"],
            "individual": item["individual_id"],
            "fitness": item["fitness"]
        })                 
    with open("GP/best_trees.json", "w") as f:
        json.dump(best, f, indent=4)
    print("Best individuals have been saved")




        