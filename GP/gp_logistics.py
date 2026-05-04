import operator
import random
import math
from deap import gp
import json
from functools import partial 
from collections import deque
import re
import os
import platform
import pydot
import numpy as np

BASE = math.e

# for primitive set
def step_penalty_multiplier(value, threshold, multiplier):
    return np.where(value > threshold, multiplier, 1.0)
def step_penalty_adder(value, threshold, penalty):
    return np.where(value > threshold, penalty, 0.0)


def protected_div(n1, n2):
    if np.isscalar(n2):
        if abs(n2) < 1e-9:
            return 1.0
        return n1 / n2

    safe_n2 = np.where(np.abs(n2) > 1e-9, n2, 1.0)
    return np.where(np.abs(n2) > 1e-9, n1 / safe_n2, 1.0)


def protected_log(x, base):
    if np.isscalar(x) and np.isscalar(base):
        safe_x = abs(x) if abs(x) > 1e-9 else 1e-9
        safe_base = abs(base)
        # vs base being 1.0
        if abs(safe_base - 1.0) < 1e-9:
            safe_base = 1.0 + 1e-9 
        elif safe_base < 1e-9:
            safe_base = 1e-9
        res = math.log(safe_x) / math.log(safe_base)
        return res if math.isfinite(res) else 1.0
    safe_x = np.where(np.abs(x) > 1e-9, np.abs(x), 1e-9)
    safe_base = np.abs(base)
    is_near_one = np.abs(safe_base - 1.0) < 1e-9
    safe_base = np.where(is_near_one, 1.0 + 1e-9, safe_base)
    safe_base = np.where(safe_base > 1e-9, safe_base, 1e-9)
    with np.errstate(divide="ignore", invalid='ignore'):
        res = np.log(safe_x) / np.log(safe_base)
    return np.where(np.isfinite(res), res, 1.0)


def protected_pow(n1, n2):
    if np.isscalar(n1) and np.isscalar(n2):
        base = abs(n1)
        exp = max(min(n2, 10), -10)
        try:
            res = math.pow(base, exp)
            return res if math.isfinite(res) else 1e10
        except:
            return 1e10

    base = np.abs(n1)
    exponent = np.clip(n2, -10, 10)
    safe_base = np.where((base < 1e-9) & (exponent < 0), 1e-9, base)

    with np.errstate(over='ignore', invalid='ignore'):
        res = np.power(safe_base, exponent)
    return np.where(np.isfinite(res), res, 1e10)


def if_then_else(condition, out_true, out_false):
    cond = np.isfinite(condition) & (condition > 0)
    return np.where(cond, out_true, out_false)

def round_random(a,b):
    return round(random.uniform(a,b), 3)

random_gen = partial(round_random, 0, 1)

# plots tree given the PrimitiveTree OR the string

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

def save_run(population, hof, diff, scen_number, gens, res, pset, path: str = "GP/res", logs = None, plot_tree = False, generation = 1, area = "Trentino"):
    title = f"{population}pop_{gens}gen_{scen_number}scenarios_res{res}_gen{generation}"
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
    tree_diz["area"] = area
    tree_diz["current_generation"] = generation
    tree_diz["total_generations"] = gens
    tree_diz["resolution"] = res
    tree_diz["population"] = population
    tree_diz["scenarios"] = scen_number
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
    
    # computes first round of fittest individuals with fixed generations-per-run equal to 15
    pop_size = [500, 1000, 2000, 2500, 5000]
    runs = [15,20,25]
    all_candidates = []

    for size in pop_size:
        for run in runs:
            # we take only the individuals in the final hall of fame
            file_path = f"GP/res/runs_15_01_2026/{size}pop_15gen_{run}run_200res/{size}pop_15gen_run{run}_res200_{run-1}subrun.json"
            try:
                with open(file_path) as f:
                    data = json.load(f)[0]
                    hof = data["hall_of_fame"]
                
                candidates_to_check = hof
                
                for entry in candidates_to_check:
                    all_candidates.append({
                        "fitness": float(entry["fitness"]),
                        "tree_string": entry["individual"],
                        "individual_id": f"size{size}, run{run}, gen15"
                    })
            except (FileNotFoundError, IndexError, KeyError):
                print("NOT WORKING") 

    # Sort by fitness (lowest is best)
    all_candidates.sort(key=lambda x: x["fitness"])

    # Filter for unique fitness values (vs. similar trees)
    unique_best = []
    seen_fitness = set()

    for cand in all_candidates:
        if cand["fitness"] not in seen_fitness:
            seen_fitness.add(cand["fitness"])
            unique_best.append(cand)
        if len(unique_best) == 5: # Stop once we have top 5
            break

    # Saves best five trees
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





