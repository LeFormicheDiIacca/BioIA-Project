import random
import math
import numpy as np
from deap import gp
import pydot
import os
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


def tree_plotter(tree, title, fitness, pset, destination = "GP/hof" ):
    if not isinstance(tree, gp.PrimitiveTree):
        try:
            tree = gp.PrimitiveTree.from_string(tree, pset)
        except TypeError:    
            tree = from_tree_to_string(tree, pset)
    nodes, edges, labels = gp.graph(tree)
    f = "digraph G {\n"
    f += "    size=\"20,20\";\n"  
    f += "    dpi=300;\n"          
    f += "    labelloc=\"t\";\n"    
    f += f"    label=\"{title, fitness}\n\\n\";\n"
    f += "    labelloc = \"t\";\n"
    f += "    size = \"20,20\";\n"
    f += "    dpi = 300;\n"     
    for node in nodes:
        f += f'    {node} [label="{labels[node]}", shape=ellipse, style=filled, fillcolor=white, fontname="Arial", fixedsize=false, margin=0.2];\n'
    for edge in edges:
        f += f"    {edge[0]} -> {edge[1]};\n"
    f += "}"
    graphs = pydot.graph_from_dot_data(f)
    graph = graphs[0]
    if not os.path.exists(destination):
        os.makedirs(destination)
    graph.write_png(f"{destination}/{title}.png")

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
def save_run(population, hof, diff, run,scenario_dur, res, pset, path: str = "GP/tree_diz.json"):
    if population >=500:
        for i in range(len(hof)):
            try:
                tree_plotter(hof[i], f"pop{population}_run{run}_res{res}_{i+1}best_tree", pset = pset)
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
    append_to_json(tree_diz, path)

def append_to_json(new_data, path: str = "GP/tree_diz.json" ):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        data = [] # Start with an empty list if file doesn't exist
    data.append(new_data)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    from GP_with_optimizations import pset

    for i in range(1,21):
        with open(f"GP/res/run_12_01_2026/GP_tree_2500pop_15gen_20runs_{i}subrun.json") as f:
            tree_diz = json.load(f)
        tree_diz = tree_diz[0]
        hof = tree_diz["hall_of_fame"]
        for j in range(len(hof)):
            title = f"subrun{i}_best_{j+1}_tree"
            fitness = f"Fitness = {hof[j]["fitness"]}"
            tree = hof[j]["individual"]
            destination = f"GP/hof_12_01/subrun{i}"
            tree_plotter(tree, title, fitness, pset, destination)
    print("All trees have been plotted")