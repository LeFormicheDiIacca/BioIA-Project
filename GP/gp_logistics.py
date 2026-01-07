import random
import math
import numpy as np
from deap import gp
import pydot
import os
import json
from functools import partial 

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


def tree_plotter(tree, title):
    nodes, edges, labels = gp.graph(tree)
    f = "digraph G {\n"
    f += "    size=\"20,20\";\n"  
    f += "    dpi=300;\n"          
    f += "    labelloc=\"t\";\n"    
    f += f"    label=\"{title}\n\\n\";\n"
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
    graph.write_png(f"hof/{title}.png")


# adds new data to a json file for finetuning

def append_to_json(new_data):
    # 1. Check if file exists and isn't empty
    if os.path.exists("tree_diz.json") and os.path.getsize("tree_diz.json") > 0:
        with open("tree_diz.json", 'r') as f:
            data = json.load(f)
    else:
        data = [] # Start with an empty list if file doesn't exist
    data.append(new_data)
    with open("tree_diz.json", 'w') as f:
        json.dump(data, f, indent=4)