import json
from gp_logistics import tree_plotter
from GP_with_optimizations import pset

with open("GP/best_trees.json") as f:
    best_trees = json.load(f)

# tree_plotter(best_trees[0]["tree_string"], "Best evolved tree", pset, "GP/best_trees/")
# tree_plotter(best_trees[1]["tree_string"], "Second-best evolved tree", pset, "GP/best_trees/")
# tree_plotter(best_trees[2]["tree_string"], "Third-best evolved tree", pset, "GP/best_trees/")

def best_CF(distance, elev_u, elev_v, steepness, is_water):
    ret = 7.828 
    if is_water==1.0:
        ret += distance + elev_v
    else:
        ret += steepness*distance*elev_v*elev_u
    
def second_best_CF(distance, elev_u, elev_v, steepness, is_water):
    if is_water == 1.0:
        ret = steepness - elev_v
    else:
        ret = ((distance - elev_v) - (steepness - elev_u))*steepness

def third_best_CF(distance, elev_u, elev_v, steepness, is_water):
    ret = elev_u*distance
    if is_water == 1.0:
        ret += 8.364
    else:
        ret += - elev_v - steepness*elev_v + steepness
        



