import math

# NB. all the parameters I used to evaluate the function can be found in the edge_info.py file,
# please choose whether YOU 

#   prefer to pass the entire edge_dict at the beginning of the run (to get immediately information 
#   about all the edges in the graph

#   OR if you want to implement get_edge_metadata edge by edge

def best_CF(distance, steepness, elev_u, elev_v, is_water):
    if is_water == 1.0:
        ret = math.log(3.148, 9.535)
    else:
        ret = distance + distance
    tmp = ((distance+3.685) + (elev_u*elev_u/4.164)) - elev_v
    if is_water == 1.0:
        tmp -= steepness
    else:
        tmp -= elev_v
    ret -= tmp
    ret -= (2.41**3.978)
    ret = (-1)*ret
    return ret


def second_best_CF(distance, steepness, elev_u, elev_v, is_water):
    ret = elev_u * steepness * distance
    if is_water == 1.0:
        ret += distance
    else:
        ret += elev_u*elev_v*steepness
    return ret

def third_best_CF(distance, steepness, elev_u, elev_v, is_water):
    ret = elev_u*steepness*distance
    if is_water == 1.0:
        ret += steepness - elev_v
    else:
        ret += elev_u - elev_v
    return ret

def fourth_best_CF(distance, steepness, elev_u, elev_v, is_water):
    ret = elev_u * steepness * distance
    if is_water == 1.0:
        ret += distance
    else:
        ret += elev_u - elev_v
    return ret
    

def fifth_best_CF(distance, steepness, elev_u, elev_v, is_water):
    if is_water == 1.0:
        tmp = distance
    else:
        tmp = steepness
    ret = distance - elev_u + tmp + elev_v
    return ret


if __name__ == "__main__":
    import json
    from gp_logistics import tree_plotter
    from GP_with_optimizations import pset
    with open("GP/best_trees.json") as f:
        best_trees = json.load(f)
    tree_combos = [(5000,15), (500,20), (1000,15)]
    durs = [20, 25]
    results = []

    for el in tree_combos:
        for dur in durs:
            # we take only individuals from the final hall of fame
            path = f"GP/res/runs_15_01_2026/{el[0]}pop_{dur}gen_{el[1]}run_200res/{el[0]}pop_{dur}gen_run{el[1]}_res200_{el[1]-1}subrun.json"   
            try:
                with open(path) as f:
                    data = json.load(f)[0]
                    hof = data["hall_of_fame"]
                
                candidates_to_check = hof
                
                for entry in candidates_to_check:
                    results.append({
                        "fitness": float(entry["fitness"]),
                        "tree_string": entry["individual"],
                        "individual_id": f"size{el[1]}, run{el[0]}, gen{dur}"
                    })
            except (FileNotFoundError, IndexError, KeyError):
                print("NOT WORKING") # Skip files that are missing or formatted incorrectly
                print(path)

    for i in range(len(best_trees)):
        results.append(best_trees[i])
    # Sort the list based on fitness (ascending)
    results.sort(key=lambda x: x["fitness"])

    # Filter for unique fitness values
    unique_results = []
    seen_fitness = set()
    for entry in results:
        fit = entry["fitness"]
        if fit not in seen_fitness:
            seen_fitness.add(fit)
            unique_results.append(entry)
        #Stop once we have our top 10 unique values, we'll exclude trees that are missing one or more inputs (and due to non-active branches have passed our checks) and choose the first five fit individuals
        if len(unique_results) == 10:
            break


    best = unique_results 
    
    for i in range(10):
        tree_plotter(best[i]["tree_string"], f"{i+1} best evolved tree", pset, "GP/potential_final_trees")

    # tree selection
    # tree 1 is missing steepness
    # tree 2 is missing elevation_u
    # tree 3 has all of them
    # tree 4 has all of them
    # tree 5 has all of them
    # tree 6 has all of them
    # tree 7 has all of them
    # we stop

    best_updated = []
    i = 1
    for el in range(3,8):
        best[el-1]["place"] = i
        best_updated.append(best[el-1])
        i += 1
    print(best_updated)

    tree_plotter(best_updated[0]["tree_string"], "Best evolved cost function tree, before pruning", pset, "GP/best_trees_updated/")
    tree_plotter(best_updated[1]["tree_string"], "Second-best evolved cost function tree, before pruning", pset, "GP/best_trees_updated/")
    tree_plotter(best_updated[2]["tree_string"], "Third-best evolved cost function tree, before pruning", pset, "GP/best_trees_updated/")
    tree_plotter(best_updated[3]["tree_string"], "Fourth-best evolved cost function tree", pset, "GP/best_trees_updated/")
    tree_plotter(best_updated[4]["tree_string"], "Fifth-best evolved cost function tree, before pruning", pset, "GP/best_trees_updated/")

    # PRUNING
    # we remove the "identity_water" function and, if present upon visual inspection, we remove nested "if_then_else" functions with "dead" (inactive)
    
    for el in best_updated:
        el["tree_string"] = el["tree_string"].replace("identity_water(is_water)", "is_water")
    
    # these have been correctly pruned
    tree_plotter(best_updated[1]["tree_string"], "Second-best evolved cost function tree, after pruning", pset, "GP/best_trees_updated/")
    tree_plotter(best_updated[2]["tree_string"], "Third-best evolved cost function tree, after pruning", pset, "GP/best_trees_updated/")    
    
    # here we need to remove an inactive branch
    best_updated[4]["tree_string"] = best_updated[4]["tree_string"].replace("if_then_else(is_water, distance, if_then_else(is_water, distance, steepness)))","if_then_else(is_water, distance, steepness))" )
    tree_plotter(best_updated[4]["tree_string"], "Fifth-best evolved cost function tree, after pruning", pset, "GP/best_trees_updated/")

    # here we need to remove multiple nested identity_water functions

    best_updated[0]["tree_string"] = best_updated[0]["tree_string"].replace("neg(add(sub(if_then_else(identity_water(identity_water(identity_water(is_water))), if_then_else(is_water, protected_log(3.148, 9.535), sub(distance, 4.705)), add(distance, neg(neg(distance)))), sub(sub(sub(neg(add(neg(distance), neg(3.685))), mul(neg(protected_div(elevation_u, 4.164)), elevation_u)), elevation_v), if_then_else(is_water, steepness, elevation_v))), neg(neg(neg(neg(neg(protected_pow(2.41, 3.978))))))))", "neg(add(sub(if_then_else(is_water, protected_log(3.148, 9.535), add(distance,distance)), sub(sub(sub(add(distance, 3.685), mul(neg(protected_div(elevation_u, 4.164)), elevation_u)), elevation_v), if_then_else(is_water, steepness, elevation_v))), neg(protected_pow(2.41, 3.978))))")
    tree_plotter(best_updated[0]["tree_string"], "Best evolved cost function tree, after pruning", pset, "GP/best_trees_updated/")



    
    with open("GP/best_trees_updated.json", "w") as f:
        json.dump(best_updated, f, indent=4)
    print("Best individuals have been saved")





        

        