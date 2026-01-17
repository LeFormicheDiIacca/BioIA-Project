import math

# NB. all the parameters I used to evaluate the function can be found in the edge_info.py file,
# please choose whether YOU 

#   prefer to pass the entire edge_dict at the beginning of the run (to get immediately information 
#   about all the edges in the graph

#   OR if you want to implement get_edge_metadata edge by edge

def best_CF(distance, elev_u, elev_v, steepness, is_water):
    if is_water == 1.0:
        ret = math.log(3.148, 9.535)
    else:
        ret = 2*distance
    tmp = ((distance+3.685) + (elev_u/4.164)*elev_u) - elev_v
    if is_water == 1.0:
        tmp -= steepness
    else:
        tmp -= elev_v
    ret -= tmp
    ret = ret + (-1)*(2.41/3.978)
    ret = (-1)*ret
    return ret


def second_best_CF(distance, elev_u, elev_v, steepness, is_water):
    ret = elev_u * steepness * distance
    if is_water == 1.0:
        ret += distance
    else:
        ret += elev_u*elev_v*steepness
    return ret

def third_best_CF(distance, elev_u, elev_v, steepness, is_water):
    ret = elev_u*steepness*distance
    if is_water == 1.0:
        ret += steepness - elev_v
    else:
        ret += elev_u - elev_v
    return ret

def fourth_best_CF(distance, elev_u, elev_v, steepness, is_water):
    if is_water == 1.0:
        ret = distance
    else:
        ret = elev_u
    if is_water == 1.0:
        ret += elev_v
    else:
        ret += 1.774
    ret -= (steepness - distance - distance)
    return ret
    

def fifth_best_CF(distance, elev_u, elev_v, steepness, is_water):
    ret = elev_u * steepness * distance
    if is_water == 1.0:
        ret += distance
    else:
        ret += elev_u - elev_v
    return ret


if __name__ == "__main__":
    import json
    from gp_logistics import tree_plotter
    from GP_with_optimizations import pset
    with open("GP/best_trees.json") as f:
        best_trees = json.load(f)
    min_string, min_ind, min_fit = best_trees[0]["tree_string"], best_trees[0]["individual"], best_trees[0]["fitness"]
    second_min_string, second_min_ind, second_min_fit = best_trees[1]["tree_string"], best_trees[1]["individual"], best_trees[1]["fitness"]
    third_min_string,  third_min_ind, third_min_fit = best_trees[2]["tree_string"], best_trees[2]["individual"], best_trees[2]["fitness"]
    fourth_min_string,  fourth_min_ind,fourth_min_fit= best_trees[3]["tree_string"], best_trees[3]["individual"], best_trees[3]["fitness"]
    fifth_min_string,  fifth_min_ind, fifth_min_fit = best_trees[4]["tree_string"], best_trees[4]["individual"], best_trees[4]["fitness"]

    tree_combos = [(1000,15), (5000,15), (500, 20)]
    durs = [20, 25]
    results = []

    for el in tree_combos:
        for dur in durs:
            for i in range(el[1]):
                path = f"GP/res/runs_15_01_2026/{el[0]}pop_{dur}gen_{el[1]}run_200res/{el[0]}pop_{dur}gen_run{i+1}_res200_{i}subrun.json"
                
                with open(path) as f:
                    diz = json.load(f)[0]
                
                # Store the current individual's data
                results.append({
                    "tree_string": diz["best_individual"],
                    "individual": f"{el[0]}, {dur}gen, {el[1]}",
                    "fitness": diz["best_individual_fitness"][0]
                })
    for i in range(len(best_trees)):
        results.append(best_trees[i])

    # 1. Sort the list based on fitness (ascending)
    results.sort(key=lambda x: x["fitness"])

    # 2. Filter for unique fitness values
    unique_results = []
    seen_fitness = set()

    for entry in results:
        fit = entry["fitness"]
        if entry["tree_string"] != "add(if_then_else(is_water, distance, neg(if_then_else(is_water, elevation_u, mul(elevation_v, steepness)))), elevation_v)" and entry["tree_string"]!= "mul(0.032, add(if_then_else(is_water, if_then_else(is_water, mul(elevation_v, elevation_u), steepness), 0.309), distance))":
            if fit not in seen_fitness:
                seen_fitness.add(fit)
                unique_results.append(entry)
            
            # Optional: Stop once we have our top 5 unique values
            if len(unique_results) == 5:
                break

    # 3. Assign rankings
    for index, item in enumerate(unique_results):
        item["place"] = index + 1

    best = unique_results # This now contains exactly 5 items with unique fitnesses
    print(best)
    with open("GP/best_trees_updated.json", "w") as f:
        json.dump(best, f, indent=4)
    print("Best individuals have been saved")
    
    tree_plotter(best[0]["tree_string"], "Best evolved cost function tree, before pruning", pset, "GP/best_trees_updated/")
    tree_plotter(best[1]["tree_string"], "Second-best evolved cost function tree, before pruning", pset, "GP/best_trees_updated/")
    tree_plotter(best[2]["tree_string"], "Third-best evolved cost function tree, before pruning", pset, "GP/best_trees_updated/")
    tree_plotter(best[3]["tree_string"], "Fourth-best evolved cost function tree", pset, "GP/best_trees_updated/")
    tree_plotter(best[4]["tree_string"], "Fifth-best evolved cost function tree", pset, "GP/best_trees_updated/")

    to_prune = best[0]["tree_string"]
    print("BEFORE PRUNING")
    print(to_prune)
    to_prune = to_prune.replace("identity_water(", "")
    to_prune = to_prune.replace("if_then_else(is_water), protected_log(3.148, 9.535), sub(distance, 4.705))", "protected_log(3.148, 9.535)")
    to_prune = to_prune.replace("if_then_else(is_water))))", "if_then_else(is_water)")
    to_prune = to_prune.replace("if_then_else(is_water, steepness, elevation_v)))", "if_then_else(is_water, steepness, elevation_v))")
    print("Have all leftover parentheses been removed?", to_prune.count("(") == to_prune.count(")"))
    print("AFTER PRUNING (eliminating non-active branches and deleting 'identity_water' functions)")
    print(to_prune)
    tree_plotter(to_prune, "Best evolved cost function tree, after pruning", pset, "GP/best_trees_updated/")
    print(best[1]["tree_string"])
    to_prune2 = best[1]["tree_string"]
    to_prune2 = to_prune2.replace("identity_water(is_water)", "is_water")
    tree_plotter(to_prune2, "Second-best evolved cost function tree, after pruning", pset, "GP/best_trees_updated/")
    print(best[2]["tree_string"])
    to_prune3 = best[2]["tree_string"]
    to_prune3 = to_prune3.replace("identity_water(is_water)", "is_water")
    tree_plotter(to_prune3, "Third-best evolved cost function tree, after pruning", pset, "GP/best_trees_updated/")







        

        