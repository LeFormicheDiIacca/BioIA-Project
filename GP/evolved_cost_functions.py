import numpy as np

from GP.gp_logistics import protected_log, protected_pow, protected_div

#Trentino
def trentino_first_cost_function(distance, steepness, is_water):
    #Run on Trento-Laives of 30/03/2026 3500pop_50gen_200res
    #Final HoF, second individual. Fitness:86.9742238105947
    term1 = protected_pow(distance,0.592)
    term_2 = np.where(is_water,protected_pow(distance,steepness-0.827), steepness)
    term_2 += 1.378+distance
    return term1 + term_2

def trentino_second_cost_function(distance, steepness, is_water):
    #Run on Trento-Laives of 30/03/2026 3500pop_50gen_200res_2
    #Final HoF, fourth individual. Fitness:86.9742280926197
    tmp_1= np.where(is_water, protected_log(distance, 0.628), steepness)
    temp2 = 1.837 + distance + protected_pow(distance,0.628)
    return tmp_1 + temp2

def trentino_third_cost_function(distance, steepness, is_water):
    tmp_1 = np.where(is_water, 1/distance, steepness)
    tmp_2 = 1.439+distance+protected_pow(distance,0.647)
    return tmp_1 + tmp_2

def trentino_fourth_cost_function(distance, steepness, is_water):
    tmp_1 = np.where(is_water,0.884, steepness)
    tmp_2 = protected_pow(distance,0.586)+1.380+distance
    return tmp_1 + tmp_2
#Naples

def naples_first_cost_function(distance, steepness, is_water):
    #Run on Mondragone-Capua of 31/03/2026 3500pop_50gen_200res_3
    #Final HoF, first individual. Fitness:97.73775153818191
    tmp_1 = np.where(is_water, 0.117, 0.11)
    tmp_2 = 0.084 *(steepness+distance)
    return  tmp_1 * tmp_2

def naples_second_cost_function(distance, steepness, is_water):
    #Run on Mondragone-Capua of 31/03/2026 3500pop_50gen_200res_4
    #Final HoF, last individual. Fitness:97.73333262244897
    tmp_1 = np.where(is_water, 0.517, steepness)
    tmp_2 = distance+0.792+protected_div(distance+steepness,0.153)
    return tmp_1 + tmp_2

def naples_third_cost_function(distance, steepness, is_water):
    #Run on Mondragone-Capua of 31/03/2026 3500pop_50gen_200res_5
    #Final HoF, last individual. Fitness:97.73775153818191
    tmp_1 =np.where(is_water, 0.215, steepness)
    return protected_pow(tmp_1+0.786+distance, 0.786)

def naples_fourth_cost_function(distance, steepness, is_water):
    #Run on Mondragone-Capua of 31/03/2026 3500pop_50gen_200res_7
    #Final HoF, second individual. Fitness:97.73775153818191
    tmp_1 = np.where(is_water, 0.629, steepness)
    return 0.009* distance+tmp_1


if __name__ == "__main__":

    # found the last, final round of best individuals

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
                        "individual_id": f"size{el[0]}, run{el[1]}, gen{dur}"
                    })
            except (FileNotFoundError, IndexError, KeyError):
                print("NOT WORKING") 
                print(path)

    for i in range(len(best_trees)):
        results.append(best_trees[i])
    # Sort the list based on fitness
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
    # we remove the "identity_water" function and, if present upon visual inspection, we remove nested "if_then_else" functions with "dead" (inactive) branches
    # we'll also remove multiple negatives and other unnecessary operations
    
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





        

        