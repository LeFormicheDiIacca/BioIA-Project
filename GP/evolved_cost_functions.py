# NB. all the parameters I used to evaluate the function can be found in the edge_info.py file,
# please choose whether YOU 

#   prefer to pass the entire edge_dict at the beginning of the run (to get immediately information 
#   about all the edges in the graph

#   OR if you want to implement get_edge_metadata edge by edge

def best_CF(distance, elev_u, elev_v, steepness, is_water):
    ret = 7.828 
    if is_water==1.0:
        ret += distance + elev_v
    else:
        ret += steepness*distance*elev_v*elev_u
    return ret

def second_best_CF(distance, elev_u, elev_v, steepness, is_water):
    if is_water == 1.0:
        ret = (distance+elev_v)*elev_u
    else:
        ret = (1.701 + steepness)*0.117*distance
    return ret

def third_best_CF(distance, elev_u, elev_v, steepness, is_water):
    if is_water == 1.0:
        ret = steepness
    else:
        ret = elev_u - distance*elev_v
    ret += 8.341 - steepness
    return ret

def fourth_best_CF(distance, elev_u, elev_v, steepness, is_water):
    ret = elev_u*distance
    if is_water == 1.0:
        ret +=7.863
    else:
        ret += -elev_v - steepness*distance + steepness
    return ret

def fifth_best_CF(distance, elev_u, elev_v, steepness, is_water):
    ret = elev_u * distance
    if is_water == 1.0:
        ret += 9.631
    else:
        ret += steepness
    ret -= (elev_v+elev_u)
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

    tree_combos = [(1000, 25), (2000,20), (2000,25), (5000, 15)]
    durs = [10, 20]
    for el in tree_combos:
        for dur in durs:
            for i in range(el[1]):
                with open(f"GP/runs_14_01_2026/{el[0]}pop_{dur}gen_{el[1]}run_200res/{el[0]}pop_{dur}gen_run{i+1}_res200_{i}subrun.json") as f:
                    diz = json.load(f)
                diz = diz[0]
                if diz["best_individual_fitness"][0] < min_fit:
                    min_fit = diz["best_individual_fitness"][0]
                    min_ind = f"{el[0]}, {dur}gen, {el[1]}.{i+1}"
                    min_string = diz["best_individual"]
                elif diz["best_individual_fitness"][0] < second_min_fit and diz["best_individual_fitness"][0] > min_fit:
                    second_min_fit = diz["best_individual_fitness"][0]
                    second_min_ind = f"{el[0]}, {dur}gen, {el[1]}.{i+1}"
                    second_min_string = diz["best_individual"]
                elif diz["best_individual_fitness"][0] < third_min_fit and diz["best_individual_fitness"][0] > second_min_fit:
                    third_min_fit = diz["best_individual_fitness"][0]
                    third_min_ind = f"{el[0]}, {dur}gen, {el[1]}.{i+1}"
                    third_min_string = diz["best_individual"]
                elif diz["best_individual_fitness"][0] < fourth_min_fit and diz["best_individual_fitness"][0] > third_min_fit:
                    fourth_min_fit = diz["best_individual_fitness"][0]
                    fourth_min_ind = f"{el[0]}, {dur}gen, {el[1]}.{i+1}"
                    fourth_min_string =diz["best_individual"]
                elif diz["best_individual_fitness"][0] < fifth_min_fit and diz["best_individual_fitness"][0] > fourth_min_fit:
                    fifth_min_fit = diz["best_individual_fitness"][0]
                    fifth_min_ind = f"{el[0]}, {dur}gen, {el[1]}.{i+1}" 
                    fifth_min_string = diz["best_individual"]
    
    best = [{"place":1, "tree_string" : min_string, "individual": min_ind, "fitness": min_fit},
            {"place":2, "tree_string" : second_min_string,"individual": second_min_ind, "fitness": second_min_fit},
            {"place":3, "tree_string" : third_min_string,"individual": third_min_ind, "fitness": third_min_fit},
            {"place":4, "tree_string" : fourth_min_string,"individual": fourth_min_ind, "fitness": fourth_min_fit},
            {"place":5, "tree_string" : fifth_min_string,"individual": fifth_min_ind, "fitness": fifth_min_fit},
            ]

    with open("GP/best_trees_updated.json", "w") as f:
        json.dump(best, f)
    print("Best individuals have been saved")
    tree_plotter(best[0]["tree_string"], "Best evolved cost function tree", pset, "GP/best_trees_updated/")
    tree_plotter(best[1]["tree_string"], "Second-best evolved cost function tree", pset, "GP/best_trees_updated/")
    tree_plotter(best[2]["tree_string"], "Third-best evolved cost function tree", pset, "GP/best_trees_updated/")
    tree_plotter(best[3]["tree_string"], "Fourth-best evolved cost function tree", pset, "GP/best_trees_updated/")
    tree_plotter(best[4]["tree_string"], "Fifth-best evolved cost function tree", pset, "GP/best_trees_updated/")





    

    