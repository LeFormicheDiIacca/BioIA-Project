import json
import pandas as pd
import matplotlib.pyplot as plt

with open("tree_diz.json", "r") as f:
    trees = json.load(f)
logs = list()


for el in trees:
    el = el["logs"]
    for run in el:
        for gen in run:
            logs.append(gen)

df_logs = pd.DataFrame(logs)

df_trees = pd.DataFrame(trees)
del df_trees["logs"]

def plot_logs(pop, runs, df_logs = df_logs, res = 80, scenario_dur = 10 ):
    best_tree_vals = df_trees[(df_trees["population"] == pop) & (df_trees["resolution"] == res) & (df_trees["runs"] == runs) & (df_trees["scenario_duration"] == scenario_dur)]["tree_fitness"].values
    best_fitness = best_tree_vals[0][0]
    best_complexity = best_tree_vals[0][1]
    fig, (ax1, ax2) = plt.subplots(2,1, figsize =(20,10), sharex = True)
    gen = df_logs["gen"]
    ax1.plot(gen, df_logs["fit_avg"], label = "average fitness", c = "red", linewidth = 3)
    ax2.plot(gen, df_logs["size_avg"], label = "average graph length", c = "blue", linewidth = 3)
    ax1.set_ylabel("fitness")
    ax2.set_ylabel("total tree nodes")
    ax2.set_xlabel("generation")
    fig.suptitle(f"Tree fitness and complexity with a population of {pop}, a resolution of {res} and {runs} runs", fontsize = 16, va = "top")
    run_start_indices = [i for i, label in enumerate(df_logs["gen"]) if label.endswith(".0")]
    for i in run_start_indices:
        ax1.axvline(x=i, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
        ax2.axvline(x=i, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
        run_num = df_logs["gen"].iloc[i].split('.')[0]
        ax1.text(i, ax1.get_ylim()[1], f'Run {run_num}', rotation=0, verticalalignment='bottom')

    ax1.axvline(x = df_logs[df_logs["fit_avg"] == min(df_logs["fit_avg"])]["gen"].iloc[0], color='red', linestyle='dotted', linewidth = 2 )
    ax2.axvline(x = df_logs[df_logs["size_avg"] == min(df_logs["size_avg"])]["gen"].iloc[0], color='blue', linestyle='dotted', linewidth = 2 )
    ax1.axhline(y = best_fitness, label = "Fitness of the best individual", color = "green", linestyle = "--" )
    ax2.axhline(y= best_complexity, label = "Complexity of the best individual", color = "green", linestyle = "--" )
    ax1.legend()
    ax2.legend()
    plt.show()
    plt.savefig(f"log_figs/log_{gen}_gen_res_{res}_pop_{pop}.png")

# to_try = [[50,3], [50, 5], [100,3], [100,5]]
# for el in to_try:
#     plot_logs(el[0], el[1])

with open("pop_info.json") as f:
    pop_info = json.load(f)
pop_df = pd.DataFrame(pop_info)
print(pop_df)
def plot_pop(pop_info):
    fig, ax = plt.subplots(figsize = (20,20))
    ax.scatter(pop_info["size"], pop_info["fitness"], color = "red")
    ax.set_xlabel("Number of nodes in the tree")
    ax.set_ylabel("Tree fitness")
    ax.set_title("Fitness by size")
    plt.savefig(f"log_figs/fitness_by_size.png")
    plt.show()

plot_pop(pop_df)
