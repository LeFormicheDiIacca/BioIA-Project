import optuna

from GP.GP_with_optimizations import run_EA


def objective(trial):
    res = 200
    pop_size = trial.suggest_int("pop_size", 100, 5000, step=100)
    generations = trial.suggest_int("generations", 50, 5000, step=10)
    mut_rate = trial.suggest_float("mut_rate", 0.05, 0.4)
    cx_rate = trial.suggest_float("cx_rate", 0.5, 0.9)
    npz_path = f"Dataset/Naples/precomputed_map_napoli_{res}.npz"
    scenario_path = f"Dataset/Naples/scenarios_napoli_{res}.json"

    best_ind, final_best_fit = run_EA(
        population=pop_size,
        generations=generations,
        mut_rate=mut_rate,
        cx_rate=cx_rate,
        res=res,
        npz_path= npz_path,
        scenario_path = scenario_path
    )

    return final_best_fit


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print(f"Best values: {study.best_params}")