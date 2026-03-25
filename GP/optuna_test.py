import optuna

from GP.GP_with_optimizations import run_EA


def objective(trial):
    pop_size = trial.suggest_int("pop_size", 100, 5000, step=100)
    generations = trial.suggest_int("generations", 50, 5000, step=10)
    mut_rate = trial.suggest_float("mut_rate", 0.05, 0.4)
    cx_rate = trial.suggest_float("cx_rate", 0.5, 0.9)

    best_ind, final_best_fit = run_EA(
        scenarios_number=10,
        population=pop_size,
        generations=generations,
        mut_rate=mut_rate,
        cx_rate=cx_rate,
        res=200,
        npz_path= f"TerrainGraph/precomputed_map_trentino_200.npz"
    )

    return final_best_fit


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print(f"Best values: {study.best_params}")