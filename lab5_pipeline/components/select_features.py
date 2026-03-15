# components/select_features.py
import argparse
import pandas as pd
import numpy as np
import os
import random
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from deap import base, creator, tools, algorithms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",  type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    # ── Load ─────────────────────────────────────────────
    print("📥 Loading features...")
    train = pd.read_parquet(f"{args.input_path}/features_train.parquet")
    test  = pd.read_parquet(f"{args.input_path}/features_test.parquet")

    y_train = train["RUL"].dropna()
    X       = train.drop(columns=["RUL", "id"], errors="ignore").loc[y_train.index]

    y_test  = test["RUL"].dropna()
    X_test  = test.drop(columns=["RUL", "id"], errors="ignore").loc[y_test.index]

    print(f"📊 Starting features: {X.shape[1]}")

    # ── Stage 1: Variance ────────────────────────────────
    sel   = VarianceThreshold(threshold=0.01)
    X_var = pd.DataFrame(sel.fit_transform(X),
                         columns=X.columns[sel.get_support()],
                         index=X.index)
    print(f"✅ After Variance    : {X_var.shape[1]}")

    # ── Stage 2: Correlation ─────────────────────────────
    corr  = X_var.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop  = [c for c in upper.columns if any(upper[c] > 0.95)]
    X_corr = X_var.drop(columns=drop)
    print(f"✅ After Correlation : {X_corr.shape[1]}")

    # ── Stage 3: Mutual Information ──────────────────────
    mi        = mutual_info_regression(X_corr, y_train, random_state=42)
    mi_series = pd.Series(mi, index=X_corr.columns).sort_values(ascending=False)
    X_filtered = X_corr[mi_series.head(50).index].copy()
    print(f"✅ After MI (top 50) : {X_filtered.shape[1]}")

    # ── Stage 4: Genetic Algorithm ───────────────────────
    X_arr      = X_filtered.values
    y_arr      = y_train.values
    n_features = X_filtered.shape[1]

    random.seed(42); np.random.seed(42)

    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool",  random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    fast_model = DecisionTreeRegressor(max_depth=5, random_state=42)

    def evaluate(individual):
        selected = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected) == 0:
            return (1e9,)
        scores = cross_val_score(
            fast_model, X_arr[:, selected], y_arr,
            cv=2, scoring="neg_root_mean_squared_error")
        return (-scores.mean() + 0.01 * len(selected) / n_features,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate",     tools.cxTwoPoint)
    toolbox.register("mutate",   tools.mutFlipBit, indpb=0.05)
    toolbox.register("select",   tools.selTournament, tournsize=3)

    print("🧬 Running GA...")
    pop   = toolbox.population(n=20)
    hof   = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)

    algorithms.eaSimple(
        pop, toolbox, cxpb=0.6, mutpb=0.2, ngen=10,
        stats=stats, halloffame=hof, verbose=True)

    selected_cols = [X_filtered.columns[i]
                     for i, bit in enumerate(hof[0]) if bit == 1]
    print(f"✅ GA selected: {len(selected_cols)} features")

    # ── Save ─────────────────────────────────────────────
    os.makedirs(args.output_path, exist_ok=True)

    X_final = X_filtered[selected_cols].copy()
    X_final["RUL"] = y_train.values
    X_final.to_parquet(f"{args.output_path}/train_final.parquet", index=False)

    X_test_final = X_test.reindex(columns=selected_cols).fillna(0)
    X_test_final["RUL"] = y_test.values
    X_test_final.to_parquet(f"{args.output_path}/test_final.parquet", index=False)

    # save selected features name
    pd.Series(selected_cols).to_csv(
        f"{args.output_path}/selected_features.csv", index=False)

    print("✅ Feature selection complete!")

if __name__ == "__main__":
    main()