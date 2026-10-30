"""Busca evolutiva de hiperparâmetros (substitui app/quantics da v1).

A v1 dependia de um pacote `quantics` que não existe no PyPI e convertia parâmetros
inteiros em float ao mutar (erro no scikit-learn). Aqui os espaços são tipados.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from app.ml.automl_engine import build_preprocessor, infer_problem_type, split_columns

SPACE = {
    "n_estimators": ("int", 20, 300),
    "max_depth": ("int", 2, 20),
    "min_samples_leaf": ("int", 1, 10),
    "max_features": ("choice", ["sqrt", "log2", None]),
}


@dataclass
class Individual:
    params: dict
    fitness: float = float("-inf")


def _sample(rng: random.Random) -> dict:
    out = {}
    for key, spec in SPACE.items():
        out[key] = rng.randint(spec[1], spec[2]) if spec[0] == "int" else rng.choice(spec[1])
    return out


def _mutate(params: dict, rng: random.Random, rate: float = 0.3) -> dict:
    child = dict(params)
    for key, spec in SPACE.items():
        if rng.random() >= rate:
            continue
        if spec[0] == "int":
            span = max(1, (spec[2] - spec[1]) // 5)
            child[key] = int(min(spec[2], max(spec[1], child[key] + rng.randint(-span, span))))
        else:
            child[key] = rng.choice(spec[1])
    return child


def _crossover(a: dict, b: dict, rng: random.Random) -> dict:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in SPACE}


def evolve(df: pd.DataFrame, target: str, generations: int = 4, population: int = 6, cv: int = 3, seed: int = 42) -> dict:
    if target not in df.columns:
        raise ValueError(f"coluna alvo '{target}' não encontrada")
    data = df.dropna(subset=[target])
    X, y = data.drop(columns=[target]), data[target]
    numeric, categorical = split_columns(X)
    X = X[numeric + categorical]
    ptype = infer_problem_type(y)
    if ptype == "classification":
        y = y.astype(str)
        cv = max(2, min(cv, int(y.value_counts().min())))
    scoring = "accuracy" if ptype == "classification" else "r2"
    Model = RandomForestClassifier if ptype == "classification" else RandomForestRegressor
    rng = random.Random(seed)

    def fitness(ind: Individual) -> float:
        pipe = Pipeline([("prep", build_preprocessor(numeric, categorical)), ("model", Model(random_state=seed, n_jobs=1, **ind.params))])
        return float(np.mean(cross_val_score(pipe, X, y, cv=cv, scoring=scoring)))

    pop = [Individual(_sample(rng)) for _ in range(population)]
    history = []
    for gen in range(generations):
        for ind in pop:
            if ind.fitness == float("-inf"):
                ind.fitness = fitness(ind)
        pop.sort(key=lambda i: i.fitness, reverse=True)
        history.append({"generation": gen, "best": round(pop[0].fitness, 4), "mean": round(float(np.mean([i.fitness for i in pop])), 4)})
        elite = pop[: max(2, population // 3)]
        children = []
        while len(elite) + len(children) < population:
            a, b = rng.sample(elite, 2)
            children.append(Individual(_mutate(_crossover(a.params, b.params, rng), rng)))
        pop = elite + children
    for ind in pop:
        if ind.fitness == float("-inf"):
            ind.fitness = fitness(ind)
    best = max(pop, key=lambda i: i.fitness)
    return {"problem_type": ptype, "metric": scoring, "best_params": best.params, "best_score": round(best.fitness, 4), "history": history}
