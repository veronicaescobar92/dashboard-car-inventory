"""Modelo global y evaluación de un paso con cortes temporales explícitos."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from repuestos.analytics import DomainError
from repuestos.data import read
from repuestos.settings import MODEL_PATH

FEATURES = ["product_id", "category", "sin_month", "cos_month", "time_index", "lag1", "lag2", "lag3", "mean3"]


def panel(bundle):
    meta = bundle["metadata"]
    months = pd.period_range(meta["start_date"], meta["end_date"], freq="M")
    if len(months) < 4:
        raise DomainError(503, "insufficient_history", "Se requieren al menos cuatro meses completos.")
    sales = pd.DataFrame(bundle["sales"], columns=["date", "product_id", "quantity"])
    sales["month"] = pd.to_datetime(sales.date).dt.to_period("M")
    grouped = sales.groupby(["product_id", "month"]).quantity.sum()
    rows = []
    for product in bundle["products"]:
        for month in months:
            rows.append(dict(product_id=product["product_id"], category=product["category"], month=month,
                             units=int(grouped.get((product["product_id"], month), 0))))
    return pd.DataFrame(rows)


def features(data):
    frame = data.sort_values(["product_id", "month"]).copy()
    for lag in (1, 2, 3):
        frame[f"lag{lag}"] = frame.groupby("product_id").units.shift(lag)
    frame["mean3"] = frame[["lag1", "lag2", "lag3"]].mean(axis=1, skipna=False)
    frame["sin_month"] = frame.month.map(lambda m: np.sin(2*np.pi*m.month/12))
    frame["cos_month"] = frame.month.map(lambda m: np.cos(2*np.pi*m.month/12))
    frame["time_index"] = frame.month.map(lambda m: m.ordinal)
    return frame.dropna(subset=FEATURES)


def make_model(depth=3, leaf=2):
    transform = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                    ["product_id", "category"])], remainder="passthrough")
    return Pipeline([("prepare", transform), ("regressor", RandomForestRegressor(
        n_estimators=100, max_depth=depth, min_samples_leaf=leaf, random_state=42, n_jobs=1))])


def scores(actual, predicted):
    return dict(mae=float(mean_absolute_error(actual, predicted)),
                rmse=float(np.sqrt(mean_squared_error(actual, predicted))))


def train(bundle, path=MODEL_PATH):
    data = features(panel(bundle))
    product_count = len(bundle["products"])
    validation_months = [pd.Period(m, "M") for m in ["2025-07", "2025-08", "2025-09"]]
    test_months = [pd.Period(m, "M") for m in ["2025-10", "2025-11", "2025-12"]]
    search = []
    for depth in (3, None):
        for leaf in (5, 2):
            folds = []
            for month in validation_months:
                past, target = data[data.month < month], data[data.month == month]
                if past.empty or len(target) != product_count:
                    raise DomainError(503, "training_period", "La evaluación requiere el conjunto completo 2024-2025.")
                model = make_model(depth, leaf).fit(past[FEATURES], past.units)
                folds.append(dict(train_end=str(month-1), target=str(month), n=len(target),
                                  **scores(target.units, model.predict(target[FEATURES]))))
            search.append(dict(depth=depth, leaf=leaf, mae=float(np.mean([f["mae"] for f in folds])), folds=folds))
    # Stable tie break: shallow trees, then larger leaves.
    best = min(search, key=lambda s: (s["mae"], s["depth"] or 999, -s["leaf"]))
    records = []
    for month in test_months:
        past, target = data[data.month < month], data[data.month == month]
        model = make_model(best["depth"], best["leaf"]).fit(past[FEATURES], past.units)
        predictions = model.predict(target[FEATURES])
        for (_, row), pred in zip(target.iterrows(), predictions):
            records.append(dict(product_id=row.product_id, month=str(month), train_end=str(month-1),
                                actual=int(row.units), model=float(max(0, pred)), baseline=float(row.mean3)))
    result = pd.DataFrame(records)
    global_scores = {k: scores(result.actual, result[k]) for k in ("model", "baseline")}
    per_product = {p: {k: scores(g.actual, g[k]) for k in ("model", "baseline")}
                   for p, g in result.groupby("product_id")}
    report = dict(global_scores=global_scores, per_product=per_product, sample_size=len(records),
                  per_product_sample_size=3, validation=search, selected={k: best[k] for k in ["depth", "leaf"]},
                  test_predictions=records, synthetic=True, seed=42,
                  warning=global_scores["model"]["mae"] >= global_scores["baseline"]["mae"],
                  limitation="Evaluación sobre simulación; no demuestra precisión ni beneficios en una empresa real.")
    final = make_model(best["depth"], best["leaf"]).fit(data[FEATURES], data.units)
    meta = dict(bundle["metadata"], model_version="random-forest-v1", sklearn_version=sklearn.__version__,
                trained_at=datetime.now(timezone.utc).isoformat(), feature_schema=FEATURES,
                target_month=str(pd.Period(bundle["metadata"]["end_date"], "M")+1))
    artifact = dict(pipeline=final, metadata=meta, report=report)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    path.with_name("metrics.json").write_text(json.dumps(dict(metadata=meta, **report), ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def predict(bundle, artifact, product_ids, target_month=None):
    meta = artifact["metadata"]
    if meta["dataset_version"] != bundle["metadata"]["dataset_version"]:
        raise DomainError(503, "incompatible_model", "Los datos cambiaron. Vuelva a entrenar el modelo.")
    if meta.get("feature_schema") != FEATURES or meta.get("sklearn_version") != sklearn.__version__:
        raise DomainError(503, "incompatible_model", "Versión del modelo incompatible. Vuelva a entrenar.")
    unknown = set(product_ids) - {p["product_id"] for p in bundle["products"]}
    if unknown:
        raise DomainError(404, "unknown_product", "Productos desconocidos: " + ", ".join(sorted(unknown)))
    if target_month and target_month != meta["target_month"]:
        raise DomainError(422, "invalid_horizon", "Solo está disponible el mes " + meta["target_month"])
    data = panel(bundle)
    future = pd.DataFrame([dict(product_id=p["product_id"], category=p["category"],
                                month=pd.Period(meta["target_month"], "M"), units=0) for p in bundle["products"]])
    prepared = features(pd.concat([data, future], ignore_index=True))
    target = prepared[(prepared.month == pd.Period(meta["target_month"], "M")) & prepared.product_id.isin(product_ids)]
    if len(target) != len(product_ids):
        raise DomainError(503, "insufficient_history", "Historial insuficiente para los productos solicitados.")
    values = artifact["pipeline"].predict(target[FEATURES])
    return [dict(product_id=p, target_month=meta["target_month"], estimated_units=float(max(0, v)))
            for p, v in zip(target.product_id, values)]


if __name__ == "__main__":
    artifact = train(read())
    print(json.dumps(artifact["report"]["global_scores"], indent=2))
