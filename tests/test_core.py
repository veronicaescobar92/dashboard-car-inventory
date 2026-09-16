from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from repuestos.analytics import Analytics, DomainError
from repuestos.api import PredictionRequest, create_app
from repuestos.data import backup, content_hash, generate, read, rebuild_inventory, save, validate
from repuestos.ml import FEATURES, features, panel, predict, scores, train


@pytest.fixture(scope="session")
def bundle():
    return generate()


@pytest.fixture(scope="session")
def trained(bundle, tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    return train(bundle, path), path


@pytest.fixture
def small(bundle):
    value = deepcopy(bundle)
    value["sales"] = [dict(sale_id=1, date="2025-01-15", product_id="P001", quantity=2, unit_price=10000, unit_cost=6000)]
    return rebuild_inventory(value)


def test_reproducible_catalog(bundle):
    assert bundle == generate()
    assert len(bundle["sales"]) > 10000
    assert len(bundle["products"]) == 30
    assert len(bundle["vehicles"]) == 30
    assert len({v["brand"] for v in bundle["vehicles"]}) >= 12
    assert len({p["part_brand"] for p in bundle["products"]}) >= 10
    assert bundle == generate()
    assert bundle["inventory_lots"] and bundle["inventory_movements"] and bundle["purchase_orders"]
    assert {m["cause"] for m in bundle["inventory_movements"]} >= {None, "damage", "expiry", "count_difference"}
    validate(bundle)


def inventory_case(daily_sales=None, initial=20, waste=2):
    daily_sales = [("2025-01-15", 8)] if daily_sales is None else daily_sales
    sales = [dict(sale_id=i+1, date=day, product_id="P001", quantity=qty,
                  unit_price=15000, unit_cost=10000) for i, (day, qty) in enumerate(daily_sales)]
    movements = [dict(movement_id=1, date="2025-01-01", product_id="P001", lot_id="L001",
                      movement_type="initial", quantity=initial, unit_cost=10000, cause=None, sale_id=None)]
    next_id = 2
    for sale in sales:
        movements.append(dict(movement_id=next_id, date=sale["date"], product_id="P001", lot_id="L001",
                              movement_type="sale", quantity=-sale["quantity"], unit_cost=10000,
                              cause=None, sale_id=sale["sale_id"]))
        next_id += 1
    if waste:
        movements.append(dict(movement_id=next_id, date="2025-01-20", product_id="P001", lot_id="L001",
                              movement_type="waste", quantity=-waste, unit_cost=10000,
                              cause="count_difference", sale_id=None))
    return dict(products=[dict(product_id="P001", sku="SKU-1", name="Producto", category="Test", part_brand="Marca")],
                vehicles=[], compatibility=[], sales=sales,
                suppliers=[dict(supplier_id="S001", name="Proveedor")],
                inventory_policies=[dict(product_id="P001", supplier_id="S001", lead_time_days=5,
                                         service_level=.95, review_window_days=90)],
                inventory_lots=[dict(lot_id="L001", product_id="P001", received_date="2025-01-01",
                                     expiry_date="2025-02-15", initial_quantity=initial, unit_cost=10000)],
                inventory_movements=movements, purchase_orders=[],
                metadata=dict(synthetic=True, start_date="2025-01-01", end_date="2025-01-31",
                              currency="CLP", coverage="complete", holding_rate=.18,
                              dataset_version="fixture"))


def test_inventory_reconciliation_costs_and_labels():
    case = inventory_case()
    result = Analytics(case).inventory_overview(as_of="2025-01-30", window_days=30, slow_days=10)
    row = result["products"][0]
    assert row["stock"] == 10
    assert row["stock_value"] == 100000
    assert result["summary"]["immobilized_capital"] == 100000
    assert result["summary"]["holding_cost"] == pytest.approx(sum(
        frame_value * .18 / 365 for frame_value in ([200000]*14 + [120000]*5 + [100000]*11)))
    assert result["waste"][0]["label"] == "Diferencia pendiente de revisión"
    assert "robo" not in str(result).lower()


def test_inventory_stockout_episodes_and_reorder_formula():
    daily_sales = [(f"2025-01-{day:02}", 1 if day % 2 else 3) for day in range(1, 31)]
    case = inventory_case(daily_sales=daily_sales, initial=68, waste=0)
    result = Analytics(case).inventory_overview(as_of="2025-01-30", window_days=30)
    order = result["reorder"][0]
    assert order["reorder_point"] == 14 and order["stock"] == 8 and order["suggested_order"] == 6
    case = inventory_case(daily_sales=[], initial=1, waste=0)
    case["inventory_movements"] += [
        dict(movement_id=2, date="2025-01-02", product_id="P001", lot_id="L001", movement_type="adjustment", quantity=-1, unit_cost=10000, cause="other", sale_id=None),
        dict(movement_id=3, date="2025-01-05", product_id="P001", lot_id="L001", movement_type="receipt", quantity=1, unit_cost=10000, cause=None, sale_id=None),
        dict(movement_id=4, date="2025-01-06", product_id="P001", lot_id="L001", movement_type="adjustment", quantity=-1, unit_cost=10000, cause="other", sale_id=None),
        dict(movement_id=5, date="2025-01-08", product_id="P001", lot_id="L001", movement_type="receipt", quantity=1, unit_cost=10000, cause=None, sale_id=None),
    ]
    outages = Analytics(case).inventory_overview(as_of="2025-01-10", window_days=30)["stockouts"][0]
    assert outages["episodes"] == 2 and outages["days"] == 5
    assert "venta perdida" in outages["note"] and "clientes" not in outages["note"]


@pytest.mark.parametrize("field,value", [("quantity", -1), ("quantity", 1.5), ("unit_cost", -10),
                                         ("date", "2030-01-01"), ("product_id", "UNKNOWN")])
def test_invalid_preserves_previous(small, tmp_path, field, value):
    db = tmp_path / "demo.sqlite"
    save(small, db)
    previous = db.read_bytes()
    broken = deepcopy(small)
    broken["sales"][0][field] = value
    with pytest.raises(ValueError):
        save(broken, db)
    assert db.read_bytes() == previous


def test_duplicates_and_hash(small):
    broken = deepcopy(small)
    broken["sales"].append(broken["sales"][0])
    with pytest.raises(ValueError, match="duplicado"):
        validate(broken)
    small["sales"][0]["quantity"] = 5
    with pytest.raises(ValueError, match="Hash"):
        validate(small)


def test_backup_restore(bundle, tmp_path):
    db, copy, restored = [tmp_path / n for n in ["original.sqlite", "copy.sqlite", "restored.sqlite"]]
    save(bundle, db)
    backup(db, copy)
    save(read(copy), restored)
    assert read(restored) == bundle
    assert Analytics(read(restored)).summary() == Analytics(bundle).summary()


def test_manual_margin_zero_ranking(small):
    a = Analytics(small)
    result = a.summary()
    assert (result["units"], result["revenue"], result["cost"], result["gross_margin"], result["margin_pct"]) == (2, 20000, 12000, 8000, 40)
    rows = a.ranking(order="asc")
    assert len(rows) == 30 and rows[0]["product_id"] == "P002" and rows[-1]["product_id"] == "P001"
    assert a.summary(product_ids=["P002"])["margin_pct"] is None


def test_compatibility_deduplicated_and_intersection(small):
    a = Analytics(small)
    assert a.summary(vehicle_model=["Yaris", "Sail"])["units"] == 2
    assert a.summary(vehicle_model=["Yaris"])["units"] == 2
    assert a.summary(vehicle_model=["Sail"])["units"] == 2
    assert a.summary(brand=["Kia"], vehicle_model=["Yaris"])["product_count"] == 0
    assert a.ranking(brand=["inexistente"]) == []


def test_monthly_zero_partial_outside_and_ties(small):
    a = Analytics(small)
    jan = a.monthly(start_date="2025-01-01", end_date="2025-01-31")[0]
    assert jan["previous_units"] == 0 and jan["difference_units"] == 2 and jan["change_pct_units"] is None
    partial = a.monthly(start_date="2025-01-10", end_date="2025-01-31")[0]
    assert partial["coverage"] == "partial" and partial["previous_units"] is None
    outside = a.monthly(start_date="2023-12-01", end_date="2023-12-31")[0]
    assert outside["units"] is None and outside["coverage"] == "unavailable"
    mins = [s for s in a.signals() if s["kind"] == "meses_menor_venta"][0]
    assert len(mins["months"]) == 23
    with pytest.raises(DomainError):
        a.summary(start_date="2025-02-01", end_date="2025-01-01")


def test_panel_and_no_future_leak(bundle):
    data = panel(bundle)
    assert len(data) == 720
    original = features(data)
    changed = data.copy()
    cutoff = pd.Period("2025-10", "M")
    changed.loc[changed.month >= cutoff, "units"] += 9999
    later = features(changed)
    pd.testing.assert_frame_equal(original[original.month == cutoff][FEATURES], later[later.month == cutoff][FEATURES])
    zero = data[(data.product_id == "P009") & (data.month == pd.Period("2025-08", "M"))]
    assert zero.units.iloc[0] == 0


def test_temporal_evaluation_and_reload(bundle, trained):
    artifact, path = trained
    report = artifact["report"]
    assert len(report["validation"]) == 4 and report["sample_size"] == 90
    for candidate in report["validation"]:
        for fold in candidate["folds"]:
            assert fold["train_end"] < fold["target"] < "2025-10" and fold["n"] == 30
    records = report["test_predictions"]
    assert all(r["train_end"] < r["month"] for r in records)
    for method in ["model", "baseline"]:
        assert scores([r["actual"] for r in records], [r[method] for r in records]) == report["global_scores"][method]
    values = predict(bundle, artifact, ["P001", "P009"])
    assert values == predict(bundle, joblib.load(path), ["P001", "P009"])
    assert all(np.isfinite(v["estimated_units"]) and v["estimated_units"] >= 0 for v in values)
    assert all(v["target_month"] == "2026-01" for v in values)
    changed = deepcopy(bundle)
    changed["metadata"]["dataset_version"] = "different"
    with pytest.raises(DomainError, match="cambiaron"):
        predict(changed, artifact, ["P001"])
    assert report["warning"] == (report["global_scores"]["model"]["mae"] >= report["global_scores"]["baseline"]["mae"])


def test_api_contract(bundle, trained, tmp_path):
    _, model_path = trained
    db = tmp_path / "demo.sqlite"
    save(bundle, db)
    with TestClient(create_app(db, model_path)) as client:
        assert client.get("/health").json()["model_ready"]
        for endpoint in ["/catalog", "/analytics/summary", "/analytics/ranking", "/analytics/monthly",
                         "/inventory/overview", "/model/metrics"]:
            r = client.get(endpoint)
            assert r.status_code == 200 and r.json()["metadata"]["synthetic"]
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").json()["paths"]["/predictions"]["post"]["responses"]["422"]
        assert client.post("/predictions", json={"product_ids": ["P001"]}).status_code == 200
        for body in [{"product_ids": []}, {"product_ids": ["P001", "P001"]},
                     {"product_ids": ["P001"], "target_month": "2026-02"}]:
            assert client.post("/predictions", json=body).status_code == 422
        assert client.post("/predictions", json={"product_ids": ["P001", "P999"]}).status_code == 404
        assert client.get("/analytics/summary?start_date=bad").status_code == 422
        assert client.get("/analytics/summary?start_date=2025-02-01&end_date=2025-01-01").status_code == 422
        assert client.get("/analytics/ranking?brand=unknown").json()["data"]["rows"] == []
        assert client.get("/analytics/ranking?metric=wrong").status_code == 422
        inventory = client.get("/inventory/overview?product_ids=P009").json()["data"]
        assert inventory["summary"]["products"] == 1 and inventory["stockouts"]
        assert client.get("/inventory/overview?window_days=2").status_code == 422
        assert client.get("/inventory/overview?as_of=2030-01-01").status_code == 422
        # Stale in-memory results must not masquerade as current data.
        db.touch()
        assert client.post("/predictions", json={"product_ids": ["P001"]}).status_code == 503


def test_missing_and_incompatible(bundle, trained, tmp_path):
    _, model_path = trained
    with TestClient(create_app(tmp_path / "missing", model_path)) as client:
        assert not client.get("/health").json()["data_ready"]
        assert client.get("/catalog").status_code == 503
    db = tmp_path / "demo.sqlite"
    save(bundle, db)
    with TestClient(create_app(db, tmp_path / "missing")) as client:
        assert client.get("/catalog").status_code == 200
        assert client.post("/predictions", json={"product_ids": ["P001"]}).status_code == 503
    altered = deepcopy(bundle)
    altered["sales"][0]["quantity"] += 1
    rebuild_inventory(altered)
    save(altered, db)
    with TestClient(create_app(db, model_path)) as client:
        assert client.get("/model/metrics").status_code == 503


def test_request_examples():
    assert PredictionRequest(product_ids=["P001"]).target_month is None
    with pytest.raises(ValidationError):
        PredictionRequest(product_ids=["P001", "P001"])
