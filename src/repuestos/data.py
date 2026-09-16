"""Generador determinista, validación y persistencia atómica de datos ficticios."""
import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from repuestos.settings import DB_PATH

VEHICLES = [
    ("Toyota", "Yaris"), ("Toyota", "Corolla"), ("Toyota", "Hilux"),
    ("Chevrolet", "Sail"), ("Chevrolet", "Spark GT"), ("Chevrolet", "Tracker"),
    ("Kia", "Rio"), ("Kia", "Morning"), ("Kia", "Sportage"),
    ("Hyundai", "Accent"), ("Hyundai", "Grand i10"), ("Hyundai", "Tucson"),
    ("Nissan", "Versa"), ("Nissan", "NP300"), ("Nissan", "Qashqai"),
    ("Suzuki", "Swift"), ("Suzuki", "Baleno"), ("Suzuki", "Vitara"),
    ("Ford", "Ranger"), ("Ford", "EcoSport"), ("Mazda", "Mazda 3"),
    ("Mazda", "CX-5"), ("Mitsubishi", "L200"), ("Volkswagen", "Gol"),
    ("Volkswagen", "T-Cross"), ("Peugeot", "208"), ("Peugeot", "Partner"),
    ("Renault", "Symbol"), ("Renault", "Duster"), ("Honda", "CR-V"),
]

# Nombre, categoría, fabricante, ventas diarias base, precio CLP y grupo de aplicación.
PRODUCT_SPECS = [
    ("Filtro de aceite M20", "Filtros", "Bosch", 5.8, 8900, "compactos"),
    ("Filtro de aceite 3/4-16", "Filtros", "Mann-Filter", 4.5, 10500, "utilitarios"),
    ("Filtro de aire rectangular", "Filtros", "Mahle", 3.8, 14900, "compactos"),
    ("Filtro de aire SUV", "Filtros", "WIX", 2.7, 18900, "suv"),
    ("Filtro de combustible diésel", "Filtros", "Bosch", 1.9, 24900, "diesel"),
    ("Filtro de habitáculo carbón activo", "Filtros", "Mann-Filter", 2.8, 15900, "pasajeros"),
    ("Pastillas de freno delanteras", "Frenos", "Brembo", 3.6, 42900, "pasajeros"),
    ("Pastillas de freno traseras", "Frenos", "TRW", 2.2, 38900, "pasajeros"),
    ("Disco de freno delantero ventilado", "Frenos", "Brembo", 1.7, 62900, "suv"),
    ("Zapatas de freno traseras", "Frenos", "Bosch", 1.3, 35900, "compactos"),
    ("Bujía de iridio", "Encendido", "NGK", 6.4, 9900, "gasolina"),
    ("Bobina de encendido", "Encendido", "Denso", 1.6, 54900, "gasolina"),
    ("Cable de bujías", "Encendido", "Bosch", 1.2, 32900, "compactos"),
    ("Amortiguador delantero", "Suspensión", "Monroe", 1.8, 74900, "pasajeros"),
    ("Amortiguador trasero", "Suspensión", "KYB", 1.5, 65900, "suv"),
    ("Rótula de suspensión", "Suspensión", "Moog", 1.4, 28900, "utilitarios"),
    ("Terminal de dirección", "Dirección", "TRW", 1.3, 26900, "pasajeros"),
    ("Rodamiento de rueda delantero", "Rodamientos", "SKF", 1.2, 58900, "pasajeros"),
    ("Kit de embrague", "Transmisión", "LuK", .65, 169900, "manuales"),
    ("Homocinética exterior", "Transmisión", "GKN", .85, 69900, "compactos"),
    ("Correa de accesorios", "Motor", "Gates", 2.1, 24900, "gasolina"),
    ("Kit correa de distribución", "Motor", "Gates", .75, 124900, "pasajeros"),
    ("Bomba de agua", "Refrigeración", "GMB", 1.0, 52900, "pasajeros"),
    ("Termostato", "Refrigeración", "Mahle", 1.15, 22900, "gasolina"),
    ("Radiador de motor", "Refrigeración", "Denso", .45, 119900, "suv"),
    ("Alternador 12 V", "Electricidad", "Bosch", .38, 189900, "utilitarios"),
    ("Motor de partida", "Electricidad", "Denso", .34, 179900, "diesel"),
    ("Sensor de oxígeno", "Sensores", "Bosch", .9, 64900, "gasolina"),
    ("Sensor ABS delantero", "Sensores", "Denso", .72, 57900, "suv"),
    ("Plumillas limpiaparabrisas 22/18", "Accesorios", "Valeo", 3.3, 19900, "pasajeros"),
]

COMPATIBILITY_GROUPS = {
    "compactos": [0, 1, 3, 4, 6, 7, 9, 10, 12, 15, 16, 20, 23, 25, 27],
    "utilitarios": [2, 13, 18, 22, 26],
    "suv": [5, 8, 11, 14, 17, 19, 21, 24, 28, 29],
    "diesel": [2, 13, 18, 22, 26],
    "pasajeros": [0, 1, 3, 5, 6, 8, 9, 11, 12, 14, 16, 17, 19, 20, 21, 24, 25, 27, 28, 29],
    "gasolina": [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 27, 28, 29],
    "manuales": [0, 3, 4, 6, 7, 9, 10, 12, 15, 16, 18, 23, 25, 26, 27, 28],
}


def content_hash(bundle):
    payload = {k: v for k, v in bundle.items() if k != "metadata"}
    payload["metadata"] = {k: v for k, v in bundle["metadata"].items() if k != "dataset_version"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


INVENTORY_TABLES = ["suppliers", "inventory_policies", "inventory_lots",
                    "inventory_movements", "purchase_orders"]


def _inventory_data(products, sales):
    """Build a deterministic FIFO inventory ledger around the sales history."""
    suppliers = [
        dict(supplier_id="S001", name="Distribuidora Norte"),
        dict(supplier_id="S002", name="Importadora Central"),
        dict(supplier_id="S003", name="Repuestos del Pacífico"),
    ]
    policies = [dict(product_id=p["product_id"], supplier_id=f"S{(i % 3)+1:03}",
                     lead_time_days=3 + i % 8, service_level=.95, review_window_days=90)
                for i, p in enumerate(products)]
    by_product = {p["product_id"]: [] for p in products}
    for sale in sales:
        by_product[sale["product_id"]].append(sale)
    lots, movements = [], []
    next_lot = next_move = 1

    def receive(product_id, received, quantity, unit_cost, expiry=None, kind="receipt"):
        nonlocal next_lot, next_move
        lot_id = f"L{next_lot:05}"
        next_lot += 1
        lots.append(dict(lot_id=lot_id, product_id=product_id, received_date=received,
                         expiry_date=expiry, initial_quantity=int(quantity), unit_cost=int(unit_cost)))
        movements.append(dict(movement_id=next_move, date=received, product_id=product_id,
                              lot_id=lot_id, movement_type=kind, quantity=int(quantity),
                              unit_cost=int(unit_cost), cause=None, sale_id=None))
        next_move += 1
        return dict(lot_id=lot_id, remaining=int(quantity), unit_cost=int(unit_cost))

    for i, product in enumerate(products):
        product_id = product["product_id"]
        product_sales = sorted(by_product[product_id], key=lambda row: (row["date"], row["sale_id"]))
        monthly = {}
        for sale in product_sales:
            monthly.setdefault(sale["date"][:7], []).append(sale)
        queues = []
        jan_units = sum(s["quantity"] for s in monthly.get("2024-01", []))
        base_cost = int(round(PRODUCT_SPECS[i][4] * .64 / 100) * 100)
        queues.append(receive(product_id, "2024-01-01", jan_units + 20, base_cost, kind="initial"))
        receipts = {}
        for month in pd.period_range("2024-02", "2025-12", freq="M"):
            key = str(month)
            quantity = sum(s["quantity"] for s in monthly.get(key, [])) + 15
            receipts.setdefault(month.start_time.date().isoformat(), []).append((quantity, base_cost, None))
        if product_id == "P025":
            receipts.setdefault("2025-10-01", []).append((50, base_cost, "2026-02-15"))
        if product_id == "P027":
            receipts.setdefault("2025-12-01", []).append((100, base_cost, "2025-12-15"))

        sales_by_date = {}
        for sale in product_sales:
            sales_by_date.setdefault(sale["date"], []).append(sale)
        dates = sorted(set(receipts) | set(sales_by_date))
        forced_outage = product_id == "P009"
        for current in dates:
            if forced_outage and current == "2025-07-01":
                for lot in queues:
                    if lot["remaining"]:
                        movements.append(dict(movement_id=next_move, date=current, product_id=product_id,
                                              lot_id=lot["lot_id"], movement_type="adjustment",
                                              quantity=-lot["remaining"], unit_cost=lot["unit_cost"],
                                              cause="count_difference", sale_id=None))
                        next_move += 1
                        lot["remaining"] = 0
            for quantity, cost, expiry in receipts.get(current, []):
                if forced_outage and "2025-07-01" <= current < "2025-09-30":
                    continue
                queues.append(receive(product_id, current, quantity, cost, expiry))
            for sale in sales_by_date.get(current, []):
                pending = sale["quantity"]
                for lot in queues:
                    if pending <= 0:
                        break
                    used = min(pending, lot["remaining"])
                    if used:
                        movements.append(dict(movement_id=next_move, date=current, product_id=product_id,
                                              lot_id=lot["lot_id"], movement_type="sale", quantity=-used,
                                              unit_cost=lot["unit_cost"], cause=None, sale_id=sale["sale_id"]))
                        next_move += 1
                        lot["remaining"] -= used
                        pending -= used
                if pending:
                    raise ValueError(f"Stock insuficiente al generar {product_id} en {current}")
            if forced_outage and current == "2025-12-31":
                for lot in queues:
                    if lot["remaining"]:
                        movements.append(dict(movement_id=next_move, date=current, product_id=product_id,
                                              lot_id=lot["lot_id"], movement_type="adjustment",
                                              quantity=-lot["remaining"], unit_cost=lot["unit_cost"],
                                              cause="count_difference", sale_id=None))
                        next_move += 1
                        lot["remaining"] = 0
        # Reproducible declared shrinkage that remains distinct from theft.
        if product_id in {"P003", "P006"}:
            cause = "damage" if product_id == "P003" else "expiry"
            for lot in queues:
                if lot["remaining"]:
                    qty = min(2, lot["remaining"])
                    movements.append(dict(movement_id=next_move, date="2025-11-15", product_id=product_id,
                                          lot_id=lot["lot_id"], movement_type="waste", quantity=-qty,
                                          unit_cost=lot["unit_cost"], cause=cause, sale_id=None))
                    next_move += 1
                    lot["remaining"] -= qty
                    break
    purchase_orders = [
        dict(order_id=f"PO{i+1:03}", product_id=p["product_id"], supplier_id=f"S{(i % 3)+1:03}",
             order_date="2025-12-28", expected_date=(date(2025, 12, 31) + timedelta(days=3+i % 8)).isoformat(),
             quantity=10+i % 6, status="open")
        for i, p in enumerate(products[:8])
    ]
    return dict(suppliers=suppliers, inventory_policies=policies, inventory_lots=lots,
                inventory_movements=movements, purchase_orders=purchase_orders)


def rebuild_inventory(bundle):
    """Refresh derived inventory tables after replacing sales in fixtures/imports."""
    bundle.update(_inventory_data(bundle["products"], bundle["sales"]))
    bundle["metadata"]["dataset_version"] = content_hash(bundle)
    return bundle


def generate(seed=42):
    rng = np.random.default_rng(seed)
    products = [dict(product_id=f"P{i+1:03}", sku=f"RPT-{i+1:04}", name=name,
                     category=category, part_brand=part_brand)
                for i, (name, category, part_brand, _, _, _) in enumerate(PRODUCT_SPECS)]
    vehicles = [dict(vehicle_id=f"V{i+1:03}", brand=brand, vehicle_model=model)
                for i, (brand, model) in enumerate(VEHICLES)]
    compatibility = []
    for i, (p, spec) in enumerate(zip(products, PRODUCT_SPECS)):
        candidates = COMPATIBILITY_GROUPS[spec[5]]
        width = min(len(candidates), 5 + i % 4)
        start = (i * 3) % len(candidates)
        selected = [candidates[(start + offset) % len(candidates)] for offset in range(width)]
        for j in sorted(selected):
            compatibility.append(dict(product_id=p["product_id"], vehicle_id=vehicles[j]["vehicle_id"]))
    sales = []
    for day in pd.date_range("2024-01-01", "2025-12-31"):
        for i, p in enumerate(products):
            # Explicit demonstration outage; not a claim about inventory or demand.
            if i == 8 and day.year == 2025 and day.month in (7, 8, 9):
                continue
            # Slow-moving inventory case: stock remains after the final observed sale.
            if i == 24 and day >= pd.Timestamp("2025-09-01"):
                continue
            season = 1 + .25 * np.sin(2*np.pi*(day.month-1)/12 + i*.25)
            trend = 1 + .08*(day.year-2024)
            rate, base_price = PRODUCT_SPECS[i][3:5]
            count = rng.poisson(rate*season*trend*(.6 if day.dayofweek == 6 else 1))
            for _ in range(count):
                price = int(round(base_price * rng.uniform(.93, 1.07) / 100) * 100)
                cost = int(round(price * rng.uniform(.53, .76) / 100) * 100)
                sales.append(dict(sale_id=len(sales)+1, date=day.date().isoformat(),
                                  product_id=p["product_id"], quantity=int(rng.choice([1, 2, 3], p=[.8, .17, .03])),
                                  unit_price=price, unit_cost=cost))
    inventory = _inventory_data(products, sales)
    bundle = dict(products=products, vehicles=vehicles, compatibility=compatibility, sales=sales, **inventory,
                  metadata=dict(synthetic=True, seed=seed, generator_version="2.0", currency="CLP",
                                start_date="2024-01-01", end_date="2025-12-31", coverage="complete",
                                tax_basis="sin impuestos", compatibility="ficticia",
                                inventory_method="FIFO", holding_rate=.18))
    bundle["metadata"]["dataset_version"] = content_hash(bundle)
    return bundle


def validate(bundle):
    meta = bundle["metadata"]
    start, end = date.fromisoformat(meta["start_date"]), date.fromisoformat(meta["end_date"])
    if start > end or not meta["synthetic"] or meta["coverage"] != "complete":
        raise ValueError("Metadatos de cobertura o simulación inválidos")
    if not bundle["products"]:
        raise ValueError("Se requiere al menos un producto")
    required = ["products", "vehicles", "compatibility", "sales", *INVENTORY_TABLES]
    if any(table not in bundle for table in required):
        raise ValueError("Faltan tablas de inventario")
    for table, field in [("products", "product_id"), ("vehicles", "vehicle_id"), ("sales", "sale_id"),
                         ("suppliers", "supplier_id"), ("inventory_lots", "lot_id"),
                         ("inventory_movements", "movement_id"), ("purchase_orders", "order_id")]:
        ids = [row[field] for row in bundle[table]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Identificador duplicado en {table}")
    skus = [product["sku"] for product in bundle["products"]]
    if len(skus) != len(set(skus)):
        raise ValueError("SKU duplicado en products")
    ids = {p["product_id"] for p in bundle["products"]}
    vehicle_ids = {v["vehicle_id"] for v in bundle["vehicles"]}
    if any(not p["product_id"] or not p["sku"] or not p["name"] or not p["category"] or not p["part_brand"]
           for p in bundle["products"]):
        raise ValueError("Producto incompleto")
    pairs = set()
    for row in bundle["compatibility"]:
        pair = (row["product_id"], row["vehicle_id"])
        if pair in pairs or pair[0] not in ids or pair[1] not in vehicle_ids:
            raise ValueError("Compatibilidad inválida o duplicada")
        pairs.add(pair)
    for row in bundle["sales"]:
        try:
            valid = (row["product_id"] in ids and start <= date.fromisoformat(row["date"]) <= end
                     and type(row["quantity"]) is int and row["quantity"] > 0
                     and all(type(row[k]) is int and row[k] >= 0 for k in ["unit_price", "unit_cost"]))
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            raise ValueError(f"Venta inválida: {row.get('sale_id', 'sin identificador')}")
    if meta["dataset_version"] != content_hash(bundle):
        raise ValueError("Hash del conjunto no coincide")
    supplier_ids = {row["supplier_id"] for row in bundle["suppliers"]}
    lot_ids = {row["lot_id"] for row in bundle["inventory_lots"]}
    sale_ids = {row["sale_id"] for row in bundle["sales"]}
    policy_products = set()
    for row in bundle["inventory_policies"]:
        valid = (row["product_id"] in ids and row["supplier_id"] in supplier_ids
                 and type(row["lead_time_days"]) is int and row["lead_time_days"] > 0
                 and 0 < row["service_level"] < 1
                 and type(row["review_window_days"]) is int and row["review_window_days"] >= 30)
        if not valid or row["product_id"] in policy_products:
            raise ValueError("Política de inventario inválida o duplicada")
        policy_products.add(row["product_id"])
    if policy_products != ids:
        raise ValueError("Cada producto requiere política de inventario")
    lot_products = {}
    for row in bundle["inventory_lots"]:
        try:
            valid = (row["product_id"] in ids and date.fromisoformat(row["received_date"]) <= end
                     and (row["expiry_date"] is None or date.fromisoformat(row["expiry_date"]) >= date.fromisoformat(row["received_date"]))
                     and type(row["initial_quantity"]) is int and row["initial_quantity"] > 0
                     and type(row["unit_cost"]) is int and row["unit_cost"] >= 0)
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            raise ValueError("Lote de inventario inválido")
        lot_products[row["lot_id"]] = row["product_id"]
    allowed_types = {"initial", "receipt", "sale", "adjustment", "waste"}
    allowed_causes = {None, "damage", "expiry", "count_difference", "other"}
    balances = {lot_id: 0 for lot_id in lot_ids}
    linked_sales = {}
    for row in sorted(bundle["inventory_movements"], key=lambda x: (x["date"], x["movement_id"])):
        try:
            valid = (row["product_id"] in ids and row["lot_id"] in lot_ids
                     and lot_products[row["lot_id"]] == row["product_id"]
                     and start <= date.fromisoformat(row["date"]) <= end
                     and row["movement_type"] in allowed_types and type(row["quantity"]) is int
                     and row["quantity"] != 0 and type(row["unit_cost"]) is int and row["unit_cost"] >= 0
                     and row["cause"] in allowed_causes
                     and (row["sale_id"] is None or row["sale_id"] in sale_ids))
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            raise ValueError("Movimiento de inventario inválido")
        if row["movement_type"] in {"initial", "receipt"} and row["quantity"] < 0:
            raise ValueError("Entrada de inventario negativa")
        if row["movement_type"] in {"sale", "waste", "adjustment"} and row["quantity"] > 0:
            raise ValueError("Salida de inventario positiva")
        if row["movement_type"] in {"waste", "adjustment"} and row["cause"] is None:
            raise ValueError("Merma o ajuste sin causa")
        balances[row["lot_id"]] += row["quantity"]
        if balances[row["lot_id"]] < 0:
            raise ValueError("Saldo de lote negativo")
        if row["sale_id"] is not None:
            linked_sales[row["sale_id"]] = linked_sales.get(row["sale_id"], 0) - row["quantity"]
    expected_sales = {row["sale_id"]: row["quantity"] for row in bundle["sales"]}
    if linked_sales != expected_sales:
        raise ValueError("Movimientos de venta no concilian")
    for row in bundle["purchase_orders"]:
        try:
            valid = (row["product_id"] in ids and row["supplier_id"] in supplier_ids
                     and date.fromisoformat(row["order_date"]) <= date.fromisoformat(row["expected_date"])
                     and type(row["quantity"]) is int and row["quantity"] > 0
                     and row["status"] in {"open", "received", "cancelled"})
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            raise ValueError("Orden de compra inválida")


def save(bundle, path=DB_PATH):
    validate(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".sqlite")
    os.close(fd)
    try:
        with closing(sqlite3.connect(tmp)) as conn:
            conn.executescript('''
                PRAGMA foreign_keys=ON;
                CREATE TABLE products(product_id TEXT PRIMARY KEY, sku TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
                                      category TEXT NOT NULL, part_brand TEXT NOT NULL);
                CREATE TABLE vehicles(vehicle_id TEXT PRIMARY KEY, brand TEXT NOT NULL, vehicle_model TEXT NOT NULL);
                CREATE TABLE compatibility(product_id TEXT REFERENCES products, vehicle_id TEXT REFERENCES vehicles,
                                           PRIMARY KEY(product_id,vehicle_id));
                CREATE TABLE sales(sale_id INTEGER PRIMARY KEY, date TEXT NOT NULL, product_id TEXT REFERENCES products,
                                   quantity INTEGER CHECK(quantity>0), unit_price INTEGER CHECK(unit_price>=0),
                                   unit_cost INTEGER CHECK(unit_cost>=0));
                CREATE INDEX sales_product_date ON sales(product_id,date);
                CREATE TABLE suppliers(supplier_id TEXT PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE inventory_policies(product_id TEXT PRIMARY KEY REFERENCES products,
                    supplier_id TEXT NOT NULL REFERENCES suppliers, lead_time_days INTEGER NOT NULL,
                    service_level REAL NOT NULL, review_window_days INTEGER NOT NULL);
                CREATE TABLE inventory_lots(lot_id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products,
                    received_date TEXT NOT NULL, expiry_date TEXT, initial_quantity INTEGER NOT NULL, unit_cost INTEGER NOT NULL);
                CREATE TABLE inventory_movements(movement_id INTEGER PRIMARY KEY, date TEXT NOT NULL,
                    product_id TEXT NOT NULL REFERENCES products, lot_id TEXT NOT NULL REFERENCES inventory_lots,
                    movement_type TEXT NOT NULL, quantity INTEGER NOT NULL, unit_cost INTEGER NOT NULL,
                    cause TEXT, sale_id INTEGER REFERENCES sales);
                CREATE INDEX inventory_product_date ON inventory_movements(product_id,date);
                CREATE TABLE purchase_orders(order_id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products,
                    supplier_id TEXT NOT NULL REFERENCES suppliers, order_date TEXT NOT NULL, expected_date TEXT NOT NULL,
                    quantity INTEGER NOT NULL, status TEXT NOT NULL);
                CREATE TABLE metadata(payload TEXT NOT NULL);
            ''')
            for table in ["products", "vehicles", "compatibility", "sales", *INVENTORY_TABLES]:
                rows = bundle[table]
                if rows:
                    keys = list(rows[0])
                    conn.executemany(f"INSERT INTO {table} ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})",
                                     [[r[k] for k in keys] for r in rows])
            conn.execute("INSERT INTO metadata VALUES (?)", [json.dumps(bundle["metadata"], ensure_ascii=False)])
            conn.commit()
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read(path=DB_PATH):
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        bundle = {t: [dict(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY rowid")]
                  for t in ["products", "vehicles", "compatibility", "sales", *INVENTORY_TABLES]}
        bundle["metadata"] = json.loads(conn.execute("SELECT payload FROM metadata").fetchone()[0])
    validate(bundle)
    return bundle


def backup(source, target):
    target = Path(target)
    if target.exists():
        raise ValueError("El destino ya existe; elija otro respaldo")
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(Path(source).resolve().as_uri()+"?mode=ro", uri=True)) as src:
        with closing(sqlite3.connect(target)) as dst:
            src.backup(dst)
    return read(target)["metadata"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "backup", "restore"])
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.action == "generate":
        bundle = generate(args.seed)
        save(bundle, args.db)
        print(json.dumps(dict(rows=len(bundle["sales"]), metadata=bundle["metadata"]), ensure_ascii=False))
    elif args.action == "backup":
        if not args.file:
            parser.error("Falta --file")
        print(backup(args.db, args.file))
    else:
        if not args.file:
            parser.error("Falta --file")
        save(read(args.file), args.db)
        print("Restaurado. Use el modelo del mismo hash o vuelva a entrenar.")


if __name__ == "__main__":
    main()
