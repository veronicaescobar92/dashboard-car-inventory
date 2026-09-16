"""Agregaciones sin duplicar ventas por compatibilidad vehicular."""
import math
from datetime import date
from statistics import NormalDist
import pandas as pd


class DomainError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


class Analytics:
    def __init__(self, bundle):
        self.bundle = bundle
        self.meta = bundle["metadata"]
        self.products = pd.DataFrame(bundle["products"])
        self.sales = pd.DataFrame(bundle["sales"], columns=["sale_id", "date", "product_id", "quantity", "unit_price", "unit_cost"])
        self.sales["date"] = pd.to_datetime(self.sales["date"])
        self.sales["units"] = self.sales.quantity.astype(int)
        self.sales["revenue"] = self.sales.units * self.sales.unit_price
        self.sales["cost"] = self.sales.units * self.sales.unit_cost
        self._inventory_cache = {}

    def selection(self, start_date=None, end_date=None, product_ids=None, brand=None, vehicle_model=None):
        start = str(start_date or self.meta["start_date"])
        end = str(end_date or self.meta["end_date"])
        if date.fromisoformat(start) > date.fromisoformat(end):
            raise DomainError(422, "invalid_period", "La fecha inicial debe ser anterior o igual a la final.")
        products = self.products.copy()
        if product_ids:
            unknown = set(product_ids) - set(products.product_id)
            if unknown:
                raise DomainError(404, "unknown_product", "Productos desconocidos: " + ", ".join(sorted(unknown)))
            products = products[products.product_id.isin(product_ids)]
        if brand or vehicle_model:
            vehicles = [v["vehicle_id"] for v in self.bundle["vehicles"]
                        if (not brand or v["brand"] in brand) and (not vehicle_model or v["vehicle_model"] in vehicle_model)]
            eligible = {c["product_id"] for c in self.bundle["compatibility"] if c["vehicle_id"] in vehicles}
            products = products[products.product_id.isin(eligible)]
        sales = self.sales[self.sales.product_id.isin(products.product_id)]
        current = sales[sales.date.between(start, end)]
        return products, current, sales, start, end

    def inventory_overview(self, as_of=None, window_days=90, slow_days=90,
                           holding_rate=None, expiry_days=90, **filters):
        if "inventory_movements" not in self.bundle or "inventory_policies" not in self.bundle:
            raise DomainError(503, "inventory_not_ready", "Prepare los datos de inventario y reinicie la API.")
        if window_days < 30 or slow_days < 1 or expiry_days < 0:
            raise DomainError(422, "invalid_inventory_parameters", "Los parámetros de inventario no son válidos.")
        as_of = pd.Timestamp(as_of or filters.get("end_date") or self.meta["end_date"]).normalize()
        coverage_end = pd.Timestamp(self.meta["end_date"])
        if as_of > coverage_end or as_of < pd.Timestamp(self.meta["start_date"]):
            raise DomainError(422, "invalid_as_of", "La fecha de inventario debe estar dentro de la cobertura.")
        rate = self.meta.get("holding_rate") if holding_rate is None else holding_rate
        if rate is not None and rate < 0:
            raise DomainError(422, "invalid_holding_rate", "La tasa de almacenamiento no puede ser negativa.")
        cache_key = (as_of.date().isoformat(), window_days, slow_days, rate, expiry_days,
                     tuple(filters.get("product_ids") or ()), tuple(filters.get("brand") or ()),
                     tuple(filters.get("vehicle_model") or ()))
        if cache_key in self._inventory_cache:
            return self._inventory_cache[cache_key]
        selection_filters = {k: v for k, v in filters.items() if k in {"product_ids", "brand", "vehicle_model"}}
        products, _, _, _, _ = self.selection(start_date=self.meta["start_date"],
                                               end_date=as_of.date().isoformat(), **selection_filters)
        eligible = set(products.product_id)
        start = max(pd.Timestamp(self.meta["start_date"]), as_of - pd.Timedelta(days=window_days-1))
        days = pd.date_range(start, as_of, freq="D")
        movements = pd.DataFrame(self.bundle["inventory_movements"])
        movements["date"] = pd.to_datetime(movements["date"])
        movements = movements[(movements.product_id.isin(eligible)) & (movements.date <= as_of)].copy()
        movements["value_delta"] = movements.quantity * movements.unit_cost
        policies = {row["product_id"]: row for row in self.bundle["inventory_policies"]}
        orders = pd.DataFrame(self.bundle["purchase_orders"])
        if not orders.empty:
            orders["expected_date"] = pd.to_datetime(orders["expected_date"])
        rows, daily_cache = [], {}
        for product in products.to_dict("records"):
            product_id = product["product_id"]
            pm = movements[movements.product_id == product_id]
            first = pm.date.min() if not pm.empty else None
            full_days = pd.date_range(first, as_of, freq="D") if first is not None else pd.DatetimeIndex([])
            daily = pd.DataFrame(index=full_days)
            if len(full_days):
                sums = pm.groupby("date")[["quantity", "value_delta"]].sum()
                daily = daily.join(sums).fillna(0).cumsum()
                daily.columns = ["stock", "value"]
                if (daily.stock < 0).any():
                    raise DomainError(503, "invalid_inventory", f"Saldo negativo para {product_id}.")
            window = daily.reindex(days).ffill().fillna(0) if len(days) else daily
            daily_cache[product_id] = window
            product_sales = self.sales[(self.sales.product_id == product_id) & self.sales.date.between(start, as_of)]
            sale_moves = pm[(pm.movement_type == "sale") & pm.date.between(start, as_of)]
            units_sold = int(product_sales.units.sum())
            cost_sold = int((-sale_moves.quantity * sale_moves.unit_cost).sum())
            stock = int(window.stock.iloc[-1]) if len(window) else 0
            stock_value = int(round(window.value.iloc[-1])) if len(window) else 0
            avg_value = float(window.value.mean()) if len(window) else 0
            turnover = cost_sold / avg_value if avg_value > 0 else None
            daily_cost = cost_sold / len(days) if len(days) else 0
            inventory_days = stock_value / daily_cost if daily_cost > 0 else None
            prior_sales = self.sales[(self.sales.product_id == product_id) & (self.sales.date <= as_of)]
            last_sale = prior_sales.date.max() if not prior_sales.empty else None
            days_since_sale = (as_of-last_sale).days if last_sale is not None else None
            lot_moves = pm.groupby("lot_id").quantity.sum() if not pm.empty else pd.Series(dtype=float)
            active_lots = [lot for lot in self.bundle["inventory_lots"]
                           if lot["product_id"] == product_id and lot_moves.get(lot["lot_id"], 0) > 0]
            oldest = min((pd.Timestamp(lot["received_date"]) for lot in active_lots), default=None)
            oldest_age = (as_of-oldest).days if oldest is not None else None
            holding_cost = float(window.value.sum() * rate / 365) if rate is not None else None
            rows.append(dict(**product, stock=stock, stock_value=stock_value, units_sold=units_sold,
                             days_since_sale=days_since_sale, oldest_stock_days=oldest_age,
                             turnover=turnover, inventory_days=inventory_days,
                             holding_cost=holding_cost, cost_sold=cost_sold))
        valid_turnover = sorted(row["turnover"] for row in rows if row["turnover"] is not None)
        fast_cutoff = valid_turnover[max(0, math.ceil(len(valid_turnover)*2/3)-1)] if valid_turnover else None
        for row in rows:
            slow = row["stock"] > 0 and ((row["days_since_sale"] is None or row["days_since_sale"] >= slow_days)
                                          or (row["turnover"] is not None and row["turnover"] <= .5))
            fast = (row["stock"] > 0 and row["days_since_sale"] is not None and row["days_since_sale"] <= 30
                    and fast_cutoff is not None and row["turnover"] is not None and row["turnover"] >= fast_cutoff)
            row["movement_class"] = "slow" if slow else "fast" if fast else "intermediate" if row["stock"] else "out_of_stock"
            row["commercial_action"] = ("Evaluar promoción o reasignación; revisar disponibilidad, precio y demanda."
                                        if slow else None)
        stockouts = []
        for row in rows:
            daily = daily_cache[row["product_id"]]
            zero = daily.stock.eq(0) if not daily.empty else pd.Series(dtype=bool)
            episodes, current_start = [], None
            for day, is_zero in zero.items():
                if is_zero and current_start is None:
                    current_start = day
                if not is_zero and current_start is not None:
                    episodes.append((current_start, day-pd.Timedelta(days=1)))
                    current_start = None
            if current_start is not None:
                episodes.append((current_start, as_of))
            if episodes:
                stockouts.append(dict(product_id=row["product_id"], name=row["name"], episodes=len(episodes),
                                      days=sum((b-a).days+1 for a, b in episodes),
                                      last_start=episodes[-1][0].date().isoformat(),
                                      last_end=episodes[-1][1].date().isoformat(),
                                      note="Riesgo de venta perdida; no existen pedidos rechazados para cuantificarla."))
        reorder = []
        z_cache = {}
        for row in rows:
            policy = policies.get(row["product_id"])
            product_sales = self.sales[(self.sales.product_id == row["product_id"]) & self.sales.date.between(start, as_of)]
            daily_units = product_sales.groupby("date").units.sum().reindex(days, fill_value=0)
            if not policy or len(days) < 30:
                reorder.append(dict(product_id=row["product_id"], available=False,
                                    reason="Falta política válida o historial mínimo de 30 días."))
                continue
            mean, std = float(daily_units.mean()), float(daily_units.std(ddof=0))
            level = float(policy["service_level"])
            z = z_cache.setdefault(level, NormalDist().inv_cdf(level))
            lead = int(policy["lead_time_days"])
            safety = z * std * math.sqrt(lead)
            point = math.ceil(mean*lead+safety)
            open_qty = 0
            if not orders.empty:
                open_qty = int(orders[(orders.product_id == row["product_id"]) & (orders.status == "open")
                                      & (orders.expected_date >= as_of)].quantity.sum())
            suggested = max(0, math.ceil(point-row["stock"]-open_qty))
            reorder.append(dict(product_id=row["product_id"], name=row["name"], available=True,
                                demand_daily=mean, demand_during_lead=mean*lead, lead_time_days=lead,
                                service_level=level, safety_stock=safety, reorder_point=point,
                                stock=row["stock"], open_purchase_units=open_qty,
                                suggested_order=suggested, alert=row["stock"]+open_qty <= point))
        lot_balances = movements.groupby("lot_id").quantity.sum() if not movements.empty else pd.Series(dtype=float)
        expiry = []
        for lot in self.bundle["inventory_lots"]:
            balance = int(lot_balances.get(lot["lot_id"], 0))
            if lot["product_id"] not in eligible or balance <= 0 or lot["expiry_date"] is None:
                continue
            remaining = (pd.Timestamp(lot["expiry_date"])-as_of).days
            if remaining <= expiry_days:
                expiry.append(dict(product_id=lot["product_id"], lot_id=lot["lot_id"], units=balance,
                                   expiry_date=lot["expiry_date"], days_remaining=remaining,
                                   status="expired" if remaining < 0 else "expiring",
                                   capital_exposed=balance*int(lot["unit_cost"])))
        waste_moves = movements[(movements.movement_type.isin(["waste", "adjustment"])) & movements.date.between(start, as_of)]
        wastes = []
        for cause, group in waste_moves.groupby("cause"):
            wastes.append(dict(cause=cause, label="Diferencia pendiente de revisión" if cause == "count_difference" else cause,
                               units=int(-group.quantity.sum()), value=int((-group.quantity*group.unit_cost).sum())))
        immobilized = sum(row["stock_value"] for row in rows if row["movement_class"] == "slow")
        summary = dict(products=len(rows), stock_units=sum(row["stock"] for row in rows),
                       stock_value=sum(row["stock_value"] for row in rows),
                       slow_products=sum(row["movement_class"] == "slow" for row in rows),
                       stockout_products=len(stockouts), immobilized_capital=immobilized,
                       holding_cost=sum(row["holding_cost"] for row in rows) if rate is not None else None,
                       expiring_units=sum(row["units"] for row in expiry),
                       waste_units=sum(row["units"] for row in wastes))
        result = dict(summary=summary, products=rows, stockouts=stockouts, reorder=reorder,
                      expiry=expiry, waste=wastes,
                      assumptions=dict(as_of=as_of.date().isoformat(), start_date=start.date().isoformat(),
                                       window_days=len(days), slow_days=slow_days, expiry_days=expiry_days,
                                       holding_rate=rate, inventory_method="FIFO",
                                       reorder_formula="ceil(demanda_diaria × plazo + stock_seguridad)",
                                       holding_formula="suma(valor_diario × tasa_anual / 365)"))
        self._inventory_cache[cache_key] = result
        return result

    def summary(self, **filters):
        products, sales, _, start, end = self.selection(**filters)
        units, revenue, cost = [int(sales[k].sum()) for k in ["units", "revenue", "cost"]]
        return dict(units=units, revenue=revenue, cost=cost, gross_margin=revenue-cost,
                    margin_pct=100*(revenue-cost)/revenue if revenue else None,
                    product_count=len(products), start_date=start, end_date=end,
                    coverage=self.coverage(pd.Timestamp(start), pd.Timestamp(end)))

    def ranking(self, metric="units", order="desc", **filters):
        if metric not in ("units", "revenue") or order not in ("asc", "desc"):
            raise DomainError(422, "invalid_order", "Métrica u orden inválidos.")
        products, sales, _, start, end = self.selection(**filters)
        grouped = sales.groupby("product_id")[["units", "revenue", "cost"]].sum()
        result = products.merge(grouped, how="left", on="product_id").fillna(0)
        for k in ["units", "revenue", "cost"]:
            result[k] = result[k].astype(int)
        result["gross_margin"] = result.revenue-result.cost
        result = result.sort_values([metric, "product_id"], ascending=[order == "asc", True])
        return result.to_dict("records")

    def coverage(self, start, end):
        a, b = pd.Timestamp(self.meta["start_date"]), pd.Timestamp(self.meta["end_date"])
        return "complete" if start >= a and end <= b else "unavailable" if end < a or start > b else "partial"

    def monthly(self, **filters):
        products, current, all_sales, start, end = self.selection(**filters)
        if not len(products):
            return []
        result = []
        for month in pd.period_range(start, end, freq="M"):
            a, b = month.start_time.normalize(), month.end_time.normalize()
            query_a, query_b = max(a, pd.Timestamp(start)), min(b, pd.Timestamp(end))
            coverage = self.coverage(query_a, query_b)
            if coverage == "complete" and (query_a != a or query_b != b):
                coverage = "partial"
            values = current[current.date.between(query_a, query_b)]
            previous = month - 12
            pa, pb = previous.start_time.normalize(), previous.end_time.normalize()
            prev_available = self.coverage(pa, pb) == "complete" and coverage == "complete"
            prev = all_sales[all_sales.date.between(pa, pb)]
            row = dict(month=str(month), coverage=coverage)
            for k in ["units", "revenue"]:
                total = int(values[k].sum()) if coverage != "unavailable" else None
                old = int(prev[k].sum()) if prev_available else None
                row.update({k: total, f"previous_{k}": old,
                            f"difference_{k}": total-old if old is not None else None,
                            f"change_pct_{k}": 100*(total-old)/old if old else None})
            result.append(row)
        return result

    def signals(self, **filters):
        ranked = self.ranking(**filters)
        _, _, _, start, end = self.selection(**filters)
        signals = []
        if ranked:
            for kind, rows, text in [
                ("alta_salida", ranked[:3], "Revisar disponibilidad antes de planificar reposición."),
                ("baja_salida", ranked[-3:], "Revisar disponibilidad, precio y demanda antes de promocionar; baja venta no prueba baja demanda.")]:
                signals.append(dict(kind=kind, criterion="Tres posiciones del ranking de unidades; empates por código",
                                    product_ids=[r["product_id"] for r in rows], units=[r["units"] for r in rows],
                                    period=f"{start} a {end}", action=text))
            zero = [r["product_id"] for r in ranked if r["units"] == 0]
            if zero:
                signals.append(dict(kind="sin_movimiento", product_ids=zero, period=f"{start} a {end}",
                                    criterion="Cero unidades en el período", action="Investigar disponibilidad y oferta antes de decidir."))
        months = [m for m in self.monthly(**filters) if m["coverage"] == "complete"]
        for kind, fn in [("meses_mayor_venta", max), ("meses_menor_venta", min)]:
            if months:
                value = fn(m["units"] for m in months)
                signals.append(dict(kind=kind, months=[m["month"] for m in months if m["units"] == value],
                                    units=value, period=f"{start} a {end}", criterion="Meses completos por unidades, incluye empates",
                                    action="Evaluar preparación de existencias o campañas; no garantiza resultados futuros."))
        return signals
