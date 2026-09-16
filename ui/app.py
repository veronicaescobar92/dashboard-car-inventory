import os
from datetime import date

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("REPUESTOS_API", "http://127.0.0.1:8000")
st.set_page_config(page_title="Repuestos | Inteligencia comercial", page_icon="📊", layout="wide")


def api(path, body=None, params=None):
    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            result = client.post(API+path, json=body) if body is not None else client.get(API+path, params=params)
        if result.status_code != 200:
            st.error(result.json().get("error", {}).get("message", "No se pudo completar la consulta."))
            st.button("Reintentar", key="retry_"+path)
            st.stop()
        return result.json()
    except (httpx.HTTPError, ValueError):
        st.error("No se puede conectar con el servicio. Comprueba que esté iniciado y vuelve a intentar.")
        st.button("Reintentar", key="retry_"+path)
        st.stop()


def money(value):
    return "$ " + f"{value:,.0f}".replace(",", ".")


def show_signals(signals):
    for s in signals:
        title = s["kind"].replace("_", " ").capitalize()
        with st.expander(title):
            st.write(s["action"])
            st.caption(s["criterion"] + " · " + s["period"])
            st.write(", ".join(s.get("product_ids", s.get("months", []))))


st.title("Repuestos · Inteligencia comercial")
st.caption("Explora tus ventas, reconoce patrones y prepara tus próximas decisiones.")
st.info("DEMOSTRACIÓN · Datos y compatibilidades ficticios. No representan transacciones de una empresa real.")
catalog = api("/catalog")
meta, data = catalog["metadata"], catalog["data"]
products = {p["product_id"]: p["name"] for p in data["products"]}
with st.sidebar:
    st.header("Tu análisis")
    page = st.radio("Vista", ["Resumen", "Productos y compatibilidad", "Evolución mensual", "Predicción", "Inventario"])
    st.caption("Historial simulado · 2024–2025 · CLP sin impuestos")
    dates = st.date_input("Período histórico", (date(2024, 1, 1), date(2025, 12, 31)),
                          min_value=date(2024, 1, 1), max_value=date(2025, 12, 31))
    ids = st.multiselect("Repuestos", list(products), format_func=lambda p: products[p], placeholder="Todos los repuestos")
    brands = st.multiselect("Marca compatible", sorted({v["brand"] for v in data["vehicles"]}), placeholder="Todas las marcas")
    models = st.multiselect("Modelo compatible", sorted({v["vehicle_model"] for v in data["vehicles"]
                                                        if not brands or v["brand"] in brands}), placeholder="Todos los modelos")
    st.caption("La compatibilidad no identifica el vehículo del comprador. Los grupos por modelo pueden superponerse.")
    st.caption("Versión de datos: " + meta["dataset_version"][:12])

if len(dates) != 2:
    st.warning("Selecciona la fecha inicial y final.")
    st.stop()
params = dict(start_date=dates[0].isoformat(), end_date=dates[1].isoformat())
for key, values in [("product_ids", ids), ("brand", brands), ("vehicle_model", models)]:
    if values:
        params[key] = values

if page == "Inventario":
    st.header("Inventario y reposición")
    st.write("Supervisa existencias, mercancía lenta, quiebres, capital expuesto y alertas de reposición.")
    settings = st.columns(3)
    window_days = settings[0].number_input("Ventana de análisis (días)", min_value=30, max_value=730, value=90)
    slow_days = settings[1].number_input("Sin venta para considerar lento (días)", min_value=1, max_value=730, value=90)
    expiry_days = settings[2].number_input("Alerta de vencimiento (días)", min_value=0, max_value=730, value=90)
    inventory_params = dict(params, as_of=dates[1].isoformat(), window_days=window_days,
                            slow_days=slow_days, expiry_days=expiry_days)
    result = api("/inventory/overview", params=inventory_params)["data"]
    summary = result["summary"]
    cards = st.columns(4)
    cards[0].metric("Unidades en stock", f'{summary["stock_units"]:,}'.replace(",", "."))
    cards[1].metric("Capital en inventario", money(summary["stock_value"]))
    cards[2].metric("Capital inmovilizado", money(summary["immobilized_capital"]))
    cards[3].metric("Costo de almacenamiento", money(summary["holding_cost"]) if summary["holding_cost"] is not None else "No disponible")
    st.caption("Valores estimados al costo. No son asientos contables ni representan flujo de caja.")
    alerts = st.columns(4)
    alerts[0].metric("Productos lentos", summary["slow_products"])
    alerts[1].metric("Con quiebres", summary["stockout_products"])
    alerts[2].metric("Unidades por vencer", summary["expiring_units"])
    alerts[3].metric("Unidades de merma", summary["waste_units"])
    product_frame = pd.DataFrame(result["products"])
    if product_frame.empty:
        st.info("No hay productos coincidentes con estos filtros.")
    else:
        st.subheader("Velocidad y capital por producto")
        labels = {"fast": "Rápido", "intermediate": "Intermedio", "slow": "Lento", "out_of_stock": "Sin stock"}
        product_frame["movement_class"] = product_frame.movement_class.map(labels)
        st.dataframe(product_frame.rename(columns={"product_id": "Código", "name": "Repuesto",
            "stock": "Stock", "stock_value": "Capital CLP", "units_sold": "Unidades vendidas",
            "days_since_sale": "Días desde última venta", "oldest_stock_days": "Antigüedad máxima",
            "turnover": "Rotación", "inventory_days": "Días de inventario",
            "holding_cost": "Costo almacenamiento CLP", "movement_class": "Movimiento",
            "commercial_action": "Acción a evaluar"}), hide_index=True)
        slow = product_frame[product_frame.movement_class.eq("Lento")]
        if not slow.empty:
            st.warning("Hay mercancía lenta. Evalúa promoción o reasignación después de revisar disponibilidad, precio y demanda; no se aplican descuentos automáticamente.")
    st.subheader("Reposición sugerida")
    reorder = pd.DataFrame([row for row in result["reorder"] if row.get("available") and row.get("alert")])
    if reorder.empty:
        st.info("No hay alertas de reposición para la selección.")
    else:
        st.dataframe(reorder.rename(columns={"product_id": "Código", "name": "Repuesto",
            "demand_daily": "Demanda diaria", "lead_time_days": "Plazo días", "safety_stock": "Stock seguridad",
            "reorder_point": "Punto de reorden", "stock": "Stock", "open_purchase_units": "En camino",
            "suggested_order": "Cantidad sugerida"}), hide_index=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Quiebres de stock")
        if result["stockouts"]:
            st.dataframe(pd.DataFrame(result["stockouts"]).rename(columns={"product_id": "Código", "name": "Repuesto",
                "episodes": "Episodios", "days": "Días", "last_start": "Último inicio", "last_end": "Último fin",
                "note": "Interpretación"}), hide_index=True)
        else:
            st.info("No se observan quiebres en la ventana.")
    with right:
        st.subheader("Vencimientos y mermas")
        if result["expiry"]:
            st.dataframe(pd.DataFrame(result["expiry"]).rename(columns={"product_id": "Código", "lot_id": "Lote",
                "units": "Unidades", "expiry_date": "Vencimiento", "days_remaining": "Días restantes",
                "status": "Estado", "capital_exposed": "Capital expuesto CLP"}), hide_index=True)
        if result["waste"]:
            st.dataframe(pd.DataFrame(result["waste"]).rename(columns={"cause": "Causa registrada", "label": "Descripción",
                "units": "Unidades", "value": "Valor CLP"}), hide_index=True)
    with st.expander("Cómo se calculan estos indicadores"):
        st.json(result["assumptions"])
        st.write("El punto de reorden es una recomendación calculada con demanda histórica, plazo y stock de seguridad; no es una cantidad exacta garantizada.")
elif page == "Predicción":
    st.header("Prepara el próximo mes")
    st.write("Estimación de unidades vendidas, no una orden de compra. Revisa stock y plazos antes de decidir.")
    metrics = api("/model/metrics")
    report, model_meta = metrics["data"], metrics["metadata"]
    st.info("Mes disponible: " + model_meta["target_month"] + " · Siguiente mes del historial simulado; independiente del filtro histórico.")
    chosen = st.multiselect("Productos a estimar", list(products), default=list(products), format_func=lambda p: products[p])
    if report["warning"]:
        st.warning("El modelo no mejora el promedio reciente en la prueba. Considera también esa referencia.")
    cols = st.columns(2)
    cols[0].metric("Error medio del modelo (MAE)", f'{report["global_scores"]["model"]["mae"]:.2f} unidades')
    cols[1].metric("Error medio del promedio reciente", f'{report["global_scores"]["baseline"]["mae"]:.2f} unidades')
    st.caption(f'Menor error es mejor. Evaluación sobre {report["sample_size"]} casos simulados; no prueba beneficios reales.')
    if st.button("Estimar ventas", type="primary", disabled=not chosen):
        prediction = api("/predictions", body=dict(product_ids=chosen))
        frame = pd.DataFrame(prediction["data"])
        frame["Repuesto"] = frame.product_id.map(products)
        st.dataframe(frame.rename(columns={"product_id": "Código", "target_month": "Mes", "estimated_units": "Unidades estimadas"}), hide_index=True)
        st.bar_chart(frame.set_index("Repuesto")["estimated_units"], color="#087f8c")
        st.caption("Modelo: " + model_meta["model_version"] + " · Entrenado: " + model_meta["trained_at"])
    with st.expander("Cómo se evaluó el modelo"):
        st.write("Se compararon meses posteriores con pronósticos hechos usando solo el pasado. La referencia es el promedio de los tres meses anteriores.")
        st.json({"métricas": report["global_scores"], "casos": report["sample_size"], "límite": report["limitation"]})
elif page == "Productos y compatibilidad":
    st.header("Qué se vende y qué necesita revisión")
    col1, col2 = st.columns(2)
    metric = col1.selectbox("Ordenar por", ["Unidades", "Ingresos"])
    order = col2.selectbox("Orden", ["Mayor a menor", "Menor a mayor"])
    result = api("/analytics/ranking", params=dict(params, metric="units" if metric == "Unidades" else "revenue",
                                                   order="desc" if order == "Mayor a menor" else "asc"))["data"]
    if not result["rows"]:
        st.info("No hay productos coincidentes con estos filtros.")
    else:
        frame = pd.DataFrame(result["rows"])
        st.dataframe(frame.rename(columns={"product_id": "Código", "sku": "SKU", "name": "Repuesto", "part_brand": "Marca repuesto", "category": "Categoría", "units": "Unidades",
                                         "revenue": "Ingresos CLP", "cost": "Costo vendido CLP", "gross_margin": "Margen bruto CLP"}), hide_index=True)
        st.bar_chart(frame.set_index("name")["units" if metric == "Unidades" else "revenue"], color="#087f8c")
        st.subheader("Señales para investigar")
        show_signals(result["signals"][:3])
elif page == "Evolución mensual":
    st.header("Reconoce los meses de mayor y menor venta")
    result = api("/analytics/monthly", params=params)["data"]
    if not result["rows"]:
        st.info("No hay productos coincidentes con estos filtros.")
    else:
        frame = pd.DataFrame(result["rows"])
        st.line_chart(frame.set_index("month")[["units", "previous_units"]].rename(columns={"units": "Unidades", "previous_units": "Mismo mes del año anterior"}))
        st.dataframe(frame.rename(columns={"month": "Mes", "coverage": "Cobertura", "units": "Unidades", "revenue": "Ingresos CLP",
            "previous_units": "Unidades año anterior", "previous_revenue": "Ingresos año anterior", "difference_units": "Diferencia unidades",
            "difference_revenue": "Diferencia ingresos", "change_pct_units": "Variación unidades %", "change_pct_revenue": "Variación ingresos %"})
            .replace({"complete": "Completo", "partial": "Parcial", "unavailable": "No disponible"}), hide_index=True)
        st.caption("Un porcentaje vacío no aplica: falta comparación completa o el valor anterior es cero. Meses parciales no participan en máximos/mínimos.")
        show_signals([s for s in result["signals"] if s["kind"].startswith("meses_")])
else:
    st.header("Una mirada al período")
    summary = api("/analytics/summary", params=params)["data"]
    if not summary["product_count"]:
        st.info("No hay productos coincidentes con estos filtros.")
    cols = st.columns(4)
    cols[0].metric("Unidades vendidas", f'{summary["units"]:,}'.replace(",", "."))
    cols[1].metric("Ingresos", money(summary["revenue"]))
    cols[2].metric("Margen bruto estimado", money(summary["gross_margin"]))
    cols[3].metric("Margen sobre ingresos", f'{summary["margin_pct"]:.1f} %' if summary["margin_pct"] is not None else "No aplica")
    st.caption("CLP sin impuestos. El margen descuenta el costo del repuesto, no gastos operativos. No equivale a utilidad neta ni flujo de caja.")
    result = api("/analytics/ranking", params=params)["data"]
    if result["rows"]:
        st.subheader("Los repuestos con mayor salida")
        st.dataframe(pd.DataFrame(result["rows"][:3])[["name", "units", "revenue"]].rename(
            columns={"name": "Repuesto", "units": "Unidades", "revenue": "Ingresos CLP"}), hide_index=True)
        show_signals(result["signals"])
