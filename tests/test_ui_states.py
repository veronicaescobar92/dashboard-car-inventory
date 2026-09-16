from pathlib import Path
from unittest.mock import patch

import httpx
from streamlit.testing.v1 import AppTest

from repuestos.data import generate


def app():
    return AppTest.from_file(str(Path('ui/app.py').resolve()), default_timeout=30)


def test_disconnected_ui_has_retry_and_no_old_results():
    with patch('httpx.Client.get', side_effect=httpx.ConnectError('offline')):
        result = app().run()
    assert not result.exception
    assert result.error and 'conectar' in result.error[0].value
    assert result.button[0].label == 'Reintentar'
    assert len(result.dataframe) == 0 and len(result.metric) == 0


def test_timeout_ui_has_retry():
    with patch('httpx.Client.get', side_effect=httpx.ReadTimeout('timeout')):
        result = app().run()
    assert not result.exception and result.error
    assert result.button[0].label == 'Reintentar'


def test_baseline_warning_is_visible():
    b = generate()
    def get(_self, url, **kwargs):
        if url.endswith('/catalog'):
            payload = dict(metadata=b['metadata'], data={k:b[k] for k in ['products','vehicles','compatibility']})
        elif url.endswith('/model/metrics'):
            payload = dict(metadata=dict(b['metadata'], target_month='2026-01'), data=dict(warning=True,
                           global_scores=dict(model=dict(mae=20),baseline=dict(mae=10)),sample_size=30,limitation='simulación'))
        elif url.endswith('/analytics/summary'):
            payload = dict(metadata=b['metadata'],data=dict(product_count=0,units=0,revenue=0,gross_margin=0,margin_pct=None))
        else:
            payload = dict(metadata=b['metadata'],data=dict(rows=[],signals=[]))
        return httpx.Response(200,json=payload)
    with patch('httpx.Client.get', get):
        result = app().run()
        result.radio[0].set_value('Predicción').run()
    assert not result.exception
    assert any('no mejora' in w.value for w in result.warning)


def test_inventory_view_exposes_actionable_indicators():
    b = generate()
    def get(_self, url, **kwargs):
        if url.endswith('/catalog'):
            payload = dict(metadata=b['metadata'], data={k:b[k] for k in ['products','vehicles','compatibility']})
        elif url.endswith('/inventory/overview'):
            payload = dict(metadata=b['metadata'], data=dict(
                summary=dict(stock_units=20, stock_value=200000, immobilized_capital=100000,
                             holding_cost=1500, slow_products=1, stockout_products=1,
                             expiring_units=2, waste_units=1),
                products=[dict(product_id='P001', name='Filtro', stock=20, stock_value=200000,
                               units_sold=0, days_since_sale=120, oldest_stock_days=180,
                               turnover=.1, inventory_days=None, holding_cost=1500,
                               movement_class='slow', commercial_action='Evaluar promoción o reasignación')],
                reorder=[dict(product_id='P001', name='Filtro', available=True, alert=True,
                              demand_daily=2, lead_time_days=5, safety_stock=4, reorder_point=14,
                              stock=8, open_purchase_units=0, suggested_order=6)],
                stockouts=[dict(product_id='P001', name='Filtro', episodes=1, days=2,
                                last_start='2025-12-01', last_end='2025-12-02',
                                note='Riesgo de venta perdida; no cuantificada.')],
                expiry=[dict(product_id='P001', lot_id='L1', units=2, expiry_date='2026-01-10',
                             days_remaining=10, status='expiring', capital_exposed=20000)],
                waste=[dict(cause='count_difference', label='Diferencia pendiente de revisión', units=1, value=10000)],
                assumptions=dict(reorder_formula='demanda × plazo + seguridad')))
        else:
            payload = dict(metadata=b['metadata'], data=dict(rows=[], signals=[]))
        return httpx.Response(200, json=payload)
    with patch('httpx.Client.get', get):
        result = app().run()
        result.radio[0].set_value('Inventario').run()
    assert not result.exception
    assert any(metric.label == 'Capital inmovilizado' for metric in result.metric)
    assert any('no se aplican descuentos' in warning.value for warning in result.warning)
    assert 'robo' not in ' '.join(frame.value.to_string() for frame in result.dataframe).lower()
