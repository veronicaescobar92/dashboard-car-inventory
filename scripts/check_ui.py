"""Recorre las vistas Streamlit contra una API HTTP en ejecución."""
import json
import os
import time
from pathlib import Path
from streamlit.testing.v1 import AppTest

EVIDENCE_DIR = Path(os.environ.get("REPUESTOS_EVIDENCE_DIR", "output/evidence"))

results = []
app = AppTest.from_file(str(Path("ui/app.py").resolve()), default_timeout=30)
start = time.perf_counter()
app.run()
assert not app.exception and len(app.metric) == 4
results.append(dict(view="Resumen", seconds=time.perf_counter()-start, passed=True))
for view in ["Productos y compatibilidad", "Evolución mensual", "Predicción", "Inventario"]:
    start = time.perf_counter()
    app.radio[0].set_value(view).run()
    assert not app.exception and not app.error
    if view == "Predicción":
        app.button[0].click().run()
        assert not app.exception and len(app.dataframe) >= 1
    else:
        assert len(app.dataframe) >= 1
    results.append(dict(view=view, seconds=time.perf_counter()-start, passed=True))
app.radio[0].set_value("Resumen").run()
app.multiselect[0].set_value(["P009"]).run()
assert not app.exception
app.date_input[0].set_value((__import__('datetime').date(2025, 7, 1), __import__('datetime').date(2025, 9, 30))).run()
assert app.metric[0].value == "0"
results.append(dict(view="Filtro sin movimiento", passed=True))
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
(EVIDENCE_DIR / "ui.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
