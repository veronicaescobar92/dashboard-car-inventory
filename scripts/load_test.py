"""Carga HTTP real: 100 solicitudes, cinco clientes, sin entrenamiento."""
import json
import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import httpx
import numpy as np

BASE = os.environ.get("REPUESTOS_API", "http://127.0.0.1:8000")
EVIDENCE_DIR = Path(os.environ.get("REPUESTOS_EVIDENCE_DIR", "output/evidence"))
LOCAL = threading.local()


def client():
    if not hasattr(LOCAL, "client"):
        LOCAL.client = httpx.Client(timeout=10, trust_env=False)
    return LOCAL.client


def request(index):
    started = time.perf_counter()
    try:
        http = client()
        if index % 3 == 0:
            response = http.get(BASE+"/inventory/overview")
        elif index % 2:
            response = http.post(BASE+"/predictions", json={"product_ids": ["P001", "P005"]})
        else:
            response = http.get(BASE+"/analytics/ranking")
        return dict(seconds=time.perf_counter()-started, status=response.status_code)
    except httpx.HTTPError as exc:
        return dict(seconds=time.perf_counter()-started, status=0, error=type(exc).__name__)


for i in range(4):
    assert request(i)["status"] == 200
with ThreadPoolExecutor(max_workers=5) as pool:
    results = list(pool.map(request, range(100)))
times = [r["seconds"] for r in results]
report = dict(requests=100, concurrent_clients=5, warmup=4, machine=platform.platform(), processor=platform.processor(),
              python=platform.python_version(), versions={p: version(p) for p in ["fastapi", "scikit-learn", "httpx"]},
              p50_seconds=float(np.percentile(times, 50)), p95_seconds=float(np.percentile(times, 95)),
              errors=sum(r["status"] != 200 for r in results), observations=results)
report["passed"] = report["errors"] == 0 and report["p95_seconds"] <= 2
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
(EVIDENCE_DIR / "load.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "observations"}, indent=2))
assert report["passed"], "La meta de carga no se cumplió"
