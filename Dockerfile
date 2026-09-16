FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.lock.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir -r requirements.lock.txt

COPY src ./src
COPY tests ./tests
COPY ui ./ui
COPY .streamlit ./.streamlit

RUN python -m pip install --no-deps --no-build-isolation .
RUN python -m repuestos.data generate \
    && python -m repuestos.ml

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "repuestos.api:app", "--host", "0.0.0.0", "--port", "8000"]
