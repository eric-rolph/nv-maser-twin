# Stage 1: builder — install deps into a venv
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e ".[api]"

# Stage 2: runtime — minimal image
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "nv_maser.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
