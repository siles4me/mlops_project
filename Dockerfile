# Stage 1: builder
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: final
FROM python:3.10-slim
WORKDIR /app
# Traer dependencias instaladas
COPY --from=builder /install /usr/local
# Copiar código fuente y modelo
COPY . .
# Exponer puerto
EXPOSE 8000
# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1
# Comando por defecto
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
