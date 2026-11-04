# --- Giai đoạn 1: build giao diện React ---
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- Giai đoạn 2: API FastAPI + giao diện đã build ---
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY app ./app
COPY models ./models
COPY reports ./reports
# Tin rao đã làm sạch: nguồn dữ liệu cho /api/recommend và /api/cars/{id}
COPY Data/processed/clean.csv ./Data/processed/clean.csv
COPY --from=web /web/dist ./frontend/dist
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
