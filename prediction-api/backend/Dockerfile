# Build from the graduation-design project root:
# docker build -f "02_系统源码/backend/Dockerfile" -t burnout-screening .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/02_系统源码/backend

COPY ["02_系统源码/backend/requirements.txt", "./requirements.txt"]
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY ["02_系统源码/backend", "/app/02_系统源码/backend"]
COPY ["02_系统源码/frontend", "/app/02_系统源码/frontend"]
COPY ["03_数据分析/artifacts", "/app/03_数据分析/artifacts"]

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
