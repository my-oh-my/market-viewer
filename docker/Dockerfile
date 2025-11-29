# Stage 1: Builder
FROM apache/airflow:2.10.2-python3.12 as builder

USER root
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

USER airflow

WORKDIR /app

COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final
FROM apache/airflow:2.10.2-python3.12

USER root
RUN apt-get update && rm -rf /var/lib/apt/lists/*

USER airflow

WORKDIR /opt/airflow

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

COPY ../dags /opt/airflow/dags
COPY ../plugins /opt/airflow/plugins
COPY ../src /opt/airflow/src



