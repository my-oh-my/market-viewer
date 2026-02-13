# Stage 1: Builder
FROM apache/airflow:2.10.2-python3.12 as builder

USER root
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

USER airflow
WORKDIR /app

# The context is the root (..), so requirements.txt is at the root
COPY requirements.txt .
# Removed --user because it conflicts with the image's internal virtualenv
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final
FROM apache/airflow:2.10.2-python3.12

USER airflow
WORKDIR /opt/airflow

# Copy the entire .local directory where dependencies were installed
COPY --from=builder /home/airflow/.local /home/airflow/.local

# Copy the application code
COPY dags /opt/airflow/dags
COPY plugins /opt/airflow/plugins
COPY src /opt/airflow/src

# Ensure the .local/bin is in PATH
ENV PATH="/home/airflow/.local/bin:${PATH}"
