FROM apache/airflow:2.10.2-python3.12

# Switch to root to install additional packages
USER root

# Install any extra dependencies here if needed
# RUN apt-get update && apt-get install -y vim

# Switch back to airflow user
USER airflow

COPY ../requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt


