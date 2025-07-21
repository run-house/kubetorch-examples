FROM apache/airflow:3.0.2
COPY docker/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
COPY dags/ /opt/airflow/dags/
