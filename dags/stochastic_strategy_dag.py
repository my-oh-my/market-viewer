"""Module containing the Stochastic Strategy DAG."""

from __future__ import annotations

import argparse
import json

import pendulum

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

from src.processor import run


def _run_strategy():
    """
    Fetches configuration from an Airflow Variable and runs the stochastic strategy.

    The DAG expects an Airflow Variable named 'stochastic_strategy_config'
    containing a JSON object with the following structure:

    {
        "symbols": "COMMA,SEPARATED,LIST,OF,SYMBOLS",
        "period": "1d",
        "intervals": "1h",
        "save_html_dir": "/path/to/save/charts",
        "plot_all": false,
        "k_window": 14,
        "d_window": 3
    }
    """
    config_str = Variable.get("stochastic_strategy_config", default_var="{}")
    config = json.loads(config_str)

    # Define default arguments
    default_args = {
        "symbols": "BTC-USD",
        "period": "1d",
        "intervals": "1h",
        "save_html_dir": "/opt/airflow/charts",
        "plot_all": False,
        "k_window": 14,
        "d_window": 3,
    }

    # Merge the provided config with defaults
    final_args = {**default_args, **config}

    # Create a Namespace object for the run function
    args = argparse.Namespace(**final_args)

    if not args.symbols:
        raise ValueError(
            "Symbols must be provided in the 'stochastic_strategy_config' Variable."
        )

    run(args)


with DAG(
    dag_id="stochastic_strategy",
    start_date=pendulum.yesterday("UTC"),
    schedule_interval="0 0 * * *",  # Run daily at midnight
    catchup=False,
    tags=["strategy"],
    doc_md=_run_strategy.__doc__,  # Display documentation in Airflow UI
) as dag:
    run_stochastic_strategy = PythonOperator(
        task_id="run_stochastic_strategy",
        python_callable=_run_strategy,
    )
