# Market Viewer

This project is a skeleton for a Python-based market analysis application, starting with a stochastic strategy.

## Setup

1.  **Create a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```

## Development

This project uses Black for code formatting and Pylint for linting.

*   **To format the code:**

    ```bash
    black src/ tests/
    ```

*   **To run the linter:**

    ```bash
    pylint src/ tests/
    ```
