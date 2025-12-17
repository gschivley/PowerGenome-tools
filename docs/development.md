# Development Guide

This guide covers how to set up the development environment for the PowerGenome tools web application.

## Installation

### Requirements

* Python 3.8+
* `uv` (recommended) or `pip`

### Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/gschivley/PowerGenome-tools.git
    cd PowerGenome-tools
    ```

2. Install dependencies using `uv`:

    ```bash
    uv sync
    ```

    Or install manually with pip:

    ```bash
    pip install networkx scipy scikit-learn pandas pyyaml numpy
    # Optional for map generation
    pip install geopandas matplotlib
    ```

## Web Application Development

The web application is located in the `web/` directory. It uses PyScript to run Python code in the browser.

### Running Locally

To test changes to the web app, you need to serve the `web/` directory via HTTP.

1. Navigate to the `web` directory:

    ```bash
    cd web
    ```

2. Start a simple Python server:

    ```bash
    python -m http.server 8000
    ```

3. Open your browser to `http://localhost:8000`.

### Key Files

* `web/index.html`: The main HTML structure and UI.
* `web/cluster_app.py`: The Python logic that runs in the browser (PyScript).
* `web/pyscript.toml`: PyScript configuration and dependencies.

## Documentation

The documentation is built with [MkDocs](https://www.mkdocs.org/).

### Building Docs

1. Install documentation dependencies:

    ```bash
    uv sync --group dev
    ```

2. Serve the documentation locally:

    ```bash
    uv run mkdocs serve
    ```

3. Build the static site:

    ```bash
    uv run mkdocs build
    ```

### Deployment

The documentation is deployed to GitHub Pages. The build process automatically includes the `web/` application directory in the deployed site, ensuring the web app remains accessible at `/web/`.

To deploy manually:

```bash
uv run mkdocs gh-deploy
```
