# PowerGenome Clustering Tools

Welcome to the documentation for the PowerGenome Clustering Tools. This project provides tools for clustering Balancing Authorities (BAs) into model regions and aggregating power plants for energy system modeling (specifically PowerGenome/GenX).

## Overview

The project consists of an interactive **[Web Application](web_app.md)** for visualizing and clustering regions, and then generating existing powerplant clusters. Built with PyScript to run entirely in-browser, it allows users to select BAs on a map, configure clustering parameters, and export results in YAML format.

## Key Features

* **Interactive Visualization**: Select regions on a map and see clustering results in real-time.
* **Multiple Algorithms**: Choose from Spectral, Louvain, and Hierarchical clustering methods.
* **Flexible Grouping**: Cluster based on various hierarchies like NERC regions, Transmission Groups, or States.
* **Plant Aggregation**: Cluster power plants within regions based on technology and efficiency. Allocate clusters based on a budget, minimizing variability within clusters and splitting large, diverse fleets.

## Getting Started

* To use the interactive tool immediately, visit the [Web App](https://gschivley.github.io/PowerGenome-tools/web/).
* To run the tools locally, check the [Development](development.md) guide.
