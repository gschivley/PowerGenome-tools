# PowerGenome Design Wizard

Welcome to the documentation for the PowerGenome Design Wizard. This project provides a comprehensive web-based interface for building complete PowerGenome settings files, guiding users through the entire process of defining model regions, configuring resources, and exporting ready-to-use configuration files for energy system modeling (specifically PowerGenome/GenX).

## Overview

The project consists of an interactive **[Web Application](web_app.md)** that walks users through a 6-step wizard to configure all aspects of a PowerGenome model. Built with PyScript to run entirely in-browser, it allows users to visually define model regions, configure new and existing resources, select fuel scenarios, and export complete settings files in YAML format.

### The 6-Step Wizard

1. **Regions** - Select Balancing Authorities and cluster them into model regions
2. **Model Setup** - Define planning years, financial parameters, and model horizon
3. **Existing Plants** - Cluster existing generators within model regions
4. **New Resources** - Select new-build technologies from NREL ATB and define modified resources
5. **Fuels** - Choose fuel price scenarios
6. **Export** - Generate and download complete PowerGenome settings YAML files

## Key Features

* **Interactive Visualization**: Select regions on a map and see clustering results in real-time with multiple algorithms (Spectral, Louvain, Hierarchical).
* **Comprehensive Configuration**: Define all major PowerGenome settings in one place, from regional boundaries to resource portfolios.
* **Flexible Grouping**: Cluster regions based on various hierarchies like NERC regions, Transmission Groups, or States.
* **Plant Aggregation**: Cluster power plants within regions based on technology and efficiency, allocating clusters based on a budget while minimizing variability.
* **NREL ATB Integration**: Select new-build resources directly from NREL's Annual Technology Baseline with support for custom modified resources.
* **Complete Settings Export**: Generate all seven major PowerGenome settings YAML files ready for use.

## Getting Started

* To use the interactive tool immediately, visit the [Web App](https://gschivley.github.io/PowerGenome-tools/web/).
* To run the tools locally, check the [Development](development.md) guide.
