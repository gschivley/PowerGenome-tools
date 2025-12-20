# PowerGenome Web Copilot Instructions

## Scope
Only the web app in `/web` matters; CLI and legacy docs are ignored.

## What the Web App Does
- Runs entirely in-browser via PyScript/Pyodide; UI scaffolding in [../web/index.html](../web/index.html), logic in [../web/cluster_app.py](../web/cluster_app.py).
- Loads BA boundaries from `web/data/US_PCA.geojson` (Leaflet), CSVs from `web/data/*.csv` (hierarchy, transmission, plants, plant→BA map), and JSON/CSV for ATB and fuel prices.
- Guides users through a multi-step wizard (currently 6 steps, with more expected in the future) to configure a PowerGenome project:
  1. **Regions**: Select BAs and cluster them into model regions.
  2. **Model Setup**: Define planning years, financial parameters, and model horizon.
  3. **Existing Plants**: Cluster existing generators within model regions.
  4. **New Resources**: Select new-build technologies from NREL ATB and define modified resources.
  5. **Fuels**: Choose fuel price scenarios.
  6. **Export**: Generate and download a complete set of PowerGenome settings YAML files.

## Runtime Flow
- `init_map()` creates Leaflet map, loads GeoJSON, binds feature events (`on_feature_click`, hover, tooltips) and populates `state.ba_layers` / `state.all_bas`.
- `load_data()` fetches CSVs; errors are surfaced via exceptions (halt init) and `set_status()` for user messages.
- `update_group_colors()` assigns outline/fill colors per grouping value; `apply_group_colors_to_map()` refreshes layer styles; tooltips show grouping + state and, after clustering, model region.
- `run_clustering()` orchestrates: respects `no_cluster` selections, chooses mode (fixed target vs auto-optimize), calls clustering, names regions, builds YAML-friendly structures, updates `state.cluster_colors` and `state.ba_to_region`, and redraws transmission lines if enabled.
- Wizard navigation (`goToStep()`) manages visibility of step panes.
- Settings generation (`on_generate_settings()`) aggregates state from all steps to produce the final YAML dictionary (`state.settings_yamls`).

## Clustering Logic (web-only)
- Graph builder: `build_transmission_graph(df, valid_bas)` sums parallel edges into `weight`; adds isolated nodes so every selected BA exists in the graph.
- Algorithms:
  - `agglomerative_cluster`: greedy merge by max edge weight until target count.
  - `louvain_cluster`: NetworkX Louvain for auto-optimize mode; merges further with agglomerative if above `max_regions`.
  - `hierarchical_cluster`: never splits a grouping value; if targets exceed groups, clusters within groups; if fewer, merges whole groups by inter-group transmission.
- Region naming: `generate_cluster_names()` walks grouping hierarchy `st → cendiv → transgrp → nercr → transreg → interconnect`, falling back to state for single BA; adds counters to avoid duplicates.

## Plant Clustering
- `prepare_plants_dataframe()` merges plant data with BA→model mapping, normalizes tech strings via `normalize_technology()` (omits tokens like "solar thermal"), and applies optional grouping (`DEFAULT_TECH_GROUPS`).
- `suggest_plant_clusters(budget=200, cap_threshold=1500, hr_iqr_threshold=0.8)` caps total clusters; certain techs in `ALWAYS_ONE_TECHS` are forced to k=1; simple k-means implementation (`run_kmeans_simple`) avoids scikit-learn.

## Settings Generation
- **New Resources**: Loads ATB data from `web/data/atb_options.json`. Users can add standard ATB resources or define "modified" resources (e.g., Hydrogen CT) with custom attributes and fuel types.
- **Fuels**: Loads scenarios from `web/data/fuel_prices.csv` (or fallback URL). Users map fuels to specific price scenarios.
- **Output**: `generate_settings_yamls()` produces:
  - `model_definition.yml`: Regions, years, financial settings.
  - `resources.yml`: Existing plant clusters, new resources, modified resources.
  - `fuels.yml`: Fuel prices and emission factors.
  - `transmission.yml`, `distributed_gen.yml`, `resource_tags.yml`, `startup_costs.yml`: Generated with defaults or derived values.

## UI/Interaction Conventions
- BA IDs are lowercase codes; model region names are CamelCase + counter. Grouping column values drive colors and “no cluster” checkboxes.
- Transmission lines are off by default; toggling rebuilds a `FeatureGroup` weighted by `firm_ttc_mw` between clustered regions.
- Box select defaults on; selection states are visually encoded (selected blue, grouped color fills otherwise).

## Performance/Constraints
- Graph ~140 nodes, ~200–300 edges; agglomerative O(n²) is acceptable in-browser. Pyodide is slower—avoid heavy loops where possible.
- Color palettes are finite: `GROUP_OUTLINE_COLORS` cycles; `CLUSTER_COLORS` (20 colors) do not cycle—too many clusters may reuse colors.

## Common Tasks
- Add a grouping column: extend `hierarchy.csv`; ensure dropdown in HTML lists it; `generate_cluster_names()` already checks `GROUPING_HIERARCHY` order.
- Add a wizard step: Add HTML in `index.html` (nav item + step pane), update `goToStep()`, and add logic in `cluster_app.py`.
- Adjust tech grouping/omits: edit `normalize_technology()` tokens and `DEFAULT_TECH_GROUPS` in [../web/cluster_app.py](../web/cluster_app.py).
- Update ATB data: regenerate `web/data/atb_options.json` using `web/build_atb_options.py` (requires `technology_costs_atb.parquet`).
- Debug styles: call `update_group_colors()` then `apply_group_colors_to_map()`; tooltips come from `update_tooltips()`.
- Serve locally for manual testing: from `/web`, run `python -m http.server 8000` (or any static server) and open `http://localhost:8000`; PyScript fetches files relative to that root.

## Gotchas
- If data fetch returns HTML (wrong path/serve), `load_data()` raises with first 100 chars sample.
- Unknown BAs in hierarchy vs. transmission: edges are skipped; plants mapped to missing BAs are dropped after merge.
- Auto-optimize: `min_regions`/`max_regions` apply after accounting for `no_cluster` exclusions; modularity stored in `info`.

## Local Development
- Edit files in `/web/cluster_app.py` and `/web/index.html`.
- Test changes by serving `/web` via local HTTP server (e.g., `python -m http.server 8000`).
- ALWAYS use the `test-writer` subagent (via the `runSubagent` tool) to write tests for any new or changed functionality.
- Use browser console for PyScript errors; use `console.log()` in Python via `js.console.log()`.
- Local environment set up using uv and .venv. All scripts and tests should use this environment.
- Dependencies are listed in pyproject.toml and uv.lock.

## PowerGenome Context
- This web app is designed to help users build their settings for PowerGenome.
- PowerGenome is an open-source ETL tool that generates inputs for capacity expansion and production cost models.
- The settings files for PowerGenome define the model regions that will be used in an analysis, how power plants are grouped within those regions, and other key parameters such as resource and fuel price scenarios.
- The documentation for PowerGenome can be found at https://powergenome.github.io/PowerGenome/beta/.

## Documentation
- When updating or adding new features to the web app, please update the documentation in the `docs/` folder accordingly.
- Follow the established documentation style and structure.
- ALWAYS use the `docs-writer` subagent (via the `runSubagent` tool) to write or update documentation.
