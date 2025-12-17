# PowerGenome Web Copilot Instructions

## Scope
Only the web app in `/web` matters; CLI and legacy docs are ignored.

## What the Web App Does
- Runs entirely in-browser via PyScript/Pyodide; UI scaffolding in [../web/index.html](../web/index.html), logic in [../web/cluster_app.py](../web/cluster_app.py).
- Loads BA boundaries from `web/data/US_PCA.geojson` (Leaflet), CSVs from `web/data/*.csv` (hierarchy, transmission, plants, plant→BA map).
- Users select BAs on the map, choose grouping column, optional “no cluster” groups, then run clustering to produce YAML (`model_regions`, `region_aggregations`).
- Optional plant clustering suggests per-(model_region, tech_group) cluster counts under a budget.

## Runtime Flow
- `init_map()` creates Leaflet map, loads GeoJSON, binds feature events (`on_feature_click`, hover, tooltips) and populates `state.ba_layers` / `state.all_bas`.
- `load_data()` fetches CSVs; errors are surfaced via exceptions (halt init) and `set_status()` for user messages.
- `update_group_colors()` assigns outline/fill colors per grouping value; `apply_group_colors_to_map()` refreshes layer styles; tooltips show grouping + state and, after clustering, model region.
- `run_clustering()` orchestrates: respects `no_cluster` selections, chooses mode (fixed target vs auto-optimize), calls clustering, names regions, builds YAML-friendly structures, updates `state.cluster_colors` and `state.ba_to_region`, and redraws transmission lines if enabled.

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

## UI/Interaction Conventions
- BA IDs are lowercase codes; model region names are CamelCase + counter. Grouping column values drive colors and “no cluster” checkboxes.
- Transmission lines are off by default; toggling rebuilds a `FeatureGroup` weighted by `firm_ttc_mw` between clustered regions.
- Box select defaults on; selection states are visually encoded (selected blue, grouped color fills otherwise).

## Performance/Constraints
- Graph ~140 nodes, ~200–300 edges; agglomerative O(n²) is acceptable in-browser. Pyodide is slower—avoid heavy loops where possible.
- Color palettes are finite: `GROUP_OUTLINE_COLORS` cycles; `CLUSTER_COLORS` (20 colors) do not cycle—too many clusters may reuse colors.

## Common Tasks
- Add a grouping column: extend `hierarchy.csv`; ensure dropdown in HTML lists it; `generate_cluster_names()` already checks `GROUPING_HIERARCHY` order.
- Adjust tech grouping/omits: edit `normalize_technology()` tokens and `DEFAULT_TECH_GROUPS` in [../web/cluster_app.py](../web/cluster_app.py).
- Debug styles: call `update_group_colors()` then `apply_group_colors_to_map()`; tooltips come from `update_tooltips()`.
- Serve locally for manual testing: from `/web`, run `python -m http.server 8000` (or any static server) and open `http://localhost:8000`; PyScript fetches files relative to that root.

## Gotchas
- If data fetch returns HTML (wrong path/serve), `load_data()` raises with first 100 chars sample.
- Unknown BAs in hierarchy vs. transmission: edges are skipped; plants mapped to missing BAs are dropped after merge.
- Auto-optimize: `min_regions`/`max_regions` apply after accounting for `no_cluster` exclusions; modularity stored in `info`.

## Local Development
- Edit files in `/web/cluster_app.py` and `/web/index.html`.
- Test changes by serving `/web` via local HTTP server (e.g., `python -m http.server 8000`).
- Use browser console for PyScript errors; use `console.log()` in Python via `js.console.log()`.
- Local environment set up using uv and .venv. All scripts and tests should use this environment.
- Dependencies are listed in pyproject.toml and uv.lock.
