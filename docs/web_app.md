# Web Application

The PowerGenome Web Clustering Tool allows users to interactively select regions, configure clustering parameters, and visualize results on a map. It runs entirely in the browser using PyScript.

[Launch Web App](https://gschivley.github.io/PowerGenome-tools/web/){ .md-button .md-button--primary }

## Features

* **Interactive Map**: View Balancing Authorities (BAs) and transmission lines.
* **Region Clustering**: Aggregate BAs into model regions based on transmission capacity.
* **Plant Clustering**: Cluster power plants within regions based on technology and efficiency.
* **YAML Export**: Generate configuration files compatible with PowerGenome.

## How to Use

### Region Clustering

1. **Select BAs**: Click on regions in the map to select them. Use **Box Select** mode to drag and select multiple BAs at once.
2. **Choose grouping**: The *Grouping Column* determines how BAs are grouped for clustering (colored outlines show groups).
3. **Set target regions**: Enter how many final regions you want after clustering. Optionally enable *Auto-optimize* to find the best number of regions within a range.
4. **Exclude groups**: Check any groups in *Groups to Keep Unclustered* to keep them as individual BAs.
5. **Run clustering**: Click *Run Clustering* to generate the region aggregations.
6. **Export**: Copy or download the YAML output to use in PowerGenome.

!!! tip
    The clustering uses transmission capacity between BAs to create aggregated regions. BAs are first clustered within their selected grouping (e.g., NERC region), then groups are merged to reach the target number of regions. The selected grouping affects the clustering results, so experiment with different options to see what works best for your case!

## Clustering Configuration Guide

Selecting the right grouping column and clustering algorithm is essential for producing model regions that are both computationally tractable and physically meaningful. This section explains how different choices affect your results.

### Choosing a Grouping Column

The **Grouping Column** determines how BAs are initially partitioned before clustering. This choice significantly influences region balance and grid operational fidelity.

| Option | Groups | Details | Strengths | Weaknesses | Best For |
|--------|--------|---------|-----------|-----------|----------|
| **Transmission Group** | 18 | ~7 BAs per group (median 6) | More balanced regions; aligns with ISOs/RTOs; better convergence | None significant | Most general analyses; grid-operational alignment |
| **Transmission Region** | 11 | ~12 BAs per group (median 11) | Still grid-aligned; moderate flexibility | Large NorthernGrid group in WECC can lead to unbalanced regions in national studies | Broader transmission boundaries |
| **Interconnect** | 3 | ~45 BAs per group (median 35) | None noted | Highly imbalanced regions; misses operational detail | Rarely used |
| **Census Division** | 9 | ~15 BAs per group (median 17) | Regional policy rollups | Doesn't reflect grid ops; splits transmission clusters | High-level regional summaries |
| **State** | 48 | ~3 BAs per group (median 2) | Aligns to state policy boundaries | Doesn't reflect grid ops; splits transmission clusters | State-level policy analysis |

**Recommendation**: Use **Transmission Group** (default) or **Transmission Region** for grid-focused analyses. These options respect ISO/RTO boundaries that reflect how the grid is actually operated and studied.

### Choosing a Clustering Algorithm and Target

Once you've selected a grouping column, you need to decide how many regions you want and which algorithm to use. The interaction between these choices matters.

#### Auto-Optimize vs. Fixed Target

**Auto-Optimize Mode**:

* Finds the number of regions that **maximizes modularity** (a measure of how well the network divides into clusters).
* Set a range (e.g., Min: 20, Max: 40), and the tool searches within that range.
* **Advantage**: Respects the natural structure of the transmission network rather than imposing an arbitrary target.
* **Disadvantage**: The optimal modularity may not align with your computational budget or modeling goals.

**Best for**: Exploratory analysis, understanding network structure, or when you're unsure how many regions you need.

**Fixed Target Mode**:

* You specify exactly how many model regions you want.
* The clustering algorithm works to achieve that target.
* **Advantage**: Direct control; use this when your computational constraints or project requirements specify a fixed region count.
* **Disadvantage**: You may override the network's natural divisions, potentially creating inefficient aggregations or unbalanced regions.

**Best for**: Production models where region count is a hard constraint, or when you're matching a predetermined regional structure.

#### Algorithm Choices

| Algorithm | How It Works | Strengths | Weaknesses | Best For |
|-----------|--------------|-----------|-----------|----------|
| **Spectral Clustering** (Default) | Uses eigenvalues of transmission graph for dimensionality reduction | Balanced regions; finds cuts minimizing flow disruption; works well with grouping constraints | — | Default choice for most analyses |
| **Louvain Community Detection** | Iteratively merges to maximize modularity | Effective in auto-optimize; finds natural network structure; identifies cohesive communities | Less coherent in fixed-target mode | Auto-optimize mode; exploratory analysis |
| **Hierarchical (Average Linkage)** | Greedy merging by edge weight, penalizing large cluster merges | Produces balanced region sizes; predictable; deterministic | — | Fixed-target mode; balanced regions needed |
| **Hierarchical (Sum Linkage)** | Merges by total transmission capacity | Captures strong corridors | Creates imbalanced "snowballing" (few very large + many small clusters) | Rarely; corridor-focused analysis only |
| **Hierarchical (Max Linkage)** | Merges by single strongest edge | — | Produces imbalanced clusters; many isolated BAs | Generally not recommended |

### Recommended Approaches

**For a new analysis**:

1. Start with **Transmission Group** grouping.
2. Enable **Auto-Optimize** with a reasonable range (e.g., 15–40 regions).
3. Try switching to a fixed number of regions. Select **Spectral** or **Louvain** algorithm.
4. Review the resulting regions and modularity score. Experiment with different ranges to see how modularity changes.

**For balanced, fixed-region models**:

1. Use **Transmission Group** grouping.
2. Set a **Fixed Target** (e.g., 20 regions).
3. Select **Spectral**, **Louvain**, or **Hierarchical Clustering (Average Linkage)** to explore how region clustering varies.

**If auto-optimize produces too many or too few regions**:

1. Try **Transmission Region** grouping (fewer initial groups = fewer final regions).
2. Or adjust your auto-optimize range to be narrower.

**Key Takeaway**: Different combinations of grouping, algorithm, and target will produce different results. **Try multiple configurations** and evaluate whether the resulting regions make sense for your use case (balanced sizes, grid-operational coherence, computational feasibility). Your judgment about the appropriateness of the results is as important as the algorithmic quality metrics.

### Plant Clustering

1. **Select technologies to omit**: Choose which technology types to exclude from clustering (e.g., "All Other", "Solar thermal", "Flywheel" are pre-selected).
2. **Group similar technologies**: Check "Group similar technologies" to automatically group related tech types (Biomass and Other peaker by default). Uncheck to treat each technology individually.
3. **Customize groups (optional)**: Add new groups and move technologies between "Available" and "In selected group" lists to create custom groupings.
4. **Set cluster budget**: Specify the total number of clusters across all technologies and regions. The default is automatically set to 15% above the minimum required (one cluster per tech/region combination).
5. **Adjust thresholds (optional)**: Modify *Capacity Threshold* (MW) and *Heat-rate IQR Threshold* to control when generators are suggested for splitting.
6. **Run clustering**: Click *Run Plant Clustering* to generate cluster assignments.
7. **Review suggestions**: Check the "Top candidates for more splits" list to identify tech/region groups that could benefit from more clusters within the current budget.
8. **Export**: Copy or download the YAML output.

!!! tip
    Plant clustering respects the model regions created in the Regions tab. If you haven't run region clustering yet, plants are grouped by their BA. The system uses heat rate variability and capacity to suggest clusters; larger, more varied generator fleets get more clusters when budget allows.

## Running Locally

Since the app uses PyScript and fetches local data files, it must be served via a local HTTP server to avoid CORS errors.

1. Navigate to the `web` directory:

    ```bash
    cd web
    ```

2. Start a simple Python server:

    ```bash
    python -m http.server 8000
    ```

3. Open your browser to `http://localhost:8000`.
