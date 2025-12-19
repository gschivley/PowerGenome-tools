# Web Application

The PowerGenome Design Wizard is a comprehensive web-based interface for building complete PowerGenome settings files. It guides users through a 7-step process to define model regions, configure resources, and export ready-to-use configuration files. The application runs entirely in the browser using PyScript—no installation required.

[Launch Web App](https://gschivley.github.io/PowerGenome-tools/web/){ .md-button .md-button--primary }

## Overview

The wizard approach ensures you configure all necessary settings in the correct order:

1. **Regions** - Define model regions by clustering Balancing Authorities
2. **Model Setup** - Configure planning years and financial parameters
3. **Existing Plants** - Aggregate existing generators within regions
4. **New Resources** - Select new-build technologies and define custom resources
5. **Fuels** - Choose fuel price scenarios
6. **ESR Policies** - Configure Energy Share Requirements for state-level policies (optional)
7. **Export** - Generate and download complete settings YAML files

Each step builds on the previous ones, with the Regions step being the foundation that determines how plants are aggregated and how model boundaries are defined.

## Step 1: Regions

The Regions step allows you to select Balancing Authorities and cluster them into model regions based on transmission capacity. This is the foundation of your PowerGenome model configuration.

### How to Use Region Clustering

1. **Select BAs**: Click on regions in the map to select them. Use **Box Select** mode to drag and select multiple BAs at once.
2. **Choose grouping**: The *Grouping Column* determines how BAs are grouped for clustering (colored outlines show groups).
3. **Set target regions**: Enter how many final regions you want after clustering. Optionally enable *Auto-optimize* to find the best number of regions within a range.
4. **Exclude groups**: Check any groups in *Groups to Keep Unclustered* to keep them as individual BAs.
5. **Run clustering**: Click *Run Clustering* to generate the region aggregations.
6. **Export**: Copy or download the YAML output to use in PowerGenome.

!!! tip
    The clustering uses transmission capacity between BAs to create aggregated regions. BAs are first clustered within their selected grouping (e.g., NERC region), then groups are merged to reach the target number of regions. The selected grouping affects the clustering results, so experiment with different options to see what works best for your case!

### Clustering Configuration Guide

Selecting the right grouping column and clustering algorithm is essential for producing model regions that are both computationally tractable and physically meaningful. This section explains how different choices affect your results.

### Choosing a Grouping Column

The **Grouping Column** determines how BAs are initially partitioned before clustering. This choice significantly influences region balance and grid operational fidelity.

| Option | Groups | Details | Strengths | Weaknesses | Best For |
| -------- | -------- | --------- | ----------- | ----------- | ---------- |
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
| ----------- | -------------- | ----------- | ----------- | ---------- |
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

### ESR-Compatible Clustering

When modeling state-level energy policies like Renewable Portfolio Standards (RPS) or Clean Energy Standards (CES), you may want model regions that respect state trading boundaries. The **ESR-compatible clustering** option ensures that Balancing Authorities are only grouped together if their states can trade renewable energy credits (RECs) or clean energy credits with each other.

#### When to Enable ESR-Compatible Clustering

This option is **not required** for ESR policy modeling—the ESR step (Step 6) will automatically handle any incompatibilities. However, enabling it during clustering can produce cleaner results:

**Enable this option when:**

* You want to prevent model regions from being split later in the ESR step
* You prefer simpler region names without `_esr1`, `_esr2` suffixes
* Your analysis heavily depends on state trading boundaries

**Leave it disabled when:**

* You prioritize transmission-based clustering over trading constraints
* You're okay with some regions being split for ESR purposes
* You're not modeling state-level energy policies

#### How It Works

When ESR-compatible clustering is enabled, the algorithm checks whether BAs can be grouped based on their states' trading relationships:

1. **Trading relationships** are defined in `rectable.csv`, which specifies which states can trade REC/ESR credits with each other.
2. **Transitive trading** is applied: if State A can trade with State C, and State B can trade with State C, then BAs in States A and B can be in the same model region. This captures indirect trading relationships through common trading partners.
3. BAs in states that cannot trade (even transitively) will **never** be placed in the same model region, regardless of transmission capacity.

#### Impact on Results

ESR-compatible clustering adds constraints that may affect your clustering results:

* The algorithm may create **more regions** than your target number if trading boundaries require separation
* When this happens, you'll see a warning explaining why the target couldn't be achieved
* Regions will be smaller but will correctly represent policy trading zones

!!! tip
    If you skip ESR-compatible clustering and your model regions contain states that cannot trade with each other, the ESR step will automatically split those regions into sub-regions (e.g., `RegionName_esr1`, `RegionName_esr2`) for policy tracking purposes.

For detailed technical information about the clustering algorithms used in this step, see the [Algorithms documentation](algorithms.md).

## Step 2: Model Setup

The Model Setup step allows you to configure the temporal and financial parameters for your PowerGenome model.

### Configuration Options

* **Target USD Year**: The dollar year for all cost values (e.g., 2024)
* **UTC Offset**: Timezone offset for demand and weather data
* **Model Years**: Comma-separated list of years to model (e.g., 2030, 2035, 2040)
* **First Planning Years**: Comma-separated list of first planning years corresponding to each model year

!!! note
    Model Years and First Planning Years must be lists of the same length. These define the temporal scope of your capacity expansion analysis.

## Step 3: Existing Plants

The Existing Plants step allows you to cluster existing generators within each model region. This reduces model complexity while preserving the operational diversity of the fleet.

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
    Plant clustering respects the model regions created in the Regions step. If you haven't run region clustering yet, plants are grouped by their BA. The system uses heat rate variability and capacity to suggest clusters; larger, more varied generator fleets get more clusters when budget allows.

## Step 4: New Resources

The New Resources step allows you to select new-build technologies from NREL's Annual Technology Baseline (ATB) and define custom modified resources.

### Standard ATB Resources

If ATB data is available, use the dropdowns to select:

* **ATB Data Year** → **Technology** → **Tech Detail** → **Cost Case**
* Specify **Size (MW)** for each resource

You can also paste resources manually, one per line:

    Technology | Tech Detail | Cost Case | Size

### Modified New Resources

Create custom resources by modifying existing ATB entries:

1. **Base Resource**: Select an ATB technology/detail/cost case/size as the starting point
2. **New Identity**: Define new technology/detail/cost case names for this modified resource
3. **Fuel Type**: Choose a standard fuel (coal, natural gas, distillate, uranium) or define a new fuel with price and emissions
4. **Resource Class Tag**: Select THERM, VRE, STOR, or other resource class
5. **Commit Tag**: For thermal resources, optionally add to the "Commit" tag
6. **Attribute Modifiers**: Add custom YAML attributes to override base resource properties

!!! note
    Modified resources are added to the `modified_new_resources` section of `resources.yml` and automatically update related settings in `fuels.yml` and `resource_tags.yml`.

## Step 5: Fuels

The Fuels step allows you to select fuel price scenarios for each fuel type in your model.

### Fuel Price Scenarios

Select a **Fuel Data Year** from the dropdown, which loads scenarios from [PowerGenome-data fuel_prices.csv](https://github.com/gschivley/PowerGenome-data/blob/main/data/fuel_prices.csv).

For each fuel (coal, natural gas, distillate, uranium), choose a price scenario:

* **Default**: Coal uses `no_111d` if available (otherwise `reference`); other fuels use `reference`
* **Available scenarios**: Varies by fuel data year and fuel type

!!! tip
    If fuel scenario options can't be loaded (offline use), the app falls back to `reference` for all fuels.

## Step 6: ESR Policies

The ESR Policies step allows you to configure Energy Share Requirements for state-level policies like Renewable Portfolio Standards (RPS) and Clean Energy Standards (CES). This step is optional—uncheck "Include ESR policies" if your analysis doesn't require policy constraints.

### How ESR Zones Work

The app automatically groups your model regions into ESR zones based on state trading rules defined in `rectable.csv`. States that can trade renewable energy credits (RECs) or clean energy credits with each other are placed in the same zone.

### Automatic Region Splitting

If a model region contains states that **cannot** trade with each other (even transitively), the app automatically splits that region into sub-regions for ESR purposes:

* Sub-regions are named with `_esr1`, `_esr2`, etc. suffixes (e.g., `MidAtlantic_esr1`, `MidAtlantic_esr2`)
* Each sub-region contains only states that can trade with each other
* The generated CSV output uses these expanded region names

!!! tip
    To avoid region splitting, enable **ESR-compatible clustering** in Step 1. This builds trading constraints into the clustering algorithm upfront, ensuring all BAs in a region can trade within the same ESR zone.

### Generated Output

The ESR step generates a CSV file (`emission_policies.csv`) containing:

* ESR zone assignments for each region (or sub-region)
* Policy requirements (RPS and CES fractions) by zone and year
* Technology qualification mappings

## Step 7: Export

The Export step generates complete PowerGenome settings files based on all previous configuration steps.

### Generated Files

The app generates seven YAML files:

* `model_definition.yml` - Model regions, years, and financial settings
* `resources.yml` - Existing plant clusters, new resources, and modified resources
* `fuels.yml` - Fuel prices and emission factors
* `transmission.yml` - Transmission line definitions
* `distributed_gen.yml` - Distributed generation settings
* `resource_tags.yml` - Resource classification tags
* `startup_costs.yml` - Startup cost parameters

The app intentionally **does not generate** these files (configure separately):

* `data.yml`
* `scenario_management.yml`
* `time_clustering.yml`
* `extra_inputs.yml`
* `demand.yml`

### How to Export

1. Review the configuration summary
2. Click **Generate Settings YAMLs**
3. Use the file dropdown to preview each settings file
4. Click **Download** to save individual files or **Download All** for a zip archive

!!! note
    `model_definition.yml` and several downstream defaults require region aggregations from Step 1.
    If you haven't clustered regions yet, the Export step will prompt you to complete Step 1 first.

---

## Additional Features

### Interactive Map

* **View Balancing Authorities**: Click on regions to select them; colored outlines show grouping
* **Box Select Mode**: Drag to select multiple BAs at once (enabled by default)
* **Transmission Lines**: Toggle to view transmission capacity between clustered regions
* **Tooltips**: Hover over regions to see BA name, state, grouping, and model region (after clustering)

---

## Running Locally

Since the app uses PyScript and fetches local data files, it must be served via a local HTTP server to avoid CORS errors.

1. Navigate to the `web` directory:

    cd web

2. Start a simple Python server:

    python -m http.server 8000

3. Open your browser to `http://localhost:8000`.
