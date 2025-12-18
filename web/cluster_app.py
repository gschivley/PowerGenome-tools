"""
PowerGenome Region Clustering - PyScript Web App

This module runs in the browser via PyScript and handles:
1. Loading and displaying the BA map
2. Managing BA selection state
3. Running clustering algorithms (agglomerative and Louvain)
4. Generating YAML output
"""

import asyncio
import html
import json
import math
import warnings
from io import StringIO

from js import L, document, fetch, window
from pyodide.ffi import create_proxy, to_js

# Suppress pandas pyarrow deprecation warning
warnings.filterwarnings("ignore", message=".*pyarrow.*", category=DeprecationWarning)

import networkx as nx
import numpy as np

# Will be imported after PyScript loads packages
import pandas as pd
import yaml

# ============================================================================
# Global State
# ============================================================================


class AppState:
    def __init__(self):
        self.map = None
        self.geojson_layer = None
        self.geojson_data = None
        self.hierarchy_df = None
        self.transmission_df = None
        self.plants_df = None  # generator-level data
        self.plant_region_map = None  # plant_id -> BA mapping
        self.plant_candidates = []  # cache of last candidate list
        self.selected_bas = set()
        self.ba_layers = {}  # ba_id -> layer
        self.all_bas = set()
        self.ba_centroids = {}  # ba_id -> (lat, lng) for box selection
        self.box_select_mode = True  # Default to box select mode
        self.box_start = None
        self.group_colors = {}  # group_value -> outline color
        self.group_fill_colors = {}  # group_value -> light fill color
        self.ba_to_group = {}  # ba_id -> group_value
        self.current_grouping = None  # current grouping column
        self.cluster_colors = {}  # ba_id -> cluster fill color (set after clustering)
        self.is_clustered = False  # True after clustering has been run
        self.ba_to_region = {}  # ba_id -> model_region name (set after clustering)
        self.transmission_lines_layer = (
            None  # Leaflet layer group for transmission lines
        )
        self.show_transmission_lines = False  # Toggle state for transmission lines
        self.region_aggregations = (
            None  # Store last clustering result for redrawing lines
        )
        self.custom_tech_groups = {}  # user-editable tech grouping map
        self.available_techs = set()  # techs not currently assigned to a group
        self.current_group = None  # currently selected group in UI
        self.omit_selected = set()  # technologies to omit (dual-list UI)
        self.omit_available = set()  # technologies available to include


state = AppState()
GROUP_OUTLINE_COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
    "#8dd3c7",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
    "#1f78b4",
    "#33a02c",
    "#fb9a99",
]

# Styling defaults
STYLE_UNSELECTED = {
    "fillColor": "#cccccc",
    "fillOpacity": 0.4,
    "color": "#666666",
    "weight": 1,
}

STYLE_SELECTED = {
    "fillColor": "#2196F3",
    "fillOpacity": 0.6,
    "color": "#1565C0",
    "weight": 2,
}

STYLE_HOVER = {
    "fillOpacity": 0.8,
    "weight": 3,
}

# Cluster colors for visualization (fill colors after clustering)
CLUSTER_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
]


def get_outline_color(ba_id):
    """Get the outline color for a BA based on its group."""
    group = state.ba_to_group.get(ba_id)
    if group and group in state.group_colors:
        return state.group_colors[group]
    return "#666666"  # default gray


def get_fill_color(ba_id):
    """Get the fill color for an unselected BA based on its group (lighter version)."""
    group = state.ba_to_group.get(ba_id)
    if group and group in state.group_fill_colors:
        return state.group_fill_colors[group]
    return "#cccccc"  # default gray


# Styling defaults
STYLE_UNSELECTED = {
    "fillColor": "#cccccc",
    "fillOpacity": 0.4,
    "color": "#666666",
    "weight": 1,
}

STYLE_SELECTED = {
    "fillColor": "#2196F3",
    "fillOpacity": 0.6,
    "color": "#1565C0",
    "weight": 2,
}

STYLE_HOVER = {
    "fillOpacity": 0.8,
    "weight": 3,
}

# ============================================================================
# Map Functions
# ============================================================================


def update_loading_text(text):
    """Update the loading indicator text."""
    el = document.getElementById("loadingText")
    if el:
        el.textContent = text


def hide_loading():
    """Hide the loading overlay."""
    el = document.getElementById("loading")
    if el:
        el.classList.add("hidden")


def set_status(message, status_type="info"):
    """Update the status box."""
    el = document.getElementById("statusBox")
    if el:
        el.textContent = message
        el.className = f"status {status_type}"


def update_selected_display():
    """Update the selected BAs display."""
    count_el = document.getElementById("selectedCount")
    list_el = document.getElementById("selectedList")

    if count_el:
        count_el.textContent = str(len(state.selected_bas))

    if list_el:
        if state.selected_bas:
            sorted_bas = sorted(state.selected_bas)
            html_list = "".join(
                f'<span class="ba-tag">{ba}</span>' for ba in sorted_bas
            )
            list_el.innerHTML = html_list
        else:
            list_el.innerHTML = "<em>None selected</em>"

    # Enable/disable run button
    run_btn = document.getElementById("runBtn")
    if run_btn:
        run_btn.disabled = len(state.selected_bas) < 2


def toggle_ba_selection(ba_id, layer):
    """Toggle selection state of a BA."""
    outline_color = get_outline_color(ba_id)
    fill_color = get_fill_color(ba_id)

    if ba_id in state.selected_bas:
        state.selected_bas.remove(ba_id)
        layer.setStyle(
            to_js(
                {
                    "fillColor": fill_color,
                    "fillOpacity": 0.5,
                    "color": outline_color,
                    "weight": 2,
                }
            )
        )
    else:
        state.selected_bas.add(ba_id)
        layer.setStyle(
            to_js(
                {
                    "fillColor": "#2196F3",
                    "fillOpacity": 0.6,
                    "color": outline_color,
                    "weight": 3,
                }
            )
        )

    update_selected_display()


def on_feature_click(e):
    """Handle click on a BA feature."""
    layer = e.target
    props = layer.feature.properties
    ba_id = props.rb
    toggle_ba_selection(ba_id, layer)


def on_feature_mouseover(e):
    """Handle mouseover on a BA feature - subtle highlight without changing colors."""
    layer = e.target
    # Only increase weight slightly, don't change fill
    layer.setStyle(to_js({"weight": 4}))
    layer.bringToFront()


def on_feature_mouseout(e):
    """Handle mouseout on a BA feature - restore original style."""
    layer = e.target
    props = layer.feature.properties
    ba_id = props.rb

    outline_color = get_outline_color(ba_id)

    # If clustering has been run, use cluster colors for selected BAs
    if state.is_clustered and ba_id in state.cluster_colors:
        layer.setStyle(
            to_js(
                {
                    "fillColor": state.cluster_colors[ba_id],
                    "fillOpacity": 0.7,
                    "color": outline_color,
                    "weight": 3,
                }
            )
        )
    elif ba_id in state.selected_bas:
        layer.setStyle(
            to_js(
                {
                    "fillColor": "#2196F3",
                    "fillOpacity": 0.6,
                    "color": outline_color,
                    "weight": 3,
                }
            )
        )
    else:
        fill_color = get_fill_color(ba_id)
        layer.setStyle(
            to_js(
                {
                    "fillColor": fill_color,
                    "fillOpacity": 0.5,
                    "color": outline_color,
                    "weight": 2,
                }
            )
        )


def on_each_feature(feature, layer):
    """Attach event handlers to each feature."""
    props = feature.properties
    ba_id = props.rb

    state.ba_layers[ba_id] = layer
    state.all_bas.add(ba_id)

    # Calculate centroid for box selection
    bounds = layer.getBounds()
    center = bounds.getCenter()
    state.ba_centroids[ba_id] = (center.lat, center.lng)

    # Initial tooltip (will be updated when data loads)
    tooltip = f"<b>{ba_id}</b><br>State: {props.st}"
    layer.bindTooltip(tooltip)

    # Events
    layer.on("click", create_proxy(on_feature_click))
    layer.on("mouseover", create_proxy(on_feature_mouseover))
    layer.on("mouseout", create_proxy(on_feature_mouseout))


def style_feature(feature):
    """Return initial style for a feature."""
    return to_js(STYLE_UNSELECTED)


async def init_map():
    """Initialize the Leaflet map."""
    update_loading_text("Initializing map...")

    # Create map centered on US
    state.map = L.map("map").setView(to_js([39.8, -98.5]), 4)

    # Add tile layer
    L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        to_js(
            {
                "attribution": '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a>',
                "maxZoom": 18,
            }
        ),
    ).addTo(state.map)

    # Create custom pane for transmission lines (above overlays, z-index 450)
    state.map.createPane("transmissionPane")
    state.map.getPane("transmissionPane").style.zIndex = 450

    # Load GeoJSON
    update_loading_text("Loading BA boundaries...")

    response = await fetch("./data/US_PCA.geojson")
    geojson_text = await response.text()
    state.geojson_data = json.loads(geojson_text)

    # Add GeoJSON layer
    state.geojson_layer = L.geoJSON(
        to_js(state.geojson_data),
        to_js(
            {
                "style": create_proxy(style_feature),
                "onEachFeature": create_proxy(on_each_feature),
            }
        ),
    ).addTo(state.map)

    # Fit bounds
    state.map.fitBounds(state.geojson_layer.getBounds())

    # Update total count
    total_el = document.getElementById("totalCount")
    if total_el:
        total_el.textContent = str(len(state.all_bas))


# ============================================================================
# Data Loading
# ============================================================================


async def load_data():
    """Load hierarchy, transmission, and plant CSVs."""
    update_loading_text("Loading hierarchy data...")

    response = await fetch("./data/hierarchy.csv")
    if not response.ok:
        raise Exception(
            f"Failed to load hierarchy.csv: {response.status} {response.statusText}"
        )
    hierarchy_text = await response.text()

    # Debug: check what we got
    if hierarchy_text.startswith("<!"):
        raise Exception(
            f"Got HTML instead of CSV. First 100 chars: {hierarchy_text[:100]}"
        )

    state.hierarchy_df = pd.read_csv(StringIO(hierarchy_text))

    update_loading_text("Loading transmission data...")

    response = await fetch("./data/transmission_capacity_reeds.csv")
    if not response.ok:
        raise Exception(
            f"Failed to load transmission CSV: {response.status} {response.statusText}"
        )
    transmission_text = await response.text()
    state.transmission_df = pd.read_csv(StringIO(transmission_text))

    update_loading_text("Loading plant data...")

    response = await fetch("./data/reeds_generators_transformed.csv")
    if not response.ok:
        raise Exception(
            f"Failed to load plant data CSV: {response.status} {response.statusText}"
        )
    plant_text = await response.text()
    state.plants_df = pd.read_csv(StringIO(plant_text))

    response = await fetch("./data/plant_region_map.csv")
    if not response.ok:
        raise Exception(
            f"Failed to load plant-region map CSV: {response.status} {response.statusText}"
        )
    plant_map_text = await response.text()
    state.plant_region_map = pd.read_csv(StringIO(plant_map_text))


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color."""
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def lighten_color(hex_color, factor=0.7):
    """Create a lighter version of a color by mixing with white."""
    r, g, b = hex_to_rgb(hex_color)
    # Mix with white (255, 255, 255)
    r = r + (255 - r) * factor
    g = g + (255 - g) * factor
    b = b + (255 - b) * factor
    return rgb_to_hex((r, g, b))


def update_group_colors():
    """Update group colors based on current grouping column and apply to map."""
    grouping_col = document.getElementById("groupingColumn").value

    if state.hierarchy_df is None:
        return

    # Skip if same grouping column (unless first time)
    if state.current_grouping == grouping_col and state.group_colors:
        return

    state.current_grouping = grouping_col

    # Get unique groups and assign colors
    unique_groups = sorted(state.hierarchy_df[grouping_col].unique())
    state.group_colors = {}
    state.group_fill_colors = {}
    for i, group in enumerate(unique_groups):
        outline_color = GROUP_OUTLINE_COLORS[i % len(GROUP_OUTLINE_COLORS)]
        state.group_colors[group] = outline_color
        # Create a light fill color (70% toward white)
        state.group_fill_colors[group] = lighten_color(outline_color, 0.75)

    # Build BA to group mapping
    state.ba_to_group = {}
    for _, row in state.hierarchy_df.iterrows():
        ba = row["ba"]
        state.ba_to_group[ba] = row[grouping_col]

    # Apply colors to all BA layers
    apply_group_colors_to_map()


def apply_group_colors_to_map():
    """Apply group outline and fill colors to all BA layers on the map."""
    for ba_id, layer in state.ba_layers.items():
        outline_color = get_outline_color(ba_id)
        fill_color = get_fill_color(ba_id)

        if ba_id in state.selected_bas:
            layer.setStyle(
                to_js(
                    {
                        "fillColor": "#2196F3",
                        "fillOpacity": 0.6,
                        "color": outline_color,
                        "weight": 3,
                    }
                )
            )
        else:
            layer.setStyle(
                to_js(
                    {
                        "fillColor": fill_color,
                        "fillOpacity": 0.5,
                        "color": outline_color,
                        "weight": 2,
                    }
                )
            )


def update_tooltips():
    """Update all BA tooltips to show the current grouping column value."""
    if state.hierarchy_df is None:
        return

    grouping_col = document.getElementById("groupingColumn").value

    # Get friendly name for the grouping column from the dropdown text
    grouping_select = document.getElementById("groupingColumn")
    selected_option = grouping_select.options.item(grouping_select.selectedIndex)
    grouping_label = selected_option.text.split(" (")[
        0
    ]  # Get just the column name part

    # Build a lookup from BA to hierarchy row
    ba_data = {}
    for _, row in state.hierarchy_df.iterrows():
        ba_data[row["ba"]] = row

    # Update each layer's tooltip
    for ba_id, layer in state.ba_layers.items():
        if ba_id in ba_data:
            row = ba_data[ba_id]
            state_val = row.get("st", "N/A")
            group_val = row.get(grouping_col, "N/A")

            # Include model region if clustering has been done
            if state.is_clustered and ba_id in state.ba_to_region:
                region_name = state.ba_to_region[ba_id]
                tooltip = f"<b>{ba_id}</b><br>State: {state_val}<br>{grouping_label}: {group_val}<br><b>Region: {region_name}</b>"
            else:
                tooltip = f"<b>{ba_id}</b><br>State: {state_val}<br>{grouping_label}: {group_val}"
            layer.unbindTooltip()
            layer.bindTooltip(tooltip)


def update_no_cluster_options():
    """Update the no-cluster checkbox options based on grouping column."""
    grouping_col = document.getElementById("groupingColumn").value
    container = document.getElementById("noClusterContainer")

    if state.hierarchy_df is None or container is None:
        return

    # Update group colors when grouping changes
    update_group_colors()

    # Update tooltips to show new grouping column
    update_tooltips()

    # Get unique values for this column
    unique_values = sorted(state.hierarchy_df[grouping_col].unique())

    # Build checkboxes with color indicators
    html = ""
    for val in unique_values:
        color = state.group_colors.get(val, "#666666")
        html += f"""
            <label>
                <input type="checkbox" name="noCluster" value="{val}">
                <span style="display:inline-block;width:12px;height:12px;background:{color};border-radius:2px;margin-right:4px;vertical-align:middle;"></span>
                {val}
            </label>
        """

    container.innerHTML = html


# ============================================================================
# Clustering Logic (adapted from cluster_regions.py)
# ============================================================================


def standardize_features(matrix):
    """Standardize columns to zero mean, unit variance."""
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    return (matrix - means) / stds


def run_kmeans_simple(features, k, weights=None, max_iter=40, seed=42):
    """Simple k-means implementation returning (inertia, centers, labels)."""
    rng = np.random.default_rng(seed)
    n_samples = features.shape[0]
    if k <= 0 or n_samples == 0:
        return 0.0, None, None

    # Initialize centers from samples
    init_idx = rng.choice(n_samples, size=min(k, n_samples), replace=False)
    centers = features[init_idx]

    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        # Assign
        dists = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2) ** 2
        labels = np.argmin(dists, axis=1)

        # Update
        new_centers = []
        for i in range(k):
            mask = labels == i
            if not np.any(mask):
                # Keep old center if empty
                new_centers.append(centers[i])
                continue

            cluster_points = features[mask]
            if weights is not None:
                cluster_weights = weights[mask][:, None]
                new_center = (cluster_points * cluster_weights).sum(
                    axis=0
                ) / cluster_weights.sum()
            else:
                new_center = cluster_points.mean(axis=0)
            new_centers.append(new_center)

        new_centers = np.vstack(new_centers)
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    # Compute inertia
    inertia = 0.0
    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            continue
        cluster_points = features[mask]
        cluster_center = centers[i]
        sq_dists = np.sum((cluster_points - cluster_center) ** 2, axis=1)
        if weights is not None:
            inertia += float((sq_dists * weights[mask]).sum())
        else:
            inertia += float(sq_dists.sum())

    return inertia, centers, labels


def build_transmission_graph(transmission_df, valid_bas):
    """Build undirected weighted graph from transmission data."""
    G = nx.Graph()

    for _, row in transmission_df.iterrows():
        region_from = row["region_from"]
        region_to = row["region_to"]
        capacity = row["firm_ttc_mw"]

        # Only include edges between valid BAs
        if region_from not in valid_bas or region_to not in valid_bas:
            continue

        if G.has_edge(region_from, region_to):
            G[region_from][region_to]["weight"] += capacity
        else:
            G.add_edge(region_from, region_to, weight=capacity)

    # Add isolated nodes for BAs with no connections
    for ba in valid_bas:
        if ba not in G:
            G.add_node(ba)

    return G


def get_regional_groups(hierarchy_df, grouping_column, valid_bas):
    """Map regional groups to their BAs."""
    groups = {}

    for _, row in hierarchy_df.iterrows():
        ba = row["ba"]
        if ba not in valid_bas:
            continue

        group = row[grouping_column]
        if group not in groups:
            groups[group] = set()
        groups[group].add(ba)

    return groups


def agglomerative_cluster(graph, n_clusters, linkage="sum"):
    """
    Perform agglomerative clustering on a graph.

    Linkage methods:
    - 'sum': Merge based on sum of edge weights (standard).
    - 'average': Merge based on average edge weight (sum / (size_a * size_b)).
    - 'max': Merge based on maximum single edge weight (single linkage).
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    if n <= n_clusters:
        # Each node is its own cluster
        return {i: {node} for i, node in enumerate(nodes)}

    # Initialize: each node is its own cluster
    clusters = {i: {node} for i, node in enumerate(nodes)}
    node_to_cluster = {node: i for i, node in enumerate(nodes)}
    cluster_sizes = {i: 1 for i in range(n)}

    # Build initial inter-cluster weights
    cluster_weights = {}
    for u, v, data in graph.edges(data=True):
        c1, c2 = node_to_cluster[u], node_to_cluster[v]
        if c1 != c2:
            key = (min(c1, c2), max(c1, c2))
            weight = data.get("weight", 1.0)

            if linkage == "max":
                current = cluster_weights.get(key, -float("inf"))
                cluster_weights[key] = max(current, weight)
            else:
                cluster_weights[key] = cluster_weights.get(key, 0) + weight

    # Merge until we reach target number of clusters
    while len(clusters) > n_clusters:
        if not cluster_weights:
            break

        # Find the pair with maximum score
        if linkage == "average":

            def get_score(k):
                c1, c2 = k
                w = cluster_weights[k]
                return w / (cluster_sizes[c1] * cluster_sizes[c2])

            best_pair = max(cluster_weights.keys(), key=get_score)
        else:
            best_pair = max(cluster_weights.keys(), key=lambda k: cluster_weights[k])

        c1, c2 = best_pair

        # Merge c2 into c1
        clusters[c1].update(clusters[c2])
        cluster_sizes[c1] += cluster_sizes[c2]
        del cluster_sizes[c2]

        for node in clusters[c2]:
            node_to_cluster[node] = c1
        del clusters[c2]

        # Update cluster weights
        new_weights = {}
        keys_to_remove = []

        for (ca, cb), weight in cluster_weights.items():
            if ca == c2 or cb == c2:
                keys_to_remove.append((ca, cb))
                # Redirect to c1
                other = cb if ca == c2 else ca
                if other != c1:
                    new_key = (min(c1, other), max(c1, other))

                    # Get existing weight between c1 and other
                    w1 = cluster_weights.get(new_key)

                    if linkage == "max":
                        val1 = w1 if w1 is not None else -float("inf")
                        new_val = max(val1, weight)
                    else:
                        val1 = w1 if w1 is not None else 0
                        new_val = val1 + weight

                    new_weights[new_key] = new_val

        for key in keys_to_remove:
            del cluster_weights[key]

        for key, weight in new_weights.items():
            cluster_weights[key] = weight

    # Renumber clusters to be sequential
    result = {}
    for i, (cluster_id, nodes_set) in enumerate(clusters.items()):
        result[i] = nodes_set

    return result


def spectral_cluster(graph, n_clusters):
    """
    Perform spectral clustering on the graph using Normalized Laplacian.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    if n <= n_clusters:
        return {i: {node} for i, node in enumerate(nodes)}

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Adjacency matrix
    A = np.zeros((n, n))
    for u, v, data in graph.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        w = data.get("weight", 1.0)
        A[i, j] = w
        A[j, i] = w

    # Degree matrix
    d = np.sum(A, axis=1)

    # Normalized Laplacian: L_sym = I - D^-1/2 * A * D^-1/2
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    d_inv_sqrt[d == 0] = 0
    D_inv_sqrt = np.diag(d_inv_sqrt)

    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # Eigen decomposition
    vals, vecs = np.linalg.eigh(L)

    # First k eigenvectors
    k = n_clusters
    X = vecs[:, :k]

    # Normalize rows
    rows_norm = np.linalg.norm(X, axis=1, keepdims=True)
    rows_norm[rows_norm == 0] = 1
    X_normalized = X / rows_norm

    # Run K-Means
    _, _, labels = run_kmeans_simple(X_normalized, k)

    # Convert labels back to clusters
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = set()
        clusters[label].add(nodes[idx])

    return clusters


def louvain_cluster(graph):
    """
    Perform Louvain community detection on a graph.

    Returns communities that maximize modularity.
    The number of clusters is determined automatically.

    This is used for auto-optimize mode.
    """
    if graph.number_of_nodes() == 0:
        return {}

    if graph.number_of_edges() == 0:
        # No edges - each node is its own community
        return {i: {node} for i, node in enumerate(graph.nodes())}

    try:
        # Use NetworkX's Louvain implementation
        communities = nx.community.louvain_communities(
            graph, weight="weight", resolution=1.0, seed=42
        )

        # Convert to our format: dict of label -> set of nodes
        return {i: set(community) for i, community in enumerate(communities)}
    except Exception:
        # Fallback: each node is its own cluster
        return {i: {node} for i, node in enumerate(graph.nodes())}


def hierarchical_cluster(
    hierarchy_df,
    transmission_df,
    cluster_bas,
    grouping_column,
    target_regions,
    method="hierarchical-sum",
):
    """
    Hierarchical clustering that respects grouping column boundaries.

    Phase 1: Cluster BAs within each grouping column region
    Phase 2: Merge entire grouping column regions together if needed

    Grouping column regions are never split across model regions.
    """
    # Parse method
    if method == "spectral":
        algo = "spectral"
        linkage = None
    elif method.startswith("hierarchical-"):
        algo = "hierarchical"
        linkage = method.split("-")[1]
    else:
        # Default
        algo = "hierarchical"
        linkage = "sum"

    # Group BAs by their grouping column value
    groups = {}
    for _, row in hierarchy_df[hierarchy_df["ba"].isin(cluster_bas)].iterrows():
        ba = row["ba"]
        group = row[grouping_column]
        if group not in groups:
            groups[group] = set()
        groups[group].add(ba)

    num_groups = len(groups)

    # If target is >= number of groups, cluster within each group
    if target_regions >= num_groups:
        # Build map of BA to group
        ba_to_group = {}
        for group, bas in groups.items():
            for ba in bas:
                ba_to_group[ba] = group

        # Build full graph
        graph = build_transmission_graph(transmission_df, cluster_bas)

        # Remove edges between different groups to enforce group boundaries
        edges_to_remove = []
        for u, v in graph.edges():
            if ba_to_group.get(u) != ba_to_group.get(v):
                edges_to_remove.append((u, v))

        graph.remove_edges_from(edges_to_remove)

        if algo == "spectral":
            # For spectral clustering, we must run it independently on each group
            # to avoid mixing eigenvectors of disconnected components.
            # We use agglomerative clustering to decide how many clusters each group gets.

            # 1. Get reference allocation using agglomerative clustering
            ref_clusters = agglomerative_cluster(
                graph, target_regions, linkage="average"
            )

            # 2. Count clusters per group
            group_allocations = {g: 0 for g in groups}
            for _, nodes in ref_clusters.items():
                # Pick a representative node to find the group
                # (All nodes in a cluster are in the same group because edges were removed)
                if not nodes:
                    continue
                rep_node = next(iter(nodes))
                grp = ba_to_group[rep_node]
                group_allocations[grp] += 1

            # 3. Run spectral clustering on each group independently
            final_clusters = {}
            cluster_id_counter = 0

            for group, count in group_allocations.items():
                if count == 0:
                    continue

                group_bas = groups[group]
                # Build subgraph for this group
                subgraph = build_transmission_graph(transmission_df, group_bas)

                # Run spectral on subgraph
                sub_clusters = spectral_cluster(subgraph, count)

                # Add to final result
                for _, nodes in sub_clusters.items():
                    final_clusters[cluster_id_counter] = nodes
                    cluster_id_counter += 1

            return final_clusters
        else:
            # Run agglomerative clustering on the whole graph
            # This prioritizes the strongest connections across all groups
            # instead of pre-allocating a fixed number of clusters per group
            return agglomerative_cluster(graph, target_regions, linkage=linkage)

    else:
        # target_regions < num_groups: need to merge entire groups
        # Phase 1: Each group becomes a single unit
        # Phase 2: Merge groups based on inter-group transmission capacity

        # Build a graph where nodes are groups and edges are total transmission between groups
        group_graph = nx.Graph()
        for group in groups:
            group_graph.add_node(group)

        # Calculate inter-group transmission capacity
        for _, row in transmission_df.iterrows():
            ba_from = row["region_from"]
            ba_to = row["region_to"]
            capacity = row["firm_ttc_mw"]

            # Find which groups these BAs belong to
            group_from = None
            group_to = None
            for group, bas in groups.items():
                if ba_from in bas:
                    group_from = group
                if ba_to in bas:
                    group_to = group

            if group_from and group_to and group_from != group_to:
                if group_graph.has_edge(group_from, group_to):
                    group_graph[group_from][group_to]["weight"] += capacity
                else:
                    group_graph.add_edge(group_from, group_to, weight=capacity)

        # Cluster groups
        if algo == "spectral":
            group_clusters = spectral_cluster(group_graph, target_regions)
        else:
            group_clusters = agglomerative_cluster(
                group_graph, target_regions, linkage=linkage
            )

        # Convert group clusters to BA clusters
        all_clusters = {}
        cluster_id = 0

        for label, group_set in group_clusters.items():
            # Combine all BAs from all groups in this cluster
            combined_bas = set()
            for group in group_set:
                combined_bas.update(groups[group])
            all_clusters[cluster_id] = combined_bas
            cluster_id += 1

        return all_clusters


def calculate_modularity(graph, clusters):
    """
    Calculate the modularity score for a clustering result.

    Modularity measures how well a network is divided into communities.
    Higher values (closer to 1) indicate better clustering.
    Values typically range from -0.5 to 1.
    """
    if not clusters or graph.number_of_edges() == 0:
        return 0.0

    # Convert clusters dict to list of sets for networkx
    communities = [clusters[label] for label in sorted(clusters.keys())]

    # Filter to only include nodes that are in the graph
    graph_nodes = set(graph.nodes())
    communities = [c & graph_nodes for c in communities]
    communities = [c for c in communities if len(c) > 0]

    if not communities:
        return 0.0

    try:
        # Use networkx modularity function
        modularity = nx.community.modularity(graph, communities, weight="weight")
        return modularity
    except Exception:
        return 0.0


def find_optimal_clusters(
    hierarchy_df,
    transmission_df,
    cluster_bas,
    grouping_column,
    min_regions,
    max_regions,
):
    """
    Find the optimal clustering using Louvain community detection.

    Uses Louvain algorithm which directly maximizes modularity.
    The min_regions and max_regions are used to constrain the result:
    - If Louvain finds fewer clusters than min_regions, we don't split further
    - If Louvain finds more clusters than max_regions, we merge using agglomerative

    Returns (best_clusters, best_n, best_modularity, all_scores)
    """
    # Group BAs by their grouping column value
    groups = {}
    for _, row in hierarchy_df[hierarchy_df["ba"].isin(cluster_bas)].iterrows():
        ba = row["ba"]
        group = row[grouping_column]
        if group not in groups:
            groups[group] = set()
        groups[group].add(ba)

    # Build graph for the cluster BAs
    graph = build_transmission_graph(transmission_df, cluster_bas)

    # Use Louvain on within-group subgraphs, respecting grouping boundaries
    all_clusters = {}
    cluster_id = 0

    for group, group_bas in groups.items():
        if len(group_bas) == 1:
            all_clusters[cluster_id] = group_bas
            cluster_id += 1
        else:
            # Build subgraph for this group
            subgraph = build_transmission_graph(transmission_df, group_bas)
            # Use Louvain to find natural communities within this group
            sub_clusters = louvain_cluster(subgraph)

            for label, nodes in sub_clusters.items():
                all_clusters[cluster_id] = nodes
                cluster_id += 1

    num_clusters = len(all_clusters)

    # Capture optimal here
    optimal_clusters = dict(all_clusters)

    # If we have fewer clusters than min_regions, split using agglomerative clustering
    if num_clusters < min_regions:
        # We need to split clusters until we reach min_regions
        # We will greedily split the largest cluster (by number of BAs)

        # Convert to mutable dict
        current_clusters = dict(all_clusters)
        next_id = max(current_clusters.keys()) + 1 if current_clusters else 0

        while len(current_clusters) < min_regions:
            # Find cluster with most BAs that has at least 2 BAs
            candidates = [
                (cid, len(nodes))
                for cid, nodes in current_clusters.items()
                if len(nodes) > 1
            ]

            if not candidates:
                break  # Cannot split further

            # Pick largest
            cid_to_split, _ = max(candidates, key=lambda x: x[1])
            nodes_to_split = current_clusters[cid_to_split]

            # Build subgraph
            subgraph = build_transmission_graph(transmission_df, nodes_to_split)

            # Split into 2 using agglomerative clustering (average linkage for balanced splits)
            split_res = agglomerative_cluster(subgraph, 2, linkage="average")

            # Update clusters
            del current_clusters[cid_to_split]
            current_clusters[cid_to_split] = split_res[0]  # Reuse ID for one
            current_clusters[next_id] = split_res[1]  # New ID for other
            next_id += 1

        all_clusters = current_clusters
        num_clusters = len(all_clusters)

    # If we have more clusters than max_regions, merge using agglomerative
    if num_clusters > max_regions:
        # Build a graph of the current clusters
        cluster_graph = nx.Graph()
        for cid in all_clusters:
            cluster_graph.add_node(cid)

        # Add edges based on transmission between clusters
        for _, row in transmission_df.iterrows():
            ba_from = row["region_from"]
            ba_to = row["region_to"]
            capacity = row["firm_ttc_mw"]

            # Find which clusters these BAs belong to
            cluster_from = None
            cluster_to = None
            for cid, bas in all_clusters.items():
                if ba_from in bas:
                    cluster_from = cid
                if ba_to in bas:
                    cluster_to = cid

            if (
                cluster_from is not None
                and cluster_to is not None
                and cluster_from != cluster_to
            ):
                if cluster_graph.has_edge(cluster_from, cluster_to):
                    cluster_graph[cluster_from][cluster_to]["weight"] += capacity
                else:
                    cluster_graph.add_edge(cluster_from, cluster_to, weight=capacity)

        # Merge clusters down to max_regions
        merged = agglomerative_cluster(cluster_graph, max_regions)

        # Convert back to BA clusters
        new_clusters = {}
        for new_id, old_cluster_ids in merged.items():
            combined_bas = set()
            for old_id in old_cluster_ids:
                combined_bas.update(all_clusters[old_id])
            new_clusters[new_id] = combined_bas

        all_clusters = new_clusters

    # Calculate final modularity
    modularity = calculate_modularity(graph, all_clusters)
    num_clusters = len(all_clusters)

    return (
        all_clusters,
        num_clusters,
        modularity,
        {num_clusters: modularity},
        optimal_clusters,
    )


def generate_cluster_names(clusters, groups):
    """Generate meaningful cluster names based on smallest containing grouping column.

    Naming rules allow only state plus one other grouping column:
    1) Use the state code when all BAs in the cluster share a state.
    2) Pick a single grouping column (other than state) that can name every
       remaining cluster; all non-state names must come from that one column.
    3) If no column satisfies (2), still pick one column (broadest available)
       and name every non-state cluster from that same column.
    """
    # Grouping columns ordered from smallest to largest geographic scope
    GROUPING_HIERARCHY = [
        "st",
        "cendiv",
        "transgrp",
        "nercr",
        "transreg",
        "interconnect",
    ]

    cluster_names = {}
    name_counts = {}  # Track counts for each base name

    def common_value(nodes_set, column):
        vals = (
            state.hierarchy_df[state.hierarchy_df["ba"].isin(nodes_set)][column]
            .dropna()
            .unique()
        )
        return vals[0] if len(vals) == 1 else None

    # First pass: identify clusters that can use state names
    single_state_labels = set()
    for label, nodes in clusters.items():
        st_val = common_value(nodes, "st")
        if st_val:
            single_state_labels.add(label)

    # Choose one naming column for the remaining clusters
    candidate_columns = [col for col in GROUPING_HIERARCHY if col != "st"]
    naming_column = None

    for col in candidate_columns:
        if col not in state.hierarchy_df.columns:
            continue

        all_cover = True
        for label, nodes in clusters.items():
            if label in single_state_labels:
                continue
            if common_value(nodes, col) is None:
                all_cover = False
                break

        if all_cover:
            naming_column = col
            break

    # Fallback: pick the broadest available column
    if naming_column is None:
        for col in reversed(candidate_columns):
            if col in state.hierarchy_df.columns:
                naming_column = col
                break

    # Assign names
    for label, nodes in clusters.items():
        nodes_list = list(nodes)

        # Single BA or single-state cluster
        st_val = common_value(nodes, "st")
        if st_val:
            base_name = st_val if len(nodes_list) > 1 else st_val
        else:
            if naming_column and naming_column in state.hierarchy_df.columns:
                vals = (
                    state.hierarchy_df[state.hierarchy_df["ba"].isin(nodes)][
                        naming_column
                    ]
                    .dropna()
                    .unique()
                )
                if len(vals) == 1:
                    base_name = vals[0]
                else:
                    # Still stick to one column; join values if mixed
                    base_name = "-".join(sorted([str(v) for v in vals])) or "Region"
            else:
                base_name = "Region"

        if base_name in name_counts:
            name_counts[base_name] += 1
            cluster_names[label] = f"{base_name}{name_counts[base_name]}"
        else:
            name_counts[base_name] = 1
            cluster_names[label] = f"{base_name}1"

    return cluster_names


def run_clustering(
    selected_bas,
    grouping_column,
    target_regions,
    no_cluster_groups,
    auto_optimize=False,
    min_regions=None,
    max_regions=None,
    method="hierarchical-sum",
):
    """
    Run the clustering algorithm.

    Returns a tuple of (model_regions, region_aggregations, error_message, info)
    where info is a dict with optional metadata like chosen_n and modularity.
    """
    try:
        info = {}

        # Filter hierarchy to selected BAs
        hierarchy = state.hierarchy_df[
            state.hierarchy_df["ba"].isin(selected_bas)
        ].copy()

        if len(hierarchy) == 0:
            return None, None, "No valid BAs selected", info

        # Identify BAs to keep unclustered
        unclustered_bas = set()
        if no_cluster_groups:
            for group in no_cluster_groups:
                group_bas = set(hierarchy[hierarchy[grouping_column] == group]["ba"])
                unclustered_bas.update(group_bas)

        # BAs to cluster
        cluster_bas = selected_bas - unclustered_bas

        if len(cluster_bas) < 2:
            # Only unclustered BAs - use state abbreviation naming
            region_aggregations = {}
            name_counts = {}

            for ba in sorted(selected_bas):
                ba_row = state.hierarchy_df[state.hierarchy_df["ba"] == ba]
                if not ba_row.empty:
                    st = ba_row.iloc[0]["st"]
                    base_name = st
                else:
                    base_name = ba

                if base_name in name_counts:
                    name_counts[base_name] += 1
                    region_name = f"{base_name}{name_counts[base_name]}"
                else:
                    name_counts[base_name] = 1
                    region_name = f"{base_name}1"

                region_aggregations[region_name] = [ba]

            model_regions = sorted(region_aggregations.keys())
            return model_regions, region_aggregations, None, info

        # Get regional groups
        groups = get_regional_groups(hierarchy, grouping_column, cluster_bas)

        # Determine number of unclustered regions
        num_unclustered = len(unclustered_bas)

        if auto_optimize and min_regions is not None and max_regions is not None:
            # Auto-optimize mode: find best number of clusters
            actual_min = max(1, min_regions - num_unclustered)
            actual_max = max(1, max_regions - num_unclustered)
            actual_max = min(actual_max, len(cluster_bas))
            actual_min = min(actual_min, actual_max)

            clusters, chosen_n, modularity, all_scores, optimal_clusters = (
                find_optimal_clusters(
                    hierarchy,
                    state.transmission_df,
                    cluster_bas,
                    grouping_column,
                    actual_min,
                    actual_max,
                )
            )

            info["chosen_n"] = chosen_n + num_unclustered
            info["modularity"] = modularity
            info["all_scores"] = all_scores
            info["optimal_n"] = len(optimal_clusters) + num_unclustered
            info["optimal_clusters"] = optimal_clusters
        else:
            # Fixed target mode
            actual_target = max(1, target_regions - num_unclustered)
            actual_target = min(actual_target, len(cluster_bas))

            if method == "louvain":
                clusters, _, modularity_val, _, _ = find_optimal_clusters(
                    hierarchy,
                    state.transmission_df,
                    cluster_bas,
                    grouping_column,
                    actual_target,  # min
                    actual_target,  # max
                )
                # find_optimal_clusters calculates modularity, but we'll recalculate it below
                # to be consistent with other methods.
            else:
                # Run hierarchical clustering that respects grouping column boundaries
                clusters = hierarchical_cluster(
                    hierarchy,
                    state.transmission_df,
                    cluster_bas,
                    grouping_column,
                    actual_target,
                    method=method,
                )

            # Calculate modularity for info
            graph = build_transmission_graph(state.transmission_df, cluster_bas)
            modularity = calculate_modularity(graph, clusters)
            info["modularity"] = modularity

        # Generate names
        cluster_names = generate_cluster_names(clusters, groups)

        # Build output
        region_aggregations = {}
        name_counts = {}  # Track name counts from cluster names

        # First, collect all base names used in cluster names to track counts
        for label, name in cluster_names.items():
            # Extract base name (remove trailing digits)
            base = name.rstrip("0123456789")
            num_str = name[len(base) :]
            if num_str:
                num = int(num_str)
                name_counts[base] = max(name_counts.get(base, 0), num)

        for label, nodes in clusters.items():
            name = cluster_names[label]
            region_aggregations[name] = sorted(nodes)

        # If we have optimal cluster info (from auto-optimize splitting), map it to region names
        if "optimal_clusters" in info and len(info["optimal_clusters"]) < len(clusters):
            optimal_combinations = []

            # Map each BA to its final region name
            ba_to_final_region = {}
            for region_name, bas in region_aggregations.items():
                for ba in bas:
                    ba_to_final_region[ba] = region_name

            # Check each optimal cluster
            for _, opt_nodes in info["optimal_clusters"].items():
                # Find which final regions are contained in this optimal cluster
                contained_regions = set()
                for ba in opt_nodes:
                    if ba in ba_to_final_region:
                        contained_regions.add(ba_to_final_region[ba])

                # If more than one final region is in this optimal cluster, they would have been combined
                if len(contained_regions) > 1:
                    optimal_combinations.append(sorted(list(contained_regions)))

            if optimal_combinations:
                info["optimal_combinations"] = optimal_combinations

            # Clean up large objects from info
            del info["optimal_clusters"]

        # Add unclustered BAs with state abbreviation naming
        for ba in sorted(unclustered_bas):
            # Get state for this BA
            ba_row = state.hierarchy_df[state.hierarchy_df["ba"] == ba]
            if not ba_row.empty:
                st = ba_row.iloc[0]["st"]
                base_name = st
            else:
                base_name = ba

            # Add counter
            if base_name in name_counts:
                name_counts[base_name] += 1
                region_name = f"{base_name}{name_counts[base_name]}"
            else:
                name_counts[base_name] = 1
                region_name = f"{base_name}1"

            region_aggregations[region_name] = [ba]

        model_regions = sorted(region_aggregations.keys())

        return model_regions, region_aggregations, None, info

    except Exception as e:
        return None, None, str(e), {}


def generate_yaml(model_regions, region_aggregations):
    """Generate YAML output."""
    output = {
        "model_regions": model_regions,
        "region_aggregations": region_aggregations,
    }
    return yaml.dump(output, default_flow_style=False, sort_keys=False)


# ============================================================================
# Plant Clustering
# ============================================================================


# Lightweight technology grouping for clustering heuristics


ALWAYS_ONE_TECHS = {
    "Conventional Hydroelectric",
    "Run of River Hydroelectric",
    "Solar Photovoltaic",
    "Onshore Wind Turbine",
    "Offshore Wind Turbine",
    "Batteries",
    "Hydroelectric Pumped Storage",
}


# Default grouping and omit behavior for plant clustering UI
DEFAULT_TECH_GROUPS = {
    "Biomass": {
        "Biomass",
        "Landfill Gas",
        "Municipal Solid Waste",
        "Other Waste Biomass",
        "Wood/Wood Waste Biomass",
    },
    "Other_peaker": {
        "Natural Gas Internal Combustion Engine",
        "Petroleum Liquids",
    },
}

DEFAULT_OMIT_TOKENS = {
    "all other",
    "flywheel",
    "solar thermal with energy storage",
    "solar thermal without energy storage",
}


def clone_group_map(group_map):
    """Shallow clone of group map with set copies."""
    return {name: set(values) for name, values in group_map.items()}


def normalize_technology(tech_name, omit_tokens=None):
    """Map technology names to canonical groups; return None to exclude."""
    if not isinstance(tech_name, str):
        return None

    name = tech_name.lower()

    default_omit = ["solar thermal", "all other", "flywheel"]
    tokens = [
        t.lower() for t in (omit_tokens if omit_tokens is not None else default_omit)
    ]

    if any(token in name for token in tokens):
        return None

    # Specific matches first
    if "pumped storage" in name:
        return "Hydroelectric Pumped Storage"
    if "run of river" in name:
        return "Run of River Hydroelectric"
    if "conventional hydro" in name or "hydroelectric" in name:
        return "Conventional Hydroelectric"
    if "landfill gas" in name:
        return "Landfill Gas"
    if "municipal solid waste" in name:
        return "Municipal Solid Waste"
    if "other waste biomass" in name:
        return "Other Waste Biomass"
    if "wood" in name:
        return "Wood/Wood Waste Biomass"
    if "biomass" in name:
        return "Biomass"
    if "geothermal" in name:
        return "Geothermal"
    if "nuclear" in name:
        return "Nuclear"
    if "combined cycle" in name:
        return "Natural Gas Fired Combined Cycle"
    if "combustion turbine" in name:
        return "Natural Gas Fired Combustion Turbine"
    if "steam turbine" in name:
        return "Natural Gas Steam Turbine"
    if "internal combustion" in name:
        return "Natural Gas Internal Combustion Engine"
    if "steam coal" in name or "coal" in name:
        return "Conventional Steam Coal"
    if "photovoltaic" in name:
        return "Solar Photovoltaic"
    if "offshore wind" in name:
        return "Offshore Wind Turbine"
    if "wind" in name:
        return "Onshore Wind Turbine"
    # Ensure solar thermal variants are classified before generic storage/battery
    if "solar thermal with energy storage" in name:
        return "Solar Thermal with Energy Storage"
    if "solar thermal without energy storage" in name:
        return "Solar Thermal without Energy Storage"
    if "battery" in name or "storage" in name:
        return "Batteries"
    if "petroleum" in name or "oil" in name:
        return "Petroleum Liquids"

    return tech_name


def weighted_quantile(values, quantile, weights):
    """Compute weighted quantile; expects numpy arrays."""
    sorter = np.argsort(values)
    v_sorted = values[sorter]
    w_sorted = weights[sorter]
    cum_weights = np.cumsum(w_sorted)
    cutoff = quantile * cum_weights[-1]
    return v_sorted[np.searchsorted(cum_weights, cutoff)]


def weighted_iqr(values, weights):
    """Weighted interquartile range (Q3 - Q1)."""
    if len(values) == 0:
        return 0.0
    return float(
        weighted_quantile(values, 0.75, weights)
        - weighted_quantile(values, 0.25, weights)
    )


def inertia_single_cluster(features, weights=None):
    """Inertia for a single cluster (k=1)."""
    center = features.mean(axis=0)
    sq_dists = np.sum((features - center) ** 2, axis=1)
    if weights is not None:
        return float((sq_dists * weights).sum())
    return float(sq_dists.sum())


def build_ba_to_model_region_map():
    """Return BA -> model region lookup using current clustering (or identity)."""
    if state.region_aggregations:
        mapping = {}
        for region_name, bas in state.region_aggregations.items():
            for ba in bas:
                mapping[ba] = region_name
        # Keep any unseen BAs mapped to themselves so plants are not dropped
        for ba in state.all_bas:
            mapping.setdefault(ba, ba)
        return mapping

    # Fallback to identity mapping (each BA is its own model region)
    return {ba: ba for ba in state.all_bas}


def apply_default_grouping(tech_group, enabled=True, group_map=None):
    """Collapse technologies into groups using provided map when enabled."""
    if not enabled:
        return tech_group
    mapping = group_map if group_map is not None else DEFAULT_TECH_GROUPS
    for group_name, members in mapping.items():
        if tech_group in members:
            return group_name
    return tech_group


def prepare_plants_dataframe(
    *,
    group_enabled=True,
    omit_tokens=None,
    group_map=None,
):
    """Merge plant data with BA mapping and apply technology grouping."""
    if state.plants_df is None or state.plant_region_map is None:
        raise Exception("Plant data not loaded yet")

    ba_to_region = build_ba_to_model_region_map()

    df = state.plants_df.merge(state.plant_region_map, on="plant_id", how="left")

    # Map BA regions to model regions
    df["model_region"] = df["region"].map(ba_to_region)
    df = df.dropna(subset=["model_region"])

    # Normalize technologies and optionally group/omit
    df["tech_group"] = df["technology"].apply(
        lambda t: normalize_technology(t, omit_tokens=omit_tokens)
    )
    df = df[df["tech_group"].notna()].copy()
    df["tech_group"] = df["tech_group"].apply(
        lambda t: apply_default_grouping(t, enabled=group_enabled, group_map=group_map)
    )

    # Ensure numeric columns are present
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce").fillna(0)
    df["heat_rate_mmbtu_mwh"] = pd.to_numeric(
        df["heat_rate_mmbtu_mwh"], errors="coerce"
    )
    df["fom_per_mwyr"] = pd.to_numeric(df["fom_per_mwyr"], errors="coerce")

    return df


def suggest_plant_clusters(
    budget=200,
    cap_threshold=1500.0,
    hr_iqr_threshold=0.8,
    *,
    group_enabled=True,
    omit_tokens=None,
    group_map=None,
):
    """Suggest cluster counts per (model_region, tech_group) under a hard budget."""
    df = prepare_plants_dataframe(
        group_enabled=group_enabled,
        omit_tokens=omit_tokens,
        group_map=group_map,
    )

    groups = []
    raw_tech_map = {}  # normalized -> set(raw tech names)

    for (model_region, tech_group), sub in df.groupby(["model_region", "tech_group"]):
        n_units = len(sub)
        total_cap = float(sub["capacity_mw"].sum())

        # Features
        heat_rate = sub["heat_rate_mmbtu_mwh"].to_numpy()
        fom = sub["fom_per_mwyr"].to_numpy()
        weights = sub["capacity_mw"].replace(0, 1e-6).to_numpy()

        # Fill missing values with medians to keep clustering stable
        hr_median = np.nanmedian(heat_rate) if not np.all(np.isnan(heat_rate)) else 0.0
        fom_median = np.nanmedian(fom) if not np.all(np.isnan(fom)) else 0.0
        hr_filled = np.where(np.isnan(heat_rate), hr_median, heat_rate)
        fom_filled = np.where(np.isnan(fom), fom_median, fom)
        features = np.column_stack([hr_filled, fom_filled])
        standardized = standardize_features(features)

        hr_iqr_val = weighted_iqr(hr_filled, weights)

        # K-means improvement for k=2
        improvement2 = 0.0
        if n_units >= 2:
            inertia1 = inertia_single_cluster(standardized, weights)
            inertia2, _, _ = run_kmeans_simple(standardized, 2, weights=weights)
            if inertia1 > 0:
                improvement2 = max(0.0, (inertia1 - inertia2) / inertia1)

        desired = 1
        # Only suggest splitting if there is actual variation in efficiency/performance
        has_variance = hr_iqr_val > 0.01 or improvement2 > 0.01

        if (
            n_units >= 2
            and has_variance
            and (
                total_cap >= cap_threshold
                or hr_iqr_val >= hr_iqr_threshold
                or improvement2 >= 0.15
            )
        ):
            desired = min(2, n_units)

        if n_units >= 3 and improvement2 >= 0.3:
            desired = min(3, n_units)

        # Allow up to 5 clusters if budget permits and variance is high
        if n_units >= 4 and improvement2 >= 0.4:
            desired = min(4, n_units)

        if n_units >= 5 and improvement2 >= 0.5:
            desired = min(5, n_units)

        # Priority scales with variance, so identical units (IQR=0) get low priority
        priority = total_cap * (hr_iqr_val + improvement2)

        raw_tech_map.setdefault(tech_group, set()).update(
            sub["technology"].dropna().unique()
        )

        groups.append(
            {
                "model_region": model_region,
                "tech_group": tech_group,
                "n_units": n_units,
                "total_capacity": total_cap,
                "hr_iqr": hr_iqr_val,
                "improvement2": improvement2,
                "desired": desired,
                "priority": priority,
            }
        )

    # Enforce single-cluster techs
    for g in groups:
        if g["tech_group"] in ALWAYS_ONE_TECHS:
            g["desired"] = 1
            g["num_clusters"] = 1
            g["alt_num_clusters"] = 1

    if not groups:
        raise Exception("No plants found for current model regions")

    # Allocate clusters within budget
    num_groups = len(groups)
    min_possible = num_groups  # at least 1 per group
    effective_budget = max(budget, min_possible)

    for g in groups:
        g["num_clusters"] = 1

    remaining = effective_budget - num_groups

    for g in sorted(groups, key=lambda x: x["priority"], reverse=True):
        if g["tech_group"] in ALWAYS_ONE_TECHS:
            continue
        extra_needed = max(0, g["desired"] - g["num_clusters"])
        if remaining <= 0 or extra_needed == 0:
            continue
        extra = min(extra_needed, remaining)
        g["num_clusters"] += extra
        remaining -= extra

    # Alt clusters are a gentle +1 where possible
    for g in groups:
        if g["tech_group"] in ALWAYS_ONE_TECHS:
            g["alt_num_clusters"] = 1
        else:
            g["alt_num_clusters"] = min(g["num_clusters"] + 1, g["n_units"])

    # Compute defaults and overrides
    defaults = {}
    tech_to_counts = {}
    tech_to_alt = {}
    for g in groups:
        tech_to_counts.setdefault(g["tech_group"], []).append(g["num_clusters"])
        tech_to_alt.setdefault(g["tech_group"], []).append(g["alt_num_clusters"])

    for tech, counts in tech_to_counts.items():
        min_count = min(counts) if counts else 1
        default_num = min_count if min_count > 1 else 1
        alt_counts = tech_to_alt.get(tech, [])
        default_alt = min(alt_counts) if alt_counts else default_num
        default_alt = max(default_alt, default_num)
        defaults[tech] = int(default_num)

    overrides = {}
    total_clusters = 0
    for g in groups:
        total_clusters += g["num_clusters"]
        d = defaults[g["tech_group"]]
        if g["num_clusters"] != d:
            overrides.setdefault(g["model_region"], {})[g["tech_group"]] = int(
                g["num_clusters"]
            )

    # Top candidates for further splitting (where desired > assigned)
    candidates = [
        g for g in groups if g["num_clusters"] < g["desired"] and g["desired"] > 1
    ]
    candidates = sorted(candidates, key=lambda x: x["priority"], reverse=True)[:10]

    state.plant_candidates = candidates

    if group_enabled:
        active_map = group_map if group_map is not None else DEFAULT_TECH_GROUPS
        tech_groups = {
            name: sorted(list(members)) for name, members in active_map.items()
        }
    else:
        tech_groups = {
            tech: sorted(list(raw_tech_map.get(tech, []))) for tech in sorted(defaults)
        }

    overrides_sorted = {
        region: {tech: int(val) for tech, val in sorted(tech_map.items())}
        for region, tech_map in sorted(overrides.items())
    }

    group_flag = group_enabled and any(len(v) > 0 for v in tech_groups.values())

    output = {
        "num_clusters": {tech: int(val) for tech, val in sorted(defaults.items())},
        "group_technologies": bool(group_flag),
        "tech_groups": tech_groups,
        "alt_num_clusters": overrides_sorted,
    }

    yaml_str = yaml.dump(output, default_flow_style=False, sort_keys=False)

    return yaml_str, total_clusters, effective_budget


# ============================================================================
# Event Handlers
# ============================================================================


def on_run_clustering(event):
    """Handle Run Clustering button click."""
    set_status("Running clustering...", "info")

    grouping_col = document.getElementById("groupingColumn").value
    method = document.getElementById("clusteringMethod").value

    # Check if auto-optimize mode is enabled
    auto_optimize_el = document.getElementById("autoOptimize")
    auto_optimize = auto_optimize_el.checked if auto_optimize_el else False

    if auto_optimize:
        min_regions = int(document.getElementById("minRegions").value)
        max_regions = int(document.getElementById("maxRegions").value)
        target_regions = None  # Will be determined by optimization
    else:
        target_regions = int(document.getElementById("targetRegions").value)
        min_regions = None
        max_regions = None

    # Get no-cluster selections
    no_cluster_groups = []
    checkboxes = document.querySelectorAll('input[name="noCluster"]:checked')
    for cb in checkboxes:
        no_cluster_groups.append(cb.value)

    # Run clustering
    model_regions, region_aggregations, error, info = run_clustering(
        state.selected_bas,
        grouping_col,
        target_regions if not auto_optimize else min_regions,  # Use min as fallback
        no_cluster_groups,
        auto_optimize=auto_optimize,
        min_regions=min_regions,
        max_regions=max_regions,
        method=method,
    )

    if error:
        set_status(f"Error: {error}", "error")
        return

    # Generate YAML
    yaml_output = generate_yaml(model_regions, region_aggregations)

    # Display
    yaml_el = document.getElementById("yamlOut")
    if yaml_el:
        yaml_el.value = yaml_output

    # Build status message
    num_regions = len(model_regions)
    modularity = info.get("modularity", 0)

    if auto_optimize:
        chosen_n = info.get("chosen_n", num_regions)
        msg = f"Clustering complete! {num_regions} regions (optimal from {min_regions}-{max_regions}). Modularity: {modularity:.3f}"

        # Add info about optimal combinations if we forced splits
        if "optimal_combinations" in info:
            optimal_n = info.get("optimal_n", "unknown")
            msg += f"\n\nOptimal number of clusters was {optimal_n}. The following regions would be combined in the optimal solution:\n"

            # Format the combinations
            combo_strs = []
            for combo in info["optimal_combinations"]:
                combo_strs.append(f"• {', '.join(combo)}")

            msg += "\n".join(combo_strs)

        set_status(msg, "success")
    else:
        if num_regions > target_regions:
            set_status(
                f"Warning: Created {num_regions} regions, which is more than the target of {target_regions}. "
                f"This can happen when 'unclustered' groups or disconnected BAs exceed the target. "
                f"Modularity: {modularity:.3f}",
                "error",
            )
        else:
            set_status(
                f"Clustering complete! {num_regions} regions created. Modularity: {modularity:.3f}",
                "success",
            )

    # Store region aggregations for transmission line drawing
    state.region_aggregations = region_aggregations

    # Update map colors to show clusters
    update_map_cluster_colors(region_aggregations)

    # Update tooltips to show model region names
    update_tooltips()

    # Update transmission lines if enabled
    update_transmission_lines()

    # Refresh plant cluster defaults based on new region mapping
    update_default_cluster_budget()


def update_map_cluster_colors(region_aggregations):
    """Update map to show cluster assignments with group outline colors preserved."""
    # Build BA -> cluster color mapping and BA -> region name mapping
    state.cluster_colors = {}
    state.ba_to_region = {}
    state.is_clustered = True

    for i, (cluster_name, bas) in enumerate(region_aggregations.items()):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        for ba in bas:
            state.cluster_colors[ba] = color
            state.ba_to_region[ba] = cluster_name

    # Update layer styles - keep group outline color
    for ba_id, layer in state.ba_layers.items():
        if ba_id in state.selected_bas:
            fill_color = state.cluster_colors.get(ba_id, "#999999")
            outline_color = get_outline_color(ba_id)
            layer.setStyle(
                to_js(
                    {
                        "fillColor": fill_color,
                        "fillOpacity": 0.7,
                        "color": outline_color,
                        "weight": 3,
                    }
                )
            )


# ============================================================================
# Transmission Lines Visualization
# ============================================================================


def get_line_weight(capacity_mw):
    """Calculate line weight based on transmission capacity."""
    # Scale: 1-8 pixels based on capacity
    # Typical range is ~100 MW to ~15000 MW
    min_weight = 1
    max_weight = 8
    min_cap = 100
    max_cap = 12000

    # Clamp and scale
    clamped = max(min_cap, min(max_cap, capacity_mw))
    normalized = (clamped - min_cap) / (max_cap - min_cap)
    return min_weight + normalized * (max_weight - min_weight)


def draw_ba_transmission_lines():
    """Draw transmission lines between BA centroids."""
    if state.transmission_df is None or not state.ba_centroids:
        return

    lines = []

    # Only show lines for selected BAs (or all if none selected)
    relevant_bas = state.selected_bas if state.selected_bas else state.all_bas

    for _, row in state.transmission_df.iterrows():
        ba_from = row["region_from"]
        ba_to = row["region_to"]
        capacity = row["firm_ttc_mw"]

        # Only draw if both BAs are in the relevant set and have centroids
        if ba_from in relevant_bas and ba_to in relevant_bas:
            if ba_from in state.ba_centroids and ba_to in state.ba_centroids:
                lat1, lng1 = state.ba_centroids[ba_from]
                lat2, lng2 = state.ba_centroids[ba_to]

                weight = get_line_weight(capacity)

                # Create polyline
                line = L.polyline(
                    to_js([[lat1, lng1], [lat2, lng2]]),
                    to_js(
                        {
                            "color": "#ff6600",
                            "weight": weight,
                            "opacity": 0.6,
                            "pane": "transmissionPane",
                        }
                    ),
                )
                line.bindTooltip(f"{ba_from} ↔ {ba_to}<br>{capacity:,.0f} MW")
                lines.append(line)

    return lines


def draw_region_transmission_lines(region_aggregations):
    """Draw transmission lines showing capacity between model regions."""
    if state.transmission_df is None or not state.ba_centroids:
        return

    # Build BA -> region mapping
    ba_to_region = {}
    for region_name, bas in region_aggregations.items():
        for ba in bas:
            ba_to_region[ba] = region_name

    # Calculate region centroids (average of BA centroids)
    region_centroids = {}
    for region_name, bas in region_aggregations.items():
        lats = []
        lngs = []
        for ba in bas:
            if ba in state.ba_centroids:
                lat, lng = state.ba_centroids[ba]
                lats.append(lat)
                lngs.append(lng)
        if lats:
            region_centroids[region_name] = (
                sum(lats) / len(lats),
                sum(lngs) / len(lngs),
            )

    # Aggregate transmission capacity between regions
    region_capacity = {}  # (region1, region2) -> total capacity

    for _, row in state.transmission_df.iterrows():
        ba_from = row["region_from"]
        ba_to = row["region_to"]
        capacity = row["firm_ttc_mw"]

        if ba_from in ba_to_region and ba_to in ba_to_region:
            region_from = ba_to_region[ba_from]
            region_to = ba_to_region[ba_to]

            # Only count inter-region connections
            if region_from != region_to:
                # Use sorted tuple as key for bidirectional
                key = tuple(sorted([region_from, region_to]))
                region_capacity[key] = region_capacity.get(key, 0) + capacity

    lines = []

    for (region1, region2), capacity in region_capacity.items():
        if region1 in region_centroids and region2 in region_centroids:
            lat1, lng1 = region_centroids[region1]
            lat2, lng2 = region_centroids[region2]

            weight = get_line_weight(capacity)

            line = L.polyline(
                to_js([[lat1, lng1], [lat2, lng2]]),
                to_js(
                    {
                        "color": "#cc0000",
                        "weight": weight,
                        "opacity": 0.8,
                        "pane": "transmissionPane",
                    }
                ),
            )
            line.bindTooltip(f"{region1} ↔ {region2}<br>{capacity:,.0f} MW")
            lines.append(line)

    return lines


def update_transmission_lines():
    """Update transmission lines based on current state."""
    # Remove existing layer if present
    if state.transmission_lines_layer is not None:
        state.map.removeLayer(state.transmission_lines_layer)
        state.transmission_lines_layer = None

    if not state.show_transmission_lines:
        return

    # Decide which type of lines to draw
    if state.is_clustered and state.region_aggregations:
        lines = draw_region_transmission_lines(state.region_aggregations)
    else:
        lines = draw_ba_transmission_lines()

    if lines:
        # Create layer group and add to map
        state.transmission_lines_layer = L.layerGroup(to_js(lines))
        state.transmission_lines_layer.addTo(state.map)


def on_toggle_transmission_lines(event):
    """Handle toggle of transmission lines checkbox."""
    checkbox = document.getElementById("showTransmissionLines")
    state.show_transmission_lines = checkbox.checked
    update_transmission_lines()


def on_select_all(event):
    """Select all BAs."""
    for ba_id, layer in state.ba_layers.items():
        if ba_id not in state.selected_bas:
            state.selected_bas.add(ba_id)
            outline_color = get_outline_color(ba_id)
            layer.setStyle(
                to_js(
                    {
                        "fillColor": "#2196F3",
                        "fillOpacity": 0.6,
                        "color": outline_color,
                        "weight": 3,
                    }
                )
            )
    update_selected_display()


def on_clear_selection(event):
    """Clear all selections."""
    # Reset cluster state
    state.cluster_colors = {}
    state.ba_to_region = {}
    state.is_clustered = False
    state.region_aggregations = None

    for ba_id, layer in state.ba_layers.items():
        if ba_id in state.selected_bas:
            outline_color = get_outline_color(ba_id)
            fill_color = get_fill_color(ba_id)
            layer.setStyle(
                to_js(
                    {
                        "fillColor": fill_color,
                        "fillOpacity": 0.5,
                        "color": outline_color,
                        "weight": 2,
                    }
                )
            )
    state.selected_bas.clear()
    update_selected_display()

    # Update tooltips to remove region names
    update_tooltips()

    # Update transmission lines (will show BA lines if toggle is on)
    update_transmission_lines()

    # Reset plant cluster defaults to BA-level mapping
    update_default_cluster_budget()


# ============================================================================
# Box Selection Mode
# ============================================================================


def set_selection_mode(mode):
    """Set selection mode: 'click' or 'box'."""
    state.box_select_mode = mode == "box"

    # Update button styles
    click_btn = document.getElementById("clickModeBtn")
    box_btn = document.getElementById("boxModeBtn")
    hint = document.getElementById("selectionHint")
    map_el = document.getElementById("map")

    if click_btn and box_btn:
        if state.box_select_mode:
            click_btn.classList.remove("active")
            box_btn.classList.add("active")
            if hint:
                hint.textContent = "Drag on the map to select multiple BAs"
            if map_el:
                map_el.classList.add("box-select-mode")
            # Disable map dragging
            state.map.dragging.disable()
        else:
            click_btn.classList.add("active")
            box_btn.classList.remove("active")
            if hint:
                hint.textContent = "Click on BAs to toggle selection"
            if map_el:
                map_el.classList.remove("box-select-mode")
            # Enable map dragging
            state.map.dragging.enable()


def on_click_mode(event):
    """Switch to click selection mode."""
    set_selection_mode("click")


def on_box_mode(event):
    """Switch to box selection mode."""
    set_selection_mode("box")


def on_map_mousedown(e):
    """Handle mousedown for box selection."""
    if not state.box_select_mode:
        return

    state.box_start = e.latlng

    # Create visual selection box
    box = document.createElement("div")
    box.id = "selectionBox"
    box.className = "selection-box"

    # Position at mouse
    container_point = state.map.latLngToContainerPoint(e.latlng)
    box.style.left = f"{container_point.x}px"
    box.style.top = f"{container_point.y}px"
    box.style.width = "0px"
    box.style.height = "0px"

    map_container = document.getElementById("map")
    map_container.appendChild(box)


def on_map_mousemove(e):
    """Handle mousemove for box selection."""
    if not state.box_select_mode or not state.box_start:
        return

    box = document.getElementById("selectionBox")
    if not box:
        return

    # Get start and current container points
    start_point = state.map.latLngToContainerPoint(state.box_start)
    current_point = state.map.latLngToContainerPoint(e.latlng)

    # Calculate box dimensions
    min_x = min(start_point.x, current_point.x)
    min_y = min(start_point.y, current_point.y)
    width = abs(current_point.x - start_point.x)
    height = abs(current_point.y - start_point.y)

    # Update box position and size
    box.style.left = f"{min_x}px"
    box.style.top = f"{min_y}px"
    box.style.width = f"{width}px"
    box.style.height = f"{height}px"


def on_map_mouseup(e):
    """Handle mouseup for box selection - select BAs in box."""
    if not state.box_select_mode or not state.box_start:
        return

    # Remove visual box
    box = document.getElementById("selectionBox")
    if box:
        box.remove()

    # Create bounds from start and end points
    end_latlng = e.latlng
    bounds = L.latLngBounds(state.box_start, end_latlng)

    # Find all BAs whose centroid is within bounds
    selected_count = 0
    for ba_id, (lat, lng) in state.ba_centroids.items():
        point = L.latLng(lat, lng)
        if bounds.contains(point):
            # Add to selection if not already selected
            if ba_id not in state.selected_bas:
                state.selected_bas.add(ba_id)
                layer = state.ba_layers.get(ba_id)
                if layer:
                    outline_color = get_outline_color(ba_id)
                    layer.setStyle(
                        to_js(
                            {
                                "fillColor": "#2196F3",
                                "fillOpacity": 0.6,
                                "color": outline_color,
                                "weight": 3,
                            }
                        )
                    )
                selected_count += 1

    state.box_start = None
    update_selected_display()

    if selected_count > 0:
        set_status(f"Added {selected_count} BAs to selection.", "info")


def on_copy_yaml(event):
    """Copy YAML to clipboard."""
    yaml_el = document.getElementById("yamlOut")
    if yaml_el and yaml_el.value:
        window.navigator.clipboard.writeText(yaml_el.value)
        set_status("YAML copied to clipboard!", "success")


def on_download_yaml(event):
    """Download YAML file."""
    yaml_el = document.getElementById("yamlOut")
    if not yaml_el or not yaml_el.value:
        set_status("No YAML to download. Run clustering first.", "error")
        return

    # Create blob and download
    blob = window.Blob.new([yaml_el.value], to_js({"type": "text/yaml"}))
    url = window.URL.createObjectURL(blob)

    a = document.createElement("a")
    a.href = url
    a.download = "region_aggregations.yml"
    a.click()

    window.URL.revokeObjectURL(url)
    set_status("YAML downloaded!", "success")


def render_plant_candidates():
    """Render the top plant split candidates list."""
    container = document.getElementById("plantCandidateList")
    if not container:
        return

    if not state.plant_candidates:
        container.innerHTML = (
            "<em>No additional splits recommended within the current budget.</em>"
        )
        return

    parts = []
    for g in state.plant_candidates:
        parts.append(
            f"<div class='candidate-item'><strong>{g['model_region']}</strong> — {g['tech_group']} (desired {g['desired']}, assigned {g['num_clusters']}; {g['total_capacity']:.0f} MW, HR IQR {g['hr_iqr']:.2f})</div>"
        )
    container.innerHTML = "".join(parts)


# --------------------------------------------------------------------------
# Interactive tech grouping UI helpers
# --------------------------------------------------------------------------


def get_normalized_techs(omit_tokens=None):
    """Return sorted list of normalized technologies from the plant data."""
    if state.plants_df is None:
        return []

    techs = set()
    for tech in state.plants_df.get("technology", []):
        normalized = normalize_technology(tech, omit_tokens=omit_tokens)
        if normalized:
            techs.add(normalized)
    return sorted(techs)


def get_selected_omit_tokens():
    """Return list of technologies currently marked for omission."""
    # Prefer state cache populated by the dual-list UI
    if state.omit_selected:
        return sorted(state.omit_selected)

    selected_el = document.getElementById("omitSelectedList")
    tokens = []
    if selected_el and hasattr(selected_el, "options"):
        tokens = [opt.value for opt in selected_el.options]

    if not tokens:
        tokens = sorted(DEFAULT_OMIT_TOKENS)

    return tokens


def calculate_min_clusters_and_default():
    """Return (min_clusters, default_clusters) based on current tech grouping/omits."""
    group_checkbox = document.getElementById("groupTechDefault")
    group_enabled = group_checkbox.checked if group_checkbox else True
    omit_tokens = get_selected_omit_tokens()

    active_group_map = None
    if group_enabled:
        active_group_map = (
            clone_group_map(state.custom_tech_groups)
            if state.custom_tech_groups
            else clone_group_map(DEFAULT_TECH_GROUPS)
        )

    df = prepare_plants_dataframe(
        group_enabled=group_enabled,
        omit_tokens=omit_tokens,
        group_map=active_group_map,
    )

    # One cluster per (model_region, tech_group) is the floor
    min_clusters = int(df.groupby(["model_region", "tech_group"]).ngroups)
    default_clusters = max(1, math.ceil(min_clusters * 1.15))
    return min_clusters, default_clusters


def update_default_cluster_budget(event=None):
    """Compute minimum clusters and set the default budget to +15%."""
    budget_input = document.getElementById("plantBudget")
    helper_text = document.getElementById("plantBudgetInfo")
    try:
        min_clusters, default_clusters = calculate_min_clusters_and_default()
    except Exception:
        # Skip updates if data not ready
        return

    if budget_input:
        budget_input.value = str(default_clusters)

    if helper_text:
        helper_text.textContent = f"Minimum clusters: {min_clusters}. Default set to {default_clusters} (+15%)."


def ensure_current_group():
    """Ensure the currently selected group exists."""
    if state.current_group and state.current_group in state.custom_tech_groups:
        return
    if state.custom_tech_groups:
        state.current_group = sorted(state.custom_tech_groups.keys())[0]
    else:
        state.current_group = None


def reset_custom_groups(omit_tokens=None):
    """Reset custom grouping to defaults and recompute available tech list."""
    state.custom_tech_groups = clone_group_map(DEFAULT_TECH_GROUPS)
    state.current_group = sorted(state.custom_tech_groups.keys())[0]
    normalized = set(get_normalized_techs(omit_tokens=omit_tokens))
    grouped = set()
    for members in state.custom_tech_groups.values():
        grouped.update(members)
    state.available_techs = normalized - grouped
    render_group_editor()
    update_default_cluster_budget()


def clear_custom_groups(omit_tokens=None):
    """Clear all groupings; make all techs available."""
    state.custom_tech_groups = {}
    state.current_group = None
    state.available_techs = set(get_normalized_techs(omit_tokens=omit_tokens))
    render_group_editor()
    update_default_cluster_budget()


def render_group_editor():
    """Render dual-list grouping UI (available vs selected for current group)."""
    group_select = document.getElementById("groupSelectDual")
    avail_list = document.getElementById("availableList")
    group_list = document.getElementById("groupList")
    empty_notice = document.getElementById("groupEmptyNotice")

    ensure_current_group()

    # Populate group dropdown
    if group_select:
        group_select.innerHTML = "".join(
            [
                f"<option value='{html.escape(name)}' {'selected' if name == state.current_group else ''}>{html.escape(name)}</option>"
                for name in sorted(state.custom_tech_groups.keys())
            ]
        )

    # Show empty notice when no groups
    if empty_notice:
        empty_notice.style.display = "block" if not state.custom_tech_groups else "none"

    # Available list
    if avail_list:
        avail_list.innerHTML = "".join(
            [
                f"<option value='{tech}'>{html.escape(tech)}</option>"
                for tech in sorted(state.available_techs)
            ]
        )

    # Current group list
    if group_list:
        members = (
            state.custom_tech_groups.get(state.current_group, set())
            if state.current_group
            else set()
        )
        group_list.innerHTML = "".join(
            [
                f"<option value='{tech}'>{html.escape(tech)}</option>"
                for tech in sorted(members)
            ]
        )


def render_omit_editor():
    """Render dual-list UI for selecting omitted technologies."""
    avail_el = document.getElementById("omitAvailableList")
    selected_el = document.getElementById("omitSelectedList")

    if state.plants_df is None or (avail_el is None and selected_el is None):
        return

    # Initialize omit sets if empty
    if not state.omit_selected and not state.omit_available:
        all_techs = set(get_normalized_techs(omit_tokens=[]))
        default_selected = {
            tech
            for tech in all_techs
            if any(tok in tech.lower() for tok in DEFAULT_OMIT_TOKENS)
        }
        state.omit_selected = default_selected
        state.omit_available = all_techs - default_selected

    if avail_el:
        avail_el.innerHTML = "".join(
            [
                f"<option value='{html.escape(tech)}'>{html.escape(tech)}</option>"
                for tech in sorted(state.omit_available)
            ]
        )

    if selected_el:
        selected_el.innerHTML = "".join(
            [
                f"<option value='{html.escape(tech)}'>{html.escape(tech)}</option>"
                for tech in sorted(state.omit_selected)
            ]
        )


def on_omit_move_to_selected(event):
    """Move technologies from available to omitted list."""
    avail_el = document.getElementById("omitAvailableList")
    if not avail_el:
        return
    chosen = [opt.value for opt in avail_el.selectedOptions]
    if not chosen:
        return
    state.omit_available -= set(chosen)
    state.omit_selected |= set(chosen)
    render_omit_editor()
    refresh_groups_for_omit_change()


def on_omit_move_to_available(event):
    """Move technologies from omitted back to available list."""
    selected_el = document.getElementById("omitSelectedList")
    if not selected_el:
        return
    chosen = [opt.value for opt in selected_el.selectedOptions]
    if not chosen:
        return
    state.omit_selected -= set(chosen)
    state.omit_available |= set(chosen)
    render_omit_editor()
    refresh_groups_for_omit_change()


def on_reset_omit_defaults(event=None):
    """Reset omitted technologies to defaults."""
    all_techs = set(get_normalized_techs(omit_tokens=[]))
    default_selected = {
        tech
        for tech in all_techs
        if any(tok in tech.lower() for tok in DEFAULT_OMIT_TOKENS)
    }
    state.omit_selected = default_selected
    state.omit_available = all_techs - default_selected
    render_omit_editor()
    refresh_groups_for_omit_change()


def on_add_group(event):
    name_input = document.getElementById("newGroupName")
    omit_tokens = get_selected_omit_tokens()
    if not name_input:
        return
    group_name = name_input.value.strip()
    if not group_name:
        return
    if group_name not in state.custom_tech_groups:
        state.custom_tech_groups[group_name] = set()
    state.current_group = group_name
    name_input.value = ""
    # Ensure available list is up to date
    state.available_techs = set(get_normalized_techs(omit_tokens=omit_tokens))
    for members in state.custom_tech_groups.values():
        state.available_techs -= members
    render_group_editor()
    update_default_cluster_budget()


def on_add_tech_to_group(event):
    avail_list = document.getElementById("availableList")
    if not avail_list:
        return
    ensure_current_group()
    if not state.current_group:
        return
    selected = [opt.value for opt in avail_list.selectedOptions]
    if not selected:
        return
    state.custom_tech_groups.setdefault(state.current_group, set()).update(selected)
    state.available_techs -= set(selected)
    render_group_editor()
    update_default_cluster_budget()


def on_remove_tech_from_group(event):
    group_list = document.getElementById("groupList")
    if not group_list:
        return
    ensure_current_group()
    if not state.current_group:
        return
    selected = [opt.value for opt in group_list.selectedOptions]
    if not selected:
        return
    for tech in selected:
        if tech in state.custom_tech_groups.get(state.current_group, set()):
            state.custom_tech_groups[state.current_group].remove(tech)
            state.available_techs.add(tech)
    render_group_editor()
    update_default_cluster_budget()


def on_group_change(event):
    select = document.getElementById("groupSelectDual")
    if select:
        state.current_group = select.value or None
    render_group_editor()


def on_reset_groups(event):
    omit_tokens = get_selected_omit_tokens()
    reset_custom_groups(omit_tokens=omit_tokens)


def on_clear_groups(event):
    omit_tokens = get_selected_omit_tokens()
    clear_custom_groups(omit_tokens=omit_tokens)


def refresh_groups_for_omit_change(event=None):
    """Recompute available techs when omit setting changes."""
    omit_tokens = get_selected_omit_tokens()
    normalized = set(get_normalized_techs(omit_tokens=omit_tokens))
    # Keep omit state in sync with current tech universe
    all_techs = set(get_normalized_techs(omit_tokens=[]))
    state.omit_selected = {t for t in state.omit_selected if t in all_techs}
    state.omit_available = all_techs - state.omit_selected
    render_omit_editor()
    # Keep existing assignments if still valid
    for group, members in list(state.custom_tech_groups.items()):
        state.custom_tech_groups[group] = {m for m in members if m in normalized}
    assigned = (
        set().union(*state.custom_tech_groups.values())
        if state.custom_tech_groups
        else set()
    )
    state.available_techs = normalized - assigned
    ensure_current_group()
    render_group_editor()
    update_default_cluster_budget()


def on_run_plant_clustering(event):
    """Handle plant clustering run."""
    try:
        budget_val = int(document.getElementById("plantBudget").value)
        cap_thresh = float(document.getElementById("capThreshold").value)
        hr_thresh = float(document.getElementById("hrThreshold").value)
        group_checkbox = document.getElementById("groupTechDefault")
        omit_tokens = get_selected_omit_tokens()
        group_enabled = group_checkbox.checked if group_checkbox else True

        active_group_map = None
        if group_enabled:
            # If user customized groups, prefer those; otherwise fall back to defaults
            active_group_map = (
                clone_group_map(state.custom_tech_groups)
                if state.custom_tech_groups
                else clone_group_map(DEFAULT_TECH_GROUPS)
            )

        yaml_str, total_clusters, effective_budget = suggest_plant_clusters(
            budget=budget_val,
            cap_threshold=cap_thresh,
            hr_iqr_threshold=hr_thresh,
            group_enabled=group_enabled,
            omit_tokens=omit_tokens,
            group_map=active_group_map,
        )
    except Exception as exc:
        set_status(f"Plant clustering error: {exc}", "error")
        result_el = document.getElementById("plantResultText")
        if result_el:
            result_el.textContent = f"Plant clustering error: {exc}"
            result_el.className = "status error"
        return

    yaml_el = document.getElementById("plantYamlOut")
    if yaml_el is not None:
        yaml_el.value = yaml_str

    render_plant_candidates()

    note = ""
    if effective_budget > budget_val:
        note = " (budget raised to minimum needed for one cluster per tech/region)"

    result_el = document.getElementById("plantResultText")
    if result_el:
        result_el.textContent = f"Plant clustering ready: {total_clusters} clusters across technologies{note}."
        result_el.className = "status success"

    set_status(
        f"Plant clustering ready: {total_clusters} clusters across techs{note}.",
        "success",
    )


def on_copy_plant_yaml(event):
    """Copy plant YAML to clipboard."""
    yaml_el = document.getElementById("plantYamlOut")
    if yaml_el and yaml_el.value:
        window.navigator.clipboard.writeText(yaml_el.value)
        set_status("Plant YAML copied to clipboard!", "success")


def on_download_plant_yaml(event):
    """Download plant YAML file."""
    yaml_el = document.getElementById("plantYamlOut")
    if not yaml_el or not yaml_el.value:
        set_status("No plant YAML to download. Run plant clustering first.", "error")
        return

    blob = window.Blob.new([yaml_el.value], to_js({"type": "text/yaml"}))
    url = window.URL.createObjectURL(blob)

    a = document.createElement("a")
    a.href = url
    a.download = "plant_clusters.yml"
    a.click()

    window.URL.revokeObjectURL(url)
    set_status("Plant YAML downloaded!", "success")


def on_grouping_change(event):
    """Handle grouping column change."""
    update_no_cluster_options()


# ============================================================================
# Initialization
# ============================================================================


async def main():
    """Main initialization function."""
    try:
        # Load data
        await load_data()

        # Initialize map
        await init_map()

        # Set up UI - this also sets up group colors after map is ready
        update_no_cluster_options()

        # Force initial group colors (in case update_no_cluster_options skipped it)
        state.current_grouping = None  # Force re-calculation
        update_group_colors()

        # Attach event handlers
        document.getElementById("runBtn").addEventListener(
            "click", create_proxy(on_run_clustering)
        )
        document.getElementById("selectAllBtn").addEventListener(
            "click", create_proxy(on_select_all)
        )
        document.getElementById("clearSelectionBtn").addEventListener(
            "click", create_proxy(on_clear_selection)
        )
        document.getElementById("copyYamlBtn").addEventListener(
            "click", create_proxy(on_copy_yaml)
        )
        document.getElementById("downloadYamlBtn").addEventListener(
            "click", create_proxy(on_download_yaml)
        )
        document.getElementById("runPlantBtn").addEventListener(
            "click", create_proxy(on_run_plant_clustering)
        )
        document.getElementById("copyPlantYamlBtn").addEventListener(
            "click", create_proxy(on_copy_plant_yaml)
        )
        document.getElementById("downloadPlantYamlBtn").addEventListener(
            "click", create_proxy(on_download_plant_yaml)
        )
        document.getElementById("groupTechDefault").addEventListener(
            "change", create_proxy(update_default_cluster_budget)
        )
        document.getElementById("groupingColumn").addEventListener(
            "change", create_proxy(on_grouping_change)
        )
        document.getElementById("addGroupBtn").addEventListener(
            "click", create_proxy(on_add_group)
        )
        document.getElementById("moveToGroupBtn").addEventListener(
            "click", create_proxy(on_add_tech_to_group)
        )
        document.getElementById("moveToAvailableBtn").addEventListener(
            "click", create_proxy(on_remove_tech_from_group)
        )
        document.getElementById("resetGroupsBtn").addEventListener(
            "click", create_proxy(on_reset_groups)
        )
        document.getElementById("clearGroupsBtn").addEventListener(
            "click", create_proxy(on_clear_groups)
        )
        document.getElementById("groupSelectDual").addEventListener(
            "change", create_proxy(on_group_change)
        )
        document.getElementById("omitMoveToSelectedBtn").addEventListener(
            "click", create_proxy(on_omit_move_to_selected)
        )
        document.getElementById("omitMoveToAvailableBtn").addEventListener(
            "click", create_proxy(on_omit_move_to_available)
        )
        document.getElementById("omitResetBtn").addEventListener(
            "click", create_proxy(on_reset_omit_defaults)
        )

        # Box selection mode buttons
        document.getElementById("clickModeBtn").addEventListener(
            "click", create_proxy(on_click_mode)
        )
        document.getElementById("boxModeBtn").addEventListener(
            "click", create_proxy(on_box_mode)
        )
        document.getElementById("showTransmissionLines").addEventListener(
            "change", create_proxy(on_toggle_transmission_lines)
        )

        # Map events for box selection
        state.map.on("mousedown", create_proxy(on_map_mousedown))
        state.map.on("mousemove", create_proxy(on_map_mousemove))
        state.map.on("mouseup", create_proxy(on_map_mouseup))

        # Start in box select mode (disable map dragging)
        state.map.dragging.disable()
        document.getElementById("map").classList.add("box-select-mode")

        # Initialize omit list and grouping editor with defaults
        render_omit_editor()
        reset_custom_groups(omit_tokens=get_selected_omit_tokens())

        # Seed the plant budget with a 15% buffer above the minimum clusters
        update_default_cluster_budget()

        # Done loading
        hide_loading()
        set_status(
            "Ready! Drag on the map to select BAs, or switch to Click mode for individual selection.",
            "info",
        )

    except Exception as e:
        set_status(f"Initialization error: {e}", "error")
        hide_loading()


# Run main
asyncio.ensure_future(main())
