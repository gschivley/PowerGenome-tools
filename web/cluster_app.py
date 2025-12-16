"""
PowerGenome Region Clustering - PyScript Web App

This module runs in the browser via PyScript and handles:
1. Loading and displaying the BA map
2. Managing BA selection state
3. Running spectral clustering
4. Generating YAML output
"""

import asyncio
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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering

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


state = AppState()

# ============================================================================
# Styling
# ============================================================================

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

# Group outline colors (for showing regional groupings)
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
            html = "".join(f'<span class="ba-tag">{ba}</span>' for ba in sorted_bas)
            list_el.innerHTML = html
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
    """Load hierarchy and transmission CSVs."""
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


def spectral_cluster(graph, n_clusters):
    """Perform spectral clustering on a graph."""
    nodes = sorted(graph.nodes())
    n = len(nodes)

    if n <= n_clusters:
        # Each node is its own cluster
        return {i: {node} for i, node in enumerate(nodes)}

    # Build adjacency matrix
    A = nx.to_numpy_array(graph, nodelist=nodes, weight="weight")

    # Check for isolated nodes
    degrees = A.sum(axis=1)
    isolated_mask = degrees == 0

    if np.all(isolated_mask):
        # All nodes isolated - assign to individual clusters
        return {i: {node} for i, node in enumerate(nodes)}

    if np.any(isolated_mask):
        # Handle isolated nodes separately
        connected_indices = np.where(~isolated_mask)[0]
        isolated_indices = np.where(isolated_mask)[0]

        connected_nodes = [nodes[i] for i in connected_indices]
        isolated_nodes = [nodes[i] for i in isolated_indices]

        if len(connected_nodes) <= n_clusters - len(isolated_nodes):
            # Give each connected node its own cluster plus isolated
            clusters = {i: {node} for i, node in enumerate(connected_nodes)}
            for i, node in enumerate(isolated_nodes):
                clusters[len(connected_nodes) + i] = {node}
            return clusters

        # Cluster connected nodes
        A_connected = A[np.ix_(connected_indices, connected_indices)]
        n_clusters_connected = max(1, n_clusters - len(isolated_nodes))

        sc = SpectralClustering(
            n_clusters=n_clusters_connected,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        labels = sc.fit_predict(A_connected)

        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = set()
            clusters[label].add(connected_nodes[idx])

        # Add isolated nodes as individual clusters
        next_label = max(clusters.keys()) + 1 if clusters else 0
        for node in isolated_nodes:
            clusters[next_label] = {node}
            next_label += 1

        return clusters

    # All nodes connected
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    labels = sc.fit_predict(A)

    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = set()
        clusters[label].add(nodes[idx])

    return clusters


def hierarchical_cluster(hierarchy_df, transmission_df, cluster_bas, grouping_column, target_regions):
    """
    Hierarchical clustering that respects grouping column boundaries.
    
    Phase 1: Cluster BAs within each grouping column region
    Phase 2: Merge entire grouping column regions together if needed
    
    Grouping column regions are never split across model regions.
    """
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
        # Distribute target across groups proportionally
        # Each group gets at least 1 region
        regions_per_group = {}
        remaining_regions = target_regions
        
        # First pass: give each group 1 region
        for group in groups:
            regions_per_group[group] = 1
            remaining_regions -= 1
        
        # Second pass: distribute remaining regions by group size
        if remaining_regions > 0:
            group_sizes = [(group, len(bas)) for group, bas in groups.items()]
            group_sizes.sort(key=lambda x: -x[1])  # Largest first
            
            for group, size in group_sizes:
                if remaining_regions <= 0:
                    break
                # Give more regions to larger groups
                max_additional = min(remaining_regions, size - 1)  # Can't have more regions than BAs
                regions_per_group[group] += max_additional
                remaining_regions -= max_additional
        
        # Cluster within each group
        all_clusters = {}
        cluster_id = 0
        
        for group, group_bas in groups.items():
            n_clusters_for_group = min(regions_per_group[group], len(group_bas))
            
            if n_clusters_for_group == 1 or len(group_bas) == 1:
                # Entire group becomes one cluster
                all_clusters[cluster_id] = group_bas
                cluster_id += 1
            else:
                # Build subgraph for this group
                subgraph = build_transmission_graph(transmission_df, group_bas)
                sub_clusters = spectral_cluster(subgraph, n_clusters_for_group)
                
                for label, nodes in sub_clusters.items():
                    all_clusters[cluster_id] = nodes
                    cluster_id += 1
        
        return all_clusters
    
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
        group_clusters = spectral_cluster(group_graph, target_regions)
        
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


def generate_cluster_names(clusters, groups):
    """Generate meaningful cluster names based on smallest containing grouping column.

    For aggregated regions: Find the smallest grouping column where all BAs share
    the same value, then name as <group_value><x> where x is an integer.
    For single BAs: Use the state abbreviation.
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

    for label, nodes in clusters.items():
        nodes_list = list(nodes)

        # Single BA - use state abbreviation
        if len(nodes_list) == 1:
            ba = nodes_list[0]
            # Get state for this BA
            ba_row = state.hierarchy_df[state.hierarchy_df["ba"] == ba]
            if not ba_row.empty:
                st = ba_row.iloc[0]["st"]
                base_name = st
            else:
                base_name = ba

            # Add counter if needed
            if base_name in name_counts:
                name_counts[base_name] += 1
                cluster_names[label] = f"{base_name}{name_counts[base_name]}"
            else:
                name_counts[base_name] = 1
                cluster_names[label] = f"{base_name}1"
            continue

        # Multiple BAs - find smallest containing grouping column
        found_group = None
        group_value = None

        for col in GROUPING_HIERARCHY:
            if col not in state.hierarchy_df.columns:
                continue

            # Get values for all BAs in this cluster
            ba_values = state.hierarchy_df[state.hierarchy_df["ba"].isin(nodes)][
                col
            ].unique()

            # If all BAs share the same value, use this column
            if len(ba_values) == 1:
                found_group = col
                group_value = ba_values[0]
                break

        if group_value:
            base_name = group_value
        else:
            # No common grouping found, use generic name
            base_name = "Region"

        # Add counter
        if base_name in name_counts:
            name_counts[base_name] += 1
            cluster_names[label] = f"{base_name}{name_counts[base_name]}"
        else:
            name_counts[base_name] = 1
            cluster_names[label] = f"{base_name}1"

    return cluster_names


def run_clustering(selected_bas, grouping_column, target_regions, no_cluster_groups):
    """
    Run the clustering algorithm.

    Returns a tuple of (model_regions, region_aggregations, error_message)
    """
    try:
        # Filter hierarchy to selected BAs
        hierarchy = state.hierarchy_df[
            state.hierarchy_df["ba"].isin(selected_bas)
        ].copy()

        if len(hierarchy) == 0:
            return None, None, "No valid BAs selected"

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
            return model_regions, region_aggregations, None

        # Build graph for clustering
        graph = build_transmission_graph(state.transmission_df, cluster_bas)

        # Get regional groups
        groups = get_regional_groups(hierarchy, grouping_column, cluster_bas)

        # Determine actual target (accounting for unclustered)
        actual_target = max(1, target_regions - len(unclustered_bas))
        actual_target = min(actual_target, len(cluster_bas))

        # Run hierarchical clustering that respects grouping column boundaries
        clusters = hierarchical_cluster(
            hierarchy, state.transmission_df, cluster_bas, grouping_column, actual_target
        )

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

        return model_regions, region_aggregations, None

    except Exception as e:
        return None, None, str(e)


def generate_yaml(model_regions, region_aggregations):
    """Generate YAML output."""
    output = {
        "model_regions": model_regions,
        "region_aggregations": region_aggregations,
    }
    return yaml.dump(output, default_flow_style=False, sort_keys=False)


# ============================================================================
# Event Handlers
# ============================================================================


def on_run_clustering(event):
    """Handle Run Clustering button click."""
    set_status("Running clustering...", "info")

    grouping_col = document.getElementById("groupingColumn").value
    target_regions = int(document.getElementById("targetRegions").value)

    # Get no-cluster selections
    no_cluster_groups = []
    checkboxes = document.querySelectorAll('input[name="noCluster"]:checked')
    for cb in checkboxes:
        no_cluster_groups.append(cb.value)

    # Run clustering
    model_regions, region_aggregations, error = run_clustering(
        state.selected_bas,
        grouping_col,
        target_regions,
        no_cluster_groups,
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

    # Check if we got more regions than targeted
    num_regions = len(model_regions)
    if num_regions > target_regions:
        set_status(
            f"Warning: Created {num_regions} regions, which is more than the target of {target_regions}. "
            f"This can happen when 'unclustered' groups or disconnected BAs exceed the target.",
            "error",
        )
    else:
        set_status(f"Clustering complete! {num_regions} regions created.", "success")

    # Store region aggregations for transmission line drawing
    state.region_aggregations = region_aggregations

    # Update map colors to show clusters
    update_map_cluster_colors(region_aggregations)

    # Update tooltips to show model region names
    update_tooltips()

    # Update transmission lines if enabled
    update_transmission_lines()


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
        document.getElementById("groupingColumn").addEventListener(
            "change", create_proxy(on_grouping_change)
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
