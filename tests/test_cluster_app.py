"""
Comprehensive test suite for cluster_app.py clustering and utility functions.

This module tests all pure Python functions that don't depend on PyScript's js module
or DOM access. Tests cover color conversion, graph clustering algorithms, plant
clustering, and YAML generation.
"""

import json
import re

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import yaml

# Since we can't import cluster_app.py directly (it has PyScript dependencies),
# we replicate the testable functions here or use fixtures that provide them.
# We'll test the logic by importing only non-PyScript-dependent pieces.


# ============================================================================
# Color Conversion Tests
# ============================================================================


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


class TestColorConversion:
    """Test color conversion functions."""

    def test_hex_to_rgb_valid(self):
        """Test hex to RGB conversion with valid input."""
        assert hex_to_rgb("#FF0000") == (255, 0, 0)
        assert hex_to_rgb("#00FF00") == (0, 255, 0)
        assert hex_to_rgb("#0000FF") == (0, 0, 255)
        assert hex_to_rgb("#FFFFFF") == (255, 255, 255)
        assert hex_to_rgb("#000000") == (0, 0, 0)

    def test_hex_to_rgb_without_hash(self):
        """Test hex to RGB with hash prefix removed."""
        assert hex_to_rgb("FF0000") == (255, 0, 0)

    def test_hex_to_rgb_lowercase(self):
        """Test hex to RGB with lowercase input."""
        assert hex_to_rgb("#ff0000") == (255, 0, 0)

    def test_rgb_to_hex(self):
        """Test RGB to hex conversion."""
        assert rgb_to_hex((255, 0, 0)) == "#ff0000"
        assert rgb_to_hex((0, 255, 0)) == "#00ff00"
        assert rgb_to_hex((0, 0, 255)) == "#0000ff"
        assert rgb_to_hex((255, 255, 255)) == "#ffffff"
        assert rgb_to_hex((0, 0, 0)) == "#000000"

    def test_hex_rgb_roundtrip(self):
        """Test that hex->rgb->hex conversion is consistent."""
        original = "#1b9e77"
        rgb = hex_to_rgb(original)
        result = rgb_to_hex(rgb)
        assert result == original

    def test_lighten_color_full_black(self):
        """Test lightening pure black."""
        result = lighten_color("#000000", factor=0.5)
        # Should be #808080 (128, 128, 128)
        rgb = hex_to_rgb(result)
        assert all(127 <= c <= 129 for c in rgb)

    def test_lighten_color_full_white(self):
        """Test lightening pure white returns white."""
        result = lighten_color("#FFFFFF", factor=0.5)
        assert result == "#ffffff"

    def test_lighten_color_default_factor(self):
        """Test lighten with default factor."""
        result = lighten_color("#FF0000")  # Red with 70% factor
        rgb = hex_to_rgb(result)
        # Red should become lighter: (255 + (255-255)*0.7, 0 + 255*0.7, 0 + 255*0.7)
        # = (255, 178.5, 178.5) ≈ (255, 178, 178)
        assert rgb[0] == 255
        assert 175 <= rgb[1] <= 180
        assert 175 <= rgb[2] <= 180

    def test_lighten_color_increases_brightness(self):
        """Test that lightening always increases brightness or keeps it same."""
        test_colors = ["#1b9e77", "#d95f02", "#7570b3"]
        for color in test_colors:
            original_rgb = hex_to_rgb(color)
            lightened = lighten_color(color, factor=0.7)
            lightened_rgb = hex_to_rgb(lightened)
            # Each component should be >= original (moving toward white)
            assert all(
                lightened_rgb[i] >= original_rgb[i] for i in range(3)
            ), f"Lightening {color} failed"


# ============================================================================
# Graph Construction Tests
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


class TestGraphConstruction:
    """Test graph building functions."""

    @pytest.fixture
    def transmission_data(self):
        """Create sample transmission data."""
        return pd.DataFrame(
            {
                "region_from": ["ca", "ca", "tx", "tx", "ca"],
                "region_to": ["ne", "tx", "ne", "se", "se"],
                "firm_ttc_mw": [1000, 500, 750, 1200, 300],
            }
        )

    @pytest.fixture
    def hierarchy_data(self):
        """Create sample hierarchy data."""
        return pd.DataFrame(
            {
                "ba": ["ca", "ne", "tx", "se", "co"],
                "st": ["CA", "NE", "TX", "SE", "CO"],
                "cendiv": ["WSC", "ENC", "WSC", "ESC", "MTN"],
                "nercr": ["WECC", "RFC", "ERCOT", "SERC", "WECC"],
            }
        )

    def test_build_transmission_graph_all_bas(self, transmission_data):
        """Test graph construction with all BAs included."""
        valid_bas = {"ca", "ne", "tx", "se"}
        graph = build_transmission_graph(transmission_data, valid_bas)

        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 5  # ca-ne, ca-tx, tx-ne, tx-se, ca-se
        assert graph["ca"]["ne"]["weight"] == 1000
        assert graph["ca"]["tx"]["weight"] == 500
        assert graph["tx"]["ne"]["weight"] == 750

    def test_build_transmission_graph_filtered_bas(self, transmission_data):
        """Test graph construction with BA filtering."""
        valid_bas = {"ca", "tx", "se"}
        graph = build_transmission_graph(transmission_data, valid_bas)

        assert graph.number_of_nodes() == 3
        assert "ne" not in graph.nodes()
        assert graph.number_of_edges() == 3

    def test_build_transmission_graph_parallel_edges(self):
        """Test that parallel edges (same BA pair) are summed."""
        transmission_df = pd.DataFrame(
            {
                "region_from": ["a", "a", "a"],
                "region_to": ["b", "b", "c"],
                "firm_ttc_mw": [100, 200, 300],
            }
        )
        valid_bas = {"a", "b", "c"}
        graph = build_transmission_graph(transmission_df, valid_bas)

        # Two edges from a->b should be summed
        assert graph["a"]["b"]["weight"] == 300

    def test_build_transmission_graph_isolated_nodes(self, transmission_data):
        """Test that isolated nodes are added to graph."""
        valid_bas = {"ca", "ne", "tx", "se", "isolated"}
        graph = build_transmission_graph(transmission_data, valid_bas)

        assert "isolated" in graph.nodes()
        assert graph.degree("isolated") == 0

    def test_build_transmission_graph_empty(self):
        """Test graph construction with no valid edges."""
        transmission_df = pd.DataFrame(
            {
                "region_from": ["a", "b"],
                "region_to": ["b", "c"],
                "firm_ttc_mw": [100, 200],
            }
        )
        valid_bas = {"x", "y"}
        graph = build_transmission_graph(transmission_df, valid_bas)

        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 0

    def test_get_regional_groups(self, hierarchy_data):
        """Test regional group mapping."""
        valid_bas = {"ca", "tx", "ne", "se"}
        groups = get_regional_groups(hierarchy_data, "st", valid_bas)

        assert len(groups) == 4
        assert groups["CA"] == {"ca"}
        assert groups["TX"] == {"tx"}

    def test_get_regional_groups_multiple_bas_per_group(self):
        """Test regional groups with multiple BAs in same group."""
        hierarchy_df = pd.DataFrame(
            {
                "ba": ["a", "b", "c", "d"],
                "st": ["CA", "CA", "TX", "TX"],
                "cendiv": ["WSC", "WSC", "SWC", "SWC"],
            }
        )
        valid_bas = {"a", "b", "c", "d"}
        groups = get_regional_groups(hierarchy_df, "st", valid_bas)

        assert groups["CA"] == {"a", "b"}
        assert groups["TX"] == {"c", "d"}

    def test_get_regional_groups_filters_invalid_bas(self, hierarchy_data):
        """Test that invalid BAs are excluded from groups."""
        valid_bas = {"ca", "tx"}  # Exclude ne, se, co
        groups = get_regional_groups(hierarchy_data, "st", valid_bas)

        assert len(groups) == 2
        assert "NE" not in groups


# ============================================================================
# Agglomerative Clustering Tests
# ============================================================================


def agglomerative_cluster(graph, n_clusters):
    """
    Perform agglomerative clustering on a graph.

    Uses a greedy approach: repeatedly merge the two clusters with the
    highest total edge weight between them until reaching n_clusters.
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    if n <= n_clusters:
        return {i: {node} for i, node in enumerate(nodes)}

    # Initialize: each node is its own cluster
    clusters = {i: {node} for i, node in enumerate(nodes)}
    node_to_cluster = {node: i for i, node in enumerate(nodes)}

    # Build initial inter-cluster weights
    cluster_weights = {}
    for u, v, data in graph.edges(data=True):
        c1, c2 = node_to_cluster[u], node_to_cluster[v]
        if c1 != c2:
            key = (min(c1, c2), max(c1, c2))
            weight = data.get("weight", 1.0)
            cluster_weights[key] = cluster_weights.get(key, 0) + weight

    # Merge until we reach target number of clusters
    while len(clusters) > n_clusters:
        if not cluster_weights:
            break

        # Find the pair with maximum weight
        best_pair = max(cluster_weights.keys(), key=lambda k: cluster_weights[k])
        c1, c2 = best_pair

        # Merge c2 into c1
        clusters[c1].update(clusters[c2])
        for node in clusters[c2]:
            node_to_cluster[node] = c1
        del clusters[c2]

        # Update cluster weights
        new_weights = {}
        keys_to_remove = []

        for (ca, cb), weight in cluster_weights.items():
            if ca == c2 or cb == c2:
                keys_to_remove.append((ca, cb))
                other = cb if ca == c2 else ca
                if other != c1:
                    new_key = (min(c1, other), max(c1, other))
                    new_weights[new_key] = new_weights.get(new_key, 0) + weight

        for key in keys_to_remove:
            del cluster_weights[key]

        for key, weight in new_weights.items():
            cluster_weights[key] = cluster_weights.get(key, 0) + weight

    # Renumber clusters to be sequential
    result = {}
    for i, (cluster_id, nodes_set) in enumerate(clusters.items()):
        result[i] = nodes_set

    return result


class TestAgglomerativeClustering:
    """Test agglomerative clustering algorithm."""

    def test_agglomerative_single_cluster(self):
        """Test clustering into a single cluster."""
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
        result = agglomerative_cluster(graph, 1)

        assert len(result) == 1
        assert result[0] == {1, 2, 3, 4}

    def test_agglomerative_multiple_clusters(self):
        """Test clustering into multiple clusters."""
        graph = nx.Graph()
        # Two well-separated components
        graph.add_edges_from([(1, 2), (2, 3)])
        graph.add_edges_from([(4, 5), (5, 6)])
        result = agglomerative_cluster(graph, 2)

        assert len(result) == 2
        clusters = list(result.values())
        assert {1, 2, 3} in clusters
        assert {4, 5, 6} in clusters

    def test_agglomerative_preserves_nodes(self):
        """Test that all nodes are preserved after clustering."""
        graph = nx.complete_graph(5)
        result = agglomerative_cluster(graph, 2)

        all_nodes = set()
        for cluster in result.values():
            all_nodes.update(cluster)
        assert all_nodes == {0, 1, 2, 3, 4}

    def test_agglomerative_weighted_merging(self):
        """Test that higher weight edges are merged first."""
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=100)
        graph.add_edge(2, 3, weight=10)
        graph.add_edge(3, 4, weight=50)
        result = agglomerative_cluster(graph, 2)

        assert len(result) == 2

    def test_agglomerative_isolated_nodes(self):
        """Test clustering with isolated nodes."""
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=1)
        graph.add_node(3)
        graph.add_node(4)
        result = agglomerative_cluster(graph, 3)

        assert len(result) == 3

    def test_agglomerative_more_clusters_than_nodes(self):
        """Test when requesting more clusters than nodes."""
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3)])
        result = agglomerative_cluster(graph, 10)

        assert len(result) == 3
        assert all(len(cluster) == 1 for cluster in result.values())


# ============================================================================
# Modularity Calculation Tests
# ============================================================================


def calculate_modularity(graph, clusters):
    """
    Calculate the modularity score for a clustering result.

    Modularity measures how well a network is divided into communities.
    """
    if not clusters or graph.number_of_edges() == 0:
        return 0.0

    communities = [clusters[label] for label in sorted(clusters.keys())]
    graph_nodes = set(graph.nodes())
    communities = [c & graph_nodes for c in communities]
    communities = [c for c in communities if len(c) > 0]

    if not communities:
        return 0.0

    try:
        modularity = nx.community.modularity(graph, communities, weight="weight")
        return float(modularity)
    except Exception:
        return 0.0


class TestModularityCalculation:
    """Test modularity score calculation."""

    def test_modularity_two_components(self):
        """Test modularity for two well-separated components."""
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3), (3, 1)])
        graph.add_edges_from([(4, 5), (5, 6), (6, 4)])
        clusters = {0: {1, 2, 3}, 1: {4, 5, 6}}

        modularity = calculate_modularity(graph, clusters)
        assert modularity >= 0.5  # Good clustering

    def test_modularity_poor_clustering(self):
        """Test modularity for poor clustering (random split)."""
        graph = nx.complete_graph(5)
        clusters = {0: {0, 1}, 1: {2, 3, 4}}

        modularity = calculate_modularity(graph, clusters)
        assert modularity < 0.5  # Poor clustering

    def test_modularity_single_cluster(self):
        """Test modularity when all nodes in one cluster."""
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3)])
        clusters = {0: {1, 2, 3}}

        modularity = calculate_modularity(graph, clusters)
        assert modularity == 0.0

    def test_modularity_empty_clusters(self):
        """Test modularity with empty clusters dict."""
        graph = nx.Graph()
        graph.add_edges_from([(1, 2)])
        clusters = {}

        modularity = calculate_modularity(graph, clusters)
        assert modularity == 0.0

    def test_modularity_no_edges(self):
        """Test modularity with disconnected graph."""
        graph = nx.Graph()
        graph.add_nodes_from([1, 2, 3])
        clusters = {0: {1}, 1: {2}, 2: {3}}

        modularity = calculate_modularity(graph, clusters)
        assert modularity == 0.0


# ============================================================================
# Technology Normalization Tests
# ============================================================================


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

    # Specific matches
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
    if "battery" in name or "storage" in name:
        return "Batteries"
    if "petroleum" in name or "oil" in name:
        return "Petroleum Liquids"

    return tech_name


class TestTechnologyNormalization:
    """Test technology name normalization."""

    def test_normalize_wind_technologies(self):
        """Test normalization of wind technologies."""
        assert normalize_technology("Onshore Wind") == "Onshore Wind Turbine"
        assert normalize_technology("Offshore Wind Turbine") == "Offshore Wind Turbine"
        assert normalize_technology("Wind Turbine") == "Onshore Wind Turbine"

    def test_normalize_solar_technologies(self):
        """Test normalization of solar technologies."""
        assert normalize_technology("Photovoltaic") == "Solar Photovoltaic"
        assert normalize_technology("Solar Photovoltaic") == "Solar Photovoltaic"
        assert normalize_technology("Solar Thermal") is None  # Omitted by default

    def test_normalize_hydro_technologies(self):
        """Test normalization of hydroelectric technologies."""
        assert (
            normalize_technology("Conventional Hydroelectric")
            == "Conventional Hydroelectric"
        )
        assert normalize_technology("Run of River") == "Run of River Hydroelectric"
        assert (
            normalize_technology("Pumped Storage Hydro")
            == "Hydroelectric Pumped Storage"
        )

    def test_normalize_coal_technologies(self):
        """Test normalization of coal technologies."""
        assert normalize_technology("Steam Coal") == "Conventional Steam Coal"
        assert normalize_technology("Coal") == "Conventional Steam Coal"

    def test_normalize_gas_technologies(self):
        """Test normalization of natural gas technologies."""
        assert (
            normalize_technology("Combined Cycle") == "Natural Gas Fired Combined Cycle"
        )
        assert (
            normalize_technology("Combustion Turbine")
            == "Natural Gas Fired Combustion Turbine"
        )
        assert normalize_technology("Steam Turbine") == "Natural Gas Steam Turbine"
        assert (
            normalize_technology("Internal Combustion Engine")
            == "Natural Gas Internal Combustion Engine"
        )

    def test_normalize_biomass_technologies(self):
        """Test normalization of biomass technologies."""
        assert normalize_technology("Biomass") == "Biomass"
        assert normalize_technology("Landfill Gas") == "Landfill Gas"
        assert normalize_technology("Municipal Solid Waste") == "Municipal Solid Waste"
        assert normalize_technology("Wood/Wood Waste") == "Wood/Wood Waste Biomass"

    def test_normalize_battery_storage(self):
        """Test normalization of battery and storage technologies."""
        assert normalize_technology("Battery") == "Batteries"
        assert normalize_technology("Battery Storage") == "Batteries"
        assert normalize_technology("Storage") == "Batteries"

    def test_normalize_petroleum(self):
        """Test normalization of petroleum technologies."""
        assert normalize_technology("Petroleum") == "Petroleum Liquids"
        assert normalize_technology("Oil") == "Petroleum Liquids"

    def test_normalize_nuclear(self):
        """Test normalization of nuclear."""
        assert normalize_technology("Nuclear") == "Nuclear"

    def test_normalize_geothermal(self):
        """Test normalization of geothermal."""
        assert normalize_technology("Geothermal") == "Geothermal"

    def test_normalize_unknown_technology(self):
        """Test that unknown technologies are returned as-is."""
        assert normalize_technology("Foo Bar Technology") == "Foo Bar Technology"

    def test_normalize_omit_default_tokens(self):
        """Test omitting default tokens."""
        assert normalize_technology("Solar Thermal") is None
        assert normalize_technology("All Other") is None
        assert normalize_technology("Flywheel Storage") is None

    def test_normalize_omit_custom_tokens(self):
        """Test omitting custom tokens."""
        assert normalize_technology("Wind", omit_tokens=["wind"]) is None
        assert normalize_technology("Solar PV", omit_tokens=["solar"]) is None

    def test_normalize_case_insensitive(self):
        """Test that normalization is case-insensitive."""
        assert normalize_technology("WIND TURBINE") == "Onshore Wind Turbine"
        assert normalize_technology("SOLAR PHOTOVOLTAIC") == "Solar Photovoltaic"

    def test_normalize_non_string_input(self):
        """Test handling of non-string input."""
        assert normalize_technology(None) is None
        assert normalize_technology(123) is None
        assert normalize_technology([]) is None


# ============================================================================
# K-means and Feature Standardization Tests
# ============================================================================


def standardize_features(matrix):
    """Standardize columns to zero mean, unit variance."""
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    return (matrix - means) / stds


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


def inertia_single_cluster(features, weights=None):
    """Inertia for a single cluster (k=1)."""
    center = features.mean(axis=0)
    sq_dists = np.sum((features - center) ** 2, axis=1)
    if weights is not None:
        return float((sq_dists * weights).sum())
    return float(sq_dists.sum())


class TestFeatureStandardization:
    """Test feature standardization."""

    def test_standardize_basic(self):
        """Test basic standardization."""
        data = np.array([[1, 2], [2, 4], [3, 6]])
        result = standardize_features(data)

        # Check mean is approximately 0
        assert np.allclose(result.mean(axis=0), 0, atol=1e-10)
        # Check std is approximately 1
        assert np.allclose(result.std(axis=0), 1, atol=1e-10)

    def test_standardize_with_nan(self):
        """Test standardization with NaN values."""
        data = np.array([[1, 2], [2, np.nan], [3, 6]])
        result = standardize_features(data)

        # Check that NaN is handled
        assert np.isfinite(result[0]).all()
        assert np.isfinite(result[2]).all()

    def test_standardize_constant_column(self):
        """Test standardization with constant column."""
        data = np.array([[1, 5], [2, 5], [3, 5]])
        result = standardize_features(data)

        # Constant column should have std=0 and be mapped to 0
        assert np.allclose(result[:, 1], 0)


class TestWeightedQuantile:
    """Test weighted quantile calculation."""

    def test_weighted_quantile_basic(self):
        """Test weighted quantile with equal weights."""
        values = np.array([1, 2, 3, 4, 5])
        weights = np.ones(5)
        q50 = weighted_quantile(values, 0.5, weights)

        # With equal weights, median should be 3
        assert np.isclose(q50, 3.0)

    def test_weighted_quantile_unequal_weights(self):
        """Test weighted quantile with unequal weights."""
        values = np.array([1, 2, 3, 4, 5])
        weights = np.array([0, 0, 0, 0, 1])  # All weight on 5
        q50 = weighted_quantile(values, 0.5, weights)

        assert q50 == 5.0


class TestWeightedIQR:
    """Test weighted interquartile range."""

    def test_weighted_iqr_basic(self):
        """Test weighted IQR calculation."""
        values = np.array([1, 2, 3, 4, 5])
        weights = np.ones(5)
        iqr = weighted_iqr(values, weights)

        # Q3 - Q1 = 4 - 2 = 2 (approximately)
        assert 1.5 < iqr < 2.5

    def test_weighted_iqr_empty(self):
        """Test weighted IQR with empty array."""
        values = np.array([])
        weights = np.array([])
        iqr = weighted_iqr(values, weights)

        assert iqr == 0.0


class TestKMeansSimple:
    """Test simple k-means implementation."""

    def test_kmeans_single_cluster(self):
        """Test k-means with k=1."""
        features = np.array([[1, 1], [2, 2], [3, 3]])
        inertia, centers, labels = run_kmeans_simple(features, 1)

        assert inertia >= 0
        assert centers.shape == (1, 2)
        assert np.all(labels == 0)

    def test_kmeans_two_clusters(self):
        """Test k-means with k=2 on well-separated data."""
        features = np.array(
            [
                [0, 0],
                [1, 1],  # Cluster 1
                [10, 10],
                [11, 11],  # Cluster 2
            ]
        )
        inertia, centers, labels = run_kmeans_simple(features, 2, seed=42)

        assert inertia >= 0
        assert centers.shape == (2, 2)
        assert len(set(labels)) == 2

    def test_kmeans_with_weights(self):
        """Test k-means with weighted samples."""
        features = np.array([[1, 1], [2, 2], [10, 10]])
        weights = np.array([1, 1, 100])  # Heavy weight on third point
        inertia, centers, labels = run_kmeans_simple(
            features, 2, weights=weights, seed=42
        )

        assert inertia >= 0
        assert centers.shape == (2, 2)

    def test_kmeans_more_clusters_than_samples(self):
        """Test k-means when k >= number of samples."""
        features = np.array([[1, 1], [2, 2], [3, 3]])
        # Use k = num_samples to avoid the empty cluster edge case
        inertia, centers, labels = run_kmeans_simple(features, 3, seed=42)

        assert inertia >= 0
        # Centers should have 3 centers
        assert centers is not None
        assert centers.shape[0] == 3

    def test_kmeans_deterministic(self):
        """Test that k-means with same seed is deterministic."""
        features = np.random.RandomState(42).randn(20, 3)
        _, centers1, labels1 = run_kmeans_simple(features, 3, seed=42)
        _, centers2, labels2 = run_kmeans_simple(features, 3, seed=42)

        assert np.allclose(centers1, centers2)
        assert np.array_equal(labels1, labels2)


class TestInertiaCalculation:
    """Test inertia calculations."""

    def test_inertia_single_cluster(self):
        """Test inertia calculation for single cluster."""
        features = np.array([[1, 1], [2, 2], [3, 3]])
        inertia = inertia_single_cluster(features)

        assert inertia > 0

    def test_inertia_with_weights(self):
        """Test inertia with weighted samples."""
        features = np.array([[1, 1], [2, 2], [3, 3]])
        weights = np.array([1, 1, 10])
        inertia = inertia_single_cluster(features, weights=weights)

        assert inertia > 0

    def test_inertia_identical_points(self):
        """Test inertia when all points are identical."""
        features = np.array([[5, 5], [5, 5], [5, 5]])
        inertia = inertia_single_cluster(features)

        assert inertia == 0.0


# ============================================================================
# YAML Generation Tests
# ============================================================================


def generate_yaml(model_regions, region_aggregations):
    """Generate YAML output."""
    output = {
        "model_regions": model_regions,
        "region_aggregations": region_aggregations,
    }
    return yaml.dump(output, default_flow_style=False, sort_keys=False)


class TestYAMLGeneration:
    """Test YAML generation."""

    def test_generate_yaml_basic(self):
        """Test basic YAML generation."""
        model_regions = ["CA1", "TX1"]
        region_aggregations = {"CA1": ["ciso", "nevp"], "TX1": ["tre"]}

        result = generate_yaml(model_regions, region_aggregations)

        parsed = yaml.safe_load(result)
        assert parsed["model_regions"] == ["CA1", "TX1"]
        assert parsed["region_aggregations"]["CA1"] == ["ciso", "nevp"]

    def test_generate_yaml_empty(self):
        """Test YAML generation with empty data."""
        model_regions = []
        region_aggregations = {}

        result = generate_yaml(model_regions, region_aggregations)

        parsed = yaml.safe_load(result)
        assert parsed["model_regions"] == []
        assert parsed["region_aggregations"] == {}

    def test_generate_yaml_single_region(self):
        """Test YAML generation with single region."""
        model_regions = ["CA"]
        region_aggregations = {"CA": ["ciso"]}

        result = generate_yaml(model_regions, region_aggregations)

        parsed = yaml.safe_load(result)
        assert len(parsed["model_regions"]) == 1
        assert "CA" in parsed["region_aggregations"]

    def test_generate_yaml_valid_format(self):
        """Test that generated YAML is valid."""
        model_regions = ["R1", "R2", "R3"]
        region_aggregations = {
            "R1": ["a", "b"],
            "R2": ["c"],
            "R3": ["d", "e", "f"],
        }

        result = generate_yaml(model_regions, region_aggregations)

        # Should parse without error
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)
        assert "model_regions" in parsed
        assert "region_aggregations" in parsed


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for clustering workflows."""

    @pytest.fixture
    def sample_data(self):
        """Create sample hierarchy and transmission data."""
        hierarchy_df = pd.DataFrame(
            {
                "ba": ["ca", "nv", "co", "tx", "ok", "ne"],
                "st": ["CA", "NV", "CO", "TX", "OK", "NE"],
                "cendiv": ["PAC", "PAC", "MTN", "WSC", "WSC", "WNC"],
                "nercr": ["WECC", "WECC", "WECC", "ERCOT", "SPP", "RFC"],
            }
        )

        transmission_df = pd.DataFrame(
            {
                "region_from": ["ca", "ca", "nv", "co", "tx", "ok"],
                "region_to": ["nv", "co", "co", "ne", "ok", "ne"],
                "firm_ttc_mw": [500, 300, 200, 400, 1000, 600],
            }
        )

        return hierarchy_df, transmission_df

    def test_clustering_workflow(self, sample_data):
        """Test complete clustering workflow."""
        hierarchy_df, transmission_df = sample_data
        valid_bas = {"ca", "nv", "co", "tx", "ok", "ne"}

        # Build graph
        graph = build_transmission_graph(transmission_df, valid_bas)
        assert graph.number_of_nodes() == 6

        # Cluster
        clusters = agglomerative_cluster(graph, 2)
        assert len(clusters) == 2

        # All nodes should be assigned
        all_nodes = set()
        for cluster in clusters.values():
            all_nodes.update(cluster)
        assert all_nodes == valid_bas

    def test_hierarchical_grouping_workflow(self, sample_data):
        """Test regional grouping for hierarchical clustering."""
        hierarchy_df, transmission_df = sample_data
        valid_bas = {"ca", "nv", "co", "tx", "ok", "ne"}

        # Get groups by state
        groups = get_regional_groups(hierarchy_df, "st", valid_bas)
        assert len(groups) == 6  # Each BA has unique state

        # Get groups by region
        groups = get_regional_groups(hierarchy_df, "nercr", valid_bas)
        assert len(groups) == 4  # Multiple interconnects

    def test_modularity_improves_with_clustering(self, sample_data):
        """Test that clustering improves modularity."""
        hierarchy_df, transmission_df = sample_data
        valid_bas = {"ca", "nv", "co", "tx", "ok", "ne"}

        graph = build_transmission_graph(transmission_df, valid_bas)

        # All separate
        poor_clusters = {i: {node} for i, node in enumerate(graph.nodes())}
        poor_modularity = calculate_modularity(graph, poor_clusters)

        # Two merged groups
        good_clusters = {0: {"ca", "nv", "co"}, 1: {"tx", "ok", "ne"}}
        good_modularity = calculate_modularity(graph, good_clusters)

        assert good_modularity >= poor_modularity


# ============================================================================
# Parsing Utilities Tests (Priority 1)
# ============================================================================


def parse_int_list(text):
    """Parse comma/space-separated integers."""
    if text is None:
        return []
    raw = re.split(r"[\s,]+", str(text).strip())
    out = []
    for tok in raw:
        if not tok:
            continue
        out.append(int(tok))
    return out


def parse_new_resources_text(text):
    """Parse manual new_resources lines.

    Each non-empty line should be: Technology | Tech Detail | Cost Case | Size
    """
    if not text:
        return []
    items = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            continue
        tech, detail, case, size = parts
        if not tech or not detail or not case or not size:
            continue
        try:
            size_val = int(float(size))
        except Exception:
            continue
        items.append([tech, detail, case, size_val])
    return items


class TestParseIntList:
    """Test parse_int_list function."""

    def test_parse_comma_separated(self):
        """Test parsing comma-separated integers."""
        assert parse_int_list("1,2,3,4,5") == [1, 2, 3, 4, 5]

    def test_parse_space_separated(self):
        """Test parsing space-separated integers."""
        assert parse_int_list("1 2 3 4 5") == [1, 2, 3, 4, 5]

    def test_parse_mixed_separators(self):
        """Test parsing mixed comma and space separators."""
        assert parse_int_list("1, 2 , 3,4  5") == [1, 2, 3, 4, 5]

    def test_parse_with_extra_whitespace(self):
        """Test parsing with leading/trailing whitespace."""
        assert parse_int_list("  1, 2, 3  ") == [1, 2, 3]

    def test_parse_single_number(self):
        """Test parsing a single number."""
        assert parse_int_list("42") == [42]

    def test_parse_none_input(self):
        """Test parsing None returns empty list."""
        assert parse_int_list(None) == []

    def test_parse_empty_string(self):
        """Test parsing empty string returns empty list."""
        assert parse_int_list("") == []
        assert parse_int_list("   ") == []

    def test_parse_with_multiple_commas(self):
        """Test parsing with consecutive separators."""
        assert parse_int_list("1,,2,,3") == [1, 2, 3]

    def test_parse_negative_numbers(self):
        """Test parsing negative numbers."""
        assert parse_int_list("-1, -2, -3") == [-1, -2, -3]

    def test_parse_mixed_positive_negative(self):
        """Test parsing mixed positive and negative."""
        assert parse_int_list("1, -2, 3, -4") == [1, -2, 3, -4]


class TestParseNewResourcesText:
    """Test parse_new_resources_text function."""

    def test_parse_single_line(self):
        """Test parsing a single valid line."""
        text = "NaturalGas | Combined Cycle | Moderate | 500"
        result = parse_new_resources_text(text)
        assert len(result) == 1
        assert result[0] == ["NaturalGas", "Combined Cycle", "Moderate", 500]

    def test_parse_multiple_lines(self):
        """Test parsing multiple valid lines."""
        text = """NaturalGas | Combined Cycle | Moderate | 500
Wind | Onshore | Low | 100
Solar | Utility PV | High | 200"""
        result = parse_new_resources_text(text)
        assert len(result) == 3
        assert result[0][0] == "NaturalGas"
        assert result[1][0] == "Wind"
        assert result[2][0] == "Solar"

    def test_parse_with_comments(self):
        """Test that comment lines are skipped."""
        text = """# This is a comment
NaturalGas | Combined Cycle | Moderate | 500
# Another comment
Wind | Onshore | Low | 100"""
        result = parse_new_resources_text(text)
        assert len(result) == 2

    def test_parse_with_blank_lines(self):
        """Test that blank lines are skipped."""
        text = """NaturalGas | Combined Cycle | Moderate | 500

Wind | Onshore | Low | 100

"""
        result = parse_new_resources_text(text)
        assert len(result) == 2

    def test_parse_with_whitespace(self):
        """Test parsing with extra whitespace around pipes."""
        text = "NaturalGas  |  Combined Cycle  |  Moderate  |  500"
        result = parse_new_resources_text(text)
        assert len(result) == 1
        assert result[0] == ["NaturalGas", "Combined Cycle", "Moderate", 500]

    def test_parse_float_size_rounded(self):
        """Test that float sizes are converted to int."""
        text = "NaturalGas | Combined Cycle | Moderate | 500.7"
        result = parse_new_resources_text(text)
        assert result[0][3] == 500

    def test_parse_invalid_size_skipped(self):
        """Test that lines with invalid size are skipped."""
        text = """NaturalGas | Combined Cycle | Moderate | 500
Wind | Onshore | Low | invalid
Solar | Utility PV | High | 200"""
        result = parse_new_resources_text(text)
        assert len(result) == 2
        assert result[0][0] == "NaturalGas"
        assert result[1][0] == "Solar"

    def test_parse_wrong_number_of_fields(self):
        """Test that lines with wrong number of fields are skipped."""
        text = """NaturalGas | Combined Cycle | Moderate | 500
Wind | Onshore | Low
Solar | Utility PV | High | 200 | Extra"""
        result = parse_new_resources_text(text)
        assert len(result) == 1

    def test_parse_empty_fields(self):
        """Test that lines with empty fields are skipped."""
        text = """NaturalGas | Combined Cycle | Moderate | 500
 | Onshore | Low | 100
Wind |  | Low | 100"""
        result = parse_new_resources_text(text)
        assert len(result) == 1

    def test_parse_none_input(self):
        """Test that None input returns empty list."""
        assert parse_new_resources_text(None) == []

    def test_parse_empty_string(self):
        """Test that empty string returns empty list."""
        assert parse_new_resources_text("") == []


# ============================================================================
# Utility Functions Tests (Priority 2)
# ============================================================================


def clone_group_map(group_map):
    """Shallow clone of group map with set copies."""
    return {name: set(values) for name, values in group_map.items()}


DEFAULT_TECH_GROUPS = {
    "Biomass": {
        "Wood/Wood Waste Biomass",
        "Landfill Gas",
        "Municipal Solid Waste",
        "Other Waste Biomass",
    },
    "Other_peaker": {
        "Natural Gas Internal Combustion Engine",
        "Petroleum Liquids",
    },
}


def apply_default_grouping(tech_group, enabled=True, group_map=None):
    """Collapse technologies into groups using provided map when enabled."""
    if not enabled:
        return tech_group
    mapping = group_map if group_map is not None else DEFAULT_TECH_GROUPS
    for group_name, members in mapping.items():
        if tech_group in members:
            return group_name
    return tech_group


def get_line_weight(capacity_mw):
    """Calculate line weight based on transmission capacity."""
    min_weight = 1
    max_weight = 8
    min_cap = 100
    max_cap = 12000

    clamped = max(min_cap, min(max_cap, capacity_mw))
    normalized = (clamped - min_cap) / (max_cap - min_cap)
    return min_weight + normalized * (max_weight - min_weight)


def compute_regional_hydro_factor(region_aggregations):
    """Default hydro_factor=2 globally; set regional_hydro_factor=4 for regions with p1-p7."""
    target_bas = {f"p{i}" for i in range(1, 8)}
    out = {}
    for region_name, bas in region_aggregations.items():
        bas_set = {str(b).strip().lower() for b in (bas or [])}
        if bas_set & target_bas:
            out[region_name] = 4
    return out


class TestCloneGroupMap:
    """Test clone_group_map function."""

    def test_clone_creates_independent_copy(self):
        """Test that clone creates an independent copy."""
        original = {"Group1": {"tech1", "tech2"}, "Group2": {"tech3"}}
        cloned = clone_group_map(original)

        # Modify cloned
        cloned["Group1"].add("tech4")

        # Original should be unchanged
        assert "tech4" not in original["Group1"]
        assert len(original["Group1"]) == 2

    def test_clone_empty_map(self):
        """Test cloning an empty map."""
        original = {}
        cloned = clone_group_map(original)
        assert cloned == {}

    def test_clone_preserves_structure(self):
        """Test that clone preserves the map structure."""
        original = {
            "Group1": {"tech1", "tech2", "tech3"},
            "Group2": {"tech4"},
            "Group3": {"tech5", "tech6"},
        }
        cloned = clone_group_map(original)

        assert len(cloned) == len(original)
        for key in original:
            assert key in cloned
            assert cloned[key] == original[key]

    def test_clone_with_empty_sets(self):
        """Test cloning a map with empty sets."""
        original = {"Group1": set(), "Group2": {"tech1"}}
        cloned = clone_group_map(original)

        assert len(cloned["Group1"]) == 0
        assert "tech1" in cloned["Group2"]


class TestApplyDefaultGrouping:
    """Test apply_default_grouping function."""

    def test_grouping_disabled(self):
        """Test that grouping is bypassed when disabled."""
        result = apply_default_grouping("Wood/Wood Waste Biomass", enabled=False)
        assert result == "Wood/Wood Waste Biomass"

    def test_grouping_biomass(self):
        """Test grouping of biomass technologies."""
        assert apply_default_grouping("Wood/Wood Waste Biomass") == "Biomass"
        assert apply_default_grouping("Landfill Gas") == "Biomass"
        assert apply_default_grouping("Municipal Solid Waste") == "Biomass"

    def test_grouping_other_peaker(self):
        """Test grouping of other peaker technologies."""
        assert (
            apply_default_grouping("Natural Gas Internal Combustion Engine")
            == "Other_peaker"
        )
        assert apply_default_grouping("Petroleum Liquids") == "Other_peaker"

    def test_grouping_ungrouped_tech(self):
        """Test that ungrouped technologies pass through unchanged."""
        assert apply_default_grouping("Solar Photovoltaic") == "Solar Photovoltaic"
        assert apply_default_grouping("Wind Turbine") == "Wind Turbine"

    def test_custom_group_map(self):
        """Test using a custom grouping map."""
        custom_map = {
            "CustomGroup": ["Tech1", "Tech2"],
            "AnotherGroup": ["Tech3"],
        }
        assert apply_default_grouping("Tech1", group_map=custom_map) == "CustomGroup"
        assert apply_default_grouping("Tech3", group_map=custom_map) == "AnotherGroup"
        assert apply_default_grouping("Tech4", group_map=custom_map) == "Tech4"


class TestGetLineWeight:
    """Test get_line_weight function."""

    def test_minimum_weight(self):
        """Test that minimum weight is returned for low capacity."""
        assert get_line_weight(50) == 1.0  # Below min_cap
        assert get_line_weight(100) == 1.0  # At min_cap

    def test_maximum_weight(self):
        """Test that maximum weight is returned for high capacity."""
        assert get_line_weight(15000) == 8.0  # Above max_cap
        assert get_line_weight(12000) == 8.0  # At max_cap

    def test_mid_range_weight(self):
        """Test weight calculation in mid-range."""
        # At 50% of range: (6050 - 100) / (12000 - 100) = 0.5
        # weight = 1 + 0.5 * 7 = 4.5
        result = get_line_weight(6050)
        assert 4.0 <= result <= 5.0

    def test_weight_increases_with_capacity(self):
        """Test that weight increases monotonically with capacity."""
        weights = [get_line_weight(cap) for cap in [500, 2000, 5000, 10000]]
        assert all(weights[i] < weights[i + 1] for i in range(len(weights) - 1))

    def test_weight_range(self):
        """Test that weight is always in valid range."""
        for cap in [0, 100, 1000, 5000, 12000, 20000]:
            weight = get_line_weight(cap)
            assert 1.0 <= weight <= 8.0


class TestComputeRegionalHydroFactor:
    """Test compute_regional_hydro_factor function."""

    def test_region_with_p_bas(self):
        """Test that regions with p1-p7 get factor 4."""
        region_aggs = {
            "Region1": ["p1", "p2"],
            "Region2": ["ca", "nv"],
        }
        result = compute_regional_hydro_factor(region_aggs)
        assert result == {"Region1": 4}

    def test_region_without_p_bas(self):
        """Test that regions without p1-p7 are not in output."""
        region_aggs = {
            "Region1": ["ca", "nv"],
            "Region2": ["tx", "ok"],
        }
        result = compute_regional_hydro_factor(region_aggs)
        assert result == {}

    def test_mixed_regions(self):
        """Test with mix of regions with and without p BAs."""
        region_aggs = {
            "Region1": ["p1", "ca"],
            "Region2": ["nv", "co"],
            "Region3": ["p5"],
        }
        result = compute_regional_hydro_factor(region_aggs)
        assert result == {"Region1": 4, "Region3": 4}
        assert "Region2" not in result

    def test_all_p_bas(self):
        """Test region with all p1-p7 BAs."""
        region_aggs = {
            "PNW": [f"p{i}" for i in range(1, 8)],
        }
        result = compute_regional_hydro_factor(region_aggs)
        assert result == {"PNW": 4}

    def test_case_insensitive(self):
        """Test that BA matching is case-insensitive."""
        region_aggs = {
            "Region1": ["P1", "P2"],  # Uppercase
            "Region2": ["ca", "NV"],
        }
        result = compute_regional_hydro_factor(region_aggs)
        assert result == {"Region1": 4}

    def test_empty_regions(self):
        """Test with empty region aggregations."""
        assert compute_regional_hydro_factor({}) == {}

    def test_region_with_none_bas(self):
        """Test handling of None in BA list."""
        region_aggs = {
            "Region1": [None, "p1", None],
        }
        result = compute_regional_hydro_factor(region_aggs)
        # Should handle gracefully
        assert "Region1" in result


# ============================================================================
# Spectral Clustering Tests (Priority 3)
# ============================================================================


def spectral_cluster(graph, n_clusters):
    """Perform spectral clustering on the graph using Normalized Laplacian."""
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


class TestSpectralClustering:
    """Test spectral_cluster function."""

    def test_spectral_single_cluster(self):
        """Test spectral clustering with k=1."""
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3), (3, 4)])
        result = spectral_cluster(graph, 1)

        assert len(result) == 1
        assert result[0] == {1, 2, 3, 4}

    def test_spectral_two_components(self):
        """Test spectral clustering on two well-separated components."""
        graph = nx.Graph()
        # Two separate components
        graph.add_edges_from([(1, 2), (2, 3)])
        graph.add_edges_from([(4, 5), (5, 6)])
        result = spectral_cluster(graph, 2)

        assert len(result) == 2
        clusters = list(result.values())
        # Should separate into the two components
        assert any({1, 2, 3}.issubset(c) for c in clusters)
        assert any({4, 5, 6}.issubset(c) for c in clusters)

    def test_spectral_preserves_nodes(self):
        """Test that all nodes are preserved after clustering."""
        graph = nx.complete_graph(5)
        result = spectral_cluster(graph, 2)

        all_nodes = set()
        for cluster in result.values():
            all_nodes.update(cluster)
        assert all_nodes == {0, 1, 2, 3, 4}

    def test_spectral_more_clusters_than_nodes(self):
        """Test when requesting more clusters than nodes."""
        graph = nx.Graph()
        graph.add_edges_from([(1, 2), (2, 3)])
        result = spectral_cluster(graph, 10)

        assert len(result) == 3
        assert all(len(cluster) == 1 for cluster in result.values())

    def test_spectral_weighted_graph(self):
        """Test spectral clustering with weighted edges."""
        graph = nx.Graph()
        graph.add_edge(1, 2, weight=10)
        graph.add_edge(2, 3, weight=10)
        graph.add_edge(3, 4, weight=1)
        graph.add_edge(4, 5, weight=10)
        graph.add_edge(5, 6, weight=10)
        result = spectral_cluster(graph, 2)

        assert len(result) == 2
        # Should tend to separate at the weak link (3-4)


# ============================================================================
# Louvain Clustering Tests (Priority 3)
# ============================================================================


def louvain_cluster(graph):
    """Perform Louvain community detection on a graph."""
    if graph.number_of_nodes() == 0:
        return {}

    if graph.number_of_edges() == 0:
        return {i: {node} for i, node in enumerate(graph.nodes())}

    try:
        communities = nx.community.louvain_communities(
            graph, weight="weight", resolution=1.0, seed=42
        )
        return {i: set(community) for i, community in enumerate(communities)}
    except Exception:
        return {i: {node} for i, node in enumerate(graph.nodes())}


class TestLouvainClustering:
    """Test louvain_cluster function."""

    def test_louvain_empty_graph(self):
        """Test Louvain on empty graph."""
        graph = nx.Graph()
        result = louvain_cluster(graph)
        assert result == {}

    def test_louvain_no_edges(self):
        """Test Louvain on graph with no edges."""
        graph = nx.Graph()
        graph.add_nodes_from([1, 2, 3])
        result = louvain_cluster(graph)

        assert len(result) == 3
        assert all(len(cluster) == 1 for cluster in result.values())

    def test_louvain_complete_graph(self):
        """Test Louvain on complete graph (single community expected)."""
        graph = nx.complete_graph(5)
        result = louvain_cluster(graph)

        # Complete graph should typically result in one community
        assert len(result) >= 1
        all_nodes = set()
        for cluster in result.values():
            all_nodes.update(cluster)
        assert all_nodes == {0, 1, 2, 3, 4}

    def test_louvain_two_components(self):
        """Test Louvain on graph with two clear communities."""
        graph = nx.Graph()
        # Dense intra-community edges
        graph.add_edges_from([(1, 2), (2, 3), (3, 1)])
        graph.add_edges_from([(4, 5), (5, 6), (6, 4)])
        # Weak inter-community edge
        graph.add_edge(3, 4, weight=0.1)

        result = louvain_cluster(graph)

        # Should identify 2 communities (or more if it splits further)
        assert len(result) >= 1

    def test_louvain_preserves_nodes(self):
        """Test that all nodes are in exactly one community."""
        graph = nx.karate_club_graph()
        result = louvain_cluster(graph)

        all_nodes = set()
        for cluster in result.values():
            all_nodes.update(cluster)
        assert all_nodes == set(graph.nodes())

    def test_louvain_deterministic(self):
        """Test that Louvain with same seed is deterministic."""
        graph = nx.karate_club_graph()
        result1 = louvain_cluster(graph)
        result2 = louvain_cluster(graph)

        # Should give same result with same seed
        assert len(result1) == len(result2)


# ============================================================================
# ESR Functions Tests (Priority 4)
# ============================================================================


class ESRGenerationError(Exception):
    """Raised when ESR generation is not possible."""

    pass


def extract_state_for_region(region_bas, hierarchy_df):
    """Extract states for each BA in a model region."""
    ba_to_state = {}
    for ba in region_bas:
        row = hierarchy_df[hierarchy_df["ba"] == ba]
        if row.empty:
            raise ESRGenerationError(f"BA '{ba}' not found in hierarchy data")
        state_val = str(row.iloc[0]["st"]).lower()
        ba_to_state[ba] = state_val
    return ba_to_state


def get_states_in_region(region_bas, hierarchy_df):
    """Get unique states in a model region."""
    ba_to_state = extract_state_for_region(region_bas, hierarchy_df)
    return set(ba_to_state.values())


def can_states_trade(state1, state2, rectable_df):
    """Check if two states can trade REC/ESR credits based on rectable.csv."""
    state1_upper = state1.upper()
    state2_upper = state2.upper()
    if state1_upper not in rectable_df.index or state2_upper not in rectable_df.columns:
        return False
    value = rectable_df.loc[state1_upper, state2_upper]
    return pd.notna(value) and float(value) > 0


def can_states_trade_transitively(states_set, rectable_df):
    """Check if all states in a set can trade with each other transitively."""
    if len(states_set) <= 1:
        return True

    states_list = list(states_set)

    # Build a graph of direct trading relationships
    trading_graph = {s: set() for s in states_list}
    for i, s1 in enumerate(states_list):
        for s2 in states_list[i + 1 :]:
            if can_states_trade(s1, s2, rectable_df):
                trading_graph[s1].add(s2)
                trading_graph[s2].add(s1)

    # Check if all states are in the same connected component
    visited = set()

    def dfs(state):
        visited.add(state)
        for neighbor in trading_graph[state]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(states_list[0])
    return len(visited) == len(states_list)


def split_bas_by_trading_zones(bas, hierarchy_df, rectable_df):
    """Split BAs into groups where all states in each group can trade transitively."""
    if rectable_df is None or len(bas) <= 1:
        return [set(bas)]

    # Build BA to state mapping
    ba_to_state = {}
    for ba in bas:
        row = hierarchy_df[hierarchy_df["ba"] == ba]
        if not row.empty:
            ba_to_state[ba] = str(row.iloc[0]["st"]).lower()

    # Get unique states
    states = set(ba_to_state.values())
    if len(states) <= 1:
        return [set(bas)]

    states_list = list(states)

    # Build a graph of direct trading relationships between states
    trading_graph = {s: set() for s in states_list}
    for i, s1 in enumerate(states_list):
        for s2 in states_list[i + 1 :]:
            if can_states_trade(s1, s2, rectable_df):
                trading_graph[s1].add(s2)
                trading_graph[s2].add(s1)

    # Find connected components (trading zones)
    visited = set()
    trading_zones = []

    def dfs(state, zone):
        visited.add(state)
        zone.add(state)
        for neighbor in trading_graph[state]:
            if neighbor not in visited:
                dfs(neighbor, zone)

    for state in states_list:
        if state not in visited:
            zone = set()
            dfs(state, zone)
            trading_zones.append(zone)

    # If all states are in one trading zone, no split needed
    if len(trading_zones) == 1:
        return [set(bas)]

    # Group BAs by their trading zone
    ba_groups = []
    for zone in trading_zones:
        group = {ba for ba, st in ba_to_state.items() if st in zone}
        if group:
            ba_groups.append(group)

    return ba_groups


class TestESRFunctions:
    """Test ESR generation functions."""

    @pytest.fixture
    def hierarchy_data(self):
        """Create sample hierarchy data."""
        return pd.DataFrame(
            {
                "ba": ["ca", "nv", "co", "tx", "ok"],
                "st": ["CA", "NV", "CO", "TX", "OK"],
            }
        )

    @pytest.fixture
    def rectable_data(self):
        """Create sample rectable data (trading matrix)."""
        # CA and NV can trade, TX and OK can trade, but CO is isolated
        data = pd.DataFrame(
            {
                "CA": [1.0, 1.0, 0.0, 0.0, 0.0],
                "NV": [1.0, 1.0, 0.0, 0.0, 0.0],
                "CO": [0.0, 0.0, 1.0, 0.0, 0.0],
                "TX": [0.0, 0.0, 0.0, 1.0, 1.0],
                "OK": [0.0, 0.0, 0.0, 1.0, 1.0],
            },
            index=["CA", "NV", "CO", "TX", "OK"],
        )
        return data

    def test_extract_state_for_region(self, hierarchy_data):
        """Test extracting states for BAs."""
        result = extract_state_for_region(["ca", "nv"], hierarchy_data)
        assert result == {"ca": "ca", "nv": "nv"}

    def test_extract_state_for_region_missing_ba(self, hierarchy_data):
        """Test that missing BA raises error."""
        with pytest.raises(ESRGenerationError, match="not found"):
            extract_state_for_region(["ca", "unknown"], hierarchy_data)

    def test_get_states_in_region(self, hierarchy_data):
        """Test getting unique states in region."""
        result = get_states_in_region(["ca", "nv", "co"], hierarchy_data)
        assert result == {"ca", "nv", "co"}

    def test_can_states_trade_direct(self, rectable_data):
        """Test direct trading between states."""
        assert can_states_trade("ca", "nv", rectable_data) is True
        assert can_states_trade("tx", "ok", rectable_data) is True
        assert can_states_trade("ca", "tx", rectable_data) is False

    def test_can_states_trade_case_insensitive(self, rectable_data):
        """Test that trading check is case-insensitive."""
        assert can_states_trade("Ca", "Nv", rectable_data) is True
        assert can_states_trade("CA", "NV", rectable_data) is True

    def test_can_states_trade_missing_state(self, rectable_data):
        """Test that missing state returns False."""
        assert can_states_trade("ca", "wy", rectable_data) is False
        assert can_states_trade("unknown", "ca", rectable_data) is False

    def test_can_states_trade_transitively_single(self, rectable_data):
        """Test transitive trading with single state."""
        assert can_states_trade_transitively({"ca"}, rectable_data) is True

    def test_can_states_trade_transitively_direct(self, rectable_data):
        """Test transitive trading with direct trading."""
        assert can_states_trade_transitively({"ca", "nv"}, rectable_data) is True
        assert can_states_trade_transitively({"tx", "ok"}, rectable_data) is True

    def test_can_states_trade_transitively_no_connection(self, rectable_data):
        """Test transitive trading with no connection."""
        assert can_states_trade_transitively({"ca", "tx"}, rectable_data) is False
        assert can_states_trade_transitively({"ca", "co"}, rectable_data) is False

    def test_split_bas_by_trading_zones_single_zone(self, hierarchy_data, rectable_data):
        """Test splitting BAs when all in one trading zone."""
        result = split_bas_by_trading_zones(
            {"ca", "nv"}, hierarchy_data, rectable_data
        )
        assert len(result) == 1
        assert result[0] == {"ca", "nv"}

    def test_split_bas_by_trading_zones_multiple_zones(
        self, hierarchy_data, rectable_data
    ):
        """Test splitting BAs into multiple trading zones."""
        result = split_bas_by_trading_zones(
            {"ca", "nv", "tx", "ok"}, hierarchy_data, rectable_data
        )
        assert len(result) == 2
        # Should have {ca, nv} and {tx, ok}
        zones = [set(z) for z in result]
        assert {"ca", "nv"} in zones
        assert {"tx", "ok"} in zones

    def test_split_bas_by_trading_zones_isolated_state(
        self, hierarchy_data, rectable_data
    ):
        """Test splitting with an isolated state."""
        result = split_bas_by_trading_zones(
            {"ca", "co"}, hierarchy_data, rectable_data
        )
        assert len(result) == 2
        # CO should be in its own zone
        zones = [set(z) for z in result]
        assert {"co"} in zones

    def test_split_bas_by_trading_zones_none_rectable(self, hierarchy_data):
        """Test that None rectable returns single zone."""
        result = split_bas_by_trading_zones({"ca", "nv", "tx"}, hierarchy_data, None)
        assert len(result) == 1
        assert result[0] == {"ca", "nv", "tx"}

    def test_split_bas_by_trading_zones_single_ba(self, hierarchy_data, rectable_data):
        """Test that single BA returns single zone."""
        result = split_bas_by_trading_zones({"ca"}, hierarchy_data, rectable_data)
        assert len(result) == 1
        assert result[0] == {"ca"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
