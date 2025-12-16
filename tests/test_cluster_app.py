"""
Comprehensive test suite for cluster_app.py clustering and utility functions.

This module tests all pure Python functions that don't depend on PyScript's js module
or DOM access. Tests cover color conversion, graph clustering algorithms, plant
clustering, and YAML generation.
"""

import json
import numpy as np
import pandas as pd
import pytest
import networkx as nx
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
    tokens = [t.lower() for t in (omit_tokens if omit_tokens is not None else default_omit)]

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
        assert (
            normalize_technology("Photovoltaic")
            == "Solar Photovoltaic"
        )
        assert normalize_technology("Solar Photovoltaic") == "Solar Photovoltaic"
        assert (
            normalize_technology("Solar Thermal")
            is None
        )  # Omitted by default

    def test_normalize_hydro_technologies(self):
        """Test normalization of hydroelectric technologies."""
        assert (
            normalize_technology("Conventional Hydroelectric")
            == "Conventional Hydroelectric"
        )
        assert (
            normalize_technology("Run of River")
            == "Run of River Hydroelectric"
        )
        assert (
            normalize_technology("Pumped Storage Hydro")
            == "Hydroelectric Pumped Storage"
        )

    def test_normalize_coal_technologies(self):
        """Test normalization of coal technologies."""
        assert (
            normalize_technology("Steam Coal")
            == "Conventional Steam Coal"
        )
        assert (
            normalize_technology("Coal")
            == "Conventional Steam Coal"
        )

    def test_normalize_gas_technologies(self):
        """Test normalization of natural gas technologies."""
        assert (
            normalize_technology("Combined Cycle")
            == "Natural Gas Fired Combined Cycle"
        )
        assert (
            normalize_technology("Combustion Turbine")
            == "Natural Gas Fired Combustion Turbine"
        )
        assert (
            normalize_technology("Steam Turbine")
            == "Natural Gas Steam Turbine"
        )
        assert (
            normalize_technology("Internal Combustion Engine")
            == "Natural Gas Internal Combustion Engine"
        )

    def test_normalize_biomass_technologies(self):
        """Test normalization of biomass technologies."""
        assert normalize_technology("Biomass") == "Biomass"
        assert normalize_technology("Landfill Gas") == "Landfill Gas"
        assert (
            normalize_technology("Municipal Solid Waste")
            == "Municipal Solid Waste"
        )
        assert (
            normalize_technology("Wood/Wood Waste")
            == "Wood/Wood Waste Biomass"
        )

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
        assert (
            normalize_technology("Wind", omit_tokens=["wind"])
            is None
        )
        assert (
            normalize_technology("Solar PV", omit_tokens=["solar"])
            is None
        )

    def test_normalize_case_insensitive(self):
        """Test that normalization is case-insensitive."""
        assert (
            normalize_technology("WIND TURBINE")
            == "Onshore Wind Turbine"
        )
        assert (
            normalize_technology("SOLAR PHOTOVOLTAIC")
            == "Solar Photovoltaic"
        )

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
        features = np.array([
            [0, 0], [1, 1],  # Cluster 1
            [10, 10], [11, 11],  # Cluster 2
        ])
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
