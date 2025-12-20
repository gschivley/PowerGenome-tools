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
import re
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

        # Settings generation (Settings tab)
        self.settings_yamls = {}  # filename -> yaml string
        self.modified_new_resources = {}  # key -> metadata + schema for resources.yml
        self.atb_options = []  # list[dict] loaded from web/data/atb_options.json
        self.atb_index = {}  # year -> tech -> detail -> sorted(list(cost_case))
        self.atb_years = []  # sorted list of years
        self.plant_cluster_settings = (
            None  # parsed YAML dict from plant clustering output
        )

        # Fuel scenario options (Settings tab)
        self.fuel_prices_df = None  # fuel price scenarios from PowerGenome-data
        self.fuel_scenario_index = {}  # data_year -> fuel -> sorted(list(scenario))

        # ESR (Energy Share Requirements) data
        self.rps_df = None  # RPS policy data
        self.ces_df = None  # CES policy data
        self.rectable_df = None  # State trading rules
        self.pop_fraction_df = None  # Population fractions for BA/state
        self.allowed_techs_df = None  # Allowed techs for RPS/CES
        # ============================================================================
        # ESR Generator (Energy Share Requirements)
        # ============================================================================

        self.esr_zones = None  # Computed ESR zones
        self.esr_map = None  # ESR constraint name -> regions mapping
        self.esr_type_map = None  # ESR constraint name -> "RPS" or "CES"
        self.emission_policies_df = None  # Generated emission_policies.csv


state = AppState()


# ============================================================================
# ESR Generator Functions (Energy Share Requirements)
# ============================================================================


class ESRGenerationError(Exception):
    """Raised when ESR generation is not possible with the given region configuration."""

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


def split_bas_by_trading_zones(bas, hierarchy_df, rectable_df):
    """Split a set of BAs into groups where all states in each group can trade transitively.

    Returns a list of sets, where each set contains BAs whose states form a connected
    trading component. BAs in different sets cannot be clustered together.
    """
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


def can_states_trade(state1, state2, rectable_df):
    """Check if two states can trade REC/ESR credits based on rectable.csv."""
    state1_upper = state1.upper()
    state2_upper = state2.upper()
    if state1_upper not in rectable_df.index or state2_upper not in rectable_df.columns:
        return False
    value = rectable_df.loc[state1_upper, state2_upper]
    return pd.notna(value) and float(value) > 0


def can_states_trade_transitively(states_set, rectable_df):
    """Check if all states in a set can trade with each other transitively.

    States can be in the same zone if they're all connected through trading partners.
    For example, if A trades with C and B trades with C, then A and B can be in the same zone.
    """
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


def build_esr_zones(region_aggregations, hierarchy_df, rectable_df):
    """Infer ESR zones (groups of regions that can trade with each other).

    Each ESR zone represents a group of regions that can all participate in
    the same REC/credit trading market. Regions are placed in separate zones
    if they cannot trade with each other.

    If a region contains states that cannot trade transitively, it is split
    into sub-regions for ESR purposes.

    Returns:
        (zones, expanded_region_aggregations): zones is a list of lists of region names,
        expanded_region_aggregations is a dict mapping region names to BA lists
        (including any _esr1, _esr2 split regions)
    """
    # First, handle regions with non-trading states by splitting them
    expanded_region_aggregations = {}
    for region_name, region_bas in region_aggregations.items():
        # Check if this region needs to be split
        trading_subgroups = split_bas_by_trading_zones(
            region_bas, hierarchy_df, rectable_df
        )
        if len(trading_subgroups) == 1:
            # No split needed
            expanded_region_aggregations[region_name] = region_bas
        else:
            # Split into sub-regions for ESR purposes
            for i, subgroup in enumerate(trading_subgroups):
                sub_name = f"{region_name}_esr{i+1}"
                expanded_region_aggregations[sub_name] = list(subgroup)

    # Build zones by grouping regions that can ALL trade with each other
    # Each region gets its own zone initially, then we try to merge compatible zones
    regions = list(expanded_region_aggregations.keys())

    # Get states for each region
    region_states = {}
    for region in regions:
        region_states[region] = get_states_in_region(
            expanded_region_aggregations[region], hierarchy_df
        )

    # Helper to check if two regions can trade (any state in region1 can trade with any in region2)
    def regions_can_trade(r1, r2):
        for s1 in region_states[r1]:
            for s2 in region_states[r2]:
                if can_states_trade(s1, s2, rectable_df):
                    return True
        return False

    # Helper to check if a region can trade with ALL regions in a zone
    def region_can_join_zone(region, zone_regions):
        for existing_region in zone_regions:
            if not regions_can_trade(region, existing_region):
                return False
        return True

    # Build zones: each region joins the first zone where it can trade with ALL members
    # If no such zone exists, create a new zone
    zones = []
    for region in regions:
        joined = False
        for zone in zones:
            if region_can_join_zone(region, zone):
                zone.append(region)
                joined = True
                break
        if not joined:
            zones.append([region])

    # Sort zones for consistent output
    zones = [sorted(zone) for zone in zones]

    return zones, expanded_region_aggregations


def get_qualified_technologies(plants_df, new_resources, allowed_techs_df):
    """Determine which technologies qualify for RPS and CES policies."""
    rps_keywords = allowed_techs_df["RPS"].dropna().str.lower().tolist()
    ces_keywords = allowed_techs_df["CES"].dropna().str.lower().tolist()

    all_techs = set()
    if plants_df is not None and not plants_df.empty:
        all_techs.update(plants_df["technology"].dropna().astype(str).tolist())
    if new_resources:
        for res in new_resources:
            if isinstance(res, (list, tuple)) and len(res) > 0:
                # Combine technology and tech_detail for matching (e.g., "NaturalGas | CCS")
                tech_name = str(res[0])
                tech_detail = str(res[1]) if len(res) > 1 else ""
                combined = f"{tech_name} {tech_detail}".strip()
                all_techs.add(combined)

    rps_qualified = set()
    ces_qualified = set()
    for tech in all_techs:
        tech_lower = str(tech).lower()
        for keyword in rps_keywords:
            if keyword in tech_lower:
                rps_qualified.add(tech)
                break
        for keyword in ces_keywords:
            if keyword in tech_lower:
                ces_qualified.add(tech)
                break
    return rps_qualified, ces_qualified


def aggregate_policy_for_region(
    region_bas, year, policy_type, hierarchy_df, pop_fraction_df, policy_df
):
    """Compute population-weighted average policy requirement for a model region in a given year.

    The result is a weighted average of each state's policy requirement, where the weights
    are the fraction of the region's total population that resides in each state.
    """
    ba_to_state_val = extract_state_for_region(region_bas, hierarchy_df)

    # First, compute total population in this region and population per state
    region_total_pop = 0.0
    state_pop_in_region = {}  # state -> total population from that state in this region

    for ba in region_bas:
        state_val = ba_to_state_val[ba]
        ba_pop_row = pop_fraction_df[
            (pop_fraction_df["region"] == ba) & (pop_fraction_df["st"] == state_val)
        ]
        if ba_pop_row.empty:
            # Fallback: assume equal population across BAs
            ba_pop = 1.0
        else:
            ba_pop = float(ba_pop_row.iloc[0]["total_population"])

        region_total_pop += ba_pop
        state_pop_in_region[state_val] = (
            state_pop_in_region.get(state_val, 0.0) + ba_pop
        )

    if region_total_pop == 0:
        return 0.0

    # Now compute weighted average of policy requirements by state
    total_requirement = 0.0
    for state_val, state_pop in state_pop_in_region.items():
        weight = state_pop / region_total_pop

        policy_row = policy_df[
            (policy_df["year"] == year) & (policy_df["st"] == state_val)
        ]
        if policy_row.empty:
            policy_value = 0.0
        else:
            col_name = "rps_all" if policy_type == "RPS" else "Value"
            policy_value = (
                float(policy_row.iloc[0][col_name])
                if col_name in policy_row.columns
                else 0.0
            )

        total_requirement += weight * policy_value

    return total_requirement


def generate_emission_policies_csv(
    region_aggregations,
    model_years,
    zones,
    hierarchy_df,
    pop_fraction_df,
    rps_df,
    ces_df,
    include_rps=True,
    include_ces=True,
    case_id="all",
):
    """Generate emission_policies.csv data.

    Returns:
        (df, esr_map, esr_type_map):
            - df: DataFrame with emission policies
            - esr_map: {ESR_1: [regions], ESR_2: [regions], ...}
            - esr_type_map: {ESR_1: "RPS", ESR_2: "CES", ...}
    """
    max_year_in_data_rps = rps_df["year"].max()
    max_year_in_data_ces = ces_df["year"].max()

    rows = []
    esr_constraint_num = 1
    esr_map = {}
    esr_type_map = {}  # Track whether each ESR is RPS or CES
    zone_esr_map = {}

    for zone_idx, zone_regions in enumerate(zones):
        zone_rps = None
        zone_ces = None
        if include_rps:
            zone_rps = f"ESR_{esr_constraint_num}"
            esr_map[zone_rps] = zone_regions
            esr_type_map[zone_rps] = "RPS"
            esr_constraint_num += 1
        if include_ces:
            zone_ces = f"ESR_{esr_constraint_num}"
            esr_map[zone_ces] = zone_regions
            esr_type_map[zone_ces] = "CES"
            esr_constraint_num += 1
        zone_esr_map[zone_idx] = (zone_rps, zone_ces)

    for region_name, region_bas in region_aggregations.items():
        region_zone = None
        for zone_idx, zone_regions in enumerate(zones):
            if region_name in zone_regions:
                region_zone = zone_idx
                break
        if region_zone is None:
            continue

        zone_rps, zone_ces = zone_esr_map[region_zone]
        for year in model_years:
            row = {"case_id": case_id, "year": int(year), "region": region_name}
            use_year_rps = min(year, max_year_in_data_rps)
            use_year_ces = min(year, max_year_in_data_ces)
            if zone_rps:
                rps_val = aggregate_policy_for_region(
                    region_bas,
                    use_year_rps,
                    "RPS",
                    hierarchy_df,
                    pop_fraction_df,
                    rps_df,
                )
                row[zone_rps] = round(float(rps_val), 3)
            if zone_ces:
                ces_val = aggregate_policy_for_region(
                    region_bas,
                    use_year_ces,
                    "CES",
                    hierarchy_df,
                    pop_fraction_df,
                    ces_df,
                )
                row[zone_ces] = round(float(ces_val), 3)
            rows.append(row)

    df = pd.DataFrame(rows)

    for zone_idx, zone_regions in enumerate(zones):
        zone_rps, zone_ces = zone_esr_map[zone_idx]
        if zone_rps and zone_ces:
            for idx, row in df.iterrows():
                if row["region"] in zone_regions:
                    rps_val = df.at[idx, zone_rps]
                    ces_val = df.at[idx, zone_ces]
                    if ces_val < rps_val:
                        df.at[idx, zone_ces] = rps_val

    # Sort ESR columns by numeric ID (ESR_1, ESR_2, ... ESR_10, ESR_11, not ESR_1, ESR_10, ESR_11, ESR_2)
    esr_cols = [c for c in df.columns if c.startswith("ESR_")]
    esr_cols_sorted = sorted(esr_cols, key=lambda x: int(x.split("_")[1]))
    columns = ["case_id", "year", "region"] + esr_cols_sorted
    df = df[columns]

    return df, esr_map, esr_type_map


SETTINGS_FILENAMES = [
    "model_definition.yml",
    "resources.yml",
    "fuels.yml",
    "transmission.yml",
    "distributed_gen.yml",
    "resource_tags.yml",
    "startup_costs.yml",
]


FUEL_PRICES_URLS = [
    # Prefer local copy if present
    "./data/fuel_prices.csv",
    # Fallback to PowerGenome-data (raw content; should be CORS-friendly)
    "https://raw.githubusercontent.com/gschivley/PowerGenome-data/main/data/fuel_prices.csv",
]


DEFAULT_RENEWABLES_CLUSTERS = [
    {
        "region": "all",
        "technology": "landbasedwind",
        "filter": [{"feature": "lcoe", "max": 75}],
        "bin": [{"feature": "lcoe", "weights": "capacity_mw", "mw_per_bin": 25000}],
        "cluster": [{"feature": "cf", "n_clusters": 2}],
    },
    {
        "region": "all",
        "technology": "utilitypv",
        "filter": [{"feature": "lcoe", "max": 40}],
        "bin": [{"feature": "lcoe", "weights": "capacity_mw", "mw_per_bin": 50000}],
    },
]


DEFAULT_GENERATOR_COLUMNS = [
    "region",
    "Resource",
    "technology",
    "cluster",
    "R_ID",
    "Zone",
    "Num_VRE_Bins",
    "CapRes_1",
    "CapRes_2",
    "THERM",
    "VRE",
    "MUST_RUN",
    "STOR",
    "FLEX",
    "LDS",
    "HYDRO",
    "ESR_1",
    "ESR_2",
    "MinCapTag_1",
    "MinCapTag_2",
    "Min_Share",
    "Max_Share",
    "Existing_Cap_MWh",
    "Existing_Cap_MW",
    "Existing_Charge_Cap_MW",
    "num_units",
    "unmodified_existing_cap_mw",
    "New_Build",
    "Cap_Size",
    "Min_Cap_MW",
    "Max_Cap_MW",
    "Max_Cap_MWh",
    "Min_Cap_MWh",
    "Max_Charge_Cap_MW",
    "Min_Charge_Cap_MW",
    "Min_Share_percent",
    "Max_Share_percent",
    "capex_mw",
    "Inv_Cost_per_MWyr",
    "Fixed_OM_Cost_per_MWyr",
    "capex_mwh",
    "Inv_Cost_per_MWhyr",
    "Fixed_OM_Cost_per_MWhyr",
    "Var_OM_Cost_per_MWh",
    "Var_OM_Cost_per_MWh_In",
    "Inv_Cost_Charge_per_MWyr",
    "Fixed_OM_Cost_Charge_per_MWyr",
    "Start_Cost_per_MW",
    "Start_Fuel_MMBTU_per_MW",
    "Heat_Rate_MMBTU_per_MWh",
    "heat_rate_mmbtu_mwh_iqr",
    "heat_rate_mmbtu_mwh_std",
    "Fuel",
    "Min_Power",
    "Self_Disch",
    "Eff_Up",
    "Eff_Down",
    "Hydro_Energy_to_Power_Ratio",
    "Ratio_power_to_energy",
    "Min_Duration",
    "Max_Duration",
    "Max_Flexible_Demand_Delay",
    "Max_Flexible_Demand_Advance",
    "Flexible_Demand_Energy_Eff",
    "Ramp_Up_Percentage",
    "Ramp_Dn_Percentage",
    "Up_Time",
    "Down_Time",
    "NACC_Eff",
    "NACC_Peak_to_Base",
    "Reg_Max",
    "Rsv_Max",
    "Reg_Cost",
    "Rsv_Cost",
    "spur_miles",
    "spur_capex",
    "offshore_spur_miles",
    "offshore_spur_capex",
    "tx_miles",
    "tx_capex",
    "interconnect_annuity",
    "Min_Retired_Cap_MW",
    "Min_Retired_Energy_Cap_MW",
    "Min_Retired_Charge_Cap_MW",
]
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
    # Expose map on window for resize/invalidate hooks
    try:
        window.appMap = state.map
    except Exception:
        pass

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

    # Load rectable for ESR-compatible clustering (optional but needed if checkbox is checked)
    update_loading_text("Loading trading rules...")
    try:
        response = await fetch("./data/state_policies/rectable.csv")
        if response.ok:
            rectable_text = await response.text()
            state.rectable_df = pd.read_csv(StringIO(rectable_text), index_col=0)
    except Exception:
        pass  # Will be loaded later in ESR step if needed


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
    esr_rectable_df=None,
):
    """
    Hierarchical clustering that respects grouping column boundaries.

    Phase 1: Cluster BAs within each grouping column region
    Phase 2: Merge entire grouping column regions together if needed

    Grouping column regions are never split across model regions.

    If esr_rectable_df is provided, also removes edges between BAs in states
    that cannot trade (even transitively), ensuring ESR-compatible clustering.
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

    # Build BA to state mapping for ESR-compatible clustering
    ba_to_state = {}
    if esr_rectable_df is not None:
        for ba in cluster_bas:
            row = hierarchy_df[hierarchy_df["ba"] == ba]
            if not row.empty:
                ba_to_state[ba] = str(row.iloc[0]["st"]).lower()

        # Pre-split grouping column groups by trading zones
        # This ensures BAs in non-trading states are never in the same group
        new_groups = {}
        for group_name, group_bas in groups.items():
            trading_subgroups = split_bas_by_trading_zones(
                group_bas, hierarchy_df, esr_rectable_df
            )
            if len(trading_subgroups) == 1:
                # No split needed
                new_groups[group_name] = group_bas
            else:
                # Split into multiple subgroups with suffixed names
                for i, subgroup in enumerate(trading_subgroups):
                    new_groups[f"{group_name}_tz{i+1}"] = subgroup
        groups = new_groups

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

        # Also remove edges between BAs in non-trading states if ESR-compatible
        if esr_rectable_df is not None:
            for u, v in graph.edges():
                if (u, v) in edges_to_remove:
                    continue  # Already marked for removal
                state_u = ba_to_state.get(u)
                state_v = ba_to_state.get(v)
                if state_u and state_v and state_u != state_v:
                    if not can_states_trade(state_u, state_v, esr_rectable_df):
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
                # If ESR-compatible, skip edges between BAs in non-trading states
                if esr_rectable_df is not None:
                    state_from = ba_to_state.get(ba_from)
                    state_to = ba_to_state.get(ba_to)
                    if state_from and state_to and state_from != state_to:
                        if not can_states_trade(state_from, state_to, esr_rectable_df):
                            continue  # Skip this edge

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
    esr_compatible=False,
):
    """
    Run the clustering algorithm.

    Returns a tuple of (model_regions, region_aggregations, error_message, info)
    where info is a dict with optional metadata like chosen_n and modularity.

    If esr_compatible=True, BAs are first split by trading zone connectivity
    to ensure all states in a resulting region can trade with each other.
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

        # If ESR-compatible clustering is enabled, split BAs by trading zones
        # This ensures that BAs whose states can't trade (even transitively) are kept separate
        if esr_compatible and state.rectable_df is not None and len(cluster_bas) > 1:
            trading_groups = split_bas_by_trading_zones(
                cluster_bas, state.hierarchy_df, state.rectable_df
            )
            # If trading creates multiple disjoint groups, treat the smaller groups as "unclustered"
            # so they don't get merged with incompatible BAs
            if len(trading_groups) > 1:
                info["trading_zone_splits"] = len(trading_groups)

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
                    esr_rectable_df=state.rectable_df if esr_compatible else None,
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
    """Return BA -> model region lookup using current clustering (or selected BAs).

    If clustering has been run, uses the region_aggregations.
    If not, but BAs are selected, maps selected BAs to themselves and excludes others.
    If nothing is selected, maps all BAs to themselves (fallback).

    Note: Only BAs that are part of the clustering (i.e., in region_aggregations)
    are included. Plants in other BAs are excluded from clustering.
    """
    if state.region_aggregations:
        mapping = {}
        for region_name, bas in state.region_aggregations.items():
            for ba in bas:
                mapping[ba] = region_name
        # Only return mapping for clustered BAs - plants in other BAs will be
        # dropped during the merge (their model_region will be NaN)
        return mapping

    # If BAs are selected but clustering hasn't been run, only include selected BAs
    if state.selected_bas:
        return {ba: ba for ba in state.selected_bas}

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

    # Check if ESR-compatible clustering is enabled
    esr_compat_el = document.getElementById("esrCompatibleClustering")
    esr_compatible = esr_compat_el.checked if esr_compat_el else False

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
        esr_compatible=esr_compatible,
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
        esr_note = " or ESR-compatible trading constraints" if esr_compatible else ""
        if num_regions > target_regions:
            set_status(
                f"Warning: Created {num_regions} regions, which is more than the target of {target_regions}. "
                f"This can happen when 'unclustered' groups{esr_note} or disconnected BAs exceed the target. "
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

        try:
            state.plant_cluster_settings = yaml.safe_load(yaml_str)
        except Exception:
            state.plant_cluster_settings = None
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


# =========================================================================
# Settings Generation (Settings tab)
# =========================================================================


async def load_atb_options():
    """Load ATB new-build options index (if present).

    Expected to live at web/data/atb_options.json. This is designed to be regenerated
    offline from technology_costs_atb.parquet; the web app only consumes the index.
    """
    try:
        response = await fetch("./data/atb_options.json")
        if not response.ok:
            state.atb_options = []
            state.atb_index = {}
            state.atb_years = []
            return

        txt = await response.text()
        payload = json.loads(txt)
        options = payload.get("options", []) if isinstance(payload, dict) else []

        # Normalize to list of dicts containing at least data_year/technology/tech_detail/cost_case
        normalized = []
        for row in options:
            if not isinstance(row, dict):
                continue
            if not all(
                k in row
                for k in ("data_year", "technology", "tech_detail", "cost_case")
            ):
                continue
            normalized.append(row)

        state.atb_options = normalized
        years = sorted(
            {
                int(r["data_year"])
                for r in normalized
                if str(r.get("data_year", "")).isdigit()
            }
        )
        state.atb_years = years

        idx = {}
        for r in normalized:
            try:
                year = int(r["data_year"])
            except Exception:
                continue
            tech = str(r.get("technology", "")).strip()
            detail = str(r.get("tech_detail", "")).strip()
            case = str(r.get("cost_case", "")).strip()
            if not tech or not detail or not case:
                continue
            idx.setdefault(year, {}).setdefault(tech, {}).setdefault(detail, set()).add(
                case
            )

        # Convert sets to sorted lists
        state.atb_index = {
            y: {
                t: {d: sorted(list(cases)) for d, cases in details.items()}
                for t, details in techs.items()
            }
            for y, techs in idx.items()
        }
    except Exception:
        state.atb_options = []
        state.atb_index = {}
        state.atb_years = []


async def load_fuel_prices():
    """Load fuel scenario options for the Settings tab.

    Tries a local `./data/fuel_prices.csv` first, then falls back to PowerGenome-data.
    Only scenario availability is used (data_year/fuel/scenario), not prices.
    """
    for url in FUEL_PRICES_URLS:
        try:
            response = await fetch(url)
            if not response.ok:
                continue
            txt = await response.text()
            if txt.startswith("<!"):
                continue
            df = pd.read_csv(StringIO(txt))
            # Must include these columns
            required = {"data_year", "fuel", "scenario"}
            if not required.issubset({c.lower() for c in df.columns}):
                # Normalize case then check
                lower_map = {c.lower(): c for c in df.columns}
                if not required.issubset(set(lower_map.keys())):
                    continue
                df = df.rename(
                    columns={
                        lower_map["data_year"]: "data_year",
                        lower_map["fuel"]: "fuel",
                        lower_map["scenario"]: "scenario",
                    }
                )
            else:
                # Normalize to expected names while preserving existing casing
                lower_map = {c.lower(): c for c in df.columns}
                df = df.rename(
                    columns={
                        lower_map.get("data_year", "data_year"): "data_year",
                        lower_map.get("fuel", "fuel"): "fuel",
                        lower_map.get("scenario", "scenario"): "scenario",
                    }
                )

            df["data_year"] = pd.to_numeric(df["data_year"], errors="coerce").astype(
                "Int64"
            )
            df["fuel"] = df["fuel"].astype(str).str.strip()
            df["scenario"] = df["scenario"].astype(str).str.strip()
            df = df.dropna(subset=["data_year"])
            df = df[(df["fuel"] != "") & (df["scenario"] != "")]

            state.fuel_prices_df = df
            state.fuel_scenario_index = build_fuel_scenario_index(df)
            return
        except Exception:
            continue

    state.fuel_prices_df = None
    state.fuel_scenario_index = {}


def build_fuel_scenario_index(df: pd.DataFrame) -> dict:
    """Build index: data_year -> fuel -> sorted scenarios."""
    idx: dict[int, dict[str, list[str]]] = {}
    if df is None or df.empty:
        return idx

    for (data_year, fuel), sub in df.groupby(["data_year", "fuel"]):
        try:
            y = int(data_year)
        except Exception:
            continue
        scenarios = sorted(
            set(sub["scenario"].dropna().astype(str).str.strip().tolist())
        )
        scenarios = [s for s in scenarios if s]
        if not scenarios:
            continue
        idx.setdefault(y, {})[str(fuel)] = scenarios

    return idx


async def load_esr_data():
    """Load ESR-related CSV files. Called when user accesses ESR step."""
    try:
        update_loading_text("Loading ESR data...")

        # Load RPS data
        response = await fetch("./data/state_policies/rps_fraction.csv")
        if response.ok:
            rps_text = await response.text()
            state.rps_df = pd.read_csv(StringIO(rps_text))
            state.rps_df["st"] = state.rps_df["st"].str.lower()
            # Column is 't' not 'year'
            if "t" in state.rps_df.columns:
                state.rps_df = state.rps_df.rename(columns={"t": "year"})
            state.rps_df["year"] = pd.to_numeric(
                state.rps_df["year"], errors="coerce"
            ).astype("Int64")

        # Load CES data
        response = await fetch("./data/state_policies/ces_fraction.csv")
        if response.ok:
            ces_text = await response.text()
            state.ces_df = pd.read_csv(StringIO(ces_text))
            state.ces_df["st"] = state.ces_df["st"].str.lower()
            # Column might be '*t' or 't' - rename to 'year'
            if "*t" in state.ces_df.columns:
                state.ces_df = state.ces_df.rename(columns={"*t": "year"})
            elif "t" in state.ces_df.columns:
                state.ces_df = state.ces_df.rename(columns={"t": "year"})
            state.ces_df["year"] = pd.to_numeric(
                state.ces_df["year"], errors="coerce"
            ).astype("Int64")

        # Load rectable (trading rules)
        response = await fetch("./data/state_policies/rectable.csv")
        if response.ok:
            rectable_text = await response.text()
            state.rectable_df = pd.read_csv(StringIO(rectable_text), index_col=0)

        # Load population fractions
        response = await fetch("./data/state_policies/state-pop-fraction.csv")
        if response.ok:
            pop_text = await response.text()
            state.pop_fraction_df = pd.read_csv(StringIO(pop_text))
            state.pop_fraction_df["st"] = state.pop_fraction_df["st"].str.lower()
            state.pop_fraction_df["region"] = (
                state.pop_fraction_df["region"].astype(str).str.lower()
            )

        # Load allowed techs
        response = await fetch("./data/state_policies/allowed_techs.csv")
        if response.ok:
            allowed_text = await response.text()
            state.allowed_techs_df = pd.read_csv(StringIO(allowed_text))

    except Exception as e:
        raise Exception(f"Error loading ESR data: {e}")


def _set_select_options_simple(
    select_el, values, *, selected_value=None, empty_label=None
):
    if not select_el:
        return
    vals = [str(v) for v in (values or []) if str(v).strip()]
    if not vals and empty_label:
        vals = [empty_label]
    if selected_value is None and vals:
        selected_value = vals[0]

    parts = []
    for v in vals:
        sel = "selected" if str(v) == str(selected_value) else ""
        parts.append(
            f"<option value='{html.escape(str(v))}' {sel}>{html.escape(str(v))}</option>"
        )
    select_el.innerHTML = "".join(parts)


def _default_scenario_for_fuel(fuel: str, scenarios: list[str]) -> str | None:
    scenarios_set = {str(s) for s in (scenarios or [])}
    if fuel == "coal" and "no_111d" in scenarios_set:
        return "no_111d"
    if "reference" in scenarios_set:
        return "reference"
    return scenarios[0] if scenarios else None


def populate_fuel_scenario_selects(event=None):
    """Populate the Fuel Scenarios selects based on the selected fuel data year."""
    year_el = document.getElementById("fuelDataYear")
    help_el = document.getElementById("fuelScenarioHelp")

    coal_el = document.getElementById("fuelScenarioCoal")
    gas_el = document.getElementById("fuelScenarioNaturalGas")
    dist_el = document.getElementById("fuelScenarioDistillate")
    ura_el = document.getElementById("fuelScenarioUranium")

    try:
        selected_year = int(_get_select_value(year_el, 0) or 0)
    except Exception:
        selected_year = 0

    if not state.fuel_scenario_index or selected_year not in state.fuel_scenario_index:
        # Fallback: just offer 'reference'
        _set_select_options_simple(coal_el, ["reference"], selected_value="reference")
        _set_select_options_simple(gas_el, ["reference"], selected_value="reference")
        _set_select_options_simple(dist_el, ["reference"], selected_value="reference")
        _set_select_options_simple(ura_el, ["reference"], selected_value="reference")
        if help_el:
            help_el.textContent = (
                "Fuel scenario options not available for this year; using 'reference'."
            )
        return

    year_map = state.fuel_scenario_index.get(selected_year, {})

    def set_for(fuel_key: str, select_el):
        scenarios = year_map.get(fuel_key, ["reference"])
        current = _get_select_value(select_el, None)
        default_val = _default_scenario_for_fuel(fuel_key, scenarios)
        chosen = current if current in scenarios else default_val
        _set_select_options_simple(select_el, scenarios, selected_value=chosen)

    set_for("coal", coal_el)
    set_for("naturalgas", gas_el)
    set_for("distillate", dist_el)
    set_for("uranium", ura_el)

    if help_el:
        # Inform about coal default if relevant
        coal_scenarios = year_map.get("coal", [])
        if "no_111d" in set(coal_scenarios):
            help_el.textContent = (
                "Coal defaults to 'no_111d' for this year (available)."
            )
        else:
            help_el.textContent = (
                "Coal 'no_111d' not available for this year; defaulting to 'reference'."
            )


def populate_fuel_data_year_select(event=None):
    """Populate the Fuel Data Year dropdown from loaded fuel_prices.csv."""
    year_el = document.getElementById("fuelDataYear")
    if not year_el:
        return

    # Gather available years from loaded index
    if not state.fuel_scenario_index:
        # Fallback to a reasonable default when fuel_prices.csv can't be loaded
        _set_select_options_simple(year_el, [2025], selected_value="2025")
        return

    years = sorted(state.fuel_scenario_index.keys())
    current = _get_select_value(year_el, None)

    # Choose a default: keep current if valid; else prefer 2025 if present; else latest.
    selected = None
    try:
        current_int = int(current) if current is not None else None
    except Exception:
        current_int = None

    if current_int in years:
        selected = current_int
    elif 2025 in years:
        selected = 2025
    elif years:
        selected = years[-1]

    _set_select_options_simple(
        year_el, years, selected_value=str(selected) if selected is not None else None
    )


def _set_select_options(select_el, values, *, selected_value=None):
    if not select_el:
        return
    safe_values = [str(v) for v in values]
    if selected_value is None and safe_values:
        selected_value = safe_values[0]
    parts = []
    for v in safe_values:
        sel = "selected" if v == str(selected_value) else ""
        parts.append(
            f"<option value='{html.escape(v)}' {sel}>{html.escape(v)}</option>"
        )
    select_el.innerHTML = "".join(parts)


def _get_select_value(el, default=None):
    if not el:
        return default
    try:
        return el.value
    except Exception:
        return default


def populate_atb_picker():
    """Populate the ATB picker selects in the Settings tab."""
    year_el = document.getElementById("atbYearSelect")
    tech_el = document.getElementById("atbTechSelect")
    detail_el = document.getElementById("atbTechDetailSelect")
    case_el = document.getElementById("atbCostCaseSelect")

    if not (year_el and tech_el and detail_el and case_el):
        return

    years = state.atb_years
    if not years:
        _set_select_options(year_el, ["(no ATB index found)"])
        _set_select_options(tech_el, [])
        _set_select_options(detail_el, [])
        _set_select_options(case_el, [])
        return

    latest_year = max(years)
    selected_year = int(_get_select_value(year_el, latest_year) or latest_year)
    if selected_year not in state.atb_index:
        selected_year = latest_year

    _set_select_options(year_el, years, selected_value=str(selected_year))

    techs = sorted(state.atb_index.get(selected_year, {}).keys())
    selected_tech = _get_select_value(tech_el, techs[0] if techs else None)
    if selected_tech not in techs and techs:
        selected_tech = techs[0]
    _set_select_options(tech_el, techs, selected_value=selected_tech)

    details = sorted(
        state.atb_index.get(selected_year, {}).get(selected_tech, {}).keys()
    )
    selected_detail = _get_select_value(detail_el, details[0] if details else None)
    if selected_detail not in details and details:
        selected_detail = details[0]
    _set_select_options(detail_el, details, selected_value=selected_detail)

    cases = (
        state.atb_index.get(selected_year, {})
        .get(selected_tech, {})
        .get(selected_detail, [])
    )
    selected_case = _get_select_value(case_el, cases[0] if cases else None)
    if selected_case not in cases and cases:
        selected_case = cases[0]
    _set_select_options(case_el, cases, selected_value=selected_case)


def on_atb_picker_change(event=None):
    populate_atb_picker()


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


def render_new_resources_list():
    container = document.getElementById("newResourcesList")
    raw_el = document.getElementById("newResourcesRaw")
    if not container or not raw_el:
        return

    items = parse_new_resources_text(raw_el.value)
    if not items:
        container.innerHTML = "<em>No new-build resources selected yet.</em>"
        return

    parts = []
    for tech, detail, case, size in items:
        parts.append(
            f"<div class='candidate-item'><strong>{html.escape(str(tech))}</strong> — {html.escape(str(detail))} — {html.escape(str(case))} — {int(size)} MW</div>"
        )
    container.innerHTML = "".join(parts)


def on_add_new_resource(event):
    raw_el = document.getElementById("newResourcesRaw")
    if not raw_el:
        return

    year_el = document.getElementById("atbYearSelect")
    tech_el = document.getElementById("atbTechSelect")
    detail_el = document.getElementById("atbTechDetailSelect")
    case_el = document.getElementById("atbCostCaseSelect")
    size_el = document.getElementById("atbSizeMw")

    tech = _get_select_value(tech_el, "").strip()
    detail = _get_select_value(detail_el, "").strip()
    case = _get_select_value(case_el, "").strip()
    try:
        size = int(float(_get_select_value(size_el, 1)))
    except Exception:
        size = 1

    if not tech or not detail or not case:
        set_status("ATB index not available; add manually below.", "error")
        return

    line = f"{tech} | {detail} | {case} | {size}"
    existing = raw_el.value.strip()
    raw_el.value = (existing + "\n" + line).strip() if existing else line
    render_new_resources_list()


def render_modified_resources_list():
    container = document.getElementById("modifiedResourcesList")
    if not container:
        return
    if not state.modified_new_resources:
        container.innerHTML = "<em>No modified resources added yet.</em>"
        return

    parts = []
    for key in sorted(state.modified_new_resources.keys()):
        item = state.modified_new_resources[key]
        new_tech = item.get("new_technology")
        fuel_desc = item.get("fuel_desc", "")
        tag_class = item.get("tag_class", "")
        try:
            n_mods = len(item.get("attr_modifiers") or {})
        except Exception:
            n_mods = 0
        parts.append(
            f"<div class='candidate-item'><strong>{html.escape(key)}</strong> — {html.escape(str(new_tech))} — {html.escape(str(tag_class))} — {html.escape(str(fuel_desc))} — {n_mods} modifiers</div>"
        )
    container.innerHTML = "".join(parts)


def on_clear_modified_resources(event):
    state.modified_new_resources = {}
    render_modified_resources_list()


def _prefix_from_new_technology(new_technology):
    t = str(new_technology).strip()
    if not t:
        return ""
    if t.endswith("_"):
        return t
    return f"{t}_"


def on_add_modified_resource(event):
    name_el = document.getElementById("modResName")
    base_tech_el = document.getElementById("modBaseTech")
    base_detail_el = document.getElementById("modBaseTechDetail")
    base_case_el = document.getElementById("modBaseCostCase")
    base_size_el = document.getElementById("modBaseSizeMw")
    new_tech_el = document.getElementById("modNewTech")
    new_detail_el = document.getElementById("modNewTechDetail")
    new_case_el = document.getElementById("modNewCostCase")

    attr_mods_el = document.getElementById("modAttrModifiers")

    fuel_type_el = document.getElementById("modFuelType")
    std_fuel_el = document.getElementById("modStandardFuel")
    new_fuel_name_el = document.getElementById("modNewFuelName")
    new_fuel_price_el = document.getElementById("modNewFuelPrice")
    new_fuel_ef_el = document.getElementById("modNewFuelEf")

    tag_class_el = document.getElementById("modTagClass")
    is_commit_el = document.getElementById("modIsCommit")

    key = str(_get_select_value(name_el, "")).strip()
    if not key:
        set_status("Modified resource needs a name/key.", "error")
        return

    base_tech = str(_get_select_value(base_tech_el, "")).strip()
    base_detail = str(_get_select_value(base_detail_el, "")).strip()
    base_case = str(_get_select_value(base_case_el, "")).strip()
    try:
        base_size = int(float(_get_select_value(base_size_el, 100)))
    except Exception:
        base_size = 100

    new_tech = str(_get_select_value(new_tech_el, "")).strip()
    new_detail = str(_get_select_value(new_detail_el, "")).strip()
    new_case = str(_get_select_value(new_case_el, "")).strip()

    # Optional attribute modifiers (YAML mapping)
    attr_modifiers = {}
    if attr_mods_el is not None:
        raw_mods = str(_get_select_value(attr_mods_el, "") or "").strip()
        if raw_mods:
            try:
                parsed = yaml.safe_load(raw_mods)
            except Exception as exc:
                set_status(f"Attribute modifiers YAML error: {exc}", "error")
                return

            if not isinstance(parsed, dict):
                set_status(
                    "Attribute modifiers must be a YAML mapping (key: value).",
                    "error",
                )
                return

            reserved_keys = {
                "technology",
                "tech_detail",
                "cost_case",
                "size_mw",
                "new_technology",
                "new_tech_detail",
                "new_cost_case",
            }
            overlap = reserved_keys & set(parsed.keys())
            if overlap:
                set_status(
                    "Attribute modifiers cannot change core resource identity fields. "
                    "These fields are managed automatically by the resource definition; "
                    "please use different keys for custom attributes.",
                    "error",
                )
                return

            attr_modifiers = parsed

    if not (
        base_tech and base_detail and base_case and new_tech and new_detail and new_case
    ):
        set_status(
            "Fill out both the base ATB resource and the new resource identity.",
            "error",
        )
        return

    fuel_type = str(_get_select_value(fuel_type_el, "standard"))
    std_fuel = str(_get_select_value(std_fuel_el, "naturalgas"))

    new_fuel_name = str(_get_select_value(new_fuel_name_el, "")).strip()
    try:
        new_fuel_price = float(_get_select_value(new_fuel_price_el, 0))
    except Exception:
        new_fuel_price = 0.0
    try:
        new_fuel_ef = float(_get_select_value(new_fuel_ef_el, 0))
    except Exception:
        new_fuel_ef = 0.0

    tag_class = str(_get_select_value(tag_class_el, "THERM"))
    is_commit = (
        bool(is_commit_el.checked) if (is_commit_el and tag_class == "THERM") else False
    )

    if fuel_type == "new":
        if not new_fuel_name:
            set_status("New fuel requires a fuel name.", "error")
            return
        if new_fuel_price < 0:
            set_status("Fuel price must be >= 0.", "error")
            return
        if new_fuel_ef < 0:
            set_status("Emission factor must be >= 0.", "error")
            return

    fuel_desc = (
        std_fuel
        if fuel_type == "standard"
        else f"{new_fuel_name} @ ${new_fuel_price}/MMBtu"
    )

    state.modified_new_resources[key] = {
        # resources.yml schema
        "technology": base_tech,
        "tech_detail": base_detail,
        "cost_case": base_case,
        "size_mw": int(base_size),
        "new_technology": new_tech,
        "new_tech_detail": new_detail,
        "new_cost_case": new_case,
        "attr_modifiers": attr_modifiers,
        # metadata for fuels.yml and resource_tags.yml
        "fuel_type": fuel_type,
        "standard_fuel": std_fuel,
        "new_fuel_name": new_fuel_name,
        "new_fuel_price": float(new_fuel_price),
        "new_fuel_emission_factor": float(new_fuel_ef),
        "tag_class": tag_class,
        "is_commit": bool(is_commit),
        "fuel_desc": fuel_desc,
    }

    # Clear modifiers editor for next entry
    if attr_mods_el is not None:
        try:
            attr_mods_el.value = ""
        except Exception:
            pass

    render_modified_resources_list()
    set_status(f"Added modified resource: {key}", "success")


def _get_region_aggregations_or_raise():
    if state.region_aggregations:
        return state.region_aggregations
    raise Exception("Run region clustering first to generate model regions.")


def compute_regional_hydro_factor(region_aggregations):
    """Default hydro_factor=2 globally; set regional_hydro_factor=4 for any model region that contains BA p1-p7."""
    target_bas = {f"p{i}" for i in range(1, 8)}
    out = {}
    for region_name, bas in region_aggregations.items():
        bas_set = {str(b).strip().lower() for b in (bas or [])}
        if bas_set & target_bas:
            out[region_name] = 4
    return out


def generate_resources_settings():
    region_aggs = _get_region_aggregations_or_raise()

    # Existing generator clustering: prefer plant clustering output if available
    cluster_settings = state.plant_cluster_settings or {}
    existing_num_clusters = cluster_settings.get("num_clusters")
    if not isinstance(existing_num_clusters, dict) or not existing_num_clusters:
        # Minimal fallback (users should run plant clustering)
        existing_num_clusters = {
            "Conventional Steam Coal": 1,
            "Natural Gas Fired Combined Cycle": 1,
            "Natural Gas Fired Combustion Turbine": 1,
            "Nuclear": 1,
            "Conventional Hydroelectric": 1,
            "Solar Photovoltaic": 1,
            "Onshore Wind Turbine": 1,
            "Batteries": 1,
        }

    group_tech = bool(cluster_settings.get("group_technologies", True))
    tech_groups = cluster_settings.get("tech_groups")
    if not isinstance(tech_groups, dict):
        tech_groups = {
            "Biomass": [
                "Wood/Wood Waste Biomass",
                "Landfill Gas",
                "Municipal Solid Waste",
                "Other Waste Biomass",
            ],
            "Other_peaker": [
                "Natural Gas Internal Combustion Engine",
                "Petroleum Liquids",
            ],
        }

    alt_num_clusters = cluster_settings.get("alt_num_clusters")
    if not isinstance(alt_num_clusters, dict) or not alt_num_clusters:
        alt_num_clusters = None

    # New-build resources come from textarea
    raw_el = document.getElementById("newResourcesRaw")
    new_resources = parse_new_resources_text(raw_el.value if raw_el else "")
    if not new_resources:
        # Seed a minimal starter set
        new_resources = [
            ["NaturalGas", "1-on-1 Combined Cycle (H-Frame)", "Moderate", 500],
            ["LandbasedWind", "Class3", "Moderate", 1],
            ["UtilityPV", "Class1", "Moderate", 1],
            ["Utility-Scale Battery Storage", "Lithium Ion", "Moderate", 1],
            ["Nuclear", "Nuclear - Large", "Moderate", 1000],
        ]

    # Hydro defaults
    hydro_factor = 2
    regional_hydro = compute_regional_hydro_factor(region_aggs)

    # Resources inputs
    resource_data_year = int(
        _get_select_value(document.getElementById("targetUsdYear"), 2024)
    )
    # Keep resource_data_year separate from target_usd_year for future; default to targetUsdYear for MVP
    resource_financial_case = "Market"
    resource_cap_recovery_years = 20
    interconnect_capex_mw = 100000

    out = {
        "cluster_with_retired_gens": True,
        "num_clusters": existing_num_clusters,
        "group_technologies": bool(group_tech),
        "tech_groups": tech_groups,
        "regional_no_grouping": None,
        "alt_num_clusters": alt_num_clusters if alt_num_clusters is not None else None,
        "hydro_factor": hydro_factor,
        "regional_hydro_factor": regional_hydro if regional_hydro else None,
        "energy_storage_duration": {
            "Hydroelectric Pumped Storage": 15.5,
            "Batteries": 2,
        },
        "resource_data_year": resource_data_year,
        "resource_financial_case": resource_financial_case,
        "resource_cap_recovery_years": resource_cap_recovery_years,
        "alt_resource_cap_recovery_years": {
            "Battery": 15,
            "Nuclear": 40,
        },
        "new_resources": new_resources,
        "interconnect_capex_mw": interconnect_capex_mw,
        "cache_resource_clusters": True,
        "use_resource_clusters_cache": True,
        "renewables_clusters": DEFAULT_RENEWABLES_CLUSTERS,
        "modified_new_resources": (
            {
                k: (
                    {
                        "technology": v["technology"],
                        "tech_detail": v["tech_detail"],
                        "cost_case": v["cost_case"],
                        "size_mw": v["size_mw"],
                        "new_technology": v["new_technology"],
                        "new_tech_detail": v["new_tech_detail"],
                        "new_cost_case": v["new_cost_case"],
                        **(
                            v.get("attr_modifiers")
                            if isinstance(v.get("attr_modifiers"), dict)
                            else {}
                        ),
                    }
                )
                for k, v in sorted(state.modified_new_resources.items())
            }
            if state.modified_new_resources
            else None
        ),
    }

    # Remove nulls to keep YAML clean
    out = {k: v for k, v in out.items() if v is not None}
    return yaml.dump(out, default_flow_style=False, sort_keys=False)


def generate_fuels_settings():
    fuel_year = int(_get_select_value(document.getElementById("fuelDataYear"), 2025))

    # Fuel scenarios: default coal to no_111d if present for selected year; otherwise reference.
    coal_sel = _get_select_value(document.getElementById("fuelScenarioCoal"), None)
    gas_sel = _get_select_value(document.getElementById("fuelScenarioNaturalGas"), None)
    dist_sel = _get_select_value(
        document.getElementById("fuelScenarioDistillate"), None
    )
    ura_sel = _get_select_value(document.getElementById("fuelScenarioUranium"), None)

    # Ensure selects are populated (e.g., if user generates settings before load finishes)
    if not coal_sel or not gas_sel or not dist_sel or not ura_sel:
        populate_fuel_scenario_selects()
        coal_sel = _get_select_value(
            document.getElementById("fuelScenarioCoal"), "reference"
        )
        gas_sel = _get_select_value(
            document.getElementById("fuelScenarioNaturalGas"), "reference"
        )
        dist_sel = _get_select_value(
            document.getElementById("fuelScenarioDistillate"), "reference"
        )
        ura_sel = _get_select_value(
            document.getElementById("fuelScenarioUranium"), "reference"
        )

    fuel_scenarios = {
        "coal": str(coal_sel or "reference"),
        "naturalgas": str(gas_sel or "reference"),
        "distillate": str(dist_sel or "reference"),
        "uranium": str(ura_sel or "reference"),
    }

    tech_fuel_map = {
        "Conventional Steam Coal": "coal",
        "Natural Gas Fired Combined Cycle": "naturalgas",
        "Natural Gas Fired Combustion Turbine": "naturalgas",
        "Natural Gas Steam Turbine": "naturalgas",
        "Natural Gas Internal Combustion Engine": "naturalgas",
        "Other_peaker": "naturalgas",
        "NaturalGas": "naturalgas",
        "Petroleum Liquids": "distillate",
        "Nuclear": "uranium",
    }

    fuel_emission_factors = {
        "naturalgas": 0.05306,
        "coal": 0.09552,
        "distillate": 0.07315,
    }

    user_fuel_price = {}

    # Modified resources can introduce new fuels and/or new mappings
    for _, item in state.modified_new_resources.items():
        prefix = _prefix_from_new_technology(item.get("new_technology"))
        if not prefix:
            continue
        if item.get("fuel_type") == "new":
            fuel_name = str(item.get("new_fuel_name") or "").strip()
            if not fuel_name:
                continue
            fuel_scenarios.setdefault(fuel_name, "reference")
            tech_fuel_map[prefix] = fuel_name
            user_fuel_price[fuel_name] = float(item.get("new_fuel_price", 0.0))
            fuel_emission_factors[fuel_name] = float(
                item.get("new_fuel_emission_factor", 0.0)
            )
        else:
            std_fuel = str(item.get("standard_fuel") or "naturalgas")
            tech_fuel_map[prefix] = std_fuel

    out = {
        "fuel_data_year": fuel_year,
        "fuel_scenarios": fuel_scenarios,
        "tech_fuel_map": tech_fuel_map,
        "fuel_emission_factors": fuel_emission_factors,
    }
    if user_fuel_price:
        out["user_fuel_price"] = user_fuel_price

    return yaml.dump(out, default_flow_style=False, sort_keys=False)


def generate_transmission_settings():
    out = {
        "tx_expansion_per_period": 1.0,
        "tx_expansion_mw_per_period": 400,
    }
    return yaml.dump(out, default_flow_style=False, sort_keys=False)


def generate_distributed_gen_settings():
    out = {
        "dg_as_resource": True,
        "avg_distribution_loss": 0.0453,
    }
    return yaml.dump(out, default_flow_style=False, sort_keys=False)


def generate_startup_costs_settings():
    out = {
        "startup_fuel_use": {
            "Conventional Steam Coal": 16.5,
            "Natural Gas Fired Combined Cycle": 2.0,
            "Natural Gas Fired Combustion Turbine": 3.5,
            "Natural Gas Steam Turbine": 13.7,
            "NaturalGas_1-on-1": 2.0,
            "NaturalGas_Combustion": 3.5,
        },
        "startup_vom_costs_mw": {
            "coal_small_sub": 2.81,
            "coal_large_sub": 2.69,
            "coal_supercritical": 2.98,
            "gas_cc": 1.03,
            "gas_large_ct": 0.77,
            "gas_aero_ct": 0.70,
            "gas_steam": 1.03,
            "nuclear": 5.4,
        },
        "startup_vom_costs_usd_year": 2011,
        "startup_costs_type": "startup_costs_per_cold_start_mw",
        "startup_costs_per_cold_start_mw": {
            "coal_small_sub": 147,
            "coal_large_sub": 105,
            "coal_supercritical": 104,
            "gas_cc": 79,
            "gas_large_ct": 103,
            "gas_aero_ct": 32,
            "gas_steam": 75,
            "nuclear": 210,
        },
        "startup_costs_per_cold_start_usd_year": 2011,
        "existing_startup_costs_tech_map": {
            "Conventional Steam Coal": "coal_large_sub",
            "Natural Gas Fired Combined Cycle": "gas_cc",
            "Natural Gas Fired Combustion Turbine": "gas_large_ct",
            "Natural Gas Steam Turbine": "gas_steam",
            "Nuclear": "nuclear",
            "Other_peaker": "gas_steam",
        },
        "new_build_startup_costs": {
            "Coal_CCS30": "coal_supercritical",
            "Coal_CCS90": "coal_supercritical",
            "Coal_IGCC": "coal_supercritical",
            "Coal_new": "coal_supercritical",
            "NaturalGas_CT": "gas_large_ct",
            "NaturalGas_CC": "gas_cc",
            "NaturalGas_CCS100": "gas_cc",
            "Nuclear_Nuclear": "nuclear",
            "NaturalGas_1-on-1": "gas_cc",
            "NaturalGas_Combustion": "gas_large_ct",
        },
    }
    return yaml.dump(out, default_flow_style=False, sort_keys=False)


def generate_resource_tags_settings():
    # Base tag names (ESR tags will be added dynamically)
    base_tag_names = [
        "THERM",
        "VRE",
        "Num_VRE_Bins",
        "MUST_RUN",
        "STOR",
        "FLEX",
        "HYDRO",
        "LDS",
        "Commit",
    ]

    # Collect ESR tag names from state (if ESR analysis has been run)
    esr_tag_names = []
    if state.esr_map:
        esr_tag_names = sorted(state.esr_map.keys(), key=lambda x: int(x.split("_")[1]))

    # Remaining tag names
    suffix_tag_names = [
        "New_Build",
        "CapRes_1",
        "CapRes_2",
        "MinCapTag_1",
        "MinCapTag_2",
        "MinCapTag_3",
        "Reg_Max",
        "Rsv_Max",
    ]

    tag_names = base_tag_names + esr_tag_names + suffix_tag_names

    values = {
        "THERM": {
            "Conventional Steam Coal": 1,
            "Natural Gas Fired Combined Cycle": 1,
            "Natural Gas Fired Combustion Turbine": 1,
            "Natural Gas Internal Combustion Engine": 1,
            "Natural Gas Steam Turbine": 1,
            "Other_peaker": 1,
            "Petroleum Liquids": 1,
            "Nuclear": 1,
            "NaturalGas_": 1,
        },
        "VRE": {
            "LandbasedWind": 1,
            "Onshore Wind": 1,
            "OffshoreWind": 1,
            "Offshore Wind Turbine": 1,
            "Solar Photovoltaic": 1,
            "UtilityPV": 1,
        },
        "Num_VRE_Bins": {
            "LandbasedWind": 1,
            "Onshore Wind": 1,
            "OffshoreWind": 1,
            "Offshore Wind Turbine": 1,
            "Solar Photovoltaic": 1,
            "UtilityPV": 1,
        },
        "STOR": {
            "Batteries": 1,
            "Battery": 1,
            "Hydroelectric Pumped Storage": 1,
        },
        "HYDRO": {
            "Conventional Hydroelectric": 1,
            "Hydropower": 1,
        },
        "MUST_RUN": {
            "Small Hydroelectric": 1,
            "Geothermal": 1,
            "Wood/Wood Waste Biomass": 1,
            "Biomass": 1,
            "distributed_gen": 1,
        },
        "Commit": {
            "Conventional Steam Coal": 1,
            "Natural Gas Fired Combined Cycle": 1,
            "Natural Gas Fired Combustion Turbine": 1,
            "Natural Gas Internal Combustion Engine": 1,
            "Natural Gas Steam Turbine": 1,
            "Other_peaker": 1,
            "Petroleum Liquids": 1,
            "Nuclear": 1,
        },
        "New_Build": {
            "NaturalGas": 1,
            "LandbasedWind": 1,
            "UtilityPV": 1,
            "Battery": 1,
            "Nuclear_Nuclear": 1,
        },
        "CapRes_1": {
            "Conventional Steam Coal": 0.9,
            "Natural Gas Fired Combined Cycle": 0.9,
            "Natural Gas Fired Combustion Turbine": 0.9,
            "Natural Gas Internal Combustion Engine": 0.9,
            "Natural Gas Steam Turbine": 0.9,
            "Other_peaker": 0.9,
            "Petroleum Liquids": 0.9,
            "Nuclear": 0.9,
            "LandbasedWind": 0.8,
            "UtilityPV": 0.8,
            "Battery": 0.95,
            "Hydroelectric Pumped Storage": 0.95,
        },
        "CapRes_2": {
            "Conventional Steam Coal": 0.9,
            "Natural Gas Fired Combined Cycle": 0.9,
            "Natural Gas Fired Combustion Turbine": 0.9,
            "Natural Gas Internal Combustion Engine": 0.9,
            "Natural Gas Steam Turbine": 0.9,
            "Other_peaker": 0.9,
            "Petroleum Liquids": 0.9,
            "Nuclear": 0.9,
            "LandbasedWind": 0.8,
            "UtilityPV": 0.8,
            "Battery": 0.95,
            "Hydroelectric Pumped Storage": 0.95,
        },
        "MinCapTag_1": {},
        "MinCapTag_2": {},
        "MinCapTag_3": {},
        "FLEX": {},
        "LDS": {},
        "Reg_Max": {},
        "Rsv_Max": {},
    }

    # Add empty ESR entries to model_tag_values (actual values are in regional_tag_values)
    for esr_name in esr_tag_names:
        values[esr_name] = {}

    # Apply modified resource tag choices
    for _, item in state.modified_new_resources.items():
        prefix = _prefix_from_new_technology(item.get("new_technology"))
        if not prefix:
            continue
        tag_class = str(item.get("tag_class") or "").strip()
        if tag_class:
            values.setdefault(tag_class, {})[prefix] = 1
        if tag_class == "THERM" and bool(item.get("is_commit")):
            values.setdefault("Commit", {})[prefix] = 1
        values.setdefault("New_Build", {})[prefix] = 1

    out = {
        "model_tag_names": tag_names,
        "default_model_tag": 0,
        "model_tag_values": {k: v for k, v in values.items() if k in tag_names},
    }

    # Generate regional_tag_values for ESR constraints
    # Maps region -> ESR_name -> {tech: 1} for qualified technologies
    if state.esr_map and state.esr_type_map:
        regional_tag_values = {}

        for esr_name, regions in state.esr_map.items():
            esr_type = state.esr_type_map.get(esr_name)
            if not esr_type:
                continue

            # Get qualified technologies based on ESR type
            if esr_type == "RPS":
                qualified_techs = getattr(state, "esr_rps_techs", set()) or set()
            else:  # CES
                qualified_techs = getattr(state, "esr_ces_techs", set()) or set()

            # Build tech map for this ESR
            tech_map = {tech: 1 for tech in sorted(qualified_techs)}

            # Apply to each region in this ESR zone
            for region in regions:
                if region not in regional_tag_values:
                    regional_tag_values[region] = {}
                regional_tag_values[region][esr_name] = tech_map

        if regional_tag_values:
            # Sort regions for consistent output
            out["regional_tag_values"] = {
                region: regional_tag_values[region]
                for region in sorted(regional_tag_values.keys())
            }

    return yaml.dump(out, default_flow_style=False, sort_keys=False)


def generate_emission_policies_settings():
    """Generate emission_policies.csv as a string."""
    if state.emission_policies_df is None:
        return None
    return state.emission_policies_df.to_csv(index=False)


def generate_model_definition_settings():
    region_aggs = _get_region_aggregations_or_raise()
    model_regions = sorted(region_aggs.keys())

    target_usd_year = int(
        _get_select_value(document.getElementById("targetUsdYear"), 2024)
    )
    utc_offset = int(_get_select_value(document.getElementById("utcOffset"), -5))
    model_years = parse_int_list(
        _get_select_value(document.getElementById("modelYears"), "")
    )
    planning_years = parse_int_list(
        _get_select_value(document.getElementById("planningYears"), "")
    )

    if not model_years or not planning_years or len(model_years) != len(planning_years):
        raise Exception("Model years and first planning years must be the same length.")

    out = {
        "model_regions": model_regions,
        "region_aggregations": region_aggs,
        "target_usd_year": target_usd_year,
        "model_year": model_years,
        "model_first_planning_year": planning_years,
        "utc_offset": utc_offset,
        "generator_columns": DEFAULT_GENERATOR_COLUMNS,
    }
    return yaml.dump(out, default_flow_style=False, sort_keys=False)


def build_settings_yamls():
    return {
        "model_definition.yml": generate_model_definition_settings(),
        "resources.yml": generate_resources_settings(),
        "fuels.yml": generate_fuels_settings(),
        "transmission.yml": generate_transmission_settings(),
        "distributed_gen.yml": generate_distributed_gen_settings(),
        "resource_tags.yml": generate_resource_tags_settings(),
        "startup_costs.yml": generate_startup_costs_settings(),
    }


def populate_settings_file_select():
    sel = document.getElementById("settingsFileSelect")
    if not sel:
        return
    files = (
        sorted(state.settings_yamls.keys())
        if state.settings_yamls
        else SETTINGS_FILENAMES
    )
    _set_select_options(sel, files, selected_value=files[0] if files else None)


def update_settings_preview():
    sel = document.getElementById("settingsFileSelect")
    out_el = document.getElementById("settingsYamlOut")
    if not out_el:
        return
    filename = _get_select_value(sel, None)
    if filename and filename in state.settings_yamls:
        out_el.value = state.settings_yamls[filename]
    else:
        out_el.value = ""


def on_generate_settings(event):
    try:
        state.settings_yamls = build_settings_yamls()
        populate_settings_file_select()
        update_settings_preview()
        set_status("Settings YAMLs generated.", "success")
    except Exception as exc:
        state.settings_yamls = {}
        set_status(f"Settings generation error: {exc}", "error")


def _download_text_file(filename, content):
    blob = window.Blob.new([content], to_js({"type": "text/yaml"}))
    url = window.URL.createObjectURL(blob)
    a = document.createElement("a")
    a.href = url
    a.download = filename
    a.click()
    window.URL.revokeObjectURL(url)


def on_download_settings_file(event):
    sel = document.getElementById("settingsFileSelect")
    filename = _get_select_value(sel, None)
    if not filename or filename not in state.settings_yamls:
        set_status("Generate settings first.", "error")
        return
    _download_text_file(filename, state.settings_yamls[filename])
    set_status(f"Downloaded {filename}", "success")


def on_settings_file_change(event):
    update_settings_preview()


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


def on_download_emission_policies(event):
    """Download emission_policies.csv file."""
    if state.emission_policies_df is None:
        set_status("Run ESR analysis first to generate emission_policies.csv.", "error")
        return

    csv_content = state.emission_policies_df.to_csv(index=False)
    blob = window.Blob.new([csv_content], to_js({"type": "text/csv"}))
    url = window.URL.createObjectURL(blob)

    a = document.createElement("a")
    a.href = url
    a.download = "emission_policies.csv"
    a.click()

    window.URL.revokeObjectURL(url)
    set_status("emission_policies.csv downloaded!", "success")


def on_grouping_change(event):
    """Handle grouping column change."""
    update_no_cluster_options()


# ============================================================================
# ESR Generation Functions
# ============================================================================


def set_esr_status(message, status_type="info"):
    """Update the ESR result text box."""
    el = document.getElementById("esrResultText")
    if el:
        el.textContent = message
        el.className = f"status {status_type}"
        el.style.display = "block"


def render_esr_results():
    """Render the ESR analysis results."""
    rps_list = document.getElementById("esrRPSTechList")
    ces_list = document.getElementById("esrCESTechList")
    zones_list = document.getElementById("esrZonesList")
    csv_preview = document.getElementById("esrCsvPreview")

    # Render RPS techs
    if rps_list and hasattr(state, "esr_rps_techs"):
        if state.esr_rps_techs:
            rps_html = "".join(
                f"<span class='ba-tag'>{html.escape(tech)}</span>"
                for tech in sorted(state.esr_rps_techs)
            )
            rps_list.innerHTML = rps_html
        else:
            rps_list.innerHTML = "<em>No RPS-qualified technologies found.</em>"

    # Render CES techs
    if ces_list and hasattr(state, "esr_ces_techs"):
        if state.esr_ces_techs:
            ces_html = "".join(
                f"<span class='ba-tag'>{html.escape(tech)}</span>"
                for tech in sorted(state.esr_ces_techs)
            )
            ces_list.innerHTML = ces_html
        else:
            ces_list.innerHTML = "<em>No CES-qualified technologies found.</em>"

    # Render zones
    if zones_list and state.esr_zones:
        zones_html = "".join(
            f"<div class='candidate-item'><strong>Zone {i+1}:</strong> {', '.join(zone)}</div>"
            for i, zone in enumerate(state.esr_zones)
        )
        zones_list.innerHTML = zones_html

    # Render CSV preview
    if csv_preview and state.emission_policies_df is not None:
        csv_str = state.emission_policies_df.to_csv(index=False)
        csv_preview.value = csv_str


def on_run_esr_analysis(event):
    """Handle Run ESR Analysis button click."""
    try:
        # Check that region clustering has been run
        if not state.region_aggregations:
            set_esr_status("Run region clustering first (Step 1).", "error")
            return

        # Check that all ESR data is loaded
        missing_data = []
        if state.rps_df is None:
            missing_data.append("RPS policies")
        if state.ces_df is None:
            missing_data.append("CES policies")
        if state.rectable_df is None:
            missing_data.append("trading rules")
        if state.pop_fraction_df is None:
            missing_data.append("population fractions")
        if state.allowed_techs_df is None:
            missing_data.append("allowed techs")

        if missing_data:
            set_esr_status(
                f"ESR data still loading ({', '.join(missing_data)}). Please wait a moment and try again.",
                "error",
            )
            return

        set_esr_status("Analyzing ESR zones...", "info")

        # Get model years
        model_years_input = document.getElementById("modelYears").value
        model_years = parse_int_list(model_years_input)
        if not model_years:
            set_esr_status("Set model years in Model Setup step (Step 2).", "error")
            return

        # Get toggle states
        include_rps = document.getElementById("esrIncludeRPS").checked
        include_ces = document.getElementById("esrIncludeCES").checked

        if not include_rps and not include_ces:
            set_esr_status("Enable at least one of RPS or CES constraints.", "error")
            return

        # Build ESR zones
        state.esr_zones, esr_region_aggregations = build_esr_zones(
            state.region_aggregations, state.hierarchy_df, state.rectable_df
        )

        # Get qualified technologies
        # Collect new resources from textarea
        raw_el = document.getElementById("newResourcesRaw")
        new_resources = parse_new_resources_text(raw_el.value if raw_el else "")

        state.esr_rps_techs, state.esr_ces_techs = get_qualified_technologies(
            state.plants_df, new_resources, state.allowed_techs_df
        )

        # Generate emission policies CSV using expanded region aggregations
        state.emission_policies_df, state.esr_map, state.esr_type_map = (
            generate_emission_policies_csv(
                esr_region_aggregations,
                model_years,
                state.esr_zones,
                state.hierarchy_df,
                state.pop_fraction_df,
                state.rps_df,
                state.ces_df,
                include_rps=include_rps,
                include_ces=include_ces,
                case_id="all",
            )
        )

        # Render results
        render_esr_results()

        set_esr_status(
            f"ESR analysis complete: {len(state.esr_zones)} zones, "
            f"{len(state.esr_rps_techs)} RPS techs, {len(state.esr_ces_techs)} CES techs",
            "success",
        )

    except ESRGenerationError as e:
        set_esr_status(f"ESR Error: {e}", "error")
    except Exception as e:
        set_esr_status(f"ESR Analysis Error: {e}", "error")


# ============================================================================
# Initialization
# ============================================================================


async def main():
    """Main initialization function."""
    try:
        # Load data
        await load_data()

        # Load ATB index for Settings tab (optional)
        await load_atb_options()

        # Load fuel scenarios for Settings tab (optional)
        await load_fuel_prices()

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

        # Settings tab
        document.getElementById("addNewResourceBtn").addEventListener(
            "click", create_proxy(on_add_new_resource)
        )
        document.getElementById("newResourcesRaw").addEventListener(
            "input", create_proxy(lambda e: render_new_resources_list())
        )
        document.getElementById("addModifiedResourceBtn").addEventListener(
            "click", create_proxy(on_add_modified_resource)
        )
        document.getElementById("clearModifiedResourcesBtn").addEventListener(
            "click", create_proxy(on_clear_modified_resources)
        )
        document.getElementById("generateSettingsBtn").addEventListener(
            "click", create_proxy(on_generate_settings)
        )
        document.getElementById("settingsFileSelect").addEventListener(
            "change", create_proxy(on_settings_file_change)
        )
        document.getElementById("downloadSettingsFileBtn").addEventListener(
            "click", create_proxy(on_download_settings_file)
        )

        # ESR step
        document.getElementById("runESRBtn").addEventListener(
            "click", create_proxy(on_run_esr_analysis)
        )
        document.getElementById("downloadEmissionPoliciesBtn").addEventListener(
            "click", create_proxy(on_download_emission_policies)
        )

        # Fuel scenario options
        document.getElementById("fuelDataYear").addEventListener(
            "change", create_proxy(populate_fuel_scenario_selects)
        )

        # ATB picker change events
        document.getElementById("atbYearSelect").addEventListener(
            "change", create_proxy(on_atb_picker_change)
        )
        document.getElementById("atbTechSelect").addEventListener(
            "change", create_proxy(on_atb_picker_change)
        )
        document.getElementById("atbTechDetailSelect").addEventListener(
            "change", create_proxy(on_atb_picker_change)
        )
        document.getElementById("atbCostCaseSelect").addEventListener(
            "change", create_proxy(on_atb_picker_change)
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

        # Initialize Settings tab widgets
        populate_atb_picker()
        populate_fuel_data_year_select()
        populate_fuel_scenario_selects()
        render_new_resources_list()
        render_modified_resources_list()
        populate_settings_file_select()
        update_settings_preview()

        # Set up deferred ESR data loading
        async def load_esr_data_deferred():
            try:
                await load_esr_data()
            except Exception as e:
                set_status(f"Failed to load ESR data: {e}", "error")

        window.loadESRDataOnDemand = create_proxy(
            lambda: asyncio.ensure_future(load_esr_data_deferred())
        )

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
