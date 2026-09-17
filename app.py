"""AIDAMS Lab 1 dashboard: global iron and steel plants and their LitPop exposure.

Run locally with:   streamlit run app.py
The files in ./data are produced by lab_1.ipynb (Part 6). The dashboard never reads the raw GEM / LitPop files.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
MAP_STYLE = "carto-positron"  # free basemap, no access token needed
STATUS_BUCKET_COLOURS = {
    "Operating": "#2a6f97",
    "Pipeline": "#e9a23b",
    "Idle": "#9aa5b1",
    "Closed or cancelled": "#c0504d",
}
ROUTE_COLOURS = {
    "BF-BOF (coal-based)": "#7a3b2e",
    "Mixed (BF-BOF + electric)": "#c98a3d",
    "DRI-EAF": "#3d8f8a",
    "EAF / IF (electric)": "#2a6f97",
    "DRI (ironmaking only)": "#8fbf9f",
    "Other / unspecified": "#b3bac2",
}
REGION_COLOURS = {
    "Asia Pacific": "#88CCEE",
    "Europe": "#CC6677",
    "North America": "#DDCC77",
    "Middle East": "#117733",
    "Africa": "#332288",
    "Eurasia": "#AA4499",
    "Central & South America": "#44AA99",
}
MIN_LITPOP_COVERAGE = 0.5  # colour a company by exposure only if LitPop covers at least half of its plants
USD_TICKS = {6: "$1M", 7: "$10M", 8: "$100M", 9: "$1bn", 10: "$10bn", 11: "$100bn", 12: "$1tn", 13: "$10tn"}
TABLE_COLUMNS = {
    "Plant name (English)": "Plant",
    "Owner": "Owner",
    "Country/Area": "Country/Area",
    "Region": "Region",
    "plant_status": "Status",
    "route": "Route",
    "capacity_operating_ttpa": "Operating capacity (ttpa)",
    "capacity_pipeline_ttpa": "Pipeline capacity (ttpa)",
    "Plant age (years)": "Plant age (years)",
    "litpop_value_25km_usd": "LitPop assets within 25 km (USD)",
    "litpop_value_nearest_usd": "LitPop nearest cell (USD)",
    "litpop_distance_km": "Distance to LitPop cell (km)",
    "GEM wiki page": "GEM wiki page",
}

# --------------------------------------------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Steel plants and exposure | AIDAMS Lab 1", page_icon="🏭", layout="wide")


# --------------------------------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------------------------------
def read_table(name: str) -> pd.DataFrame:
    """Read an export of the notebook: Parquet when a Parquet engine is installed, otherwise the CSV twin."""
    parquet_path, csv_path = DATA_DIR / f"{name}.parquet", DATA_DIR / f"{name}.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except ImportError:  # no Parquet engine installed: use the CSV twin
            pass
    if not csv_path.exists():
        raise FileNotFoundError(f"'{csv_path}' is missing. Run lab_1.ipynb (Part 6) to create the dashboard data.")
    return pd.read_csv(csv_path, keep_default_na=False, na_values=[""])


@st.cache_data(show_spinner="Loading plant data...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load the processed tables once per session and add the display helpers used by the charts."""
    plants = read_table("plants_litpop")
    companies = read_table("companies")
    litpop_grid = read_table("litpop_grid")
    metadata_path = DATA_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    plants["litpop_matched"] = plants["litpop_matched"].fillna(False).astype(bool)
    plants["capacity_for_filter"] = plants["capacity_operating_ttpa"].fillna(0.0)
    plants["marker_size"] = plants["capacity_for_filter"].clip(lower=200.0)  # keeps non-operating plants visible
    plants["capacity_label"] = np.select(
        [plants["capacity_operating_ttpa"].isna(), plants["capacity_operating_ttpa"].gt(0)],
        ["operating steel capacity not reported", plants["capacity_for_filter"].map("{:,.0f} ttpa".format)],
        default="no operating steel capacity on record",
    )
    if "route" not in plants.columns:
        plants["route"] = "Other / unspecified"
    return plants, companies, litpop_grid, metadata


def show_chart(fig: go.Figure, key: str) -> None:
    """Render a Plotly figure at full container width."""
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, key=key, width="stretch")
    else:
        st.plotly_chart(fig, key=key, use_container_width=True)


def show_table(table: pd.DataFrame, height: int, link_column: str | None = None) -> None:
    """Render a dataframe at full width, including on Streamlit 1.37."""
    column_config = (
        {link_column: st.column_config.LinkColumn(link_column, display_text="open")} if link_column else None
    )
    if inspect.signature(st.dataframe).parameters["width"].default == "stretch":
        st.dataframe(table, hide_index=True, height=height, column_config=column_config, width="stretch")
    else:
        st.dataframe(table, hide_index=True, height=height, column_config=column_config, use_container_width=True)


# --------------------------------------------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------------------------------------------
def sidebar_filters(plants: pd.DataFrame, companies: pd.DataFrame) -> pd.DataFrame:
    """Draw the sidebar widgets and return the plants that pass every filter. An empty selection means 'all'."""
    st.sidebar.header("Filters")
    st.sidebar.caption("Leave a box empty to keep everything.")

    regions = st.sidebar.multiselect("Region", sorted(plants["Region"].dropna().unique()), placeholder="All regions")
    in_region = plants[plants["Region"].isin(regions)] if regions else plants

    countries = st.sidebar.multiselect(
        "Country / area", sorted(in_region["Country/Area"].dropna().unique()), placeholder="All countries"
    )
    in_country = in_region[in_region["Country/Area"].isin(countries)] if countries else in_region

    # Companies are listed by global operating capacity, largest first, so the big names are easy to find
    ranking = companies.set_index("Owner")["total_capacity_ttpa"].fillna(0)
    owner_options = sorted(in_country["Owner"].dropna().unique(), key=lambda owner: (-ranking.get(owner, 0.0), owner))
    owners = st.sidebar.multiselect("Company (owner)", owner_options, placeholder="All companies")

    statuses = st.sidebar.multiselect(
        "Plant status", sorted(plants["plant_status"].dropna().unique()), placeholder="All statuses"
    )
    routes = st.sidebar.multiselect(
        "Steelmaking route", [r for r in ROUTE_COLOURS if r in set(plants["route"])], placeholder="All routes"
    )

    max_capacity = int(np.ceil(plants["capacity_for_filter"].max() / 100.0) * 100)
    capacity_range = st.sidebar.slider(
        "Operating capacity (thousand tonnes per year)", 0, max_capacity, (0, max_capacity), step=100
    )
    only_exposure = st.sidebar.checkbox("Only plants with LitPop exposure (China, India, Japan)", value=False)

    selected = in_country
    if owners:
        selected = selected[selected["Owner"].isin(owners)]
    if statuses:
        selected = selected[selected["plant_status"].isin(statuses)]
    if routes:
        selected = selected[selected["route"].isin(routes)]
    selected = selected[selected["capacity_for_filter"].between(*capacity_range)]
    if only_exposure:
        selected = selected[selected["litpop_matched"]]
    st.sidebar.caption("A lower bound above 0 keeps only plants with reported positive operating steel capacity.")
    return selected


# --------------------------------------------------------------------------------------------------------------
# Building blocks of the main area
# --------------------------------------------------------------------------------------------------------------
def kpi_row(selected: pd.DataFrame, plants: pd.DataFrame) -> None:
    """Headline numbers for the current selection."""
    operating_mtpa = selected["capacity_operating_ttpa"].sum(min_count=1) / 1000
    global_mtpa = plants["capacity_operating_ttpa"].sum(min_count=1) / 1000
    known_ages = selected["Plant age (years)"].dropna()
    median_age = known_ages.median() if len(known_ages) else np.nan
    columns = st.columns(6)
    columns[0].metric("Plants", f"{len(selected):,}", help="Plants tracked by GEM in the selection, all statuses")
    columns[1].metric(
        "Operating capacity (Mtpa)",
        f"{operating_mtpa:,.0f}" if pd.notna(operating_mtpa) else "n/a",
        help="Operating + operating pre-retirement crude steel capacity, million tonnes per year",
    )
    columns[2].metric(
        "Share of world capacity",
        f"{operating_mtpa / global_mtpa:.1%}" if pd.notna(operating_mtpa) and global_mtpa > 0 else "n/a",
    )
    pipeline_mtpa = selected["capacity_pipeline_ttpa"].sum(min_count=1) / 1000
    columns[3].metric(
        "Pipeline (Mtpa)",
        f"{pipeline_mtpa:,.0f}" if pd.notna(pipeline_mtpa) else "n/a",
        help="Announced + under construction",
    )
    n_companies = selected.loc[selected["Owner"].astype(str).str.lower() != "unknown", "Owner"].nunique()
    columns[4].metric("Companies", f"{n_companies:,}", help="Distinct immediate owners ('unknown' excluded)")
    columns[5].metric("Median plant age (years)", "n/a" if pd.isna(median_age) else f"{median_age:.0f}")


def zoom_for(selected: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Map centre and a zoom level that roughly fits the selected plants."""
    lat_span = selected["Latitude"].max() - selected["Latitude"].min()
    lon_span = selected["Longitude"].max() - selected["Longitude"].min()
    span = max(lat_span * 2.0, lon_span, 0.5)
    zoom = float(np.clip(np.log2(360.0 / span) - 0.3, 0.6, 9.0))
    centre = {"lat": float(selected["Latitude"].mean()), "lon": float(selected["Longitude"].mean())}
    return centre, zoom


def plant_map(selected: pd.DataFrame, colour_by: str, top_owners: list[str]) -> go.Figure:
    """Scatter map: one marker per plant, area proportional to operating capacity."""
    data = selected.sort_values("marker_size", ascending=False).copy()
    palette: dict[str, str] | None = None
    colour_sequence: list[str] | None = None
    if colour_by == "Company (top 10)":
        data["colour"] = data["Owner"].where(data["Owner"].isin(top_owners), "Other owners")
        palette = dict(zip(top_owners, px.colors.qualitative.Bold + px.colors.qualitative.Vivid))
        palette["Other owners"] = "#c3c9d1"
        colour = "colour"
    elif colour_by == "Steelmaking route":
        colour, palette = "route", ROUTE_COLOURS
    elif colour_by == "Plant status":
        colour, colour_sequence = "plant_status", px.colors.qualitative.Safe
    else:
        colour, palette = "Region", REGION_COLOURS
    centre, zoom = zoom_for(data)
    fig = px.scatter_map(
        data,
        lat="Latitude",
        lon="Longitude",
        size="marker_size",
        size_max=24,
        hover_name="Plant name (English)",
        hover_data={
            "Owner": True,
            "Country/Area": True,
            "capacity_label": True,
            "plant_status": True,
            "route": True,
            "marker_size": False,
            "Latitude": False,
            "Longitude": False,
            "colour": False,
        }
        if "colour" in data
        else {
            "Owner": True,
            "Country/Area": True,
            "capacity_label": True,
            "plant_status": True,
            "route": True,
            "marker_size": False,
            "Latitude": False,
            "Longitude": False,
        },
        labels={"capacity_label": "Operating capacity", "plant_status": "Status", "route": "Route", "colour": "Owner"},
        center=centre,
        zoom=zoom,
        height=600,
        opacity=0.8,
        color=colour,
        color_discrete_map=palette,
        color_discrete_sequence=colour_sequence,
        map_style=MAP_STYLE,
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={
            "title_text": "",
            "itemsizing": "constant",
            "bgcolor": "rgba(255,255,255,0.8)",
            "font": {"color": "#202124"},
            "x": 0.01,
            "y": 0.99,
        },
    )
    return fig


def density_map(selected: pd.DataFrame, weight_by_capacity: bool) -> go.Figure:
    """Heatmap of plant concentration, optionally weighted by capacity."""
    centre, zoom = zoom_for(selected)
    fig = px.density_map(
        selected,
        lat="Latitude",
        lon="Longitude",
        z="capacity_for_filter" if weight_by_capacity else None,
        radius=12,
        hover_name="Plant name (English)",
        hover_data={"Country/Area": True, "capacity_label": True, "Latitude": False, "Longitude": False},
        labels={"capacity_label": "Operating capacity"},
        center=centre,
        zoom=zoom,
        height=600,
        color_continuous_scale="Inferno",
        map_style=MAP_STYLE,
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, coloraxis_colorbar={"title": {"text": "Relative<br>density"}}
    )
    return fig


def exposure_map(covered: pd.DataFrame, litpop_grid: pd.DataFrame, show_grid: bool) -> go.Figure:
    """Plants coloured by the LitPop asset value within 25 km, with the coarse exposure grid as optional background."""
    data = covered.sort_values("marker_size", ascending=False).copy()
    data["log10_exposure"] = np.log10(data["litpop_value_25km_usd"].clip(lower=1))
    data["exposure_25km_bn"] = data["litpop_value_25km_usd"] / 1e9
    data["exposure_cell_m"] = data["litpop_value_nearest_usd"] / 1e6
    centre, zoom = zoom_for(data)
    fig = px.scatter_map(
        data,
        lat="Latitude",
        lon="Longitude",
        size="marker_size",
        size_max=24,
        color="log10_exposure",
        color_continuous_scale="Plasma",
        hover_name="Plant name (English)",
        hover_data={
            "Owner": True,
            "Country/Area": True,
            "capacity_label": True,
            "exposure_25km_bn": ":,.1f",
            "exposure_cell_m": ":,.0f",
            "litpop_distance_km": ":.1f",
            "log10_exposure": False,
            "marker_size": False,
            "Latitude": False,
            "Longitude": False,
        },
        labels={
            "capacity_label": "Operating capacity",
            "exposure_25km_bn": "Assets within 25 km (USD bn)",
            "exposure_cell_m": "Nearest cell value (USD M)",
            "litpop_distance_km": "Distance to cell centre (km)",
        },
        center=centre,
        zoom=zoom,
        height=600,
        opacity=0.9,
        map_style=MAP_STYLE,
    )
    ticks = [v for v in USD_TICKS if data["log10_exposure"].min() - 1 < v < data["log10_exposure"].max() + 1]
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        coloraxis_colorbar={
            "title": {"text": "Assets within<br>25 km (USD)"},
            "tickvals": ticks,
            "ticktext": [USD_TICKS[v] for v in ticks],
        },
    )
    if show_grid and len(litpop_grid):
        grid = litpop_grid[
            litpop_grid["lat_bin"].between(data["Latitude"].min() - 3, data["Latitude"].max() + 3)
            & litpop_grid["lon_bin"].between(data["Longitude"].min() - 3, data["Longitude"].max() + 3)
        ]
        background = go.Scattermap(
            lat=grid["lat_bin"],
            lon=grid["lon_bin"],
            mode="markers",
            name="LitPop grid",
            hoverinfo="skip",
            showlegend=False,
            marker={
                "size": 7,
                "opacity": 0.35,
                "color": grid["log10_value"],
                "colorscale": "Greys",
                "cmin": 6,
                "cmax": 13,
                "showscale": False,
            },
        )
        fig.add_trace(background)
        fig.data = fig.data[::-1]  # draw the grid underneath the single plant trace
    return fig


def summarise_owners(selected: pd.DataFrame) -> pd.DataFrame:
    """Company-level view of the current selection (same definitions as Part 5 of the notebook)."""
    known = selected[selected["Owner"].astype(str).str.lower() != "unknown"]
    largest = known.sort_values("capacity_for_filter", ascending=False).drop_duplicates("Owner").set_index("Owner")
    summary = known.groupby("Owner").agg(
        plants=("Plant ID", "nunique"),
        capacity_mtpa=("capacity_operating_ttpa", lambda s: s.sum(min_count=1) / 1000),
        pipeline_mtpa=("capacity_pipeline_ttpa", lambda s: s.sum(min_count=1) / 1000),
        countries=("Country/Area", "nunique"),
        median_age=("Plant age (years)", "median"),
        plants_with_litpop=("litpop_matched", "sum"),
        avg_assets_25km_bn=("litpop_value_25km_usd", lambda s: s.mean() / 1e9),
    )
    # An average based on a minority of a company's plants would not represent the company: keep it only above the threshold
    summary.loc[summary["plants_with_litpop"] / summary["plants"] < MIN_LITPOP_COVERAGE, "avg_assets_25km_bn"] = np.nan
    summary["largest_plant"] = largest["Plant name (English)"]
    summary["lat"], summary["lon"] = largest["Latitude"], largest["Longitude"]
    return summary.sort_values("capacity_mtpa", ascending=False).reset_index()


def company_map(owner_table: pd.DataFrame) -> go.Figure:
    """One marker per company at its largest plant: size = capacity, colour = average LitPop exposure (grey if coverage is insufficient)."""
    data = owner_table[owner_table["capacity_mtpa"] > 0].copy()
    data["exposure_label"] = np.where(
        data["avg_assets_25km_bn"].notna(),
        data["avg_assets_25km_bn"].map("{:,.1f} USD bn".format),
        "insufficient LitPop coverage (<50% of plants)",
    )
    data["log10_exposure"] = np.log10((data["avg_assets_25km_bn"] * 1e9).clip(lower=1))
    centre, zoom = zoom_for(data.rename(columns={"lat": "Latitude", "lon": "Longitude"}))
    fig = go.Figure()
    size_ref = 2.0 * data["capacity_mtpa"].max() / 40**2
    for subset, name, marker in (
        (
            data[data["avg_assets_25km_bn"].isna()],
            "LitPop covers less than half of the plants (or none)",
            {"color": "#aab2bb"},
        ),
        (data[data["avg_assets_25km_bn"].notna()], "LitPop exposure available", None),
    ):
        if subset.empty:
            continue
        if marker is None:
            ticks = [
                v for v in USD_TICKS if subset["log10_exposure"].min() - 1 < v < subset["log10_exposure"].max() + 1
            ]
            marker = {
                "color": subset["log10_exposure"],
                "colorscale": "Plasma",
                "colorbar": {
                    "title": {"text": "Avg assets within<br>25 km (USD)"},
                    "tickvals": ticks,
                    "ticktext": [USD_TICKS[v] for v in ticks],
                },
            }
        fig.add_trace(
            go.Scattermap(
                lat=subset["lat"],
                lon=subset["lon"],
                mode="markers",
                name=name,
                text=subset["Owner"],
                customdata=np.stack(
                    [
                        subset["plants"],
                        subset["capacity_mtpa"],
                        subset["countries"],
                        subset["largest_plant"],
                        subset["exposure_label"],
                    ],
                    axis=-1,
                ),
                marker=dict(
                    size=subset["capacity_mtpa"], sizemode="area", sizeref=size_ref, sizemin=3, opacity=0.8, **marker
                ),
                hovertemplate="<b>%{text}</b><br>Plants: %{customdata[0]} in %{customdata[2]} countries/areas<br>Operating capacity: %{customdata[1]:,.1f} Mtpa"
                "<br>Largest plant: %{customdata[3]}<br>Avg LitPop assets within 25 km: %{customdata[4]}<extra></extra>",
            )
        )
    fig.update_layout(
        map={"style": MAP_STYLE, "center": centre, "zoom": zoom},
        height=560,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={
            "orientation": "h",
            "y": 0.01,
            "x": 0.01,
            "bgcolor": "rgba(255,255,255,0.8)",
            "font": {"color": "#202124"},
        },
    )
    return fig


def insight_charts(selected: pd.DataFrame) -> None:
    """Charts carried over from the exploratory analysis (Part 2 of the notebook), recomputed on the selection."""
    left, right = st.columns(2)

    by_country = (
        selected.groupby("Country/Area")[["capacity_operating_ttpa", "capacity_pipeline_ttpa"]]
        .sum(min_count=1)
        .div(1000)
    )
    country_order = (
        by_country.fillna(0)
        .sort_values(["capacity_operating_ttpa", "capacity_pipeline_ttpa"], ascending=False)
        .head(15)
        .index
    )
    by_country = by_country.loc[country_order].rename(
        columns={
            "capacity_operating_ttpa": "Operating",
            "capacity_pipeline_ttpa": "Pipeline (announced + construction)",
        }
    )
    fig = px.bar(
        by_country.reset_index().melt(id_vars="Country/Area", var_name="Bucket", value_name="Mtpa"),
        x="Mtpa",
        y="Country/Area",
        color="Bucket",
        barmode="group",
        orientation="h",
        title="Capacity by country/area (top 15)",
        color_discrete_map={"Operating": "#2a6f97", "Pipeline (announced + construction)": "#e9a23b"},
        height=480,
    )
    fig.update_layout(
        yaxis={
            "categoryorder": "array",
            "categoryarray": by_country.index[::-1].tolist(),
            "title": "",
            "automargin": True,
        },
        legend={"orientation": "h", "y": 1.1, "x": 0, "title_text": ""},
        xaxis_title="Crude steel capacity (Mtpa)",
    )
    with left:
        show_chart(fig, "chart_country")

    buckets = (
        pd.Series(
            {
                "Operating": selected["capacity_operating_ttpa"].sum(min_count=1),
                "Pipeline": selected["capacity_pipeline_ttpa"].sum(min_count=1),
                "Idle": selected["capacity_idle_ttpa"].sum(min_count=1),
                "Closed or cancelled": selected["capacity_closed_ttpa"].sum(min_count=1),
            }
        )
        .div(1000)
        .rename("Mtpa")
        .rename_axis("Bucket")
        .reset_index()
    )
    fig = px.bar(
        buckets,
        x="Bucket",
        y="Mtpa",
        color="Bucket",
        color_discrete_map=STATUS_BUCKET_COLOURS,
        text="Mtpa",
        title="Capacity by operational status",
        height=480,
    )
    fig.update_traces(texttemplate="%{text:,.0f}")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Crude steel capacity (Mtpa)")
    with right:
        show_chart(fig, "chart_status")

    top_owners = (
        selected.groupby("Owner")["capacity_operating_ttpa"]
        .sum(min_count=1)
        .div(1000)
        .nlargest(15)
        .rename("Mtpa")
        .reset_index()
    )
    fig = px.bar(
        top_owners,
        x="Mtpa",
        y="Owner",
        orientation="h",
        title="Top 15 owners by operating capacity",
        height=480,
        color_discrete_sequence=["#2a6f97"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending", "title": ""}, xaxis_title="Operating capacity (Mtpa)")
    with left:
        show_chart(fig, "chart_owner")

    by_route = (
        selected.groupby("route")["capacity_operating_ttpa"].sum(min_count=1).div(1000).rename("Mtpa").reset_index()
    )
    fig = px.pie(
        by_route[by_route["Mtpa"] > 0],
        names="route",
        values="Mtpa",
        color="route",
        color_discrete_map=ROUTE_COLOURS,
        hole=0.45,
        title="Operating capacity by steelmaking route",
        height=480,
    )
    fig.update_traces(textinfo="percent+label", showlegend=False)
    with right:
        show_chart(fig, "chart_route")

    aged = selected.dropna(subset=["Plant age (years)"])
    if len(aged):
        fig = px.histogram(
            aged,
            x="Plant age (years)",
            color="Region",
            nbins=50,
            title="Plant age distribution by region",
            color_discrete_map=REGION_COLOURS,
            height=420,
        )
        fig.update_layout(bargap=0.05, yaxis_title="Number of plants", legend_title_text="")
        show_chart(fig, "chart_age")


# --------------------------------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------------------------------
def main() -> None:
    """Assemble the page: title, sidebar, KPI row, tabs and footer."""
    plants, companies, litpop_grid, metadata = load_data()
    release = metadata.get("gem_release", "release unknown")

    st.title("🏭 Global iron and steel plants: capacity, ownership and exposure")
    st.markdown(
        f"Asset-level view of the **{len(plants):,} plants** in Global Energy Monitor's Global Iron and Steel Tracker ({release}), "
        "enriched with **LitPop** asset exposure for plants in China, India and Japan. "
        "Use the sidebar to filter by company, geography, status, route and capacity: every number and chart follows the selection."
    )

    selected = sidebar_filters(plants, companies)
    if selected.empty:
        st.warning("No plant matches the current filters. Widen the selection in the sidebar.")
        st.stop()

    kpi_row(selected, plants)
    tab_map, tab_exposure, tab_companies, tab_insights, tab_data = st.tabs(
        ["🗺️ Plant map", "🏙️ Exposure (LitPop)", "🏢 Companies", "📊 Insights", "📋 Data"]
    )

    with tab_map:
        controls = st.columns([2, 2, 3])
        view = controls[0].radio("View", ["Plants", "Density heatmap"], horizontal=True)
        if view == "Plants":
            colour_by = controls[1].selectbox(
                "Colour by", ["Region", "Steelmaking route", "Plant status", "Company (top 10)"]
            )
            top_owners = selected.groupby("Owner")["capacity_for_filter"].sum().nlargest(10).index.tolist()
            show_chart(plant_map(selected, colour_by, top_owners), "map_plants")
            st.caption(
                "Marker area is proportional to operating crude steel capacity. Plants with zero or unreported operating capacity get the smallest marker."
            )
        else:
            weighted = controls[1].checkbox(
                "Weight by capacity", value=False, help="Off: every plant counts once. On: large plants weigh more."
            )
            show_chart(density_map(selected, weighted), "map_density")
            st.caption(
                "Bright areas concentrate many plants (or much capacity when weighted). Zoom in to separate neighbouring clusters."
            )

    with tab_exposure:
        covered = selected[selected["litpop_matched"]]
        if covered.empty:
            st.info(
                "LitPop exposure is only available for plants in China, India and Japan (the sample provided for the lab). "
                "Select one of these countries to see this tab."
            )
        else:
            show_grid = st.checkbox("Show the LitPop exposure grid in the background (0.5-degree blocks)", value=False)
            show_chart(exposure_map(covered, litpop_grid, show_grid), "map_exposure")
            columns = st.columns(3)
            columns[0].metric("Plants with exposure", f"{len(covered):,}")
            columns[1].metric(
                "Median assets within 25 km", f"${covered['litpop_value_25km_usd'].median() / 1e9:,.1f} bn"
            )
            covered_mtpa = covered["capacity_operating_ttpa"].sum(min_count=1) / 1000
            columns[2].metric("Capacity covered (Mtpa)", f"{covered_mtpa:,.0f}" if pd.notna(covered_mtpa) else "n/a")
            scatter = covered[covered["capacity_operating_ttpa"] > 0].assign(
                exposure_bn=lambda t: t["litpop_value_25km_usd"] / 1e9
            )
            if len(scatter) > 2:
                fig = px.scatter(
                    scatter,
                    x="capacity_operating_ttpa",
                    y="exposure_bn",
                    color="Country/Area",
                    log_x=True,
                    log_y=True,
                    hover_name="Plant name (English)",
                    opacity=0.7,
                    height=430,
                    labels={
                        "capacity_operating_ttpa": "Operating capacity (ttpa, log)",
                        "exposure_bn": "LitPop assets within 25 km (USD bn, log)",
                    },
                    title="Plant size versus surrounding asset value",
                )
                show_chart(fig, "chart_exposure_scatter")
            st.caption(
                "LitPop spreads national produced capital over a grid in proportion to night lights x population. "
                "'Within 25 km' sums every grid cell whose centre lies within 25 km of the plant."
            )

    with tab_companies:
        owner_table = summarise_owners(selected)
        if owner_table["capacity_mtpa"].gt(0).any():
            show_chart(company_map(owner_table), "map_companies")
            st.caption(
                "One marker per company, placed at its largest plant in the selection. Grey: LitPop covers less than half of the company's plants (or none)."
            )
        show_table(owner_table.drop(columns=["lat", "lon"]).round(1), height=380)

    with tab_insights:
        insight_charts(selected)

    with tab_data:
        table = selected[[c for c in TABLE_COLUMNS if c in selected.columns]].rename(columns=TABLE_COLUMNS)
        table = table.sort_values("Operating capacity (ttpa)", ascending=False)
        st.write(f"{len(table):,} plants in the selection")
        show_table(table, height=520, link_column="GEM wiki page")
        st.download_button(
            "Download the selection as CSV",
            table.to_csv(index=False).encode("utf-8"),
            "steel_plants_selection.csv",
            "text/csv",
        )

    # Footer ----------------------------------------------------------------------------------------------------
    st.divider()
    st.caption(
        f"**Sources.** Plants: Global Energy Monitor, Global Iron and Steel Tracker, {release} (CC BY 4.0). "
        "Exposure: LitPop, Eberenz et al. (2020), Earth Syst. Sci. Data 12, 817-833, doi:10.5194/essd-12-817-2020; "
        "course sample for China, India and Japan (300 arc-seconds, reference year 2018, produced capital in USD).  \n"
        "**Notes.** Capacity = nominal crude steel capacity with status operating or operating pre-retirement, in thousand tonnes per year (ttpa); "
        "pipeline = announced + construction. GEM only tracks plants of 500 ttpa or more. "
        f"A plant is linked to its nearest LitPop cell when the cell centre is within {metadata.get('max_match_distance_km', 13.1)} km, "
        f"and to all cells within {metadata.get('neighbourhood_radius_km', 25):.0f} km for the neighbourhood sum. "
        "'Company' is GEM's immediate owner, which is often a subsidiary of a larger group."
    )


main()
