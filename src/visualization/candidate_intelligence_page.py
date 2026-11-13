
from pathlib import Path
import json
import html

import streamlit as st
import pandas as pd
import plotly.express as px
import folium
import geopandas as gpd

from streamlit_folium import st_folium


# ============================================================
# DATA ACCESS
# ============================================================

@st.cache_data(show_spinner=False)
def load_profiles(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_candidate_boundaries(path):
    gdf = gpd.read_file(path)

    if gdf.crs is None:
        raise ValueError(
            "Candidate boundary GeoJSON has no declared CRS. "
            "Cannot safely display its coordinates."
        )

    gdf = gdf.to_crs("EPSG:4326")

    return gdf.__geo_interface__


# ============================================================
# PRESENTATION HELPERS
# ============================================================

def candidate_section(label, title, description=None):
    if label:
        st.markdown(
            f'<div class="eyebrow">{html.escape(label)}</div>',
            unsafe_allow_html=True
        )

    st.markdown(f"### {title}")

    if description:
        st.caption(description)


def candidate_value(value, decimals=2, suffix=""):
    if isinstance(value, (int, float)):
        return f"{value:,.{decimals}f}{suffix}"
    return f"—{suffix}"


def candidate_integer(value):
    if isinstance(value, (int, float)):
        return f"{value:,.0f}"
    return "—"


def candidate_status(value):
    if value is True:
        return "Within interval"
    if value is False:
        return "Outside interval"
    return "Not available"


# ============================================================
# GEOGRAPHIC VISUALISATION
# ============================================================

def render_candidate_map(profile, geojson):
    geography = profile["geography"]

    latitude = geography["latitude"]
    longitude = geography["longitude"]
    candidate_id = profile["candidate_id"]

    map_object = folium.Map(
        location=[latitude, longitude],
        zoom_start=10,
        tiles=None,
        control_scale=True
    )

    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap"
    ).add_to(map_object)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri",
        name="Satellite imagery",
        overlay=False
    ).add_to(map_object)

    matching_features = [
        feature
        for feature in geojson.get("features", [])
        if str(
            feature.get("properties", {}).get("candidate_id", "")
        ) == str(candidate_id)
    ]

    if matching_features:
        candidate_feature_collection = {
            "type": "FeatureCollection",
            "features": matching_features
        }

        folium.GeoJson(
            candidate_feature_collection,
            name=f"{candidate_id} boundary",
            style_function=lambda feature: {
                "fillColor": "#55D6A0",
                "color": "#55D6A0",
                "weight": 2,
                "fillOpacity": 0.28,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["candidate_id"],
                aliases=["Candidate:"]
            )
        ).add_to(map_object)

        bounds = gpd.GeoDataFrame.from_features(
            matching_features,
            crs="EPSG:4326"
        ).total_bounds

        min_lon, min_lat, max_lon, max_lat = bounds

        map_object.fit_bounds([
            [min_lat, min_lon],
            [max_lat, max_lon]
        ])

    else:
        folium.Marker(
            [latitude, longitude],
            tooltip=f"{candidate_id} centroid",
            popup=(
                f"{candidate_id}<br>"
                f"Area: {geography['area_km2']} km²"
            ),
            icon=folium.Icon(
                color="green",
                icon="globe"
            )
        ).add_to(map_object)

    folium.LayerControl().add_to(map_object)

    return map_object, bool(matching_features)


# ============================================================
# CANDIDATE PROFILE
# ============================================================

def render_profile(profile, geojson, root):

    candidate_id = profile["candidate_id"]

    geography = profile["geography"]
    terrain = profile["terrain"]
    geology = profile["geology"]
    screening = profile["screening"]
    coverage = geology["mapped_coverage"]
    membership = profile["strict_interval_membership"]

    strict = profile["jezero_reference"]["strict_p40_p60"]

    # --------------------------------------------------------
    # PROFILE HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="page-title">{html.escape(candidate_id)}</div>
        <div class="page-description">
            Individual terrestrial candidate intelligence profile.
            Area rank: {profile['area_rank']} — ranking by connected-region
            area, not scientific suitability.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        f"{geography['region']} · "
        f"{geography['latitude']:.5f}° latitude · "
        f"{geography['longitude']:.5f}° longitude"
    )

    # --------------------------------------------------------
    # CANDIDATE OVERVIEW
    # --------------------------------------------------------

    candidate_section(
        None,
        "Candidate characteristics",
        "Basic spatial and geological attributes of the selected terrestrial region."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Candidate area",
        candidate_value(geography["area_km2"], 2, " km²")
    )

    c2.metric(
        "Geological coverage",
        candidate_value(coverage["coverage_percent"], 2, "%")
    )

    c3.metric(
        "Geological records",
        candidate_integer(
            geology["number_of_intersecting_geological_records"]
        )
    )

    c4.metric(
        "Raster pixels",
        candidate_integer(geography["pixel_count"])
    )

    st.divider()

    # --------------------------------------------------------
    # MAP AND TERRAIN CHARACTERISTICS
    # --------------------------------------------------------

    candidate_section(
        None,
        "Candidate geography",
        "Inspect the mapped candidate boundary and its derived terrain measurements."
    )

    map_col, terrain_col = st.columns([1.35, 1])

    with map_col:

        candidate_map, has_polygon = render_candidate_map(
            profile,
            geojson
        )

        st_folium(
            candidate_map,
            width=None,
            height=460,
            returned_objects=[],
            key=f"candidate_map_{candidate_id}"
        )

        if has_polygon:
            st.caption(
                "Mapped boundary: verified candidate polygon."
            )
        else:
            st.warning(
                "Candidate polygon not found in the GeoJSON. "
                "Showing the candidate centroid instead."
            )

    with terrain_col:

        st.markdown("#### Terrain characterization")

        terrain_rows = [
            {
                "Metric": "Slope",
                "Mean": terrain["slope"]["mean_deg"],
                "Median": terrain["slope"]["median_deg"],
                "Unit": "°"
            },
            {
                "Metric": "Roughness",
                "Mean": terrain["roughness"]["mean_m"],
                "Median": terrain["roughness"]["median_m"],
                "Unit": "m"
            },
            {
                "Metric": "Local relief",
                "Mean": terrain["local_relief"]["mean_m"],
                "Median": terrain["local_relief"]["median_m"],
                "Unit": "m"
            }
        ]

        terrain_df = pd.DataFrame(terrain_rows)

        st.dataframe(
            terrain_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Jezero reference intervals")

        st.caption(
            "Strict P40–P60 reference ranges. These are measurement "
            "comparisons, not a validated analogue suitability test."
        )

        interval_rows = [
            {
                "Metric": "Slope",
                "Candidate median": terrain["slope"]["median_deg"],
                "Lower": strict["slope_deg"][0],
                "Upper": strict["slope_deg"][1],
                "Unit": "°",
                "Within interval": membership[
                    "slope_median_within_strict_interval"
                ]
            },
            {
                "Metric": "Roughness",
                "Candidate median": terrain["roughness"]["median_m"],
                "Lower": strict["roughness_m"][0],
                "Upper": strict["roughness_m"][1],
                "Unit": "m",
                "Within interval": membership[
                    "roughness_median_within_strict_interval"
                ]
            },
            {
                "Metric": "Local relief",
                "Candidate median": terrain["local_relief"]["median_m"],
                "Lower": strict["local_relief_m"][0],
                "Upper": strict["local_relief_m"][1],
                "Unit": "m",
                "Within interval": membership[
                    "local_relief_median_within_strict_interval"
                ]
            }
        ]

        for item in interval_rows:

            status = candidate_status(
                item["Within interval"]
            )

            if item["Within interval"] is True:
                st.success(
                    f"{item['Metric']} · {status}"
                )
            elif item["Within interval"] is False:
                st.warning(
                    f"{item['Metric']} · {status}"
                )
            else:
                st.info(
                    f"{item['Metric']} · {status}"
                )

            st.caption(
                f"Candidate median: "
                f"{item['Candidate median']:.3f} {item['Unit']} | "
                f"Jezero P40–P60: "
                f"{item['Lower']:.3f}–{item['Upper']:.3f} "
                f"{item['Unit']}"
            )

    st.divider()

    # --------------------------------------------------------
    # GEOLOGICAL CHARACTERIZATION
    # --------------------------------------------------------

    candidate_section(
        None,
        "Geological characterization",
        "Mapped geological records intersecting the candidate region."
    )

    geo_col, geo_summary_col = st.columns([1.45, 1])

    units = geology["all_intersecting_units"]

    with geo_col:

        if units:

            geological_df = pd.DataFrame(units)

            geological_df = geological_df.rename(
                columns={
                    "code": "Geological code",
                    "lithology": "Mapped lithology",
                    "intersection_area_km2": "Area (km²)",
                    "candidate_coverage_percent": "Candidate coverage (%)",
                }
            )

            fig = px.bar(
                geological_df,
                x="Area (km²)",
                y="Geological code",
                orientation="h",
                hover_data=[
                    "Mapped lithology",
                    "Candidate coverage (%)"
                ],
                title="Mapped geological units by intersection area",
                color="Area (km²)"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#EDF4F8"),
                xaxis=dict(
                    gridcolor="#263746",
                    title="Intersection area (km²)"
                ),
                yaxis=dict(
                    gridcolor="#263746",
                    title=""
                ),
                coloraxis_showscale=False,
                margin=dict(
                    l=10,
                    r=10,
                    t=45,
                    b=10
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info(
                "No geological intersection records were found."
            )

    with geo_summary_col:

        st.markdown("#### Geological map coverage")

        st.metric(
            "Mapped area",
            candidate_value(
                coverage["mapped_area_km2"],
                2,
                " km²"
            )
        )

        st.metric(
            "Unmapped area",
            candidate_value(
                coverage["unmapped_area_km2"],
                3,
                " km²"
            )
        )

        st.metric(
            "Mapped coverage",
            candidate_value(
                coverage["coverage_percent"],
                2,
                "%"
            )
        )

        st.markdown("#### Dominant mapped units")

        for unit in geology["dominant_units"]:

            st.markdown(
                f"**{unit['code']}** — "
                f"{unit['candidate_coverage_percent']:.2f}%"
            )

            st.caption(
                unit["lithology"] or "Lithology not specified"
            )

    if units:
        with st.expander(
            "Inspect complete geological unit inventory"
        ):
            st.dataframe(
                pd.DataFrame(units),
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    # --------------------------------------------------------
    # SCREENING EVIDENCE
    # --------------------------------------------------------

    candidate_section(
        None,
        "Terrain screening evidence",
        "Compare candidate terrain measurements with broader Jezero reference scenarios."
    )

    broad_col, moderate_col = st.columns(2)

    with broad_col:

        st.metric(
            "Broad screening overlap",
            candidate_value(
                screening["broad_overlap_percent"],
                2,
                "%"
            )
        )

        st.caption(
            "Based on the broad Jezero P10–P90 screening scenario."
        )

    with moderate_col:

        st.metric(
            "Moderate screening overlap",
            candidate_value(
                screening["moderate_overlap_percent"],
                2,
                "%"
            )
        )

        st.caption(
            "Based on the moderate Jezero P25–P75 screening scenario."
        )

    st.info(
        "These percentages are inherited screening statistics. "
        "They are not probabilities, independent validation results, "
        "or overall analogue similarity scores."
    )

    st.markdown("#### Scientific limitations")

    for limitation in profile["scientific_limitations"]:
        st.markdown(f"- {limitation}")

    st.divider()

    # --------------------------------------------------------
    # PROFILE EXPORTS
    # --------------------------------------------------------

    candidate_section(
        None,
        "Export candidate profile",
        "Download the selected candidate's underlying profile or its prepared research report."
    )

    json_bytes = json.dumps(
        profile,
        indent=2,
        ensure_ascii=False
    ).encode("utf-8")

    markdown_path = (
        root
        / "outputs"
        / "reports"
        / "candidate_profiles"
        / f"{candidate_id}_profile.md"
    )

    export_col1, export_col2 = st.columns(2)

    with export_col1:

        st.download_button(
            "Download JSON profile",
            data=json_bytes,
            file_name=f"{candidate_id}_profile.json",
            mime="application/json",
            use_container_width=True,
            key=f"json_download_{candidate_id}"
        )

    with export_col2:

        if (
            markdown_path.exists()
            and markdown_path.is_file()
            and markdown_path.stat().st_size > 0
        ):

            st.download_button(
                "Download research report",
                data=markdown_path.read_bytes(),
                file_name=f"{candidate_id}_profile.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"md_download_{candidate_id}"
            )


# ============================================================
# MAIN CANDIDATE INTELLIGENCE PAGE
# ============================================================

def candidate_intelligence_page(root):

    profiles_path = (
        root
        / "outputs"
        / "dashboard"
        / "candidate_profiles.json"
    )

    boundaries_path = (
        root
        / "outputs"
        / "maps"
        / "geological_characterization"
        / "verified_candidate_polygons.geojson"
    )

    comparison_path = (
        root
        / "outputs"
        / "dashboard"
        / "candidate_master_comparison.csv"
    )

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    st.markdown(
        '<div class="page-title">Candidate Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">'
        'Explore terrestrial candidate regions through spatial '
        'boundaries, terrain measurements, geological characterization '
        'and transparent Jezero reference comparisons.'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "EARTH STUDY REGION: ATACAMA DESERT, CHILE "
        "· MARS REFERENCE: JEZERO CRATER"
    )

    # --------------------------------------------------------
    # DATA AVAILABILITY
    # --------------------------------------------------------

    if not profiles_path.exists():

        st.error(
            "Candidate profile dataset not found. "
            "Run generate_candidate_profiles.py first."
        )
        return

    if not boundaries_path.exists():

        st.error(
            "Verified candidate boundary GeoJSON not found. "
            "Run geological_characterization.py first."
        )
        return

    try:

        data = load_profiles(str(profiles_path))

        geojson = load_candidate_boundaries(
            str(boundaries_path)
        )

    except Exception as error:

        st.error(
            f"Could not load candidate intelligence data: {error}"
        )
        return

    profiles = data.get("profiles", [])

    if not profiles:

        st.warning(
            "No candidate profiles are available."
        )
        return

    # --------------------------------------------------------
    # RESEARCH SUMMARY
    # --------------------------------------------------------

    st.markdown("### Research inventory")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Available candidate profiles",
        len(profiles)
    )

    c2.metric(
        "Planetary reference",
        "Mars"
    )

    c3.metric(
        "Reference region",
        "Jezero Crater"
    )

    st.caption(
        "Candidate ordering reflects connected-region area only. "
        "It does not represent scientific suitability or a validated "
        "ranking of planetary analogues."
    )

    st.divider()

    # --------------------------------------------------------
    # CANDIDATE SELECTION
    # --------------------------------------------------------

    candidate_section(
        None,
        "Explore a terrestrial region",
        "Select a candidate to inspect its individual spatial, terrain and geological evidence."
    )

    candidate_lookup = {
        profile["candidate_id"]: profile
        for profile in profiles
    }

    selected_id = st.selectbox(
        "Candidate region",
        options=list(candidate_lookup.keys()),
        format_func=lambda candidate_id: (
            f"{candidate_id} — "
            f"{candidate_lookup[candidate_id]['geography']['area_km2']:.2f} km²"
        ),
        key="terraform_candidate_selector"
    )

    selected_profile = candidate_lookup[selected_id]

    render_profile(
        selected_profile,
        geojson,
        root
    )

    # --------------------------------------------------------
    # MASTER COMPARISON TABLE
    # --------------------------------------------------------

    st.divider()

    candidate_section(
        None,
        "All candidate regions",
        "Review the generated master comparison dataset across the available candidate regions."
    )

    if comparison_path.exists():

        comparison_df = pd.read_csv(comparison_path)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download all candidate comparisons",
            data=comparison_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="terraform_candidate_comparison.csv",
            mime="text/csv",
            key="download_all_candidate_comparisons",
            use_container_width=True
        )

    else:

        st.warning(
            "Master comparison CSV is missing. "
            "Regenerate the candidate profiles."
        )

    st.caption(
        "CANDIDATE INTELLIGENCE · "
        "REGIONAL TERRAIN AND GEOLOGICAL SCREENING"
    )
