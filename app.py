from pathlib import Path
import base64
import html
import json
import mimetypes
import streamlit as st
import pandas as pd
import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None
import plotly.express as px
import folium

from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from src.data.dashboard_pipeline import load_and_process
from src.visualization.candidate_intelligence_page import candidate_intelligence_page


# ============================================================
# TERRAFORM | PLANETARY ANALOG INTELLIGENCE
# ============================================================

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

PLANETARY_DATASETS = {
    "Earth": {
        "source": "NOAA ETOPO 2022",
        "raster": "data/Global/Earth/earth_etopo_2022_60s.tif",
        "report": "Global/Earth/earth_global_report.json",
        "reference": "WGS84 + EGM2008 height",
        "description": "Global topography and bathymetry",
    },
    "Mars": {
        "source": "NASA PDS MOLA MEGDR",
        "raster": "data/Global/Mars/megt90n000eb.img",
        "report": "Global/Mars/mars_global_report.json",
        "reference": "MOLA areoid",
        "description": "Global median planetary topography",
    },
    "Moon": {
        "source": "NASA PDS LRO LOLA LDEM_16",
        "raster": "data/Global/Moon/LDEM_16.LBL",
        "report": "Global/Moon/moon_global_report.json",
        "reference": "Lunar reference sphere, radius 1,737,400 m",
        "description": "Global lunar digital elevation model",
    },
}

GLOBAL_COMPARISON_REPORT_PATH = (
    OUTPUTS / "Global/Comparisons/global_planetary_comparison_report.json"
)

ACCENT = "#55D6A0"
BACKGROUND = "#0A1018"
PANEL = "#111C27"
BORDER = "#263746"
MUTED = "#8FA3B5"
WHITE = "#EDF4F8"

NUMERIC_COLUMNS = [
    "latitude", "longitude", "elevation_m", "slope_deg",
    "roughness_m", "local_relief_m", "surface_temp_c",
    "precipitation_mm", "similarity_score"
]

COMPARISON_METRICS = [
    "slope_deg", "roughness_m", "local_relief_m",
    "precipitation_mm"
]

RESEARCH_CSV_FILES = {
    "Candidate inventory": "maps/candidate_regions/atacama_candidate_inventory.csv",
    "Master candidate comparison": "dashboard/candidate_master_comparison.csv",
    "Geological coverage": "maps/geological_characterization/candidate_geological_coverage.csv",
    "Geological units": "maps/geological_characterization/candidate_geological_units.csv",
    "Sensitivity summary": "maps/sensitivity_analysis/sensitivity_summary.csv",
}

RESEARCH_JSON_FILES = {
    "Candidate profiles": "dashboard/candidate_profiles.json",
    "Terrain comparison statistics": "maps/terrain_comparison/terrain_visualisation_statistics.json",
    "Earth-Mars comparison report": "reports/earth_mars_terrain_comparison.json",
    "Earth terrain analysis": "reports/earth_terrain_analysis.json",
    "Jezero terrain analysis": "reports/jezero_terrain_analysis.json",
    "Earth validation report": "reports/earth_validation_report.json",
    "Sensitivity analysis report": "reports/sensitivity_analysis_report.json",
}

RESEARCH_RASTER_FILES = {
    "Candidate region IDs": "maps/candidate_regions/atacama_candidate_region_ids.tif",
    "Broad screening mask": "maps/sensitivity_analysis/broad_three_metric_mask.tif",
    "Moderate screening mask": "maps/sensitivity_analysis/moderate_three_metric_mask.tif",
    "Strict screening mask": "maps/sensitivity_analysis/strict_three_metric_mask.tif",
    "4-neighbour cluster map": "maps/spatial_clusters/atacama_clusters_4_connectivity.tif",
    "8-neighbour cluster map": "maps/spatial_clusters/atacama_clusters_8_connectivity.tif",
    "Slope screening mask": "maps/terrain_range_screening/atacama_slope_range_mask.tif",
    "Roughness screening mask": "maps/terrain_range_screening/atacama_roughness_range_mask.tif",
    "Local relief screening mask": "maps/terrain_range_screening/atacama_local_relief_range_mask.tif",
    "Three-metric intersection": "maps/terrain_range_screening/atacama_three_metric_intersection.tif",
}

MARS_JSON_PATH = OUTPUTS / "global_mars/terrain_metrics/mars_global_terrain_report.json"
MARS_AREA_REPORT_PATH = OUTPUTS / "global_mars/mars_area_weighted_report.txt"

MARS_IMAGES = {
    "Global elevation": "global_mars/mars_global_elevation.png",
    "Elevation distribution": "global_mars/mars_global_elevation_distribution.png",
    "Observation classification": "global_mars/mars_global_observation_classification.png",
    "Observation density": "global_mars/mars_global_observation_density.png",
}

MARS_METRIC_IMAGES = {
    "Slope": "global_mars/terrain_metrics/mars_global_slope_degrees.png",
    "Roughness": "global_mars/terrain_metrics/mars_global_roughness_3x3_m.png",
    "Local relief": "global_mars/terrain_metrics/mars_global_local_relief_3x3_m.png",
    "Terrain quality overview": "global_mars/terrain_metrics/mars_global_terrain_quality.png",
}

VISUALISATION_GROUPS = {
    "Global Mars terrain": [
        ("Global Mars elevation", "global_mars/mars_global_elevation.png"),
        ("Mars elevation distribution", "global_mars/mars_global_elevation_distribution.png"),
        ("Mars observation classification", "global_mars/mars_global_observation_classification.png"),
        ("Mars observation density", "global_mars/mars_global_observation_density.png"),
    ],
    "Global Mars terrain metrics": [
        ("Global Mars slope", "global_mars/terrain_metrics/mars_global_slope_degrees.png"),
        ("Global Mars roughness", "global_mars/terrain_metrics/mars_global_roughness_3x3_m.png"),
        ("Global Mars local relief", "global_mars/terrain_metrics/mars_global_local_relief_3x3_m.png"),
        ("Global Mars terrain quality overview", "global_mars/terrain_metrics/mars_global_terrain_quality.png"),
    ],
    "Atacama–Jezero regional study": [
        ("Atacama candidate regions", "maps/candidate_regions/atacama_candidate_regions_map.png"),
        ("Broad screening map", "maps/sensitivity_analysis/broad_screening_map.png"),
        ("Moderate screening map", "maps/sensitivity_analysis/moderate_screening_map.png"),
        ("Strict screening map", "maps/sensitivity_analysis/strict_screening_map.png"),
        ("Screening sensitivity comparison", "maps/sensitivity_analysis/sensitivity_comparison.png"),
        ("Largest spatial clusters", "maps/spatial_clusters/atacama_largest_matching_clusters.png"),
        ("Earth–Mars terrain overview", "maps/terrain_comparison/earth_mars_terrain_overview.png"),
        ("Local relief distributions", "maps/terrain_comparison/local_relief_distribution.png"),
        ("Roughness distributions", "maps/terrain_comparison/roughness_distribution.png"),
        ("Slope distributions", "maps/terrain_comparison/slope_distribution.png"),
        ("Atacama terrain screening", "maps/terrain_range_screening/atacama_terrain_screening_overview.png"),
    ],
    "Global planetary visualisations": [
        ("Global Earth elevation", "Visualisations/Global/earth_global_elevation.png"),
        ("Global elevation distribution comparison", "Visualisations/Global/global_elevation_distribution_comparison.png"),
        ("Global normalised terrain comparison", "Visualisations/Global/global_normalized_terrain_comparison.png"),
        ("Global Mars elevation", "Visualisations/Global/mars_global_elevation.png"),
        ("Global Moon elevation", "Visualisations/Global/moon_global_elevation.png"),
    ],
}


st.set_page_config(
    page_title="TERRAFORM | Planetary Analog Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL STYLING
# ============================================================


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --background: #070B11;
    --navy: #0B1420;
    --panel: #101B27;
    --elevated: #142332;
    --border: #22364A;
    --accent: #55D6A0;
    --accent-2: #2F9F77;
    --cyan: #62C9E8;
    --text: #EDF4F8;
    --secondary: #A7B7C5;
    --muted: #74899A;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: rgba(10, 16, 24, 0.97);
}

[data-testid="stMainBlockContainer"],
.main .block-container,
main .block-container {
    box-sizing: border-box;
    width: 100%;
    max-width: 1600px;
    padding: 5rem 2.5rem 3rem 2.5rem !important;
}

[data-testid="stSidebar"] {
    background: #0D1721;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

h1, h2, h3 {
    color: var(--text);
    letter-spacing: -0.6px;
}

p, li, label {
    color: #BDCAD5;
}

small {
    color: var(--muted);
}

hr {
    border-color: var(--border);
}

.terraform-brand {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: 2px;
    color: white;
}

.terraform-subbrand {
    font-family: 'DM Mono', monospace;
    color: var(--accent);
    font-size: 9px;
    letter-spacing: 2.3px;
    margin-top: -5px;
}

.eyebrow {
    color: var(--accent);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 9px;
}

.page-title {
    color: white;
    font-size: 34px;
    line-height: 1.2;
    font-weight: 800;
    letter-spacing: -1.3px;
    margin-top: 0;
    margin-bottom: 8px;
    overflow-wrap: break-word;
}

.page-description {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 27px;
}

.hero {
    background:
        radial-gradient(circle at 90% 15%, rgba(85,214,160,.13), transparent 28%),
        linear-gradient(120deg, #142633, #0D1721 70%);
    border: 1px solid #304553;
    border-radius: 17px;
    padding: 31px 34px;
    margin-bottom: 24px;
}

.hero-title {
    font-size: 39px;
    font-weight: 800;
    color: white;
    letter-spacing: -1.5px;
    line-height: 1.15;
}

.hero-description {
    color: #A9BAC7;
    font-size: 14px;
    max-width: 720px;
    margin-top: 12px;
    line-height: 1.8;
}

.hero-tag {
    display: inline-block;
    color: var(--accent);
    background: rgba(85,214,160,.09);
    border: 1px solid rgba(85,214,160,.25);
    border-radius: 5px;
    padding: 5px 9px;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    margin-bottom: 17px;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 19px;
    margin-bottom: 14px;
}

.panel-title {
    color: white;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 5px;
}

.panel-description {
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 14px;
}

div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 17px 20px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

div[data-testid="stMetricValue"] {
    color: var(--text);
    font-weight: 700;
}

.stButton > button,
.stDownloadButton > button {
    background: #162B32;
    color: var(--accent);
    border: 1px solid #326452;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #1C3B36;
    border-color: var(--accent);
    color: white;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background: #111C27;
    border-color: var(--border);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 18px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted);
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
}

.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent);
    margin-right: 7px;
}

.status-text {
    color: var(--accent);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
}

.mono {
    font-family: 'DM Mono', monospace;
    color: var(--muted);
    font-size: 11px;
}

.empty-state {
    background: #111C27;
    border: 1px dashed #354B5B;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    color: var(--muted);
}

.empty-state h3 {
    color: #EAF1F5;
    font-size: 17px;
}

.footer {
    border-top: 1px solid var(--border);
    margin-top: 40px;
    padding-top: 17px;
    color: #718394;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
}

[data-testid="stAlert"] {
    border-radius: 9px;
}

img {
    max-width: 100%;
    height: auto;
}

@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"],
    .main .block-container,
    main .block-container {
        padding: 4.5rem 1.25rem 2rem 1.25rem !important;
    }

    .page-title {
        font-size: clamp(26px, 4vw, 32px);
    }

    .hero {
        padding: 24px;
    }

    .hero-title {
        font-size: clamp(29px, 5vw, 36px);
    }

    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.75rem !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex: 1 1 calc(50% - 0.75rem) !important;
        min-width: min(100%, 240px) !important;
    }
}

@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"],
    .main .block-container,
    main .block-container {
        padding: 4rem 0.85rem 1.5rem 0.85rem !important;
    }

    .terraform-brand {
        font-size: 21px;
    }

    .page-title {
        font-size: 27px;
        line-height: 1.2;
    }

    .page-description {
        font-size: 13px;
        margin-bottom: 19px;
    }

    .hero {
        padding: 19px;
        border-radius: 13px;
    }

    .hero-title {
        font-size: 29px;
        letter-spacing: -0.8px;
    }

    .hero-description {
        font-size: 13px;
        line-height: 1.65;
    }

    .panel {
        padding: 15px;
    }

    div[data-testid="stMetric"] {
        padding: 13px 14px;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
        width: 100% !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        overflow-x: auto;
    }

    .stTabs [data-baseweb="tab"] {
        white-space: nowrap;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .page-title, .hero-title, .tf-hero-title { font-family: 'Space Grotesk', 'Inter', sans-serif; }

.stApp {
    background:
        linear-gradient(rgba(98,201,232,.035) 1px, transparent 1px) 0 0 / 56px 56px,
        linear-gradient(90deg, rgba(98,201,232,.035) 1px, transparent 1px) 0 0 / 56px 56px,
        radial-gradient(ellipse at 85% -10%, rgba(85,214,160,.10), transparent 45%),
        radial-gradient(ellipse at 0% 100%, rgba(98,201,232,.06), transparent 40%),
        #070B11;
    background-attachment: fixed;
}
header[data-testid="stHeader"] { background: rgba(7,11,17,.85); backdrop-filter: blur(8px); }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1420 0%, #080E16 100%);
    border-right: 1px solid var(--border);
}
.tf-logo { display:flex; align-items:center; gap:11px; }
.tf-logo-mark {
    width:30px; height:30px; border-radius:50%;
    background: radial-gradient(circle at 32% 30%, #9CF0CB 0%, #55D6A0 28%, #1E6B55 62%, #0B1420 100%);
    box-shadow: 0 0 18px rgba(85,214,160,.35), inset -4px -4px 8px rgba(0,0,0,.45);
    position:relative; flex:none;
}
.tf-logo-mark::after {
    content:""; position:absolute; inset:-5px -9px; border:1px solid rgba(98,201,232,.45);
    border-radius:50%; transform: rotate(-24deg) scaleY(.32);
}
.tf-nav-label {
    font-family:'DM Mono',monospace; font-size:9.5px; letter-spacing:2.4px;
    color: var(--muted); margin: 18px 0 6px 2px; text-transform:uppercase;
}
[data-testid="stSidebar"] .stButton > button {
    width:100%; justify-content:flex-start; text-align:left;
    background: transparent; border: 1px solid transparent; border-left: 2px solid transparent;
    border-radius: 6px; color: var(--secondary); font-weight:500; font-size:13.5px;
    padding: 6px 12px; min-height: 0; transition: all .18s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(85,214,160,.06); color:#fff; border-left-color: rgba(85,214,160,.6);
    transform: translateX(2px);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, rgba(85,214,160,.14), rgba(85,214,160,.02));
    border-left: 2px solid var(--accent); color: var(--accent); font-weight:600;
}

/* ---------- global header ---------- */
.tf-header {
    display:flex; align-items:center; justify-content:space-between; gap:14px;
    padding: 9px 16px; margin-bottom: 22px;
    background: rgba(16,27,39,.62); backdrop-filter: blur(10px);
    border:1px solid var(--border); border-radius: 10px;
}
.tf-header-brand { font-family:'Space Grotesk',sans-serif; font-weight:700; letter-spacing:3px; color:#fff; font-size:14px; }
.tf-header-brand span { color: var(--accent); }
.tf-header-section { font-family:'DM Mono',monospace; font-size:11px; letter-spacing:2px; color: var(--secondary); text-transform:uppercase; text-align:center; }
.tf-header-status { font-family:'DM Mono',monospace; font-size:10px; letter-spacing:1.6px; color: var(--muted); white-space:nowrap; }
.tf-header-status i { display:inline-block; width:6px; height:6px; border-radius:50%; background: var(--accent); margin-right:7px; }

/* ---------- surfaces ---------- */
.panel, div[data-testid="stMetric"] {
    background: linear-gradient(160deg, rgba(20,35,50,.78), rgba(16,27,39,.78));
    border: 1px solid var(--border); border-radius: 10px;
    transition: border-color .2s ease, transform .2s ease, box-shadow .2s ease;
}
.panel:hover, div[data-testid="stMetric"]:hover {
    border-color: rgba(85,214,160,.38); box-shadow: 0 8px 28px rgba(0,0,0,.35);
}
.eyebrow { position:relative; padding-left:26px; }
.eyebrow::before { content:""; position:absolute; left:0; top:50%; width:18px; height:1px; background: var(--accent); }
.page-title { font-family:'Space Grotesk','Inter',sans-serif; }
.page-title::after {
    content:""; display:block; width:64px; height:2px; margin-top:12px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.stButton > button, .stDownloadButton > button { transition: all .18s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(85,214,160,.14); }
[data-testid="stExpander"] { border:1px solid var(--border); border-radius:10px; background: rgba(16,27,39,.5); }
[data-testid="stImage"] img { border:1px solid var(--border); border-radius:8px; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent); }

/* ---------- home page ---------- */
.tf-hero {
    position:relative; overflow:hidden; border:1px solid #2B4357; border-radius:16px;
    min-height: 470px; padding: 64px 52px; margin-bottom: 26px;
    background-color:#070B11;
    background-image:
        linear-gradient(90deg, rgba(7,11,17,.96) 0%, rgba(7,11,17,.78) 38%, rgba(7,11,17,.15) 75%),
        radial-gradient(circle at 82% 105%, #b5532f 0%, #7a3322 22%, #3a1a1a 44%, transparent 62%),
        radial-gradient(1.5px 1.5px at 12% 18%, #fff 50%, transparent 52%),
        radial-gradient(1px 1px at 33% 52%, #cfe 50%, transparent 52%),
        radial-gradient(1.5px 1.5px at 58% 14%, #fff 50%, transparent 52%),
        radial-gradient(1px 1px at 72% 36%, #fff 50%, transparent 52%),
        radial-gradient(1px 1px at 90% 20%, #cfe 50%, transparent 52%),
        linear-gradient(#070B11, #0B1420);
}
.tf-hero.has-img {
    background-image:
        linear-gradient(90deg, rgba(7,11,17,.96) 0%, rgba(7,11,17,.80) 40%, rgba(7,11,17,.10) 78%),
        var(--hero-img),
        radial-gradient(circle at 82% 105%, #b5532f 0%, #7a3322 22%, #3a1a1a 44%, transparent 62%),
        linear-gradient(#070B11, #0B1420);
    background-size: cover, cover, cover, cover;
    background-position: center, 78% 55%, center, center;
}
.tf-hero::after {
    content:""; position:absolute; left:0; right:0; bottom:0; height:2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    animation: tf-scan 5s ease-in-out infinite;
}
@keyframes tf-scan { 0%,100% { opacity:.25; } 50% { opacity:1; } }
.tf-hero-inner { position:relative; z-index:2; max-width: 640px; }
.tf-hero-tag { font-family:'DM Mono',monospace; font-size:10.5px; letter-spacing:3px; color:var(--accent); margin-bottom:20px; }
.tf-hero-title { font-size: clamp(42px, 8vw, 92px); font-weight:700; letter-spacing: clamp(4px, 1.2vw, 12px); color:#fff; line-height:1; margin:0; }
.tf-hero-sub { font-family:'DM Mono',monospace; letter-spacing: 4px; color: var(--accent); font-size: 13px; margin-top: 14px; }
.tf-hero-statement { font-size: 19px; color: var(--text); margin-top: 28px; line-height:1.5; font-weight:500; }
.tf-hero-desc { font-size: 14px; color: var(--secondary); margin-top: 14px; line-height: 1.8; max-width: 560px; }
.tf-credit { position:absolute; right:14px; bottom:9px; z-index:3; font-family:'DM Mono',monospace; font-size:9px; color: rgba(167,183,197,.6); letter-spacing:.8px; }

.tf-step { position:relative; padding:18px 18px 16px 18px; background: rgba(16,27,39,.65); border:1px solid var(--border); border-radius:10px; min-height:118px; height:100%; }
.tf-step .n { font-family:'DM Mono',monospace; font-size:10px; letter-spacing:2px; color: var(--accent); }
.tf-step .t { color:#fff; font-weight:600; font-size:15px; margin-top:8px; }
.tf-step .d { color: var(--muted); font-size:12px; margin-top:5px; line-height:1.55; }

.tf-world { border:1px solid var(--border); border-radius:12px; overflow:hidden; background: var(--panel); transition: border-color .2s ease, transform .25s ease; }
.tf-world:hover { border-color: rgba(85,214,160,.45); transform: translateY(-3px); }
.tf-world-img { height:210px; background-size:cover; background-position:center; transition: transform .6s ease; }
.tf-world:hover .tf-world-img { transform: scale(1.04); }
.tf-world-wrap { overflow:hidden; }
.tf-world-body { padding:16px 18px 14px 18px; }
.tf-world-body .name { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:22px; letter-spacing:3px; color:#fff; }
.tf-world-body .src { font-family:'DM Mono',monospace; font-size:10.5px; color: var(--accent); letter-spacing:1px; margin-top:4px; }
.tf-world-body .desc { font-size:12.5px; color: var(--secondary); margin-top:9px; line-height:1.6; }

.tf-cap { padding:16px 18px; border:1px solid var(--border); border-left:2px solid var(--accent-2); border-radius:8px; background: rgba(16,27,39,.55); margin-bottom:12px; min-height:92px; }
.tf-cap .t { color:#fff; font-weight:600; font-size:14.5px; }
.tf-cap .d { color: var(--muted); font-size:12px; margin-top:5px; line-height:1.55; }

.tf-cta { text-align:center; padding: 54px 24px 38px 24px; margin-top:14px; border:1px solid var(--border); border-radius:16px;
    background: radial-gradient(ellipse at 50% 120%, rgba(85,214,160,.16), transparent 60%), rgba(16,27,39,.55); }
.tf-cta h2 { font-family:'Space Grotesk',sans-serif; letter-spacing: 3px; font-size: clamp(20px, 3.4vw, 32px); color:#fff; margin:0 0 6px 0; }

@media (max-width: 900px) {
    .tf-hero { padding: 40px 26px; min-height: 400px; }
    .tf-header { flex-wrap: wrap; }
    .tf-header-section { text-align:left; }
}
@media (max-width: 640px) {
    .tf-hero { padding: 30px 20px; min-height: 360px; border-radius: 12px; }
    .tf-hero-statement { font-size: 16px; }
    .tf-world-img { height: 170px; }
    .tf-header-status { display:none; }
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
/* ===== Consistent card system ===== */
[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
[data-testid="stColumn"], [data-testid="column"] { display: flex; flex-direction: column; min-width: 0; }
[data-testid="stColumn"] > [data-testid="stVerticalBlock"],
[data-testid="column"] > [data-testid="stVerticalBlock"] { flex: 1 1 auto; min-width: 0; }

/* a lone panel in a column fills the row height, so sibling cards match */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child:has(.panel),
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > .element-container:only-child:has(.panel),
[data-testid="column"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child:has(.panel),
[data-testid="column"] > [data-testid="stVerticalBlock"] > .element-container:only-child:has(.panel),
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.tf-world),
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > .element-container:has(.tf-world),
[data-testid="column"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.tf-world),
[data-testid="column"] > [data-testid="stVerticalBlock"] > .element-container:has(.tf-world) {
    flex: 1 1 auto; display: flex;
}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child:has(.panel) > [data-testid="stMarkdown"],
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > .element-container:only-child:has(.panel) > [data-testid="stMarkdown"],
[data-testid="column"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child:has(.panel) > [data-testid="stMarkdown"],
[data-testid="column"] > [data-testid="stVerticalBlock"] > .element-container:only-child:has(.panel) > [data-testid="stMarkdown"],
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.tf-world) > [data-testid="stMarkdown"],
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > .element-container:has(.tf-world) > [data-testid="stMarkdown"],
[data-testid="column"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.tf-world) > [data-testid="stMarkdown"],
[data-testid="column"] > [data-testid="stVerticalBlock"] > .element-container:has(.tf-world) > [data-testid="stMarkdown"] {
    flex: 1 1 auto; display: flex; width: 100%;
}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child:has(.panel) [data-testid="stMarkdownContainer"],
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > .element-container:only-child:has(.panel) [data-testid="stMarkdownContainer"],
[data-testid="column"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child:has(.panel) [data-testid="stMarkdownContainer"],
[data-testid="column"] > [data-testid="stVerticalBlock"] > .element-container:only-child:has(.panel) [data-testid="stMarkdownContainer"],
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.tf-world) [data-testid="stMarkdownContainer"],
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > .element-container:has(.tf-world) [data-testid="stMarkdownContainer"],
[data-testid="column"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(.tf-world) [data-testid="stMarkdownContainer"],
[data-testid="column"] > [data-testid="stVerticalBlock"] > .element-container:has(.tf-world) [data-testid="stMarkdownContainer"] {
    flex: 1 1 auto; display: flex; flex-direction: column; width: 100%;
}
[data-testid="stMarkdownContainer"] > .panel,
[data-testid="stMarkdownContainer"] > .tf-world { flex: 1 1 auto; box-sizing: border-box; }
[data-testid="stMarkdownContainer"] > .panel:last-child { margin-bottom: 0; }

.panel, .tf-step, .tf-cap, .tf-world { box-sizing: border-box; overflow-wrap: break-word; word-break: normal; }
.panel .mono, .panel div { overflow-wrap: break-word; }

/* HTML grids: every card in a group is exactly the same size */
.tf-grid { display: grid; gap: 14px; grid-auto-rows: 1fr; margin: 6px 0 4px 0; }
.tf-grid-6 { grid-template-columns: repeat(6, minmax(0, 1fr)); }
.tf-grid-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.tf-step, .tf-cap { display: flex; flex-direction: column; justify-content: flex-start; height: 100%; margin: 0; }
.tf-step { padding: 18px; min-height: 150px; }
.tf-cap { padding: 16px 18px; min-height: 108px; }
.tf-step .t, .tf-cap .t { line-height: 1.3; }
@media (max-width: 1200px) { .tf-grid-6 { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 1000px) { .tf-grid-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 640px)  { .tf-grid-6, .tf-grid-4 { grid-template-columns: repeat(1, minmax(0, 1fr)); } .tf-step, .tf-cap { min-height: 0; } }

/* planet cards: identical photo frame + body */
.tf-world { display: flex; flex-direction: column; margin: 0; }
/* the button that follows a planet card always sits on its own row, clear of the card */
[data-testid="stElementContainer"]:has(.tf-world) + [data-testid="stElementContainer"],
.element-container:has(.tf-world) + .element-container { margin-top: 1.1rem; position: relative; z-index: 1; }
[data-testid="stElementContainer"]:has(.tf-world), .element-container:has(.tf-world) { min-height: 0; }

/* ===== Final call-to-action: heading and button share one bordered box ===== */
[data-testid="stVerticalBlockBorderWrapper"]:has(.tf-cta-inner) {
    border: 1px solid var(--border); border-radius: 16px; margin-top: 14px;
    padding: 44px 24px 48px 24px !important;
    background: radial-gradient(ellipse at 50% 120%, rgba(85,214,160,.16), transparent 60%), rgba(16,27,39,.55);
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.tf-cta-inner) > [data-testid="stVerticalBlock"] { gap: 1.6rem; align-items: stretch; }
.tf-cta-inner { text-align: center; }
.tf-cta-inner h2 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 3px; font-size: clamp(20px, 3.4vw, 32px); color: #fff; margin: 0 0 10px 0; padding: 0; }

.tf-world-wrap { flex: none; }
.tf-world-img { height: auto !important; aspect-ratio: 16 / 10; width: 100%; background-color: #05080d;
    background-size: cover; background-position: center center; background-repeat: no-repeat; }
.tf-world-body { flex: 1 1 auto; min-height: 168px; }
@media (max-width: 900px) { .tf-world-img { max-height: 300px; } }

/* ===== Metrics: never truncate numbers ===== */
div[data-testid="stMetric"] { box-sizing: border-box; width: 100%; min-height: 116px; padding: 16px 18px; overflow: visible; }
div[data-testid="stMetric"] > div { width: 100%; }
[data-testid="stMetricLabel"] { min-height: 2.6em; align-items: flex-start; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
    overflow-wrap: break-word; max-width: 100%;
}
[data-testid="stMetricValue"] { font-size: clamp(1.2rem, 1.7vw, 1.75rem); line-height: 1.2; }
[data-testid="stMetricLabel"] { line-height: 1.25; }

/* ===== Buttons: no overflow, no overlap ===== */
.stButton > button, .stDownloadButton > button {
    height: auto; min-height: 2.6rem; white-space: normal; line-height: 1.25;
    padding: .55rem .9rem; max-width: 100%;
}
.stButton > button p, .stDownloadButton > button p { white-space: normal; overflow-wrap: break-word; }
[data-testid="stSidebar"] .stButton > button { min-height: 0; padding: 6px 12px; }
[data-testid="stColumn"] .stButton, [data-testid="column"] .stButton { width: 100%; }
[data-testid="stColumn"] .stButton > button.tf-x, [data-testid="stColumn"] .stDownloadButton > button { width: 100%; }

/* ===== Images in multi-column groups share one frame ===== */
[data-testid="stHorizontalBlock"] [data-testid="stImage"] img {
    width: 100% !important; height: auto !important; aspect-ratio: 16 / 10; object-fit: contain;
    background: #0B1420; border: 1px solid var(--border); border-radius: 8px;
}
[data-testid="stImage"] { width: 100%; }
[data-testid="stImageCaption"] { text-align: left; margin-top: 6px; min-height: 2.4em; overflow-wrap: break-word; }

.tf-viz-title { font-weight: 600; color: var(--text); font-size: 14px; line-height: 1.3; min-height: 2.7em;
    display: flex; align-items: flex-end; margin: 4px 0 6px 0; }
.tf-viz-missing { aspect-ratio: 16 / 10; display: flex; align-items: center; justify-content: center; text-align: center;
    padding: 16px; box-sizing: border-box; color: var(--muted); font-size: 13px; background: #0B1420;
    border: 1px dashed #354B5B; border-radius: 8px; }

/* ===== Sidebar research target ===== */
.tf-target { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; background: rgba(16,27,39,.55); }
.tf-target-note { font-size: 11px; line-height: 1.5; color: var(--muted); margin-bottom: 8px; }
.tf-target-row { display: grid; grid-template-columns: 46px 1fr auto; gap: 8px; align-items: start;
    padding: 7px 0; border-top: 1px solid rgba(34,54,74,.9); }
.tf-target-row .b { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 1.5px; color: var(--accent); padding-top: 2px; }
.tf-target-row .r { font-size: 12px; line-height: 1.4; color: var(--text); overflow-wrap: break-word; min-width: 0; }
.tf-target-row .k { font-family: 'DM Mono', monospace; font-size: 8.5px; letter-spacing: 1px; color: var(--muted); padding-top: 3px; }

/* ===== Hero spacing so footer credit never collides with copy ===== */
.tf-hero { padding-bottom: 72px; }

/* responsive: both old and new Streamlit column test ids */
@media (max-width: 900px) {
    [data-testid="stColumn"] { flex: 1 1 calc(50% - 0.75rem) !important; min-width: min(100%, 240px) !important; }
}
@media (max-width: 640px) {
    [data-testid="stColumn"] { flex: 1 1 100% !important; min-width: 100% !important; width: 100% !important; }
    div[data-testid="stMetric"] { min-height: 0; }
    .tf-hero { padding-bottom: 60px; }
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* vertical rhythm between stacked widgets */
div[data-testid="stMetric"] { margin-bottom: .35rem; }
[data-testid="stElementContainer"]:has([data-testid="stMetric"]) { margin-bottom: .25rem; }
iframe[title*="st_folium"], iframe[title*="streamlit_folium"] { display: block; margin: 0; }

/* equal card grids */
.tf-grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.tf-grid .panel { margin: 0; height: 100%; min-height: 138px; padding: 18px; }
.tf-card-title { font-size: 17px; font-weight: 700; color: #fff; margin-top: 8px; line-height: 1.3; }
.tf-card-text { font-size: 12px; color: var(--muted); margin-top: 7px; line-height: 1.6; }
@media (max-width: 900px) { .tf-grid-3 { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 640px) { .tf-grid-3 { grid-template-columns: 1fr; } .tf-grid .panel { min-height: 0; } }

/* ===== Technical data presentation ===== */
.tf-kv { display: grid; grid-template-columns: minmax(130px, 220px) minmax(0, 1fr); border: 1px solid var(--border);
    border-radius: 8px; overflow: hidden; margin: 4px 0 12px 0; background: rgba(11,20,32,.7); }
.tf-kv .k, .tf-kv .v { padding: 8px 12px; border-top: 1px solid rgba(34,54,74,.8); font-size: 12.5px; line-height: 1.5; overflow-wrap: anywhere; }
.tf-kv .k:nth-child(1), .tf-kv .v:nth-child(2) { border-top: none; }
.tf-kv .k { font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: .6px; color: var(--muted); text-transform: uppercase; background: rgba(20,35,50,.6); }
.tf-kv .v { color: var(--text); }
@media (max-width: 640px) { .tf-kv { grid-template-columns: 1fr; } .tf-kv .v { border-top: none; padding-top: 0; } }

[data-testid="stJson"], [data-testid="stCode"] {
    max-height: 420px; overflow: auto; border: 1px solid var(--border); border-radius: 8px; background: #0B1420;
}
[data-testid="stCode"] pre, [data-testid="stCode"] code { white-space: pre-wrap !important; overflow-wrap: anywhere; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# GENERAL UTILITIES
# ============================================================

def format_number(value, suffix=""):
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):,.2f}{suffix}"
    except (TypeError, ValueError):
        return str(value)

        
def format_file_size(size_bytes):
    """Format file sizes using readable binary units."""
    size_bytes = int(size_bytes)

    if size_bytes < 1024:
        return f"{size_bytes:,} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:,.2f} KB"
    if size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):,.2f} MB"
    return f"{size_bytes / (1024 ** 3):,.2f} GB"



def section_header(label, title, description=""):
    st.markdown(
        f'<div class="eyebrow">{html.escape(label)}</div>'
        f'<div class="page-title">{html.escape(title)}</div>'
        f'<div class="page-description">{html.escape(description)}</div>',
        unsafe_allow_html=True
    )


def empty_state(title, message):
    st.markdown(
        f'<div class="empty-state"><h3>{html.escape(title)}</h3>'
        f'<p>{html.escape(message)}</p></div>',
        unsafe_allow_html=True
    )


def metric_card(label, value, help_text=""):
    st.markdown(
        f'<div class="panel"><div class="mono">{html.escape(label.upper())}</div>'
        f'<div style="font-size:26px;font-weight:800;color:#FFFFFF;margin-top:9px;">'
        f'{html.escape(str(value))}</div><div style="font-size:11px;color:#8FA3B5;margin-top:5px;">'
        f'{html.escape(help_text)}</div></div>',
        unsafe_allow_html=True
    )


def load_research_csv(relative_path):
    path = OUTPUTS / relative_path
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as error:
        print(f"Could not load research CSV {path}: {error}")
        return None


def load_research_json(relative_path):
    path = OUTPUTS / relative_path
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        print(f"Could not load research JSON {path}: {error}")
        return None


def safe_metric_value(row, column, suffix="", digits=2):
    if row is None or column not in row.index:
        return "—"
    value = row[column]
    if pd.isna(value):
        return "—"
    try:
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def get_profile_data(candidate_id):
    if not isinstance(CANDIDATE_PROFILES, dict):
        return None
    for profile in CANDIDATE_PROFILES.get("profiles", []):
        if str(profile.get("candidate_id")) == str(candidate_id):
            return profile
    return None


def load_text_report(path):
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as error:
        return f"Could not read report: {error}"


def inspect_raster(path):
    """Read raster metadata only; never load the raster band into memory."""
    path = Path(path)
    if not path.exists():
        return {"status": "Missing", "path": str(path)}
    if rasterio is None:
        return {"status": "Rasterio unavailable", "path": str(path)}

    try:
        with rasterio.open(path) as dataset:
            return {
                "status": "Available",
                "filename": path.name,
                "size_mb": path.stat().st_size / (1024 * 1024),
                "width": dataset.width,
                "height": dataset.height,
                "bands": dataset.count,
                "crs": str(dataset.crs) if dataset.crs else "Not declared",
                "resolution": tuple(dataset.res),
                "bounds": tuple(dataset.bounds),
                "dtype": ", ".join(dataset.dtypes),
                "nodata": dataset.nodata,
                "scales": tuple(dataset.scales),
                "offsets": tuple(dataset.offsets),
                "driver": dataset.driver,
            }
    except Exception as error:
        return {"status": "Read error", "path": str(path), "error": str(error)}


def show_image(relative_path, caption=None, width="stretch"):
    path = OUTPUTS / relative_path
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=(width == "stretch"))
        return True
    st.warning("This visualisation is currently unavailable.")
    return False


def kv_table(rows):
    """Clean label / value list. Values are shown in full (wrapping allowed)."""
    cells = "".join(
        f'<div class="k">{html.escape(str(label).rstrip(":"))}</div>'
        f'<div class="v">{html.escape(str(value))}</div>'
        for label, value in rows
    )
    st.markdown(f'<div class="tf-kv">{cells}</div>', unsafe_allow_html=True)


def show_stat_values(values):
    """Scalars as a tidy table; nested structures as collapsed JSON."""
    if not isinstance(values, dict) or not values:
        st.caption("No values available.")
        return
    scalars = [(k, v) for k, v in values.items() if not isinstance(v, (dict, list))]
    nested = {k: v for k, v in values.items() if isinstance(v, (dict, list))}
    if scalars:
        kv_table([(str(k).replace("_", " "), v) for k, v in scalars])
    if nested:
        st.json(nested, expanded=False)


def viz_tile(title, relative_path):
    """Uniform gallery tile: fixed title band + fixed-ratio image frame."""
    path = OUTPUTS / relative_path
    st.markdown(
        f'<div class="tf-viz-title">{html.escape(title)}</div>',
        unsafe_allow_html=True
    )
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.markdown(
            '<div class="tf-viz-missing">This visualisation is currently unavailable.</div>',
            unsafe_allow_html=True
        )


def show_report_file(path, title):
    if path.exists():
        with st.expander(title):
            content = load_text_report(path)
            st.code(content or "Report is empty.", language="text")
            st.download_button(
                f"Download {title}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/plain",
                key=f"download_{path.name}"
            )


# ============================================================
# LOAD RESEARCH OUTPUTS
# ============================================================

RESEARCH_CSV = {
    name: load_research_csv(path)
    for name, path in RESEARCH_CSV_FILES.items()
}
RESEARCH_JSON = {
    name: load_research_json(path)
    for name, path in RESEARCH_JSON_FILES.items()
}

CANDIDATE_INVENTORY = RESEARCH_CSV.get("Candidate inventory")
MASTER_CANDIDATES = RESEARCH_CSV.get("Master candidate comparison")
GEOLOGICAL_COVERAGE = RESEARCH_CSV.get("Geological coverage")
GEOLOGICAL_UNITS = RESEARCH_CSV.get("Geological units")
SENSITIVITY_SUMMARY = RESEARCH_CSV.get("Sensitivity summary")

CANDIDATE_PROFILES = RESEARCH_JSON.get("Candidate profiles")
TERRAIN_COMPARISON_STATS = RESEARCH_JSON.get("Terrain comparison statistics")
EARTH_MARS_REPORT = RESEARCH_JSON.get("Earth-Mars comparison report")
EARTH_TERRAIN_REPORT = RESEARCH_JSON.get("Earth terrain analysis")
JEZERO_TERRAIN_REPORT = RESEARCH_JSON.get("Jezero terrain analysis")
EARTH_VALIDATION_REPORT = RESEARCH_JSON.get("Earth validation report")
SENSITIVITY_REPORT = RESEARCH_JSON.get("Sensitivity analysis report")

MARS_TERRAIN_REPORT = load_research_json(
    "global_mars/terrain_metrics/mars_global_terrain_report.json"
)
MARS_AREA_REPORT = load_text_report(MARS_AREA_REPORT_PATH)

PLANETARY_REPORTS = {
    name: load_research_json(spec["report"])
    for name, spec in PLANETARY_DATASETS.items()
}
GLOBAL_PLANETARY_REPORT = load_research_json(
    "Global/Comparisons/global_planetary_comparison_report.json"
)
PLANETARY_RASTER_METADATA = {
    name: inspect_raster(ROOT / spec["raster"])
    for name, spec in PLANETARY_DATASETS.items()
}


def repository_statistics():
    csv_count = sum(frame is not None for frame in RESEARCH_CSV.values())
    json_count = sum(report is not None for report in RESEARCH_JSON.values())
    raster_count = sum(
        (OUTPUTS / path).exists()
        for path in RESEARCH_RASTER_FILES.values()
    )
    return {
        "inventory": len(CANDIDATE_INVENTORY) if CANDIDATE_INVENTORY is not None else 0,
        "profiles": len(MASTER_CANDIDATES) if MASTER_CANDIDATES is not None else 0,
        "geology": len(GEOLOGICAL_UNITS) if GEOLOGICAL_UNITS is not None else 0,
        "sensitivity": len(SENSITIVITY_SUMMARY) if SENSITIVITY_SUMMARY is not None else 0,
        "csv": csv_count,
        "json": json_count,
        "rasters": raster_count,
        "assets": csv_count + json_count + raster_count,
    }


REPOSITORY_STATS = repository_statistics()


# ============================================================
# SIDEBAR AND NAVIGATION
# ============================================================

PAGES = [
    "Mission Overview",
    "Planetary Environments",
    "Moon Explorer",
    "Discovery Explorer",
    "Global Mars Explorer",
    "Regional Analogue Study",
    "Candidate Intelligence",
    "Global Planetary Comparison",
    "Research Visualisations",
    "Data Laboratory",
    "Technical Deep Dive",
]

HOME_PAGE = "Home"

NAV_GROUPS = [
    ("OVERVIEW", [HOME_PAGE, "Mission Overview"]),
    ("PLANETARY EXPLORATION", ["Planetary Environments", "Moon Explorer", "Discovery Explorer", "Global Mars Explorer"]),
    ("RESEARCH", ["Regional Analogue Study", "Candidate Intelligence", "Global Planetary Comparison", "Research Visualisations"]),
    ("SYSTEM", ["Data Laboratory", "Technical Deep Dive"]),
]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = HOME_PAGE


def go_to_page(target):
    """Navigation callback: route to any existing page (or Home)."""
    st.session_state["nav_page"] = target


with st.sidebar:
    st.markdown(
        '<div class="tf-logo"><div class="tf-logo-mark"></div><div>'
        '<div class="terraform-brand">TERRAFORM<span style="color:#55D6A0">.</span></div>'
        '<div class="terraform-subbrand">PLANETARY ANALOG INTELLIGENCE</div>'
        '</div></div>',
        unsafe_allow_html=True
    )
    st.write("")
    st.markdown(
        '<span class="status-dot"></span><span class="status-text">RESEARCH ENVIRONMENT</span>',
        unsafe_allow_html=True
    )
    st.divider()
    for group_label, group_pages in NAV_GROUPS:
        st.markdown(f'<div class="tf-nav-label">{group_label}</div>', unsafe_allow_html=True)
        for nav_target in group_pages:
            st.button(
                nav_target,
                key=f"nav_{nav_target}",
                on_click=go_to_page,
                args=(nav_target,),
                type="primary" if st.session_state["nav_page"] == nav_target else "secondary",
                use_container_width=True,
            )
    page = st.session_state["nav_page"]
    st.divider()
    st.markdown("##### DATA CONNECTIONS")
    earth_upload = st.file_uploader(
        "Additional Earth dataset", type=["csv"], key="earth_upload",
        help="Optional custom Earth CSV containing location, latitude and longitude."
    )
    mars_upload = st.file_uploader(
        "Additional Mars dataset", type=["csv"], key="mars_upload",
        help="Optional custom Mars CSV containing documented measurements."
    )
    st.caption(
        "Existing TERRAFORM research outputs load automatically. "
        "Uploads are optional additional datasets."
    )
    st.divider()
    st.markdown("##### RESEARCH TARGET")
    st.markdown(
        '<div class="tf-target">'
        '<div class="tf-target-note">Terrestrial analogue environments compared with Mars and Moon reference datasets</div>'
        '<div class="tf-target-row"><span class="b">EARTH</span><span class="r">Atacama Desert<br>Death Valley</span><span class="k">ANALOGUE</span></div>'
        '<div class="tf-target-row"><span class="b">MARS</span><span class="r">MOLA Region<br>Jezero Crater</span><span class="k">REFERENCE</span></div>'
        '<div class="tf-target-row"><span class="b">MOON</span><span class="r">LOLA Region</span><span class="k">REFERENCE</span></div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.divider()
    st.caption("Independent research platform")


# ============================================================
# BACKEND DATA INTEGRATION
# ============================================================

try:
    earth_result = load_and_process(earth_upload, "earth")
    mars_result = load_and_process(mars_upload, "mars")

    earth_df = earth_result["cleaned"]
    mars_df = mars_result["cleaned"]

    earth_source = earth_result["source"]
    mars_source = mars_result["source"]

    earth_report = earth_result["report"]
    mars_report = mars_result["report"]

    earth_rejected = earth_result["rejected"]
    mars_rejected = mars_result["rejected"]

    earth_saved_paths = earth_result["saved_paths"]
    mars_saved_paths = mars_result["saved_paths"]

except Exception as error:
    st.error(f"Backend data integration error: {error}")
    st.exception(error)
    st.stop()


# ============================================================
# PLOTLY THEME
# ============================================================

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=WHITE, size=11),
    margin=dict(l=15, r=15, t=35, b=15),
    xaxis=dict(gridcolor="#263746", zerolinecolor="#263746"),
    yaxis=dict(gridcolor="#263746", zerolinecolor="#263746"),
    legend=dict(bgcolor="rgba(0,0,0,0)")
)


# ============================================================
# MAP GENERATORS
# ============================================================

def create_candidate_map(df):
    if df is None or df.empty:
        return None
    required = ["latitude", "longitude", "location"]
    if not all(column in df.columns for column in required):
        return None

    map_df = df.copy()
    for col in ["latitude", "longitude"]:
        map_df[col] = pd.to_numeric(map_df[col], errors="coerce")
    map_df = map_df.dropna(subset=["latitude", "longitude"])
    if map_df.empty:
        return None

    map_object = folium.Map(
        location=[map_df["latitude"].mean(), map_df["longitude"].mean()],
        zoom_start=2, tiles=None, control_scale=True
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=True).add_to(map_object)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri", name="Satellite Imagery", show=False
    ).add_to(map_object)
    folium.TileLayer(
        "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap", name="Topographic Map", show=False
    ).add_to(map_object)

    cluster = MarkerCluster(name="Earth candidate locations").add_to(map_object)
    for _, row in map_df.iterrows():
        location_name = html.escape(str(row["location"]))
        popup = (
            f"<b>{location_name}</b><br>"
            f"Latitude: {row['latitude']:.4f}<br>"
            f"Longitude: {row['longitude']:.4f}"
        )
        folium.Marker(
            [row["latitude"], row["longitude"]],
            tooltip=location_name,
            popup=folium.Popup(popup, max_width=250),
            icon=folium.Icon(color="green", icon="globe")
        ).add_to(cluster)
    folium.LayerControl().add_to(map_object)
    return map_object


def create_research_candidate_map(df):
    """Map extracted Atacama candidate centroids; not a suitability map."""
    if df is None or df.empty:
        return None
    required = ["candidate_id", "centroid_latitude", "centroid_longitude"]
    if not all(column in df.columns for column in required):
        return None

    map_df = df.copy()
    for col in ["centroid_latitude", "centroid_longitude"]:
        map_df[col] = pd.to_numeric(map_df[col], errors="coerce")
    map_df = map_df.dropna(subset=["centroid_latitude", "centroid_longitude"])
    if map_df.empty:
        return None

    map_object = folium.Map(
        location=[
            map_df["centroid_latitude"].mean(),
            map_df["centroid_longitude"].mean()
        ],
        zoom_start=6, tiles=None, control_scale=True
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=True).add_to(map_object)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri", name="Satellite imagery", show=False
    ).add_to(map_object)

    cluster = MarkerCluster(name="Extracted candidate regions").add_to(map_object)
    for _, row in map_df.iterrows():
        candidate_id = html.escape(str(row["candidate_id"]))
        area = row.get("area_km2", None)
        area_text = f"{float(area):,.2f} km²" if pd.notna(area) else "Not available"
        popup = (
            f"<b>{candidate_id}</b><br>Extracted Atacama candidate region<br><br>"
            f"Area: {area_text}<br>"
            f"Latitude: {row['centroid_latitude']:.5f}<br>"
            f"Longitude: {row['centroid_longitude']:.5f}"
        )
        folium.CircleMarker(
            [row["centroid_latitude"], row["centroid_longitude"]],
            radius=5, color="#55D6A0", fill=True,
            fill_color="#55D6A0", fill_opacity=.75, weight=1,
            tooltip=candidate_id,
            popup=folium.Popup(popup, max_width=280)
        ).add_to(cluster)
    folium.LayerControl().add_to(map_object)
    return map_object



# ============================================================
# PAGE 1 — PLANETARY ENVIRONMENTS
# ============================================================

def planetary_environments_page():

    section_header(
        "PLANETARY DATA",
        "Planetary Environments",
        "Explore global elevation datasets, measured terrain statistics "
        "and the scientific reference systems of Earth, Mars and the Moon."
    )

    reports = globals().get("PLANETARY_REPORTS", {})
    raster_metadata = globals().get("PLANETARY_RASTER_METADATA", {})

    if not isinstance(reports, dict):
        reports = {}

    if not isinstance(raster_metadata, dict):
        raster_metadata = {}

    planet = st.selectbox(
        "SELECT PLANETARY BODY",
        ["Earth", "Mars", "Moon"],
        key="planetary_environment_selector"
    )

    config = PLANETARY_DATASETS[planet]
    report = reports.get(planet)

    if not isinstance(report, dict):
        empty_state(
            f"{planet} report unavailable",
            "The expected planetary JSON report could not be loaded."
        )
        return

    dataset = report.get("dataset", {})
    statistics = (
        report.get("statistics")
        if planet == "Earth"
        else report.get("elevation_statistics")
    )

    if not isinstance(dataset, dict):
        dataset = {}

    if not isinstance(statistics, dict):
        statistics = {}

    raster = raster_metadata.get(planet, {})

    if not isinstance(raster, dict) or not raster:
        raster = inspect_raster(ROOT / config["raster"])

    if not isinstance(raster, dict):
        raster = {"status": "Metadata unavailable"}

    st.markdown(
        f'<div class="panel">'
        f'<div class="eyebrow">PLANETARY DATASET / {html.escape(planet.upper())}</div>'
        f'<div style="font-size:25px;font-weight:800;color:#EDF4F8;margin:8px 0;">'
        f'{html.escape(str(config["source"]))}</div>'
        f'<div style="color:#8FA3B5;">{html.escape(str(config["description"]))}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    def display_number(value, suffix="", digits=2):
        try:
            number = float(value)
            if not np.isfinite(number):
                return "—"
            return f"{number:,.{digits}f}{suffix}"
        except (TypeError, ValueError):
            return "—"

    # --------------------------------------------------------
    # Elevation statistics
    # --------------------------------------------------------

    st.markdown("### Elevation intelligence")

    if planet == "Earth":
        metric_definitions = [
            ("Global minimum", "global_elevation_min_m", " m"),
            ("Global maximum", "global_elevation_max_m", " m"),
            ("Global mean", "global_elevation_mean_m", " m"),
            ("Land mean", "land_elevation_mean_m", " m"),
        ]
    else:
        metric_definitions = [
            ("Minimum elevation", "minimum_m", " m"),
            ("Maximum elevation", "maximum_m", " m"),
            ("Mean elevation", "mean_m", " m"),
            ("Median elevation", "median_m", " m"),
        ]

    metric_columns = st.columns(4)

    for column, (label, key, suffix) in zip(
        metric_columns, metric_definitions
    ):
        column.metric(
            label,
            display_number(statistics.get(key), suffix)
        )

    if planet == "Earth":
        st.markdown("### Surface classification")

        land_col, ocean_col, valid_col = st.columns(3)

        land_col.metric(
            "Land cells",
            display_number(statistics.get("land_cells"), digits=0)
        )
        ocean_col.metric(
            "Ocean cells",
            display_number(statistics.get("ocean_cells"), digits=0)
        )
        valid_col.metric(
            "Land share",
            display_number(
                statistics.get("land_percentage_of_valid_cells"), "%"
            )
        )

        st.caption(
            "Earth's land statistics use the report's elevation-positive "
            "land proxy. This is a preliminary classification, not a "
            "complete geological land-cover classification."
        )

    else:
        st.markdown("### Statistical distribution")

        percentiles = statistics.get("percentiles_m", {})

        if isinstance(percentiles, dict) and percentiles:
            percentile_rows = [
                {
                    "Percentile": str(key).upper(),
                    "Elevation (m)": display_number(value)
                }
                for key, value in percentiles.items()
            ]

            st.dataframe(
                pd.DataFrame(percentile_rows),
                use_container_width=True,
                hide_index=True
            )

        deviation = statistics.get("standard_deviation_m")

        if deviation is not None:
            st.caption(
                "Standard deviation: "
                + display_number(deviation, " m")
            )

    # --------------------------------------------------------
    # Dataset and raster metadata
    # --------------------------------------------------------

    st.markdown("### Dataset configuration")

    metadata_columns = st.columns(3)

    metadata_columns[0].markdown("**DATA SOURCE**")
    metadata_columns[0].write(
        str(dataset.get("provider") or dataset.get("source") or
            dataset.get("product") or config["source"])
    )

    metadata_columns[1].markdown("**VERTICAL REFERENCE**")
    metadata_columns[1].write(str(config["reference"]))

    metadata_columns[2].markdown("**RASTER STATUS**")
    metadata_columns[2].write(str(raster.get("status", "Unknown")))

    dimensions = raster.get("width")
    height = raster.get("height")

    if dimensions is not None and height is not None:
        grid_text = f"{int(dimensions):,} × {int(height):,}"
    else:
        grid_text = "Not available"

    resolution = dataset.get(
        "source_resolution",
        dataset.get(
            "pixel_resolution_degrees",
            raster.get("resolution", "Not declared")
        )
    )

    raster_columns = st.columns(4)

    raster_columns[0].metric("Grid dimensions", grid_text)
    raster_columns[1].metric(
        "Data type",
        str(raster.get("dtype", dataset.get("data_type", "—")))
    )
    raster_columns[2].metric(
        "Raster driver",
        str(raster.get("driver", "—"))
    )
    raster_columns[3].metric(
        "Bands",
        str(raster.get("bands", "—"))
    )

    st.caption(f"Source resolution: {resolution}")

    if planet == "Moon":
        st.info(
            "The lunar source is a PDS label-backed raster. "
            "The label defines the image dimensions, data type, scaling "
            "and reference radius. The displayed raster file size, if "
            "inspected through the label, is not the binary image size."
        )

        st.caption(
            "Physical elevation is calculated from the stored digital "
            "number using the documented 0.5 m scale. The 1,737,400 m "
            "reference radius defines the reference sphere; it is not "
            "an elevation offset to add to the displayed height."
        )

    elif planet == "Mars":
        st.info(
            "The MOLA elevation product contains median observed "
            "topography where measurements exist, alongside interpolated "
            "values in areas without observations."
        )

        st.caption(
            "The generated ENVI header declares zero as an ignore value, "
            "while the original PDS label does not declare zero as "
            "NoData. TERRAFORM's global analysis retains zero-valued "
            "cells pending metadata reconciliation."
        )

    # --------------------------------------------------------
    # Methodology and limitations
    # --------------------------------------------------------

    st.markdown("### Scientific documentation")

    limitations = report.get(
        "limitations",
        report.get("scientific_limitations", [])
    )

    if not isinstance(limitations, list):
        limitations = []

    with st.expander("Methodology and interpretation"):
        methodology = report.get("methodology", {})

        if isinstance(methodology, dict):
            methodology_rows = [
                (str(key).replace('_', ' ').title(), value)
                for key, value in methodology.items()
                if isinstance(value, (str, int, float, bool))
            ]
            if methodology_rows:
                kv_table(methodology_rows)

        processing = report.get("processing", {})

        if isinstance(processing, dict):
            st.markdown("**Processing configuration**")
            processing_rows = [
                (str(key).replace('_', ' ').title(), value)
                for key, value in processing.items()
                if isinstance(value, (str, int, float, bool))
            ]
            if processing_rows:
                kv_table(processing_rows)

    with st.expander("Scientific limitations"):
        if limitations:
            for limitation in limitations:
                st.markdown(f"- {limitation}")
        else:
            st.caption("No limitations list is available in this report.")

        if planet == "Moon":
            validation = report.get("raw_data_validation", {})

            if isinstance(validation, dict):
                if validation.get("range_matches_label") is False:
                    st.warning(
                        "Observed raw extrema differ from the PDS label "
                        "extrema. The observed values remain within the "
                        "declared label range; the discrepancy is unresolved."
                    )

    # --------------------------------------------------------
    # Original report download
    # --------------------------------------------------------

    report_path = ROOT / "outputs" / config["report"]

    if report_path.exists():
        st.download_button(
            "Download original planetary report",
            data=report_path.read_bytes(),
            file_name=f"{planet.lower()}_global_report.json",
            mime="application/json",
            key=f"download_planetary_report_{planet.lower()}"
        )

    st.caption(
        "TERRAFORM presents the existing source reports without "
        "recalculating or modifying planetary elevation data."
    )



# ============================================================
# PAGE 1 — MISSION OVERVIEW
# ============================================================


def overview_page():
    # ========================================================
    # TERRAFORM — MISSION OVERVIEW
    # ========================================================

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">PLANETARY ANALOG INTELLIGENCE</div>
        <div class="hero-title">Compare Worlds.<br>Discover Analogues.</div>
        <div class="hero-description">
            A geospatial research platform investigating planetary terrain,
            geological characteristics and measurable environmental similarities
            across Earth, Mars and the Moon.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # MISSION SCOPE
    # --------------------------------------------------------

    st.markdown(
        '<div class="eyebrow">01 / PLANETARY RESEARCH SCOPE</div>',
        unsafe_allow_html=True
    )

    planet_columns = st.columns(3)

    planetary_cards = [
        (
            "EARTH",
            "Terrestrial environments",
            "Global topography and regional analogue investigation.",
            "NOAA ETOPO 2022"
        ),
        (
            "MARS",
            "Planetary reference",
            "Global MOLA topography and the Jezero Crater study region.",
            "NASA MOLA MEGDR"
        ),
        (
            "MOON",
            "Lunar terrain",
            "Global lunar elevation analysis using LOLA observations.",
            "NASA LRO LOLA"
        ),
    ]

    for column, (planet, heading, description, source) in zip(
        planet_columns, planetary_cards
    ):
        with column:
            st.markdown(
                f"""
                <div class="panel" style="min-height:190px;">
                    <div class="mono">{planet} / DATA DOMAIN</div>
                    <div style="font-size:19px;color:#55D6A0;
                                font-weight:750;margin-top:12px;">
                        {heading}
                    </div>
                    <div style="font-size:12px;color:#8FA3B5;
                                line-height:1.7;margin-top:9px;">
                        {description}
                    </div>
                    <div class="mono" style="margin-top:13px;">
                        SOURCE / {source}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.caption(
        "Global planetary comparison and regional analogue investigation "
        "are separate research scopes with different datasets, resolutions "
        "and processing methods."
    )

    st.divider()

    # --------------------------------------------------------
    # RESEARCH REPOSITORY
    # --------------------------------------------------------

    st.markdown(
        '<div class="eyebrow">02 / RESEARCH REPOSITORY</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Extracted regions",
        REPOSITORY_STATS["inventory"]
    )
    c2.metric(
        "Profiled candidates",
        REPOSITORY_STATS["profiles"]
    )
    c3.metric(
        "Geological records",
        REPOSITORY_STATS["geology"]
    )
    c4.metric(
        "Sensitivity scenarios",
        REPOSITORY_STATS["sensitivity"]
    )

    st.caption(
        "Repository statistics are loaded from the existing TERRAFORM "
        "research inventory. These counts describe available research "
        "outputs, not confirmed planetary analogue discoveries."
    )

    st.divider()

    # --------------------------------------------------------
    # REGIONAL ANALOGUE INVESTIGATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="eyebrow">03 / REGIONAL ANALOGUE INVESTIGATION</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.55, 1])

    with left:
        st.markdown("### Atacama candidate geography")

        st.caption(
            "Geographical distribution of extracted candidate-region "
            "centroids within the Atacama Desert, Chile."
        )

        if (
            CANDIDATE_INVENTORY is not None
            and not CANDIDATE_INVENTORY.empty
        ):
            candidate_map = create_research_candidate_map(
                CANDIDATE_INVENTORY
            )

            if candidate_map is not None:
                st_folium(
                    candidate_map,
                    width=None,
                    height=440,
                    returned_objects=[],
                    key="overview_research_map"
                )
            else:
                empty_state(
                    "Candidate coordinates unavailable",
                    "The inventory does not contain valid centroid coordinates."
                )
        else:
            empty_state(
                "Research inventory unavailable",
                "The Atacama candidate inventory could not be loaded."
            )

    with right:
        st.markdown("### Regional study configuration")

        repository_status = (
            "Research outputs connected"
            if REPOSITORY_STATS["inventory"] > 0
            else "Research inventory unavailable"
        )

        status_color = (
            "#55D6A0"
            if REPOSITORY_STATS["inventory"] > 0
            else "#E6B566"
        )

        st.markdown(
            f"""
            <div class="panel">
                <div class="mono">EARTH ANALOGUE REGIONS</div>
                <div style="font-size:19px;color:#55D6A0;
                            font-weight:700;margin-top:8px;">
                    Atacama Desert · Death Valley
                </div>
                <div class="mono">Earth / terrestrial terrain studies</div>
            </div>

            <div class="panel">
                <div class="mono">MARS REFERENCE REGIONS</div>
                <div style="font-size:19px;color:#55D6A0;
                            font-weight:700;margin-top:8px;">
                    MOLA Region · Jezero Crater
                </div>
                <div class="mono">Mars / MOLA terrain reference</div>
            </div>

            <div class="panel">
                <div class="mono">LUNAR REFERENCE REGION</div>
                <div style="font-size:19px;color:#55D6A0;
                            font-weight:700;margin-top:8px;">
                    LOLA Region
                </div>
                <div class="mono">Moon / LOLA terrain reference</div>
            </div>

            <div class="panel">
                <div class="mono">RESEARCH REPOSITORY</div>
                <div style="font-size:16px;color:{status_color};
                            font-weight:700;margin-top:8px;">
                    {html.escape(repository_status)}
                </div>
                <div class="mono">
                    {REPOSITORY_STATS['assets']} research assets detected
                </div>
            </div>

            <div class="panel">
                <div class="mono">REGIONAL WORKING RESOLUTION</div>
                <div style="font-size:19px;color:#55D6A0;
                            font-weight:700;margin-top:8px;">
                    ~463 metres
                </div>
                <div class="mono">
                    Atacama–Jezero regional comparison only
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # --------------------------------------------------------
    # RESEARCH DIMENSIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="eyebrow">04 / ANALYTICAL FRAMEWORK</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Research dimensions")

    dimensions = [
        (
            "01",
            "Terrain morphology",
            "Surface shape, elevation patterns and landforms."
        ),
        (
            "02",
            "Slope analysis",
            "Surface inclination and terrain gradients."
        ),
        (
            "03",
            "Surface roughness",
            "Local terrain variability and texture."
        ),
        (
            "04",
            "Local relief",
            "Elevation variation across surrounding terrain."
        ),
        (
            "05",
            "Geological context",
            "Mapped geological units and regional characteristics."
        ),
        (
            "06",
            "Data integrity",
            "Source validation, processing assumptions and limitations."
        ),
    ]

    st.markdown(
        '<div class="tf-grid tf-grid-3">'
        + "".join(
            f'<div class="panel"><div class="mono">{number} / RESEARCH VARIABLE</div>'
            f'<div class="tf-card-title">{html.escape(name)}</div>'
            f'<div class="tf-card-text">{html.escape(description)}</div></div>'
            for number, name, description in dimensions
        )
        + '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # --------------------------------------------------------
    # SENSITIVITY ANALYSIS
    # --------------------------------------------------------

    st.markdown(
        '<div class="eyebrow">05 / CURRENT RESEARCH OUTPUTS</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Terrain screening sensitivity")

    st.caption(
        "Existing sensitivity scenarios from the regional research "
        "repository. Matching percentages and areas are screening outputs, "
        "not probabilities or validated suitability scores."
    )

    if (
        SENSITIVITY_SUMMARY is not None
        and not SENSITIVITY_SUMMARY.empty
    ):
        scenario_cols = st.columns(3)

        for index, (_, row) in enumerate(
            SENSITIVITY_SUMMARY.iterrows()
        ):
            scenario = str(
                row.get("scenario", "Scenario")
            ).title()

            percentage = safe_metric_value(
                row, "matching_percentage", "%"
            )

            area = safe_metric_value(
                row, "matching_area_km2", " km²"
            )

            with scenario_cols[index % 3]:
                st.metric(
                    f"{scenario} matching area",
                    percentage
                )
                st.caption(
                    f"Matching area: {area}"
                )

    else:
        st.info(
            "Sensitivity summary is not available in the research repository."
        )

    st.divider()

    # --------------------------------------------------------
    # SCIENTIFIC INTERPRETATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="eyebrow">06 / SCIENTIFIC INTERPRETATION</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Evidence before interpretation")

    st.info(
        "TERRAFORM distinguishes measured observations, processed "
        "datasets and research hypotheses. Terrain-range overlap is an "
        "empirical screening result, not a probability of finding a "
        "suitable Martian analogue. No universal similarity score or "
        "scientific suitability ranking is assigned."
    )

    with st.expander("Understanding the research boundaries"):
        st.markdown("""
        - Global Earth, Mars and Moon elevation distributions are
          comparative planetary analyses.
        - The Atacama–Jezero investigation is a separate regional study.
        - Different planetary bodies use different elevation reference
          systems, source products and processing resolutions.
        - Statistical similarity does not establish geological,
          environmental or operational equivalence.
        - Candidate regions require further scientific investigation
          before any suitability conclusions can be made.
        """)

    st.caption(
        "TERRAFORM / PLANETARY ANALOG INTELLIGENCE · "
        "INDEPENDENT RESEARCH PROJECT · NASA SPACE APPS CHALLENGE 2026"
    )


# ============================================================
# PAGE 2 — DISCOVERY EXPLORER
# ============================================================

def discovery_page():
    section_header(
        "EARTH EXPLORATION",
        "Discovery Explorer",
        "Explore extracted Atacama candidate regions, inspect their geography and examine their measured terrain characteristics."
    )
    if CANDIDATE_INVENTORY is None or CANDIDATE_INVENTORY.empty:
        empty_state("Candidate inventory unavailable", "The Atacama candidate inventory could not be loaded.")
        return

    df = CANDIDATE_INVENTORY.copy()
    if "candidate_id" not in df.columns:
        st.error("The candidate inventory does not contain a candidate_id column.")
        return

    st.markdown("### Explore extracted regions")
    f1, f2, f3 = st.columns([1.2, 1, 1])
    with f1:
        search = st.text_input("Search candidate ID", placeholder="Example: TF-001")
    with f2:
        area_range = None
        if "area_km2" in df.columns:
            areas = pd.to_numeric(df["area_km2"], errors="coerce").dropna()
            if not areas.empty:
                lo, hi = float(areas.min()), float(areas.max())
                area_range = st.slider(
                    "Area range (km²)", min_value=lo, max_value=hi,
                    value=(lo, hi), key="discovery_area_range"
                )
    with f3:
        sort_order = st.selectbox(
            "Sort candidates",
            ["Largest area first", "Smallest area first", "Candidate ID"]
        )

    filtered_df = df.copy()
    if search:
        filtered_df = filtered_df[
            filtered_df["candidate_id"].astype(str).str.contains(search, case=False, na=False)
        ]
    if area_range is not None and "area_km2" in filtered_df.columns:
        filtered_df = filtered_df[
            pd.to_numeric(filtered_df["area_km2"], errors="coerce").between(*area_range)
        ]
    if sort_order == "Largest area first" and "area_km2" in filtered_df.columns:
        filtered_df = filtered_df.sort_values("area_km2", ascending=False)
    elif sort_order == "Smallest area first" and "area_km2" in filtered_df.columns:
        filtered_df = filtered_df.sort_values("area_km2", ascending=True)
    else:
        filtered_df = filtered_df.sort_values("candidate_id")

    a, b, c = st.columns(3)
    a.metric("Extracted inventory", len(df))
    b.metric("Filtered candidates", len(filtered_df))
    c.metric("Profiled candidates", REPOSITORY_STATS["profiles"])

    st.caption(
        "Geographic scope: this extracted inventory covers the Atacama "
        "Desert study area in Chile only. It is not a global Earth inventory, "
        "a suitability ranking, or confirmation of planetary analogues."
    )
    st.divider()

    map_col, profile_col = st.columns([1.45, 1])
    with map_col:
        st.markdown("### Candidate geography")
        if not filtered_df.empty:
            candidate_map = create_research_candidate_map(filtered_df)
            if candidate_map is not None:
                st_folium(
                    candidate_map, width=None, height=560,
                    returned_objects=[], key="discovery_research_map"
                )
            else:
                empty_state("Candidate coordinates unavailable", "No valid centroid coordinates were found.")
        else:
            empty_state("No matching candidates", "Change your search or area filters.")

    with profile_col:
        st.markdown("### Candidate profile")
        if not filtered_df.empty:
            if isinstance(CANDIDATE_PROFILES, dict):
                available_profile_ids = {
                    str(profile.get("candidate_id"))
                    for profile in CANDIDATE_PROFILES.get("profiles", [])
                    if isinstance(profile, dict)
                    and profile.get("candidate_id") is not None
                }
            else:
                available_profile_ids = set()

            profile_df = filtered_df[
                filtered_df["candidate_id"].astype(str).isin(available_profile_ids)
            ].copy()

            if not profile_df.empty:
                selected_id = st.selectbox(
                    "Select detailed profile",
                    profile_df["candidate_id"].astype(str).tolist(),
                    key="research_discovery_candidate"
                )
                selected = profile_df[
                    profile_df["candidate_id"].astype(str) == selected_id
                ].iloc[0]

                st.markdown(
                    f'<div class="panel"><div class="eyebrow">SELECTED REGION</div>'
                    f'<div style="font-size:25px;font-weight:800;color:white;">{html.escape(selected_id)}</div>'
                    f'<div class="mono">ATACAMA DESERT / CHILE</div></div>',
                    unsafe_allow_html=True
                )

                profile_metrics = [
                    (label, column, suffix)
                    for label, column, suffix in [
                        ("Area", "area_km2", " km²"),
                        ("Mean slope", "slope_mean_deg", "°"),
                        ("Mean roughness", "roughness_mean_m", " m"),
                        ("Mean local relief", "local_relief_mean_m", " m"),
                    ]
                    if column in selected.index
                ]
                for pair_start in range(0, len(profile_metrics), 2):
                    metric_pair = st.columns(2)
                    for metric_col, (label, column, suffix) in zip(
                        metric_pair, profile_metrics[pair_start:pair_start + 2]
                    ):
                        metric_col.metric(label, safe_metric_value(selected, column, suffix))

                lat = selected.get("centroid_latitude", np.nan)
                lon = selected.get("centroid_longitude", np.nan)

                if pd.notna(lat) and pd.notna(lon):
                    st.caption(
                        f"Latitude: {float(lat):.5f} | Longitude: {float(lon):.5f}"
                    )

                detailed_profile = get_profile_data(selected_id)
                if detailed_profile:
                    with st.expander("Detailed research profile"):
                        st.json(detailed_profile)

                with st.expander("Complete candidate record"):
                    st.dataframe(
                        selected.to_frame("Value"),
                        use_container_width=True
                    )
            else:
                st.info(
                    "No detailed research profiles match the current search "
                    "and area filters. The full extracted candidate inventory "
                    "remains available below."
                )
        else:
            empty_state("No candidate selected", "There are no matching records.")

    st.divider()
    st.markdown("### Extracted candidate records")
    display_columns = [
        col for col in [
            "candidate_id", "rank_by_area", "pixel_count", "area_km2",
            "centroid_latitude", "centroid_longitude", "slope_mean_deg",
            "roughness_mean_m", "local_relief_mean_m",
            "broad_screening_overlap_percent", "moderate_screening_overlap_percent"
        ] if col in filtered_df.columns
    ]
    st.dataframe(filtered_df[display_columns], use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered candidate inventory",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="terraform_filtered_candidate_inventory.csv",
        mime="text/csv"
    )
    st.info(
        "The inventory contains connected regions extracted from the strict three-metric "
        "terrain intersection. Candidate identifiers and area ranks describe extracted "
        "components, not scientific suitability or confirmed planetary analogy."
    )


# ============================================================
# PAGE 3 — GLOBAL MARS EXPLORER
# ============================================================



def global_mars_page():
    # ========================================================
    # GLOBAL MARS EXPLORER
    # ========================================================

    section_header(
        "PLANETARY-SCALE ANALYSIS",
        "Global Mars Explorer",
        "Investigate the global topography, surface characteristics and observation coverage of Mars using NASA MOLA MEGDR data."
    )

    if not isinstance(MARS_TERRAIN_REPORT, dict):
        st.error(
            "The global Mars terrain metrics JSON report could not be loaded."
        )
        st.caption(f"Expected: `{MARS_JSON_PATH}`")
        return

    report = MARS_TERRAIN_REPORT
    source = report.get("source", {})
    grid = report.get("grid", {})
    geometry = report.get("geometry", {})
    stats = report.get("statistics", {})
    processing = report.get("processing", {})

    def display_number(value, suffix="", decimals=2):
        if isinstance(value, (int, float)):
            return f"{value:,.{decimals}f}{suffix}"
        return f"—{suffix}"

    def display_integer(value):
        if isinstance(value, (int, float)):
            return f"{value:,.0f}"
        return "—"

    # --------------------------------------------------------
    # 01 / DATASET OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="eyebrow">01 / PLANETARY DATASET</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Global Mars terrain model")

    rows = display_integer(grid.get("rows"))
    columns = display_integer(grid.get("columns"))

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Grid dimensions",
        f"{rows} × {columns}"
    )
    c2.metric(
        "Angular spacing",
        display_number(
            grid.get("angular_spacing_degrees"),
            "°",
            4
        )
    )
    c3.metric(
        "Reference radius",
        display_number(
            geometry.get("reference_radius_km"),
            " km",
            1
        )
    )
    c4.metric(
        "Analysed reference area",
        display_number(
            geometry.get("analysed_reference_area_km2"),
            " km²",
            0
        )
    )

    st.markdown(
        f"""
        <div class="panel">
            <div class="mono">DATA PRODUCT IDENTIFICATION</div>
            <div style="font-size:17px;color:#55D6A0;
                        font-weight:700;margin-top:9px;">
                {html.escape(str(source.get('dataset_id', 'MOLA MEGDR')))}
            </div>
            <div style="font-size:12px;color:#8FA3B5;
                        line-height:1.8;margin-top:8px;">
                Elevation source: {html.escape(str(source.get('elevation', 'Not specified')))}
                <br>
                Observation source: {html.escape(str(source.get('observation_count', 'Not specified')))}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "The global terrain model and observation-support information "
        "are distinct data products. Their measurements and coverage "
        "denominators must not be treated as interchangeable."
    )

    st.divider()

    # --------------------------------------------------------
    # 02 / GLOBAL SURFACE VISUALISATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="eyebrow">02 / GLOBAL SURFACE VISUALISATION</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Elevation and observation context")

    st.caption(
        "Explore the global elevation model alongside the spatial "
        "distribution of recorded MOLA observations."
    )

    image_tabs = st.tabs([
        "Elevation",
        "Elevation distribution",
        "Observation classification",
        "Observation density"
    ])

    image_items = list(MARS_IMAGES.items())

    for tab, (label, relative_path) in zip(
        image_tabs, image_items
    ):
        with tab:
            show_image(relative_path, label)

    st.info(
        "MOLA observation count represents recorded measurements, "
        "not a direct measure of elevation accuracy. Areas without "
        "recorded observations may still contain interpolated elevations."
    )

    st.divider()

    # --------------------------------------------------------
    # 03 / TERRAIN METRIC MAPS
    # --------------------------------------------------------

    st.markdown(
        '<div class="eyebrow">03 / DERIVED TERRAIN ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Global terrain characteristics")

    st.caption(
        "Derived terrain metrics describe different aspects of "
        "surface morphology and should be interpreted independently."
    )

    metric_image_tabs = st.tabs([
        "Slope",
        "Roughness",
        "Local relief",
        "Terrain overview"
    ])

    for tab, (label, relative_path) in zip(
        metric_image_tabs, MARS_METRIC_IMAGES.items()
    ):
        with tab:
            show_image(relative_path, label)

            if label == "Terrain overview":
                st.caption(
                    "The combined terrain-quality visualisation is "
                    "a derived overview, not an independently validated "
                    "accuracy or suitability map."
                )

    st.divider()

    # --------------------------------------------------------
    # 04 / AREA-WEIGHTED TERRAIN STATISTICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="eyebrow">04 / QUANTITATIVE TERRAIN ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Area-weighted terrain statistics")

    st.caption(
        "These statistics use the polar-excluded analysis domain "
        "(absolute planetocentric latitude below 89.5°). "
        "This is not the same denominator as the full-sphere "
        "observation-coverage analysis."
    )

    metric_definitions = [
        ("slope_degrees", "Slope", "°"),
        ("roughness_3x3_m", "3×3 roughness", " m"),
        ("local_relief_3x3_m", "3×3 local relief", " m"),
    ]

    metric_tabs = st.tabs([
        item[1] for item in metric_definitions
    ])

    for tab, (key, label, unit) in zip(
        metric_tabs, metric_definitions
    ):
        with tab:
            metric_data = stats.get(key, {})
            all_data = metric_data.get("all_analysed", {})
            observed_data = metric_data.get("observed_only", {})

            st.markdown(f"#### {label}")

            x1, x2, x3 = st.columns(3)

            x1.metric(
                "Area-weighted mean",
                display_number(
                    all_data.get("area_weighted_mean"),
                    unit
                )
            )

            x2.metric(
                "Minimum",
                display_number(
                    all_data.get("minimum"),
                    unit
                )
            )

            x3.metric(
                "Maximum",
                display_number(
                    all_data.get("maximum"),
                    unit
                )
            )

            st.markdown(
                "**All analysed cells — approximate area-weighted percentiles**"
            )

            percentiles = all_data.get(
                "area_weighted_percentiles_approx", {}
            )

            percentile_columns = st.columns(5)

            for column, percentile in zip(
                percentile_columns,
                ["P01", "P25", "P50", "P75", "P99"]
            ):
                column.metric(
                    percentile,
                    display_number(
                        percentiles.get(percentile),
                        unit
                    )
                )

            observed_mean = observed_data.get(
                "area_weighted_mean"
            )

            all_mean = all_data.get(
                "area_weighted_mean"
            )

            if observed_mean is not None:
                st.divider()

                st.markdown(
                    "**Observed-centre cell comparison**"
                )

                st.metric(
                    "Observed-only area-weighted mean",
                    display_number(observed_mean, unit),
                    help=(
                        "Centre cells with one or more recorded observations. "
                        "Neighbouring cells used in 3×3 metrics may still "
                        "include interpolated elevations."
                    )
                )

                comparison_frame = pd.DataFrame([
                    {
                        "Statistic": "Area-weighted mean",
                        "All analysed": all_mean,
                        "Observed-only": observed_mean,
                    },
                    {
                        "Statistic": "Cell count",
                        "All analysed": all_data.get("cell_count"),
                        "Observed-only": observed_data.get("cell_count"),
                    },
                    {
                        "Statistic": "Reference area (km²)",
                        "All analysed": all_data.get("reference_area_km2"),
                        "Observed-only": observed_data.get("reference_area_km2"),
                    },
                ])

                st.dataframe(
                    comparison_frame,
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "Percentiles are histogram-based approximations. "
                "The reported maximum is an extreme value, not "
                "a representative terrain condition."
            )

    st.divider()

    # --------------------------------------------------------
    # 05 / FULL-SPHERE OBSERVATION COVERAGE
    # --------------------------------------------------------

    st.markdown(
        '<div class="eyebrow">05 / OBSERVATION COVERAGE</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Global MOLA observation coverage")

    st.caption(
        "These values come from the separate full-sphere area-weighted "
        "elevation report. They do not use the polar-excluded "
        "terrain-metric analysis domain."
    )

    if MARS_AREA_REPORT:

        coverage_columns = st.columns(4)

        coverage_columns[0].metric(
            "Area-weighted observed coverage",
            "96.43865%"
        )

        coverage_columns[1].metric(
            "Area-weighted zero-observation",
            "3.56135%"
        )

        coverage_columns[2].metric(
            "Observed cells",
            "15,685,242"
        )

        coverage_columns[3].metric(
            "Zero-observation cells",
            "903,558"
        )

        with st.expander(
            "Inspect full-sphere coverage report"
        ):
            st.code(
                MARS_AREA_REPORT,
                language="text"
            )

            st.download_button(
                "Download area-weighted report",
                data=MARS_AREA_REPORT.encode("utf-8"),
                file_name="mars_area_weighted_report.txt",
                mime="text/plain"
            )

    else:
        st.warning(
            f"Area-weighted report not found: `{MARS_AREA_REPORT_PATH}`"
        )

    st.divider()

    # --------------------------------------------------------
    # 06 / DATA INTERPRETATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="eyebrow">06 / DATA QUALITY AND INTERPRETATION</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Zero-value metadata discrepancy")

    st.warning(
        "**Source metadata and processing interpretation require "
        "careful distinction.** The generated ENVI header identifies "
        "raw value `0` as its data-ignore value. However, the original "
        "NASA PDS MOLA elevation label does not declare zero as a "
        "special NoData value."
    )

    with st.expander(
        "Read the complete zero-value handling disclosure",
        expanded=True
    ):
        st.markdown(
            """
            TERRAFORM's Mars processing reads the original big-endian
            elevation data and retains zero-valued cells.

            The global planetary comparison report records **2,128
            zero-valued cells retained**, with no histogram cells excluded.

            **Interpretation:** this is a discrepancy between generated
            raster metadata and the processing pipeline's treatment of
            source values. The conflict is disclosed rather than silently
            treating every zero as missing data or claiming that every
            zero is independently confirmed physical elevation.
            """
        )

        st.caption(
            "Do not confuse these 2,128 zero-valued elevation cells "
            "with the 903,558 zero-observation cells in the separate "
            "full-sphere MOLA observation-coverage analysis. "
            "They represent different measurements."
        )

    st.markdown("### Processing and interpretation limitations")

    limitations = report.get("limitations", [])

    if limitations:
        for index, limitation in enumerate(
            limitations, start=1
        ):
            st.markdown(f"**{index:02d}.** {limitation}")
    else:
        st.info(
            "No limitations were provided in the loaded report."
        )

    # --------------------------------------------------------
    # 07 / TECHNICAL DOCUMENTATION
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="eyebrow">07 / TECHNICAL DOCUMENTATION</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Reproducibility and source records")

    with st.expander("Technical processing metadata"):
        st.json(processing)

    with st.expander("Complete Mars terrain metrics report"):
        st.json(report)

        st.download_button(
            "Download Mars terrain metrics JSON",
            data=json.dumps(
                report,
                indent=2,
                ensure_ascii=False
            ).encode("utf-8"),
            file_name="mars_global_terrain_report.json",
            mime="application/json"
        )

    st.caption(
        "TERRAFORM / MARS GLOBAL TERRAIN ANALYSIS · "
        "NASA MOLA MEGDR · SCIENTIFIC INTERPRETATION "
        "SUBJECT TO SOURCE DATA LIMITATIONS"
    )


# ============================================================
# PAGE 4 — EARTH / MARS COMPARISON
# ============================================================


def comparison_page():
    import plotly.graph_objects as go

    project_root = Path(__file__).resolve().parent
    reports_dir = project_root / "outputs" / "reports"
    global_dir = project_root / "outputs" / "Global"

    global_report_path = (
        global_dir / "Comparisons"
        / "global_planetary_comparison_report.json"
    )

    moon_report_path = (
        global_dir / "Moon"
        / "moon_global_terrain_analysis.json"
    )

    earth_moon_report_path = (
        global_dir / "Comparisons"
        / "earth_moon_common_scale_report.json"
    )

    # ========================================================
    # PAGE HEADER
    # ========================================================

    section_header(
        "REGIONAL ANALOGUE RESEARCH",
        "Regional Analogue Study",
        "Compare terrestrial environments against regional and "
        "global planetary terrain references."
    )

    # ========================================================
    # EARTH REGION SELECTOR
    # ========================================================

    st.divider()
    st.markdown("### Select terrestrial study region")

    region_options = {
        "Atacama Desert, Chile": {
            "report_path": reports_dir
            / "earth_mars_terrain_comparison.json",
            "earth_label": "Atacama, Earth",
            "region_key": "Atacama",
            "report": EARTH_MARS_REPORT,
            "filename": "terraform_atacama_comparison.json",
        },

        "Death Valley, USA": {
            "report_path": reports_dir
            / "death_valley_jezero_terrain_comparison.json",
            "earth_label": "Death Valley, Earth",
            "region_key": "Death Valley",
            "report": None,
            "filename": "terraform_death_valley_comparison.json",
        },
    }

    selected_region = st.radio(
        "Earth study region",
        list(region_options.keys()),
        horizontal=True,
        key="regional_analogue_region_selector",
    )

    config = region_options[selected_region]
    earth_label = config["earth_label"]

    # ========================================================
    # PLANETARY REFERENCE SELECTOR
    # ========================================================

    st.divider()
    st.markdown("### Select planetary reference")

    planetary_reference = st.radio(
        "Planetary reference dataset",
        [
            "Jezero Crater — Regional",
            "Global Mars — MOLA",
            "Global Moon — LOLA",
        ],
        horizontal=True,
        key="regional_planetary_reference_selector",
    )

    is_jezero = (
        planetary_reference == "Jezero Crater — Regional"
    )

    is_global_mars = (
        planetary_reference == "Global Mars — MOLA"
    )

    is_global_moon = (
        planetary_reference == "Global Moon — LOLA"
    )

    # ========================================================
    # LOAD SELECTED REPORTS
    # ========================================================

    report = None
    active_report = None

    comparison = {}
    terrain_metrics = {}
    histograms = {}
    elevation_context = {}

    global_report = None
    global_mars = None

    moon_report = None
    moon_comparison_report = None
    moon_region_data = None

    # --------------------------------------------------------
    # JEZERO REGIONAL REPORT
    # --------------------------------------------------------

    if is_jezero:

        report = config["report"]

        if report is None:

            if not config["report_path"].exists():

                st.error(
                    f"Regional comparison report not found: "
                    f"{config['report_path']}"
                )

                st.info(
                    "Run the corresponding regional comparison engine first."
                )

                return

            try:

                with open(
                    config["report_path"],
                    "r",
                    encoding="utf-8",
                ) as file:

                    report = json.load(file)

            except (json.JSONDecodeError, OSError) as error:

                st.error(
                    f"Could not load regional report: {error}"
                )

                return

        if not isinstance(report, dict):

            st.error("The selected regional report is invalid.")
            return

        active_report = report

        comparison = report.get("comparison", {})
        terrain_metrics = report.get("terrain_metrics", {})
        histograms = report.get("histograms", {})
        elevation_context = report.get("elevation_context", {})

    # --------------------------------------------------------
    # GLOBAL MARS REPORT
    # --------------------------------------------------------

    elif is_global_mars:

        if not global_report_path.exists():

            st.error(
                "Global planetary comparison report was not found."
            )

            return

        try:

            with open(
                global_report_path,
                "r",
                encoding="utf-8",
            ) as file:

                global_report = json.load(file)

            global_mars = (
                global_report
                .get("planetary_results", {})
                .get("Mars", {})
            )

            if not global_mars:

                st.error(
                    "The global report does not contain Mars statistics."
                )

                return

            active_report = global_report

        except (json.JSONDecodeError, OSError) as error:

            st.error(
                f"Could not load global Mars report: {error}"
            )

            return

    # --------------------------------------------------------
    # GLOBAL MOON REPORTS
    # --------------------------------------------------------

    elif is_global_moon:

        if not moon_report_path.exists():

            st.error(
                "Global lunar terrain report was not found."
            )

            st.info(
                "Run: python src/analysis/moon_terrain_analysis.py"
            )

            return

        if not earth_moon_report_path.exists():

            st.error(
                "Earth-Moon common-scale comparison report was not found."
            )

            st.info(
                "Run: python src/analysis/earth_moon_comparison.py"
            )

            return

        try:

            with open(
                moon_report_path,
                "r",
                encoding="utf-8",
            ) as file:

                moon_report = json.load(file)

            with open(
                earth_moon_report_path,
                "r",
                encoding="utf-8",
            ) as file:

                moon_comparison_report = json.load(file)

            moon_region_data = (
                moon_comparison_report
                .get("regions", {})
                .get(config["region_key"])
            )

            if not moon_region_data:

                st.error(
                    f"No Earth-Moon comparison was found for "
                    f"{selected_region}."
                )

                return

            active_report = moon_report

        except (json.JSONDecodeError, OSError) as error:

            st.error(
                f"Could not load lunar comparison data: {error}"
            )

            return

    st.divider()

    # ========================================================
    # JEZERO CRATER — REGIONAL COMPARISON
    # ========================================================

    if is_jezero:

        st.markdown("### Study overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Earth analogue",
            selected_region.split(",")[0],
        )

        c2.metric(
            "Planetary reference",
            "Jezero Crater",
        )

        c3.metric(
            "Working resolution",
            f"{comparison.get('target_resolution_metres', 463):,.0f} m",
        )

        c4.metric(
            "Analysis neighbourhood",
            str(comparison.get("neighbourhood", "3 x 3 pixels")),
        )

        st.caption(
            "Earth and Jezero terrain distributions are compared "
            "at a common working resolution. This is statistical "
            "comparison, not pixel-to-pixel spatial matching."
        )

        st.divider()

        # ----------------------------------------------------
        # TERRAIN METRICS
        # ----------------------------------------------------

        st.markdown("### Comparative terrain measurements")

        metric_descriptions = {
            "slope_degrees": ("Slope", "degrees"),
            "roughness_m": ("Terrain roughness", "metres"),
            "local_relief_m": ("Local relief", "metres"),
        }

        available_metrics = [
            key for key in metric_descriptions
            if key in terrain_metrics
        ]

        if available_metrics:

            tabs = st.tabs([
                metric_descriptions[key][0]
                for key in available_metrics
            ])

            for tab, key in zip(tabs, available_metrics):

                with tab:

                    metric_data = terrain_metrics[key]

                    earth_stats = metric_data.get("earth", {})
                    mars_stats = metric_data.get("mars", {})

                    earth_col, mars_col = st.columns(2)

                    with earth_col:

                        st.markdown(f"**{earth_label}**")

                        st.metric(
                            "Mean",
                            f"{earth_stats.get('mean', 0):,.3f}",
                        )

                        st.metric(
                            "Median",
                            f"{earth_stats.get('median', 0):,.3f}",
                        )

                        st.caption(
                            f"Valid pixels: "
                            f"{earth_stats.get('valid_pixels', 0):,}"
                        )

                    with mars_col:

                        st.markdown("**Jezero Crater, Mars**")

                        st.metric(
                            "Mean",
                            f"{mars_stats.get('mean', 0):,.3f}",
                        )

                        st.metric(
                            "Median",
                            f"{mars_stats.get('median', 0):,.3f}",
                        )

                        st.caption(
                            f"Valid pixels: "
                            f"{mars_stats.get('valid_pixels', 0):,}"
                        )

                    earth_percentiles = earth_stats.get(
                        "percentiles", {}
                    )

                    mars_percentiles = mars_stats.get(
                        "percentiles", {}
                    )

                    percentile_keys = [
                        "p10", "p25", "p50", "p75", "p90"
                    ]

                    if earth_percentiles and mars_percentiles:

                        percentile_frame = pd.DataFrame({
                            "Percentile": percentile_keys,

                            earth_label: [
                                earth_percentiles.get(k)
                                for k in percentile_keys
                            ],

                            "Jezero, Mars": [
                                mars_percentiles.get(k)
                                for k in percentile_keys
                            ],
                        })

                        st.markdown("**Percentile comparison**")

                        st.dataframe(
                            percentile_frame,
                            use_container_width=True,
                            hide_index=True,
                        )

                    with st.expander("Technical details — JSON"):
                        st.json(metric_data)

        else:

            st.warning(
                "The selected report does not contain expected terrain metrics."
            )

        # ----------------------------------------------------
        # REGIONAL HISTOGRAMS
        # ----------------------------------------------------

        st.divider()
        st.markdown("### Terrain distribution visualisations")

        chart_settings = {
            "slope_degrees": (
                "Terrain Slope Distribution",
                "Slope (degrees)",
            ),

            "roughness_m": (
                "Surface Roughness Distribution",
                "Local elevation standard deviation (m)",
            ),

            "local_relief_m": (
                "Local Terrain Relief Distribution",
                "Local elevation range (m)",
            ),
        }

        chart_keys = [
            key for key in chart_settings
            if key in histograms
        ]

        if chart_keys:

            chart_tabs = st.tabs([
                chart_settings[key][0]
                for key in chart_keys
            ])

            for tab, metric_key in zip(chart_tabs, chart_keys):

                with tab:

                    histogram = histograms.get(metric_key, {})

                    edges = histogram.get("bin_edges", [])
                    earth_counts = histogram.get("earth_counts", [])
                    mars_counts = histogram.get("mars_counts", [])

                    if (
                        len(edges) < 2
                        or len(earth_counts) != len(edges) - 1
                        or len(mars_counts) != len(edges) - 1
                    ):

                        st.info(
                            f"Histogram data for {metric_key} is incomplete."
                        )

                        continue

                    widths = [
                        edges[i + 1] - edges[i]
                        for i in range(len(edges) - 1)
                    ]

                    centres = [
                        (edges[i + 1] + edges[i]) / 2
                        for i in range(len(edges) - 1)
                    ]

                    earth_total = sum(earth_counts)
                    mars_total = sum(mars_counts)

                    if earth_total == 0 or mars_total == 0:

                        st.info(
                            "Not enough valid histogram observations."
                        )

                        continue

                    earth_density = [
                        count / (earth_total * width)
                        for count, width in zip(earth_counts, widths)
                    ]

                    mars_density = [
                        count / (mars_total * width)
                        for count, width in zip(mars_counts, widths)
                    ]

                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x=centres,
                        y=earth_density,
                        width=widths,
                        name=earth_label,
                        opacity=0.58,
                    ))

                    fig.add_trace(go.Bar(
                        x=centres,
                        y=mars_density,
                        width=widths,
                        name="Jezero, Mars",
                        opacity=0.58,
                    ))

                    metric_data = terrain_metrics.get(
                        metric_key, {}
                    )

                    earth_median = (
                        metric_data.get("earth", {}).get("median")
                    )

                    mars_median = (
                        metric_data.get("mars", {}).get("median")
                    )

                    if earth_median is not None:

                        fig.add_vline(
                            x=earth_median,
                            line_dash="dash",
                            annotation_text="Earth median",
                        )

                    if mars_median is not None:

                        fig.add_vline(
                            x=mars_median,
                            line_dash="dot",
                            annotation_text="Mars median",
                        )

                    title, xlabel = chart_settings[metric_key]

                    fig.update_layout(
                        title=title,
                        xaxis_title=xlabel,
                        yaxis_title="Probability density",
                        barmode="overlay",
                        legend_title="Study region",
                        height=480,
                        template="plotly_white",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

        else:

            st.info("No regional histogram data is available.")

    # ========================================================
    # GLOBAL MARS — MOLA
    # ========================================================

    elif is_global_mars:

        mars_stats = global_mars.get("statistics", {})

        mars_diagnostics = global_mars.get(
            "histogram_diagnostics", {}
        )

        mars_dimensions = global_mars.get(
            "grid_dimensions", {}
        )

        mars_percentiles = mars_stats.get(
            "weighted_percentiles_m", {}
        )

        # ----------------------------------------------------
        # OVERVIEW
        # ----------------------------------------------------

        st.markdown("### Global Mars study overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Earth region",
            selected_region.split(",")[0],
        )

        c2.metric(
            "Planetary reference",
            "Global Mars",
        )

        c3.metric(
            "MOLA spacing",
            f"{global_mars.get('approximate_equatorial_spacing_km', 3.704):.3f} km",
        )

        c4.metric(
            "Accepted grid cells",
            f"{mars_diagnostics.get('accepted_cells', 0):,}",
        )

        st.caption(
            "Reference dataset: NASA Mars Global Surveyor MOLA MEGDR."
        )

        st.divider()

        # ----------------------------------------------------
        # GLOBAL ELEVATION
        # ----------------------------------------------------

        st.markdown("### Global Mars elevation statistics")

        stat1, stat2, stat3, stat4 = st.columns(4)

        stat1.metric(
            "Weighted mean",
            f"{mars_stats.get('weighted_mean_m', 0):,.1f} m",
        )

        stat2.metric(
            "Weighted median",
            f"{mars_percentiles.get('P50', 0):,.1f} m",
        )

        stat3.metric(
            "Weighted standard deviation",
            f"{mars_stats.get('weighted_standard_deviation_m', 0):,.1f} m",
        )

        stat4.metric(
            "Accepted cells",
            f"{mars_diagnostics.get('accepted_cells', 0):,}",
        )

        st.caption(
            "Global statistics use latitude-weighted elevation "
            "distributions. Percentiles are interpolated within "
            "10-metre histogram bins."
        )

        # ----------------------------------------------------
        # EARTH ELEVATION CONTEXT
        # ----------------------------------------------------

        st.markdown("### Earth regional elevation context")

        earth_elevation = (
            report.get("elevation_context", {}).get("earth", {})
            if report else {}
        )

        if earth_elevation:

            earth_col, mars_col = st.columns(2)

            with earth_col:

                st.markdown(f"**{earth_label}**")

                st.metric(
                    "Mean elevation",
                    f"{earth_elevation.get('mean', 0):,.1f} m",
                )

                st.metric(
                    "Median elevation",
                    f"{earth_elevation.get('median', 0):,.1f} m",
                )

            with mars_col:

                st.markdown("**Global Mars — MOLA**")

                st.metric(
                    "Weighted mean elevation",
                    f"{mars_stats.get('weighted_mean_m', 0):,.1f} m",
                )

                st.metric(
                    "Weighted median elevation",
                    f"{mars_percentiles.get('P50', 0):,.1f} m",
                )

            st.info(
                "These values provide statistical context only. "
                "Earth and Mars use different vertical reference systems "
                "and geographic scopes. Absolute elevation is not a "
                "direct terrain-similarity measure."
            )

        else:

            st.info(
                "Regional elevation context is unavailable."
            )

        # ----------------------------------------------------
        # GLOBAL MARS VISUALISATIONS
        # ----------------------------------------------------

        st.divider()
        st.markdown("### Global planetary distribution visualisations")

        visualisations = global_report.get("visualisations", {})

        distribution_image = visualisations.get(
            "elevation_distributions"
        )

        normalized_image = visualisations.get(
            "normalized_distributions"
        )

        for label, relative_path in [
            ("Global elevation distributions", distribution_image),
            ("Normalised planetary distributions", normalized_image),
        ]:

            if not relative_path:
                continue

            image_path = project_root / relative_path

            if image_path.exists():

                st.markdown(f"**{label}**")

                st.image(
                    str(image_path),
                    use_container_width=True,
                )

            else:

                st.warning(f"{label} image was not found.")

        st.caption(
            "These charts show global elevation distribution patterns, "
            "not spatially matched regional terrain."
        )

        # ----------------------------------------------------
        # TECHNICAL DETAILS
        # ----------------------------------------------------

        with st.expander("Global Mars technical details"):

            kv_table([
                ("Dataset", global_mars.get("dataset", "MOLA MEGDR")),
                ("Input file", global_mars.get("input_file", "Not specified")),
                ("Grid rows", mars_dimensions.get("rows", "Not specified")),
                ("Grid columns", mars_dimensions.get("columns", "Not specified")),
                ("Vertical reference", global_mars.get("vertical_reference", "Not specified")),
                ("Histogram coverage", f"{mars_diagnostics.get('cell_coverage_percent', 0):.2f}%"),
                ("Weighted coverage", f"{mars_diagnostics.get('weighted_coverage_percent', 0):.2f}%"),
            ])

            st.markdown("**Complete record**")
            st.json(global_mars, expanded=False)

        with st.expander("Global comparison methodology"):
            st.json(global_report.get("method", {}))

        with st.expander("Global scientific interpretation"):

            for item in global_report.get(
                "scientific_interpretation", []
            ):

                st.markdown(f"- {item}")

        with st.expander("Global Mars limitations"):

            for item in global_mars.get("limitations", []):

                st.markdown(f"- {item}")

        with st.expander("Complete global planetary report"):

            st.download_button(
                "Download global planetary report",

                data=json.dumps(
                    global_report,
                    indent=4,
                    ensure_ascii=False,
                ).encode("utf-8"),

                file_name="terraform_global_planetary_comparison.json",

                mime="application/json",

                key="download_global_planetary_report",
            )

    # ========================================================
    # GLOBAL MOON — LOLA
    # ========================================================

    elif is_global_moon:

        dataset = moon_report.get("dataset", {})
        elevation_stats = moon_report.get(
            "elevation_statistics", {}
        )

        terrain_stats = moon_report.get(
            "terrain_statistics", {}
        )

        lunar_reference = moon_region_data.get(
            "lunar_reference", {}
        )

        earth_statistics = moon_region_data.get(
            "earth_statistics", {}
        )

        lunar_statistics = lunar_reference.get(
            "metrics", {}
        )

        centre = moon_region_data.get("centre", {})
        scale = moon_region_data.get(
            "comparison_scale", {}
        )

        # ----------------------------------------------------
        # OVERVIEW
        # ----------------------------------------------------

        st.markdown("### Global lunar study overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Earth study region",
            selected_region.split(",")[0],
        )

        c2.metric(
            "Planetary reference",
            "Global Moon",
        )

        c3.metric(
            "LOLA equatorial spacing",
            f"{dataset.get('equatorial_pixel_scale_m', 1895.21):,.0f} m",
        )

        c4.metric(
            "Global elevation cells",
            f"{dataset.get('dimensions', {}).get('total_cells', 0):,}",
        )

        st.caption(
            "Reference dataset: NASA Lunar Reconnaissance Orbiter "
            "LOLA LDEM_16, Version 3.1."
        )

        st.divider()

        # ----------------------------------------------------
        # GLOBAL LUNAR ELEVATION
        # ----------------------------------------------------

        st.markdown("### Global lunar elevation statistics")

        e1, e2, e3, e4 = st.columns(4)

        e1.metric(
            "Mean elevation",
            f"{elevation_stats.get('mean', 0):,.1f} m",
        )

        e2.metric(
            "Median elevation",
            f"{elevation_stats.get('median', 0):,.1f} m",
        )

        e3.metric(
            "Standard deviation",
            f"{elevation_stats.get('standard_deviation', 0):,.1f} m",
        )

        e4.metric(
            "Valid cells",
            f"{elevation_stats.get('valid_pixels', 0):,}",
        )

        elevation_percentiles = elevation_stats.get(
            "percentiles", {}
        )

        if elevation_percentiles:

            elevation_frame = pd.DataFrame({
                "Percentile": list(elevation_percentiles.keys()),
                "Elevation (m)": list(elevation_percentiles.values()),
            })

            st.markdown("**Global elevation percentiles**")

            st.dataframe(
                elevation_frame,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # GLOBAL LUNAR TERRAIN METRICS
        # ----------------------------------------------------

        st.divider()
        st.markdown("### Global lunar terrain characteristics")

        moon_metric_labels = {
            "slope_degrees": ("Slope", "degrees"),
            "roughness_m": ("Roughness", "metres"),
            "local_relief_m": ("Local relief", "metres"),
        }

        moon_metric_keys = [
            key for key in moon_metric_labels
            if key in terrain_stats
        ]

        if moon_metric_keys:

            tabs = st.tabs([
                moon_metric_labels[key][0]
                for key in moon_metric_keys
            ])

            for tab, key in zip(tabs, moon_metric_keys):

                with tab:

                    stats = terrain_stats[key]

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Mean",
                        f"{stats.get('mean', 0):,.3f}",
                    )

                    col2.metric(
                        "Median",
                        f"{stats.get('median', 0):,.3f}",
                    )

                    col3.metric(
                        "Valid pixels",
                        f"{stats.get('valid_pixels', 0):,}",
                    )

                    percentiles = stats.get("percentiles", {})

                    if percentiles:

                        frame = pd.DataFrame({
                            "Percentile": list(percentiles.keys()),
                            "Value": list(percentiles.values()),
                        })

                        st.dataframe(
                            frame,
                            use_container_width=True,
                            hide_index=True,
                        )

        # ----------------------------------------------------
        # EARTH-MOON COMMON-SCALE COMPARISON
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            f"### {selected_region} vs lunar latitude reference"
        )

        latitude = centre.get("latitude", 0)
        longitude = centre.get("longitude", 0)

        st.caption(
            f"Earth reference centre: {latitude:.4f}° latitude, "
            f"{longitude:.4f}° longitude."
        )

        st.markdown("**Common spatial scale**")

        sc1, sc2, sc3 = st.columns(3)

        sc1.metric(
            "Earth east-west spacing",
            f"{scale.get('east_west_pixel_size_m', 0):,.1f} m",
        )

        sc2.metric(
            "Earth north-south spacing",
            f"{scale.get('north_south_pixel_size_m', 0):,.1f} m",
        )

        sc3.metric(
            "Neighbourhood",
            str(scale.get("neighbourhood", "3 x 3")),
        )

        st.caption(
            "Earth DEMs are aggregated before terrain metrics are "
            "recalculated. The lunar reference uses a latitude band "
            "centred on the corresponding Earth study region."
        )

        # ----------------------------------------------------
        # LATITUDE BAND
        # ----------------------------------------------------

        band = lunar_reference.get(
            "latitude_band_degrees", []
        )

        if len(band) == 2:

            st.info(
                f"Lunar reference latitude band: "
                f"{band[0]:.2f}° to {band[1]:.2f}°. "
                "This band includes all lunar longitudes."
            )

        # ----------------------------------------------------
        # METRIC COMPARISON TABS
        # ----------------------------------------------------

        st.markdown("### Earth–Moon terrain comparison")

        comparison_keys = [
            key for key in moon_metric_labels
            if key in earth_statistics and key in lunar_statistics
        ]

        if comparison_keys:

            comparison_tabs = st.tabs([
                moon_metric_labels[key][0]
                for key in comparison_keys
            ])

            for tab, key in zip(comparison_tabs, comparison_keys):

                with tab:

                    earth_stats = earth_statistics[key]
                    moon_stats = lunar_statistics[key]

                    earth_col, moon_col = st.columns(2)

                    with earth_col:

                        st.markdown(f"**{earth_label}**")

                        st.metric(
                            "Mean",
                            f"{earth_stats.get('mean', 0):,.3f}",
                        )

                        st.metric(
                            "Median",
                            f"{earth_stats.get('median', 0):,.3f}",
                        )

                        st.caption(
                            f"Valid pixels: "
                            f"{earth_stats.get('valid_pixels', 0):,}"
                        )

                    with moon_col:

                        st.markdown("**Moon — LOLA latitude band**")

                        st.metric(
                            "Mean",
                            f"{moon_stats.get('mean', 0):,.3f}",
                        )

                        st.metric(
                            "Median",
                            f"{moon_stats.get('median', 0):,.3f}",
                        )

                        st.caption(
                            f"Valid pixels: "
                            f"{moon_stats.get('valid_pixels', 0):,}"
                        )

                    earth_percentiles = earth_stats.get(
                        "percentiles", {}
                    )

                    moon_percentiles = moon_stats.get(
                        "percentiles", {}
                    )

                    percentile_keys = [
                        "p05", "p25", "p50", "p75", "p95"
                    ]

                    if earth_percentiles and moon_percentiles:

                        frame = pd.DataFrame({
                            "Percentile": percentile_keys,

                            earth_label: [
                                earth_percentiles.get(k)
                                for k in percentile_keys
                            ],

                            "Moon LOLA": [
                                moon_percentiles.get(k)
                                for k in percentile_keys
                            ],
                        })

                        st.markdown("**Percentile comparison**")

                        st.dataframe(
                            frame,
                            use_container_width=True,
                            hide_index=True,
                        )

                    # Median comparison chart.
                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x=[earth_label, "Moon LOLA"],
                        y=[
                            earth_stats.get("median", 0),
                            moon_stats.get("median", 0),
                        ],
                        text=[
                            f"{earth_stats.get('median', 0):,.2f}",
                            f"{moon_stats.get('median', 0):,.2f}",
                        ],
                        textposition="outside",
                        name="Median",
                    ))

                    fig.update_layout(
                        title=f"{moon_metric_labels[key][0]} — Median comparison",
                        yaxis_title=moon_metric_labels[key][1],
                        height=380,
                        template="plotly_white",
                        showlegend=False,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

        else:

            st.warning(
                "Terrain comparison statistics are missing for this region."
            )

        # ----------------------------------------------------
        # LUNAR ELEVATION VISUALISATION
        # ----------------------------------------------------

        st.divider()
        st.markdown("### Global lunar visualisation")

        moon_image_path = (
            project_root
            / "outputs"
            / "Visualisations"
            / "Global"
            / "moon_global_elevation.png"
        )

        if moon_image_path.exists():

            st.image(
                str(moon_image_path),
                use_container_width=True,
                caption="Global LOLA lunar elevation distribution map.",
            )

        else:

            st.info(
                "Global lunar elevation visualisation is not available."
            )

        # ----------------------------------------------------
        # DATASET AND PROCESSING DETAILS
        # ----------------------------------------------------

        with st.expander("LOLA dataset technical details"):

            kv_table([
                ("Product", dataset.get("product", "LOLA Global DEM")),
                ("Product ID", dataset.get("product_id", "LDEM_16")),
                ("Data set ID", dataset.get("data_set_id", "LRO-L-LOLA-4-GDR-V1.0")),
                ("Product version", dataset.get("product_version", "V3.1")),
                ("Projection", dataset.get("projection", "Simple cylindrical")),
                ("Reference radius", f"{dataset.get('reference_radius_m', 1737400):,.0f} m"),
                ("Pixel resolution", f"{dataset.get('pixel_resolution_degrees', 0.0625)} degrees"),
            ])

            st.markdown("**Complete record**")
            st.json(dataset, expanded=False)

        with st.expander("Lunar processing methodology"):
            st.json(moon_report.get("processing", {}))

        with st.expander("Earth–Moon comparison methodology"):

            st.markdown("**Methodology**")
            st.json(
                moon_comparison_report.get("methodology", {}), expanded=False
            )

            st.markdown("**Comparison scale**")
            st.json(
                moon_region_data.get("comparison_scale", {}), expanded=False
            )

        # ----------------------------------------------------
        # DATA VALIDATION
        # ----------------------------------------------------

        with st.expander("LOLA source-data validation"):

            validation = moon_report.get(
                "raw_data_validation", {}
            )

            st.json(validation)

            if not validation.get("range_matches_label", True):

                st.warning(
                    "Observed raster extrema differ from the PDS label "
                    "extrema. This discrepancy remains unresolved and "
                    "has not been silently corrected."
                )

        # ----------------------------------------------------
        # SCIENTIFIC LIMITATIONS
        # ----------------------------------------------------

        with st.expander("Global lunar scientific limitations"):

            for item in moon_report.get(
                "scientific_limitations", []
            ):

                st.markdown(f"- {item}")

        with st.expander("Earth–Moon comparison limitations"):

            for item in moon_comparison_report.get(
                "scientific_limitations", []
            ):

                st.markdown(f"- {item}")

        # ----------------------------------------------------
        # DOWNLOAD REPORTS
        # ----------------------------------------------------

        with st.expander("Download lunar research reports"):

            st.download_button(
                "Download global lunar terrain report",

                data=json.dumps(
                    moon_report,
                    indent=4,
                    ensure_ascii=False,
                ).encode("utf-8"),

                file_name="terraform_global_moon_terrain_report.json",

                mime="application/json",

                key="download_global_moon_report",
            )

            st.download_button(
                "Download Earth-Moon comparison report",

                data=json.dumps(
                    moon_comparison_report,
                    indent=4,
                    ensure_ascii=False,
                ).encode("utf-8"),

                file_name="terraform_earth_moon_comparison.json",

                mime="application/json",

                key="download_earth_moon_report",
            )

    # ========================================================
    # REFERENCE-SPECIFIC METHODOLOGY
    # ========================================================

    st.divider()

    st.markdown("### Processing methodology")

    if is_jezero:

        methods = report.get("processing_methods", {})

        if methods:

            with st.expander("Regional processing methodology"):
                st.json(methods)

        if elevation_context:

            with st.expander("Regional elevation context"):
                st.json(elevation_context)

        with st.expander("Regional scientific limitations"):

            limitations = report.get(
                "scientific_limitations", []
            )

            for limitation in limitations:
                st.markdown(f"- {limitation}")

        with st.expander("Underlying regional histogram data"):
            st.json(histograms)

        with st.expander("Complete regional comparison report"):

            st.json(report)

            st.download_button(
                "Download selected regional report",

                data=json.dumps(
                    report,
                    indent=4,
                    ensure_ascii=False,
                ).encode("utf-8"),

                file_name=config["filename"],

                mime="application/json",

                key="download_selected_regional_report",
            )

    elif is_global_mars:

        st.caption(
            "Global Mars methodology and limitations are documented "
            "in the Global Mars technical details above."
        )

    elif is_global_moon:

        st.caption(
            "Global Moon and Earth–Moon methodologies and limitations "
            "are documented in the lunar technical details above."
        )

    # ========================================================
    # SCIENTIFIC DISCLAIMER
    # ========================================================

    st.warning(
        "Earth terrain is derived from Copernicus DEM GLO-30, a "
        "digital surface model. Mars terrain is based on NASA MOLA "
        "MEGDR data. Lunar terrain is based on NASA LOLA LDEM_16. "
        "Their vertical reference systems, spatial resolutions, "
        "measurement characteristics and physical environments differ. "
        "Global statistical comparisons are not spatially matched "
        "analogue searches. Terrain similarity alone does not establish "
        "geological equivalence, habitability, resource potential, "
        "landing safety or mission-site suitability."
    )

    # ========================================================
    # OPTIONAL UPLOADED DATASET COMPARISON
    # ========================================================

    with st.expander("Additional uploaded dataset comparison"):

        if earth_df.empty or mars_df.empty:

            st.info(
                "Optional custom comparison is available when additional "
                "Earth and Mars CSV files are uploaded through the sidebar. "
                "The main research comparison does not require uploads."
            )

        elif "location" not in earth_df.columns:

            st.warning(
                "The uploaded Earth dataset needs a location column."
            )

        else:

            earth_name = st.selectbox(
                "Earth candidate",
                earth_df["location"].astype(str).tolist(),
                key="custom_comparison_earth",
            )

            earth_row = earth_df[
                earth_df["location"].astype(str) == earth_name
            ].iloc[0]

            if "location" in mars_df.columns:

                mars_name = st.selectbox(
                    "Mars reference region",
                    mars_df["location"].astype(str).tolist(),
                    key="custom_comparison_mars",
                )

                mars_row = mars_df[
                    mars_df["location"].astype(str) == mars_name
                ].iloc[0]

            else:

                mars_name = "Mars reference record"
                mars_row = mars_df.iloc[0]

            common = [
                metric
                for metric in COMPARISON_METRICS
                if metric in earth_row.index
                and metric in mars_row.index
                and pd.notna(earth_row[metric])
                and pd.notna(mars_row[metric])
            ]

            if common:

                frame = pd.DataFrame({
                    "Measurement": common,

                    str(earth_name): [
                        earth_row[m] for m in common
                    ],

                    str(mars_name): [
                        mars_row[m] for m in common
                    ],
                })

                st.dataframe(
                    frame,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.warning(
                    "The uploaded records do not share enough valid "
                    "numeric measurements."
                )


# ============================================================
# PAGE 5 — CANDIDATE INTELLIGENCE
# ============================================================

# This page intentionally uses your existing dedicated module.
# Its original functionality is retained.
def candidate_page():
    candidate_intelligence_page(ROOT)


# ============================================================
# PAGE 6 — RESEARCH VISUALISATIONS
# ============================================================



# ============================================================
# PAGE 7 — DATA LABORATORY
# ============================================================



def global_planetary_comparison_page():
    section_header(
        "PLANETARY-SCALE COMPARISON",
        "Global Planetary Comparison",
        "A statistical comparison of global elevation distributions across Earth, Mars and the Moon."
    )

    report_path = GLOBAL_COMPARISON_REPORT_PATH

    if not report_path.exists():
        st.error("The global planetary comparison report is unavailable.")
        return

    try:
        with open(report_path, "r", encoding="utf-8") as file:
            report = json.load(file)
    except Exception as error:
        st.error(f"Could not load global comparison report: {error}")
        return

    results = report.get("planetary_results", {})
    planets = ["Earth", "Mars", "Moon"]


    st.markdown("### Global elevation comparison")
    columns = st.columns(3)

    for column, planet in zip(columns, planets):
        data = results.get(planet, {})
        stats = data.get("statistics", {})
        diagnostics = data.get("histogram_diagnostics", {})
        percentiles = stats.get("weighted_percentiles_m", {})

        with column:
            st.markdown(f"#### {planet}")

            mean = stats.get("weighted_mean_m")
            std = stats.get("weighted_standard_deviation_m")
            median = percentiles.get("P50")
            coverage = diagnostics.get("weighted_coverage_percent")

            st.metric("Weighted mean", f"{mean:,.2f} m" if mean is not None else "—")
            st.metric("Weighted median", f"{median:,.2f} m" if median is not None else "—")
            st.metric("Standard deviation", f"{std:,.2f} m" if std is not None else "—")
            st.metric("Histogram coverage", f"{coverage:.4f}%" if coverage is not None else "—")
            st.caption(f"Accepted cells: {diagnostics.get('accepted_cells', 0):,}")

    st.divider()
    st.markdown("### Weighted elevation percentiles")

    percentile_rows = []
    for percentile in ["P05", "P10", "P25", "P50", "P75", "P90", "P95"]:
        row = {"Percentile": percentile}
        for planet in planets:
            values = results.get(planet, {}).get("statistics", {}).get("weighted_percentiles_m", {})
            row[planet] = values.get(percentile)
        percentile_rows.append(row)

    percentile_df = pd.DataFrame(percentile_rows)

    st.dataframe(
        percentile_df.style.format(
            {planet: "{:,.2f} m" for planet in planets},
            na_rep="—"
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Percentiles are interpolated within 10 m histogram bins, assuming "
        "uniform within-bin weight distribution."
    )

    st.divider()
    st.markdown("### Global elevation distributions")

    visualisations = report.get("visualisations", {})
    images = [
        ("Elevation distribution", "elevation_distributions"),
        ("Normalised terrain distribution", "normalized_distributions")
    ]

    for label, key in images:
        image_value = visualisations.get(key)

        if not image_value:
            st.info(f"No report image registered for {label}.")
            continue

        image_path = Path(image_value)

        if image_path.exists():
            st.markdown(f"**{label}**")
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning(f"{label} image is unavailable: {image_path}")

    st.divider()
    st.markdown("### Planetary surface scope")

    columns = st.columns(3)

    for column, planet in zip(columns, planets):
        data = results.get(planet, {})

        with column:
            st.markdown(f"**{planet}**")
            st.write(data.get("surface_scope", "Not specified"))
            st.caption(data.get("vertical_reference", "Reference not specified"))
            st.caption(
                f"Equatorial spacing: "
                f"{data.get('approximate_equatorial_spacing_km', '—')} km"
            )

    earth = results.get("Earth", {})
    mars = results.get("Mars", {})
    moon = results.get("Moon", {})

    st.divider()
    st.markdown("### Scientific interpretation and limitations")

    with st.expander("Earth — land-proxy qualification", expanded=True):
        st.write(earth.get("land_proxy_rule", "Land-proxy rule not specified."))
        st.warning(
            "Positive elevation is a screening proxy, not an authoritative "
            "coastline or land mask. Earth therefore uses a different surface "
            "selection rule from the full planetary grids."
        )

    with st.expander("Mars — zero-value metadata discrepancy", expanded=True):
        zero_count = mars.get("source_raster_diagnostics", {}).get("zero_cells")
        st.write(mars.get("zero_value_handling", "Not specified."))

        if zero_count is not None:
            st.info(
                f"The report records {zero_count:,} zero-valued source cells retained. "
                "The generated ENVI header marks zero as its ignore value, while "
                "the original PDS elevation label does not declare zero as a "
                "special NoData value. This conflict remains disclosed."
            )

    with st.expander("Moon — reference and source-range qualification"):
        st.write(moon.get("height_conversion", "Height conversion not specified."))
        st.write(moon.get("vertical_reference", "Reference not specified."))

        moon_diag = moon.get("source_raster_diagnostics", {})

        if moon_diag.get("raw_extrema_match_pds_label") is False:
            st.warning(
                "Observed raw extrema do not exactly match the PDS label extrema, "
                "although they remain within the label's declared range."
            )

    with st.expander("Comparison methodology and limitations"):
        method = report.get("method", {})

        st.markdown(f"**Comparison type:** {method.get('comparison_type', 'Not specified')}")
        st.markdown(f"**Area weighting:** {method.get('area_weighting', 'Not specified')}")
        st.markdown(f"**Resolution:** {method.get('resolution_note', 'Not specified')}")
        st.markdown(f"**Statistics:** {method.get('statistics_precision', 'Not specified')}")
        st.markdown(f"**Normalisation:** {method.get('normalization', 'Not specified')}")

        for planet in planets:
            st.markdown(f"**{planet} limitations**")

            for limitation in results.get(planet, {}).get("limitations", []):
                st.markdown(f"- {limitation}")

        for limitation in report.get("scientific_interpretation", []):
            st.markdown(f"- {limitation}")

    st.download_button(
        "Download global planetary comparison report",
        data=json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="terraform_global_planetary_comparison_v1.1.json",
        mime="application/json",
        key="download_dedicated_global_planetary_report"
    )






def moon_explorer_page():
    section_header(
        "LUNAR TERRAIN INTELLIGENCE",
        "Moon Explorer",
        "Explore the global lunar elevation model and its scientific characteristics."
    )

    report_path = OUTPUTS / "Global" / "Moon" / "moon_global_report.json"
    image_path = OUTPUTS / "Visualisations" / "Global" / "moon_global_elevation.png"

    if not report_path.exists():
        st.error("Moon global report not found.")
        return

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not read the Moon report: {exc}")
        return

    dataset = report.get("dataset", {})
    dimensions = dataset.get("dimensions", {})
    stats = report.get("elevation_statistics", {})
    percentiles = stats.get("percentiles_m", {})
    processing = report.get("processing", {})
    limitations = report.get("scientific_limitations", [])
    validation = report.get("raw_data_validation", {})

    def moon_num(value, digits=1):
        try:
            return f"{float(value):,.{digits}f}"
        except (TypeError, ValueError):
            return "N/A"

    st.markdown("### Global lunar elevation")

    if image_path.exists():
        st.image(
            str(image_path),
            caption="Global lunar elevation ? LOLA LDEM_16",
            use_container_width=True
        )
    else:
        st.warning("Global Moon elevation image not found.")

    st.markdown("### Elevation statistics")

    metrics = [
        ("Minimum", stats.get("minimum_m")),
        ("Maximum", stats.get("maximum_m")),
        ("Mean", stats.get("mean_m")),
        ("Median", stats.get("median_m")),
    ]

    cols = st.columns(4)
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, f"{moon_num(value)} m")

    cols = st.columns(3)
    cols[0].metric(
        "Standard deviation",
        f"{moon_num(stats.get('standard_deviation_m'))} m"
    )
    cols[1].metric(
        "Grid cells",
        f"{dimensions.get('total_cells', 0):,}"
    )
    cols[2].metric(
        "Zero-height cells",
        f"{stats.get('zero_height_cells', 'N/A'):,}"
        if isinstance(stats.get("zero_height_cells"), int)
        else "N/A"
    )

    percentile_values = {
        "5th": percentiles.get("p05"),
        "25th": percentiles.get("p25"),
        "50th": stats.get("median_m"),
        "75th": percentiles.get("p75"),
        "95th": percentiles.get("p95"),
    }

    available = {
        key: value for key, value in percentile_values.items()
        if isinstance(value, (int, float))
    }

    if available:
        st.markdown("### Elevation distribution")
        df = pd.DataFrame({
            "Percentile": list(available.keys()),
            "Elevation (m)": list(available.values())
        })
        fig = px.bar(
            df,
            x="Percentile",
            y="Elevation (m)",
            color="Elevation (m)",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset information")

    info = {
        "Dataset": dataset.get("product", "LOLA Global DEM"),
        "Product ID": dataset.get("product_id", "LDEM_16"),
        "PDS Data Set ID": dataset.get("data_set_id", "N/A"),
        "Product version": dataset.get("product_version", "N/A"),
        "Source": dataset.get("source", "NASA PDS"),
        "Filename": dataset.get("filename", "ldem_16.img"),
        "Grid dimensions": (
            f"{dimensions.get('columns', 'N/A')} ? "
            f"{dimensions.get('rows', 'N/A')}"
        ),
        "Data type": dataset.get("data_type", "N/A"),
        "Byte order": dataset.get("byte_order", "N/A"),
        "Elevation scale": f"{moon_num(dataset.get('scaling_factor'), 2)} m",
        "Reference radius": f"{moon_num(dataset.get('reference_radius_m'), 0)} m",
        "Pixel resolution": (
            f"{moon_num(dataset.get('pixel_resolution_degrees'), 4)}?"
        ),
        "Equatorial pixel scale": (
            f"{moon_num(dataset.get('equatorial_pixel_scale_m'), 2)} m"
        ),
        "Projection": dataset.get("projection", "N/A"),
        "Coordinate system": dataset.get("coordinate_system", "N/A"),
    }

    st.table(pd.DataFrame([
        {"Property": key, "Value": value}
        for key, value in info.items()
    ]))

    st.markdown("### Raw data validation")

    cols = st.columns(2)
    cols[0].metric(
        "Observed raw minimum",
        str(validation.get("observed_raw_minimum", "N/A"))
    )
    cols[1].metric(
        "Observed raw maximum",
        str(validation.get("observed_raw_maximum", "N/A"))
    )

    if validation.get("range_matches_label") is False:
        st.warning(
            "UNRESOLVED METADATA DISCREPANCY: Observed raw extrema do not "
            "match the extrema stated in the PDS label. The source data "
            "has not been modified."
        )

    st.markdown("### Processing methodology")
    kv_table([
        ("Conversion", processing.get("conversion", "Not specified")),
        ("Display downsampling", processing.get("display_downsampling", "Not specified")),
        ("NoData mask applied", processing.get("nodata_mask_applied", "Not specified")),
        ("Zero-height cells retained", processing.get("zero_height_cells_retained", "Not specified")),
    ])

    st.markdown("### Scientific limitations")

    if isinstance(limitations, list):
        for item in limitations:
            st.warning(str(item))
    else:
        st.warning(str(limitations))

    st.info(
        "Elevation describes surface topography. It does not independently "
        "establish geological similarity, resource availability, or suitability "
        "for a permanent lunar base."
    )

    st.download_button(
        "Download Moon analysis report",
        data=report_path.read_bytes(),
        file_name="moon_global_report.json",
        mime="application/json"
    )


def visualisations_page():
    section_header(
        "SCIENTIFIC VISUALISATION",
        "Research Visualisations",
        "Explore global Earth, Mars and Moon analysis alongside the Atacama–Jezero regional investigation."
    )

    # ========================================================
    # VISUALISATION INVENTORY
    # ========================================================

    st.markdown("## Research Visualisation Archive")

    total_images = sum(
        len(items) for items in VISUALISATION_GROUPS.values()
    )

    available_images = sum(
        (OUTPUTS / path).exists()
        for items in VISUALISATION_GROUPS.values()
        for _, path in items
    )

    a, b, c = st.columns(3)

    a.metric("Visual assets catalogued", total_images)

    st.caption(
        "The gallery displays existing research outputs. "
        "Images are not regenerated or modified by this page."
    )

    st.divider()

    for group_name, images in VISUALISATION_GROUPS.items():

        st.markdown(f"### {group_name}")

        for row_start in range(0, len(images), 2):

            row_cols = st.columns(2)

            for col, (title, relative_path) in zip(
                row_cols, images[row_start:row_start + 2]
            ):
                with col:
                    viz_tile(title, relative_path)

        st.divider()

    # ========================================================
    # INVENTORY TABLE
    # ========================================================

    st.markdown("### Visualisation Inventory")

    inventory_rows = []

    for group_name, images in VISUALISATION_GROUPS.items():

        for title, relative_path in images:

            path = OUTPUTS / relative_path

            inventory_rows.append({
                "Research stream": group_name,
                "Visualisation": title,
                "Status": (
                    "Available" if path.exists() else "Missing"
                ),
            })

    inventory_df = pd.DataFrame(inventory_rows)

    st.dataframe(
        inventory_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download visualisation inventory",
        data=inventory_df.to_csv(index=False).encode("utf-8"),
        file_name="terraform_visualisation_inventory.csv",
        mime="text/csv",
        key="download_visualisation_inventory"
    )


# ============================================================
# PAGE 8 — TECHNICAL DEEP DIVE
# ============================================================

def technical_deep_dive_page():
    section_header(
        "METHODS, DATA AND REPRODUCIBILITY",
        "Technical Deep Dive",
        "Inspect the scientific foundations, data products, processing methods, assumptions and limitations behind TERRAFORM."
    )

    st.markdown("### Research architecture")
    stages = [
        ("01", "Data acquisition", "Open Earth observation products and planetary datasets."),
        ("02", "Data harmonisation", "Coordinate systems, spatial resolution, units and vertical references."),
        ("03", "Terrain processing", "Slope, roughness, local relief and elevation statistics."),
        ("04", "Analogue discovery", "Empirical percentile-range screening and spatial intersection."),
        ("05", "Candidate extraction", "Connected-component extraction and candidate profiling."),
        ("06", "Scientific interpretation", "Geological context, sensitivity analysis and limitations."),
    ]
    st.markdown(
        '<div class="tf-grid tf-grid-3">'
        + "".join(
            f'<div class="panel"><div class="mono">STAGE {number}</div>'
            f'<div class="tf-card-title">{html.escape(title)}</div>'
            f'<div class="tf-card-text">{html.escape(description)}</div></div>'
            for number, title, description in stages
        )
        + '</div>',
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("### Global Mars terrain analysis")

    if isinstance(MARS_TERRAIN_REPORT, dict):
        source = MARS_TERRAIN_REPORT.get("source", {})
        grid = MARS_TERRAIN_REPORT.get("grid", {})
        geometry = MARS_TERRAIN_REPORT.get("geometry", {})
        processing = MARS_TERRAIN_REPORT.get("processing", {})

        st.markdown("#### Source product and grid")
        c1, c2, c3 = st.columns(3)
        c1.metric("Grid", f"{grid.get('rows', 2880):,} × {grid.get('columns', 5760):,}")
        c2.metric("Grid spacing", f"{grid.get('angular_spacing_degrees', 0.0625)}°")
        c3.metric("Reference radius", f"{geometry.get('reference_radius_km', 3396):,.0f} km")

        st.markdown(
            f"""
            - **Dataset:** `{source.get('dataset_id', 'MGS-M-MOLA-5-MEGDR-L3-V1.0')}`
            - **Elevation raster:** `{source.get('elevation', 'megt90n000eb.img')}`
            - **Observation-count raster:** `{source.get('observation_count', 'megc90n000eb.img')}`
            - **Longitude convention:** 0–360° east
            - **Latitude convention:** planetocentric
            - **Grid:** simple cylindrical, 0.0625° angular spacing
            - **Elevation reference:** MOLA areoid based on the documented GMM3 model context
            """
        )

        st.markdown("#### Spherical reference-cell area")
        st.latex(r"A = R^2 \Delta\lambda \left(\sin\phi_N-\sin\phi_S\right)")
        st.caption(
            "The area calculation uses a spherical reference geometry. "
            "It is a reference-cell area, not an exact terrain-surface area."
        )

        st.markdown("#### Terrain metric methods")
        st.markdown(
            """
            **Slope:** central finite differences on the spherical reference geometry,
            with east–west pixel distance adjusted for latitude.

            **Roughness:** population standard deviation of elevation values in a
            3×3 angular grid-cell neighbourhood.

            **Local relief:** maximum elevation minus minimum elevation in a
            3×3 angular grid-cell neighbourhood.
            """
        )
        st.latex(r"\text{Local relief}_{3\times3}=\max(z)-\min(z)")
        st.caption(
            "The 3×3 windows are defined in angular grid cells, not constant physical-distance "
            "windows. Their east–west physical footprint changes with latitude."
        )

        st.markdown("#### Processing configuration")
        with st.expander("Advanced technical details — JSON"):
            st.json(processing, expanded=False)

        st.markdown("#### Interpretation of observation support")
        st.markdown(
            """
            - The observation-count raster records the number of MOLA observations contributing to a cell.
            - A zero count does not mean the corresponding elevation value is zero or missing.
            - The global MOLA elevation product retains interpolated elevation values in zero-observation cells.
            - Observed-only terrain statistics select centre cells with at least one recorded observation.
            - Neighbouring cells used by 3×3 metrics can still include interpolated elevations.
            - Observation count is sampling support, not a direct accuracy measurement.
            """
        )

        st.markdown("#### Mars terrain metric results")
        terrain_stats = MARS_TERRAIN_REPORT.get("statistics", {})
        for metric_key, metric_label in [
            ("slope_degrees", "Slope (degrees)"),
            ("roughness_3x3_m", "Roughness (metres)"),
            ("local_relief_3x3_m", "Local relief (metres)"),
        ]:
            with st.expander(metric_label):
                data = terrain_stats.get(metric_key, {})
                for subset, title in [
                    ("all_analysed", "All analysed cells"),
                    ("observed_only", "Observed-centre cells"),
                ]:
                    values = data.get(subset, {})
                    st.markdown(f"**{title}**")
                    show_stat_values(values)

        st.markdown("#### Terrain analysis limitations")
        for limitation in MARS_TERRAIN_REPORT.get("limitations", []):
            st.markdown(f"- {limitation}")
    else:
        st.warning("Global Mars terrain metrics report is not available.")

    if MARS_AREA_REPORT:
        with st.expander("Full-sphere area-weighted elevation and coverage report"):
            st.code(MARS_AREA_REPORT, language="text")

    st.divider()
    st.markdown("### Atacama–Jezero regional study")

    st.markdown(
        """
        **Earth terrain source:** Copernicus DEM GLO-30 digital surface model.

        **Mars terrain source:** MOLA MEGDR elevation data for the Jezero reference region.

        **Common working resolution:** approximately 463 metres.

        **Terrain screening intervals:**
        - Broad: Jezero P10–P90
        - Moderate: Jezero P25–P75
        - Strict: Jezero P40–P60

        The screening uses the intersection of slope, roughness and local-relief
        interval masks. Connected components are extracted from the strict
        three-metric intersection and profiled by region.
        """
    )

    st.markdown("#### Reported regional screening results")
    screening_results = pd.DataFrame([
        {"Scenario": "Broad", "Jezero percentile interval": "P10–P90", "Matching pixels (%)": 73.30, "Approx. matching area (km²)": 8249.78},
        {"Scenario": "Moderate", "Jezero percentile interval": "P25–P75", "Matching pixels (%)": 42.03, "Approx. matching area (km²)": np.nan},
        {"Scenario": "Strict", "Jezero percentile interval": "P40–P60", "Matching pixels (%)": 12.59, "Approx. matching area (km²)": np.nan},
    ])
    st.dataframe(screening_results, use_container_width=True, hide_index=True)
    st.caption(
        "Percentages are the reported matching-pixel proportions for the relevant screening "
        "scenario. The broad area figure is approximately 8,249.78 km². Do not interpret "
        "these values as probabilities or as a universal similarity score."
    )

    st.markdown("#### Candidate extraction and geological context")
    st.markdown(
        """
        - The strict three-metric intersection produced **873 connected components**.
        - The ten largest components were profiled, ordered by connected-region area.
        - Candidate identifiers and area ranks describe extracted geometry, not suitability.
        - Geological intersections use SERNAGEOMIN regional geological mapping at approximately 1:1,000,000 scale.
        - Geological coverage is a map-intersection result, not field verification of lithology or mineral composition.
        - Earth and Mars use different coordinate and vertical-reference systems; the regional comparison is distribution-based, not pixel-to-pixel alignment.
        """
    )

    st.divider()
    st.markdown("### Coordinate systems and reference surfaces")
    st.markdown(
        """
        The global Mars raster uses planetocentric latitude and east longitude in a
        0–360° convention, on a simple cylindrical grid. Its elevation reference is
        the MOLA areoid context documented with the source product.

        The regional Earth and Mars products have separate spatial reference systems
        and vertical references. They are harmonised for the documented regional
        terrain-distribution comparison; this does not make their elevations directly
        interchangeable or establish exact spatial correspondence.
        """
    )

    st.divider()
    st.markdown("### Research reports and audit records")

    report_paths = [
        ("Global Mars terrain metrics JSON", OUTPUTS / "global_mars/terrain_metrics/mars_global_terrain_report.json"),
        ("Global Mars area-weighted report", OUTPUTS / "global_mars/mars_area_weighted_report.txt"),
        ("Global Mars analysis report", OUTPUTS / "global_mars/mars_global_analysis_report.txt"),
        ("Observation threshold report", OUTPUTS / "global_mars/mars_observation_threshold_report.txt"),
        ("Outlier validation report", OUTPUTS / "global_mars/mars_outlier_validation_report.txt"),
        ("Strict maxima audit", OUTPUTS / "global_mars/mars_strict_maxima_audit.txt"),
        ("Strict observation report", OUTPUTS / "global_mars/mars_strict_observation_report.txt"),
        ("Outlier inspection report", OUTPUTS / "global_mars/outlier_inspection_report.txt"),
        ("Terrain metrics text report", OUTPUTS / "global_mars/terrain_metrics/mars_global_terrain_report.txt"),
    ]
    for title, path in report_paths:
        show_report_file(path, title)

    regional_report_paths = [
        ("Earth-Mars terrain comparison", OUTPUTS / "reports/earth_mars_terrain_comparison.json"),
        ("Earth terrain analysis", OUTPUTS / "reports/earth_terrain_analysis.json"),
        ("Jezero terrain analysis", OUTPUTS / "reports/jezero_terrain_analysis.json"),
        ("Earth validation", OUTPUTS / "reports/earth_validation_report.json"),
        ("Sensitivity analysis", OUTPUTS / "reports/sensitivity_analysis_report.json"),
    ]
    for title, path in regional_report_paths:
        if path.exists():
            with st.expander(title):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    st.json(data)
                    st.download_button(
                        f"Download {title}",
                        data=json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"),
                        file_name=path.name,
                        mime="application/json",
                        key=f"deep_dive_{path.name}"
                    )
                except Exception:
                    st.code(load_text_report(path), language="text")

    st.divider()
    st.markdown("### Scientific limitations and interpretation")
    st.warning(
        "Earth analogues cannot reproduce the complete Martian environment. "
        "Gravity, atmospheric pressure, radiation, atmospheric chemistry and "
        "other planetary conditions remain fundamentally different."
    )
    limitations = [
        "The spherical Mars reference geometry is not an exact model of the physical terrain surface.",
        "The cylindrical grid becomes singular near the poles; terrain-metric cells at and beyond 89.5° absolute planetocentric latitude are excluded.",
        "The maximum slope is an extreme, outlier-sensitive value and should not be treated as representative.",
        "The 3×3 roughness and relief windows are angular grid-cell neighbourhoods rather than fixed-distance windows.",
        "Interpolated elevations remain in zero-observation cells in the source product.",
        "Observed-only centre-cell selection does not guarantee that every neighbouring cell used in a metric was directly observed.",
        "Area-weighted percentile values are approximate histogram-bin estimates.",
        "Regional geological mapping is not field verification.",
        "Percentile overlap is descriptive screening, not a probability of planetary analogy.",
        "No universal similarity score or validated site-suitability ranking is produced.",
    ]
    for item in limitations:
        st.markdown(f"- {item}")

    st.divider()
    st.markdown("### Research provenance")
    provenance = pd.DataFrame([
        ("Earth terrain", "Copernicus DEM GLO-30 DSM"),
        ("Mars global elevation", "MOLA MEGDR / MGS-M-MOLA-5-MEGDR-L3-V1.0"),
        ("Mars regional reference", "Jezero Crater terrain analysis"),
        ("Mars global reference radius", "3,396 km spherical reference radius"),
        ("Geological mapping", "SERNAGEOMIN regional geological map"),
        ("Regional common working resolution", "Approximately 463 m"),
    ], columns=["Component", "Documented source / context"])
    st.dataframe(provenance, use_container_width=True, hide_index=True)

    st.caption(
        "TERRAFORM is an independent research platform. It is not an official NASA product, "
        "a NASA mission instrument, or a validated planetary site-selection system."
    )


# ============================================================
# MAIN PAGE ROUTER
# ============================================================



# ============================================================
# PAGE ? DATA LABORATORY
# ============================================================


def data_lab_page():
    section_header(
        "DATA OPERATIONS",
        "Data Laboratory",
        "Inspect planetary elevation datasets, CSV processing results and research reports."
    )

    datasets = PLANETARY_DATASETS

    selected_planet = st.radio(
        "SELECT PLANETARY DATASET",
        ["Earth", "Mars", "Moon"],
        horizontal=True,
        key="data_lab_planet"
    )

    config = datasets[selected_planet]

    raster_path = Path(ROOT) / config["raster"]
    report_path = Path(OUTPUTS) / config["report"]

    if selected_planet == "Earth":
        dataframe = earth_df
        rejected = earth_rejected
        processing_report = earth_report
        source = earth_source
        saved_paths = earth_saved_paths

    elif selected_planet == "Mars":
        dataframe = mars_df
        rejected = mars_rejected
        processing_report = mars_report
        source = mars_source
        saved_paths = mars_saved_paths

    else:
        dataframe = pd.DataFrame()
        rejected = pd.DataFrame()
        processing_report = None
        source = config["source"]
        saved_paths = {}

    dataframe = (
        dataframe
        if isinstance(dataframe, pd.DataFrame)
        else pd.DataFrame()
    )

    rejected = (
        rejected
        if isinstance(rejected, pd.DataFrame)
        else pd.DataFrame()
    )

    st.markdown(
        f'<div class="panel">'
        f'<div class="mono">ACTIVE PLANETARY DATA SOURCE</div>'
        f'<div class="panel-title">{html.escape(str(config["source"]))}</div>'
        f'<div class="panel-description">'
        f'{html.escape(str(config["description"]))}'
        f'</div></div>',
        unsafe_allow_html=True
    )

    st.divider()

    raster_tab, processing_tab, reports_tab = st.tabs(
        [
            "Raster Inspection",
            "CSV Processing",
            "Research Reports"
        ]
    )

    with raster_tab:
        st.markdown("### Raster metadata")

        if not raster_path.is_file():
            st.warning(
                "The registered planetary raster file is unavailable. "
                "Check the dataset path before attempting inspection."
            )

        elif rasterio is None:
            st.error(
                "Rasterio is not installed or available. "
                "Raster metadata cannot be inspected."
            )

        else:
            try:
                with rasterio.open(raster_path) as dataset:
                    width = dataset.width
                    height = dataset.height
                    band_count = dataset.count
                    driver = dataset.driver
                    data_types = ", ".join(dataset.dtypes)
                    crs = str(dataset.crs) if dataset.crs else "Not declared"
                    nodata = dataset.nodata
                    bounds = dataset.bounds
                    resolution = dataset.res
                    transform = dataset.transform

                    file_size = raster_path.stat().st_size

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric("Raster width", f"{width:,} px")
                    col2.metric("Raster height", f"{height:,} px")
                    col3.metric("Bands", f"{band_count:,}")
                    col4.metric(
                        "File size",
                        f"{file_size / (1024 ** 2):,.2f} MB"
                    )

                    st.divider()

                    metadata = pd.DataFrame([
                        ["Dataset", selected_planet],
                        ["Product", config["source"]],
                        ["File", raster_path.name],
                        ["Driver", driver],
                        ["Dimensions", f"{width:,} × {height:,}"],
                        ["Data type", data_types],
                        ["Coordinate reference system", crs],
                        ["Reference surface", config["reference"]],
                        ["Resolution", str(resolution)],
                        ["NoData value", str(nodata)],
                        ["Upper-left / lower-right bounds",
                         f"{bounds}"],
                        ["Affine transform", str(transform)],
                    ], columns=["Property", "Value"])

                    st.dataframe(
                        metadata,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Metadata is read from the registered raster. "
                        "No elevation values are changed or reprocessed."
                    )

            except Exception as error:
                st.error("Raster metadata inspection failed.")
                st.caption("Detailed error information is available in local application logs.")

    with processing_tab:
        st.markdown("### CSV processing results")

        has_csv_processing_result = (
            not dataframe.empty
            or not rejected.empty
            or processing_report is not None
        )

        if has_csv_processing_result:
            row_count = len(dataframe)
            rejected_count = len(rejected)
            column_count = len(dataframe.columns)
            missing_count = (
                int(dataframe.isna().sum().sum())
                if not dataframe.empty else 0
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accepted records", f"{row_count:,}")
            c2.metric("Rejected records", f"{rejected_count:,}")
            c3.metric("Data columns", f"{column_count:,}")
            c4.metric("Missing values", f"{missing_count:,}")

        if dataframe.empty:
            if processing_report is None and rejected.empty:
                st.info(
                    f"No optional CSV dataset has been uploaded for {selected_planet}. "
                    "Upload a custom CSV using the sidebar to run validation "
                    "and inspect records. The planetary raster dataset is "
                    "separate and can be inspected in the Raster Inspection tab."
                )
            else:
                st.warning(
                    f"No accepted CSV records are available for {selected_planet}. "
                    "Review the validation report and rejected records below."
                )

                if processing_report is not None:
                    with st.expander("CSV validation report"):
                        st.json(processing_report)

                if not rejected.empty:
                    with st.expander("Inspect rejected records"):
                        st.dataframe(
                            rejected.head(100),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download rejected records",
                            data=rejected.to_csv(index=False).encode("utf-8"),
                            file_name=f"terraform_{selected_planet.lower()}_rejected.csv",
                            mime="text/csv",
                            key=f"download_rejected_{selected_planet}"
                        )

        else:
            st.markdown("#### Dataset explorer")

            selected_columns = st.multiselect(
                "Visible columns",
                options=list(dataframe.columns),
                default=list(dataframe.columns),
                key=f"data_lab_columns_{selected_planet}"
            )

            preview_limit = st.slider(
                "Preview records",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                key=f"data_lab_preview_{selected_planet}"
            )

            if selected_columns:
                st.dataframe(
                    dataframe[selected_columns].head(preview_limit),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Select at least one column.")

            st.markdown("#### Column intelligence")

            metadata = pd.DataFrame({
                "Column": dataframe.columns.astype(str),
                "Data type": [
                    str(dataframe[column].dtype)
                    for column in dataframe.columns
                ],
                "Non-null records": [
                    int(dataframe[column].notna().sum())
                    for column in dataframe.columns
                ],
                "Missing records": [
                    int(dataframe[column].isna().sum())
                    for column in dataframe.columns
                ],
                "Unique values": [
                    int(dataframe[column].nunique(dropna=True))
                    for column in dataframe.columns
                ],
            })

            st.dataframe(
                metadata,
                use_container_width=True,
                hide_index=True
            )

            completeness = (
                (1 - missing_count / (row_count * column_count)) * 100
                if row_count and column_count else 0
            )

            quality_left, quality_right = st.columns(2)

            quality_left.metric(
                "Dataset completeness",
                f"{completeness:.2f}%"
            )

            quality_right.metric(
                "Rejected records",
                f"{rejected_count:,}"
            )

            if not rejected.empty:
                with st.expander("Inspect rejected records"):
                    st.dataframe(
                        rejected.head(100),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download rejected records",
                        data=rejected.to_csv(index=False).encode("utf-8"),
                        file_name=f"terraform_{selected_planet.lower()}_rejected.csv",
                        mime="text/csv",
                        key=f"download_rejected_{selected_planet}"
                    )

            st.download_button(
                f"Export cleaned {selected_planet} CSV",
                data=dataframe.to_csv(index=False).encode("utf-8"),
                file_name=f"terraform_{selected_planet.lower()}_cleaned.csv",
                mime="text/csv",
                key=f"download_cleaned_{selected_planet}"
            )

        if saved_paths:
            with st.expander("Saved CSV processing outputs"):
                for name, path in saved_paths.items():
                    st.markdown(f"**{html.escape(str(name))}**")
                    st.caption("Saved in the TERRAFORM project workspace.")

    with reports_tab:
        st.markdown("### Planetary research report")

        if report_path.is_file():
            try:
                with open(report_path, "r", encoding="utf-8") as file:
                    planetary_report = json.load(file)


                with st.expander("Planetary research report — JSON"):
                    st.json(planetary_report)

                st.download_button(
                    "Download planetary research report",
                    data=json.dumps(
                        planetary_report,
                        indent=2,
                        ensure_ascii=False
                    ).encode("utf-8"),
                    file_name=f"terraform_{selected_planet.lower()}_global_report.json",
                    mime="application/json",
                    key=f"download_global_report_{selected_planet}"
                )

            except Exception as error:
                st.error("The planetary report could not be read.")
                st.caption("Detailed error information is available in local application logs.")

        else:
            st.info("The planetary research report is not available.")

    st.divider()

    st.caption(
        "Raster inspection displays metadata from the registered planetary "
        "elevation products. CSV processing results and planetary research "
        "reports are presented separately. Dataset inspection does not "
        "independently validate scientific suitability or planetary analogy."
    )




# ============================================================
# CINEMATIC HOME PAGE (presentation only)
# ============================================================

HOME_IMAGES = {
    # Real photographs from Wikimedia Commons (NASA / ESA public imagery).
    # Rendered as CSS background layers: if a URL fails, the gradient beneath shows instead.
    "Mars": "https://commons.wikimedia.org/wiki/Special:FilePath/OSIRIS_Mars_true_color.jpg?width=1600",
    "Earth": "https://commons.wikimedia.org/wiki/Special:FilePath/The_Earth_seen_from_Apollo_17.jpg?width=1000",
    "Moon": "https://commons.wikimedia.org/wiki/Special:FilePath/FullMoon2010.jpg?width=1000",
}

# ------------------------------------------------------------
# CUSTOM PLANET CARD IMAGES
# Point each planet at your own image. Each value can be:
#   * a path relative to app.py, e.g. "assets/planets/earth.jpg"
#   * a full URL, e.g. "https://example.org/earth.jpg"
# If the entry is None, or a local file is missing, the default photo is used.
# ------------------------------------------------------------
PLANET_IMAGE_SOURCES = {
    "Earth": "assets/planets/earth.jpg",
    "Mars": "assets/planets/mars.jpg",
    "Moon": "assets/planets/moon.jpg",
}


@st.cache_data(show_spinner=False)
def _local_image_data_uri(path_str, modified_time):
    """Read a local image once and return it as an inline data URI (cached)."""
    mime = mimetypes.guess_type(path_str)[0] or "image/jpeg"
    encoded = base64.b64encode(Path(path_str).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _find_local_planet_image(key):
    """Find EARTH / MARS / MOON by name in the project, any case, any common extension."""
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for folder in (ROOT / "assets" / "planets", ROOT / "assets", ROOT):
        if not folder.is_dir():
            continue
        for candidate in sorted(folder.iterdir()):
            if (
                candidate.is_file()
                and candidate.stem.lower() == key.lower()
                and candidate.suffix.lower() in extensions
            ):
                return candidate
    return None


def resolve_planet_image(key):
    source = PLANET_IMAGE_SOURCES.get(key)
    if source:
        source = str(source)
        if source.startswith(("http://", "https://")):
            return source.replace("'", "%27")
        local_path = ROOT / source
        if local_path.is_file():
            return _local_image_data_uri(str(local_path), local_path.stat().st_mtime)
    found = _find_local_planet_image(key)
    if found is not None:
        return _local_image_data_uri(str(found), found.stat().st_mtime)
    return HOME_IMAGES[key]


WORLD_FALLBACKS = {
    "Earth": "radial-gradient(circle at 50% 50%, #4aa3d8 0, #1b4f7a 30%, #0a1a2c 31%, #05080d 100%)",
    "Mars": "radial-gradient(circle at 50% 50%, #c0623b 0, #7a3322 30%, #1c0f10 31%, #05080d 100%)",
    "Moon": "radial-gradient(circle at 50% 50%, #b9bcc0 0, #6c7076 30%, #15181c 31%, #05080d 100%)",
}


def project_image_exists(relative_path):
    return (OUTPUTS / relative_path).exists()


def home_page():
    hero_url = HOME_IMAGES["Mars"]
    st.markdown(
        f"""
        <div class="tf-hero has-img" style="--hero-img: url('{hero_url}');">
            <div class="tf-hero-inner">
                <div class="tf-hero-tag">// PLANETARY RESEARCH SYSTEM</div>
                <h1 class="tf-hero-title">TERRAFORM</h1>
                <div class="tf-hero-sub">PLANETARY ANALOG INTELLIGENCE</div>
                <div class="tf-hero-statement">Exploring terrestrial environments through planetary-scale geospatial intelligence.</div>
                <div class="tf-hero-desc">
                    TERRAFORM is an independent research platform for analysing terrestrial and
                    planetary terrain datasets, identifying measurable similarities and
                    investigating potential analogue environments.
                </div>
            </div>
            <div class="tf-credit">MARS: ESA / MPS OSIRIS TEAM · DECORATIVE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2, _spacer = st.columns([1.5, 1.5, 2])
    b1.button("ENTER RESEARCH ENVIRONMENT", key="home_enter", on_click=go_to_page,
              args=("Mission Overview",), type="primary", use_container_width=True)
    b2.button("EXPLORE THE METHODOLOGY", key="home_method", on_click=go_to_page,
              args=("Technical Deep Dive",), use_container_width=True)

    st.write("")

    # ---- Research mission ----
    st.markdown('<div class="eyebrow">01 / RESEARCH MISSION</div>', unsafe_allow_html=True)
    st.markdown("### From observation to analogue investigation")
    steps = [
        ("01", "Earth observation", "Terrestrial elevation and regional study data."),
        ("02", "Planetary elevation", "Global Earth, Mars and Moon elevation datasets."),
        ("03", "Terrain characterisation", "Slope, roughness and local relief metrics."),
        ("04", "Statistical comparison", "Measured terrain distributions compared across worlds."),
        ("05", "Candidate identification", "Connected regions extracted from terrain screening."),
        ("06", "Geospatial visualisation", "Interactive maps and research figures."),
    ]
    st.markdown(
        '<div class="tf-grid tf-grid-6">'
        + "".join(
            f'<div class="tf-step"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
            for n, t, d in steps
        )
        + '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- Planetary worlds ----
    st.markdown('<div class="eyebrow">02 / PLANETARY WORLDS</div>', unsafe_allow_html=True)
    st.markdown("### Three worlds, three reference datasets")
    worlds = [
        ("Earth", "EARTH", "NOAA ETOPO 2022", "Global topography and bathymetry.", "WGS84 + EGM2008 height"),
        ("Mars", "MARS", "NASA MOLA MEGDR", "Global median planetary topography.", "MOLA areoid"),
        ("Moon", "MOON", "NASA LRO LOLA", "Global lunar digital elevation model.", "Lunar reference sphere, radius 1,737,400 m"),
    ]
    world_cols = st.columns(3)
    for col, (key, name, source, desc, reference) in zip(world_cols, worlds):
        with col:
            st.markdown(
                f"""
                <div class="tf-world">
                    <div class="tf-world-wrap">
                        <div class="tf-world-img" style="background-image: url('{resolve_planet_image(key)}'), {WORLD_FALLBACKS[key]};" role="img" aria-label="{name} photograph"></div>
                    </div>
                    <div class="tf-world-body">
                        <div class="name">{name}</div>
                        <div class="src">{html.escape(source)}</div>
                        <div class="desc">{html.escape(desc)}<br><span class="mono">VERTICAL REFERENCE / {html.escape(reference)}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(f"Open {key} data", key=f"home_world_{key}", on_click=go_to_page,
                      args=("Moon Explorer" if key == "Moon" else "Planetary Environments",),
                      use_container_width=True)
    st.caption("Photographs: NASA Apollo 17 (Earth), ESA/MPS OSIRIS Team (Mars), Luc Viatour / Wikimedia Commons (Moon). Illustrative imagery; not project data.")

    st.divider()

    # ---- Capabilities ----
    st.markdown('<div class="eyebrow">03 / RESEARCH CAPABILITIES</div>', unsafe_allow_html=True)
    st.markdown("### What the platform does")
    capabilities = [
        ("Global planetary terrain analysis", "Earth, Mars and Moon elevation statistics and comparisons."),
        ("Regional analogue comparison", "Atacama–Jezero terrain comparison."),
        ("Terrain characterisation", "Slope, roughness and local relief measurements."),
        ("Candidate identification", "Connected candidate regions extracted from strict terrain screening."),
        ("Sensitivity screening", "Broad, moderate and strict screening scenarios."),
        ("Geological characterisation", "Geological unit coverage for candidate regions."),
        ("Spatial analysis", "Spatial cluster analysis with 4- and 8-neighbour connectivity."),
        ("Interactive mapping", "Searchable, filterable candidate maps with satellite layers."),
    ]
    st.markdown(
        '<div class="tf-grid tf-grid-4">'
        + "".join(
            f'<div class="tf-cap"><div class="t">{t}</div><div class="d">{d}</div></div>'
            for t, d in capabilities
        )
        + '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- Featured research ----
    st.markdown('<div class="eyebrow">04 / FEATURED RESEARCH</div>', unsafe_allow_html=True)
    st.markdown("### Atacama–Jezero regional demonstration")
    st.caption("Earth study environment: Atacama Desert, Chile. Mars reference: Jezero Crater. "
               "Figures below are existing project outputs; no findings are implied.")
    left, right = st.columns([1.4, 1])
    with left:
        if CANDIDATE_INVENTORY is not None and not CANDIDATE_INVENTORY.empty:
            candidate_map = create_research_candidate_map(CANDIDATE_INVENTORY)
            if candidate_map is not None:
                st_folium(candidate_map, width=None, height=400,
                          returned_objects=[], key="home_research_map")
            else:
                empty_state("Candidate coordinates unavailable",
                            "The inventory does not contain valid centroid coordinates.")
        else:
            empty_state("Research inventory unavailable",
                        "The Atacama candidate inventory could not be loaded.")
    with right:
        shown = False
        for rel, cap in [
            ("maps/terrain_comparison/earth_mars_terrain_overview.png", "Earth–Mars terrain overview"),
            ("maps/candidate_regions/atacama_candidate_regions_map.png", "Atacama candidate regions"),
        ]:
            if project_image_exists(rel):
                show_image(rel, cap)
                shown = True
        if not shown:
            empty_state("Figures unavailable", "Project visualisations were not found in the outputs directory.")

    # ---- Final CTA ----
    with st.container(border=True):
        st.markdown(
            '<div class="tf-cta-inner"><h2>THE PLANETARY RESEARCH ENVIRONMENT AWAITS.</h2>'
            '<div class="mono">INDEPENDENT RESEARCH · SCIENTIFIC OBSERVATIONS ONLY</div></div>',
            unsafe_allow_html=True,
        )
        _l, mid, _r = st.columns([1, 1, 1])
        mid.button("LAUNCH TERRAFORM", key="home_launch", on_click=go_to_page,
                   args=("Mission Overview",), type="primary", use_container_width=True)


def render_global_header(current_page):
    st.markdown(
        f"""
        <div class="tf-header">
            <div class="tf-header-brand">TERRAFORM<span>.</span></div>
            <div class="tf-header-section">{html.escape(current_page)}</div>
            <div class="tf-header-status"><i></i>PLANETARY RESEARCH SYSTEM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN PAGE ROUTER
# ============================================================

render_global_header(page)

if page == HOME_PAGE:
    home_page()

elif page == "Mission Overview":
    overview_page()

elif page == "Planetary Environments":
    planetary_environments_page()

elif page == "Moon Explorer":
    moon_explorer_page()

elif page == "Discovery Explorer":
    discovery_page()

elif page == "Global Mars Explorer":
    global_mars_page()

elif page == "Regional Analogue Study":
    comparison_page()

elif page == "Candidate Intelligence":
    candidate_page()

elif page == "Global Planetary Comparison":
    global_planetary_comparison_page()

elif page == "Research Visualisations":
    visualisations_page()

elif page == "Data Laboratory":
    data_lab_page()

elif page == "Technical Deep Dive":
    technical_deep_dive_page()


# ============================================================
# GLOBAL FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    TERRAFORM // PLANETARY ANALOG INTELLIGENCE
    <br><br>
    INDEPENDENT RESEARCH PROJECT · NASA SPACE APPS CHALLENGE 2026
    <br>
    SCIENTIFIC OBSERVATIONS ONLY · NO FABRICATED DISCOVERIES
</div>
""", unsafe_allow_html=True)
