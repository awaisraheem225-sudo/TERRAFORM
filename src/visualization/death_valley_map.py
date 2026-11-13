
from pathlib import Path

import numpy as np
import rasterio
import folium

from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)
from rasterio.transform import array_bounds
from rasterio import Affine
from matplotlib import colormaps


ROOT = Path(__file__).resolve().parents[2]

RASTER_DIR = (
    ROOT / "data" / "processed" / "earth" / "death_valley"
)

TERRAIN_LAYERS = {
    "Elevation": {
        "file": "death_valley_dem_projected_utm11n.tif",
        "palette": "terrain",
        "unit": "metres",
        "description": "Digital elevation model",
    },
    "Slope": {
        "file": "death_valley_slope_deg.tif",
        "palette": "inferno",
        "unit": "degrees",
        "description": "Terrain slope",
    },
    "Roughness": {
        "file": "death_valley_roughness_std_3x3_m.tif",
        "palette": "magma",
        "unit": "metres",
        "description": "3 × 3 neighbourhood elevation roughness",
    },
    "Local relief": {
        "file": "death_valley_local_relief_3x3_m.tif",
        "palette": "viridis",
        "unit": "metres",
        "description": "3 × 3 neighbourhood local relief",
    },
}


def _prepare_raster(path, max_dimension=1200):
    """Reproject a terrain raster to geographic coordinates for web mapping."""

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {path.name}")

        source = src.read(1).astype(np.float32)
        source_nodata = src.nodata

        valid = np.isfinite(source)

        if source_nodata is not None:
            valid &= source != source_nodata

        if not np.any(valid):
            raise ValueError(f"No valid raster cells: {path.name}")

        dst_crs = "EPSG:4326"

        base_transform, base_width, base_height = (
            calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,
            )
        )

        scale = min(
            1.0,
            max_dimension / max(base_width, base_height),
        )

        dst_width = max(1, int(base_width * scale))
        dst_height = max(1, int(base_height * scale))

        dst_transform = base_transform * Affine.scale(
            base_width / dst_width,
            base_height / dst_height,
        )

        destination = np.full(
            (dst_height, dst_width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=source_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    west_south_north_east = array_bounds(
        dst_height,
        dst_width,
        dst_transform,
    )

    west, south, east, north = west_south_north_east

    return destination, [[south, west], [north, east]]


def create_death_valley_map(layer="Elevation"):
    """Create an interactive Death Valley terrain map."""

    if layer not in TERRAIN_LAYERS:
        raise ValueError(f"Unsupported terrain layer: {layer}")

    config = TERRAIN_LAYERS[layer]
    raster_path = RASTER_DIR / config["file"]

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Death Valley raster not found: {raster_path}"
        )

    values, bounds = _prepare_raster(raster_path)

    valid = np.isfinite(values)

    if not np.any(valid):
        raise ValueError("No valid pixels available for map rendering.")

    low, high = np.nanpercentile(values, [2, 98])

    if high <= low:
        high = low + 1

    normalized = np.clip(
        (values - low) / (high - low),
        0,
        1,
    )

    rgba = colormaps[config["palette"]](normalized)
    rgba[..., 3] = np.where(valid, 0.88, 0.0)

    rgba = (rgba * 255).astype(np.uint8)

    south_west, north_east = bounds

    centre = [
        (south_west[0] + north_east[0]) / 2,
        (south_west[1] + north_east[1]) / 2,
    ]

    map_object = folium.Map(
        location=centre,
        zoom_start=10,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        "OpenStreetMap",
        name="Street map",
        show=True,
    ).add_to(map_object)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri",
        name="Satellite imagery",
        show=False,
    ).add_to(map_object)

    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=bounds,
        opacity=0.88,
        name=f"{layer} terrain",
        interactive=True,
        cross_origin=False,
        zindex=2,
    ).add_to(map_object)

    folium.LayerControl(
        collapsed=False,
    ).add_to(map_object)

    map_object.fit_bounds(bounds)

    colormap = colormaps[config["palette"]]

    legend_steps = np.linspace(low, high, 6)

    legend_rows = ""

    for value in legend_steps:
        color = colormap(
            (value - low) / (high - low)
        )
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255),
        )

        legend_rows += (
            f'<div style="display:flex;align-items:center;'
            f'gap:7px;margin:4px 0;">'
            f'<span style="background:{hex_color};width:15px;'
            f'height:11px;display:inline-block;"></span>'
            f'<span>{value:,.1f} {config["unit"]}</span>'
            f'</div>'
        )

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 25px;
        left: 25px;
        z-index: 9999;
        background: rgba(15, 23, 32, 0.94);
        color: white;
        padding: 12px 15px;
        border-radius: 8px;
        border: 1px solid #52616b;
        font-family: Arial, sans-serif;
        font-size: 11px;
        min-width: 145px;
    ">
        <b style="font-size:12px;">DEATH VALLEY · {layer.upper()}</b>
        <div style="margin:5px 0 8px;color:#b7c5ce;">
            {config["description"]}
        </div>
        {legend_rows}
        <div style="margin-top:7px;color:#91a1ac;">
            Copernicus DEM GLO-30 · ~30 m source
        </div>
    </div>
    """

    map_object.get_root().html.add_child(
        folium.Element(legend_html)
    )

    return map_object
