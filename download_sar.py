from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import planetary_computer
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds


# ============================================================
# YOUR SENTINEL-2 FOOTPRINT
# ============================================================

MIN_LON = 88.3056678751018
MIN_LAT = 22.532985835658778
MAX_LON = 88.38672637729906
MAX_LAT = 22.595759959903916

BBOX = [
    MIN_LON,
    MIN_LAT,
    MAX_LON,
    MAX_LAT,
]

TARGET_DATE = datetime(
    2026,
    6,
    17,
    tzinfo=timezone.utc,
)

OUTPUT = Path(
    "uploads/sentinel1_kolkata_vv_rtc.tif"
)


def main():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Opening Planetary Computer STAC..."
    )

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    print(
        "Searching Sentinel-1 RTC..."
    )

    search = catalog.search(
        collections=[
            "sentinel-1-rtc"
        ],
        bbox=BBOX,
        datetime=(
            "2026-06-01T00:00:00Z/"
            "2026-06-30T23:59:59Z"
        ),
    )

    items = list(
        search.items()
    )

    print(
        "Matching RTC scenes:",
        len(items),
    )

    if not items:

        raise RuntimeError(
            "No Sentinel-1 RTC scene was found "
            "for the requested area and date."
        )

    items.sort(
        key=lambda item: (
            abs(
                (
                    item.datetime
                    - TARGET_DATE
                ).total_seconds()
            )
            if item.datetime
            else float("inf")
        )
    )

    item = items[0]

    print()
    print(
        "Selected RTC scene:"
    )
    print(
        item.id
    )

    print(
        "Acquisition:"
    )
    print(
        item.datetime
    )

    print(
        "Available assets:"
    )
    print(
        list(
            item.assets.keys()
        )
    )

    if "vv" not in item.assets:

        raise RuntimeError(
            "The selected RTC scene does not "
            "contain a VV asset."
        )

    asset = item.assets["vv"]

    print()
    print(
        "Opening RTC VV asset..."
    )

    with rasterio.open(
        asset.href
    ) as src:

        print(
            "Full size:",
            src.width,
            "x",
            src.height,
        )

        print(
            "CRS:",
            src.crs,
        )

        print(
            "Transform:",
            src.transform,
        )

        print(
            "Bounds:",
            src.bounds,
        )

        if src.crs is None:

            raise RuntimeError(
                "RTC asset has no CRS. "
                "Refusing to create a fake "
                "georeferenced SAR product."
            )

        # ----------------------------------------------------
        # Convert the Sentinel-2 lon/lat footprint
        # into the RTC raster CRS.
        # ----------------------------------------------------

        transformer = Transformer.from_crs(
            "EPSG:4326",
            src.crs,
            always_xy=True,
        )

        x1, y1 = transformer.transform(
            MIN_LON,
            MIN_LAT,
        )

        x2, y2 = transformer.transform(
            MAX_LON,
            MAX_LAT,
        )

        left = min(
            x1,
            x2,
        )

        right = max(
            x1,
            x2,
        )

        bottom = min(
            y1,
            y2,
        )

        top = max(
            y1,
            y2,
        )

        print()
        print(
            "AOI in RTC CRS:"
        )

        print(
            left,
            bottom,
            right,
            top,
        )

        # ----------------------------------------------------
        # Make sure requested AOI intersects the raster.
        # ----------------------------------------------------

        raster_left = src.bounds.left
        raster_right = src.bounds.right
        raster_bottom = src.bounds.bottom
        raster_top = src.bounds.top

        intersects = (
            right > raster_left
            and left < raster_right
            and top > raster_bottom
            and bottom < raster_top
        )

        if not intersects:

            raise RuntimeError(
                "The selected Sentinel-1 RTC scene "
                "does not overlap the Sentinel-2 AOI."
            )

        left = max(
            left,
            raster_left,
        )

        right = min(
            right,
            raster_right,
        )

        bottom = max(
            bottom,
            raster_bottom,
        )

        top = min(
            top,
            raster_top,
        )

        window = from_bounds(
            left,
            bottom,
            right,
            top,
            transform=src.transform,
        )

        window = window.round_offsets().round_lengths()

        data = src.read(
            1,
            window=window,
        )

        transform = src.window_transform(
            window
        )

        profile = src.profile.copy()

        profile.update(
            {
                "driver": "GTiff",
                "height": data.shape[0],
                "width": data.shape[1],
                "count": 1,
                "transform": transform,
                "crs": src.crs,
                "compress": "deflate",
            }
        )

        with rasterio.open(
            OUTPUT,
            "w",
            **profile,
        ) as dst:

            dst.write(
                data,
                1,
            )

            dst.set_band_description(
                1,
                "VV",
            )

    print()
    print(
        "======================================"
    )
    print(
        "SUCCESS"
    )
    print(
        "Georeferenced SAR written to:"
    )
    print(
        OUTPUT.resolve()
    )
    print(
        "======================================"
    )


if __name__ == "__main__":
    main()