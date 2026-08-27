from pathlib import Path

import rasterio
from rasterio.transform import from_bounds


INPUT = Path(
    "uploads/d32885b32880498cb70fc53f9446e43b.jpeg"
)

OUTPUT = Path(
    "uploads/d32885b32880498cb70fc53f9446e43b_georef.tif"
)


# Geographic footprint from the original Sentinel-2 filename.
MIN_LON = 88.3056678751018
MIN_LAT = 22.532985835658778
MAX_LON = 88.38672637729906
MAX_LAT = 22.595759959903916


with rasterio.open(INPUT) as src:

    image = src.read()

    height = src.height
    width = src.width

    transform = from_bounds(
        MIN_LON,
        MIN_LAT,
        MAX_LON,
        MAX_LAT,
        width,
        height,
    )

    profile = src.profile.copy()

    profile.update(
        {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 3,
            "dtype": image.dtype,
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "deflate",
        }
    )

    with rasterio.open(
        OUTPUT,
        "w",
        **profile,
    ) as dst:

        dst.write(image)

        dst.set_band_description(
            1,
            "Red",
        )

        dst.set_band_description(
            2,
            "Green",
        )

        dst.set_band_description(
            3,
            "Blue",
        )


print("SUCCESS")
print("Created:")
print(OUTPUT.resolve())


with rasterio.open(OUTPUT) as check:

    print()
    print("Verification:")
    print("size =", check.width, check.height)
    print("count =", check.count)
    print("crs =", check.crs)
    print("transform =", check.transform)
    print("bounds =", check.bounds)
    print("descriptions =", check.descriptions)