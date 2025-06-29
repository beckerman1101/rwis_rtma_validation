"""
Half‑hourly RWIS + RTMA snapshot logger
--------------------------------------More actions
• Fetches RTMA 2.5 km analysis closest to now (–20 min safety lag)
• Interpolates RTMA fields to every station in rwis_metadata.csv
• Pulls CoTrip road‑weather JSON, pivots sensors → columns
• Spatially pairs CoTrip points to nearest RWIS station (≤0.005° ≈ 500 m)
• Merges and trims columns, then appends to a daily NetCDF file
"""

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import xarray as xr
import requests
from scipy.spatial import cKDTree
import cfgrib

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
RWIS_META_CSV = "rwis_metadata.csv"  # local metadata file
COTRIP_URL = "https://data.cotrip.org/api/v1/weatherStations"
PAIR_TOLERANCE_DEG = 0.005           # KD‑tree pairing radius (≈500 m)
RECENT_MIN = 20                      # CoTrip obs must be ≤ 20 min old
RTMA_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtma/prod"
API_KEY_ENV = "COTRIP_API_KEY"       # set as secret in GitHub Actions
# ---------------------------------------------------------------------------


def download_rtma_grib() -> str:
    """Download the latest RTMA RU 2‑D var‑grid file; return local filename."""
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y%m%d")

    # choose previous 15‑min interval ≥ 20 min ago
    buffer = now - timedelta(minutes=20)
    cycle_stamp = f"{buffer:%H}{(buffer.minute // 15) * 15:02d}"

    url = f"{RTMA_BASE}/rtma2p5_ru.{day}/rtma2p5_ru.t{cycle_stamp}z.2dvarges_ndfd.grb2"
    fn = "rtma.bin"

    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(fn, "wb") as fh:
        for chunk in r.iter_content(8192):
            fh.write(chunk)
    return fn


def interpolate_rtma_to_points(grib_file: str, rwis: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of RTMA values at each RWIS station."""
    datasets = cfgrib.open_datasets(grib_file, indexpath=None)
    ds0 = datasets[0]  # total cloud cover
    ds2 = datasets[2]  # wind
    ds3 = datasets[3]   # tcc, wind, thermo

    tcc = ds0.tcc
    tmp_f = (ds3.t2m - 273.15) * 9 / 5 + 32
    dpt_f = (ds3.d2m - 273.15) * 9 / 5 + 32
    wdir = ds2.wdir10
    wgust = ds2.i10fg
    wspd = ds2.si10

    lat, lon = tcc.latitude.values, tcc.longitude.values - 360
    tree = cKDTree(np.column_stack((lat.ravel(), lon.ravel())))

    records = []
    for _, stn in rwis.iterrows():
        _, flat = tree.query([stn.lat, stn.lon])
        iy, ix = np.unravel_index(flat, lat.shape)
        records.append(
            {
                "station_id": stn.stid,
                "station_name": stn.station_name,
                "lat": stn.lat,
                "lon": stn.lon,
                "cloud_cover": float(tcc.values[iy, ix]),
                "rtma_temps": float(tmp_f.values[ix, iy]),
                "rtma_dp": float(dpt_f.values[ix, iy]),
                "rtma_wind_direction": float(wdir.values[ix, iy]),
                "rtma_wind_gust": float(wgust.values[ix, iy]),
                "rtma_wind_speed": float(wspd.values[ix, iy]),
            }
        )
    return pd.DataFrame.from_records(records)


def fetch_cotrip(api_key: str) -> pd.DataFrame:
    """Download CoTrip JSON, pivot sensors → columns, parse timestamps."""
    r = requests.get(f"{COTRIP_URL}?apiKey={api_key}", timeout=60)
    r.raise_for_status()
    df = pd.json_normalize(r.json()["features"]).explode("properties.sensors")

    sensors = pd.json_normalize(df["properties.sensors"]).rename(columns={"type": "sensor_type"})
    meta = df.drop(columns=["properties.sensors"]).reset_index(drop=True)
    merged = pd.concat([meta, sensors], axis=1)

    merged[["lon", "lat"]] = merged["geometry.coordinates"].apply(pd.Series)

    pivot = (
        merged.pivot_table(
            index=[
                "properties.name",
                "lon",
                "lat",
                "geometry.type",
                "geometry.srid",
                "properties.lastUpdated",
                "properties.nativeId",
                "properties.direction",
            ],
            columns="sensor_type",
            values="currentReading",
            aggfunc="first",
        )
        .reset_index()
        .copy()
    )

    pivot["properties.lastUpdated"] = pd.to_datetime(
        pivot["properties.lastUpdated"], utc=True, errors="coerce"
    )
    return pivot


def pair_and_merge(rwis_pts: pd.DataFrame, cotrip: pd.DataFrame) -> pd.DataFrame:
    """Spatial/temporal join: nearest RWIS within tolerance + recent timestamp."""
    rwis_xy = rwis_pts[["lat", "lon"]].to_numpy()
    cotrip_xy = cotrip[["lat", "lon"]].to_numpy()

    tree = cKDTree(rwis_xy)
    dist, idx = tree.query(cotrip_xy, distance_upper_bound=PAIR_TOLERANCE_DEG)

    recent_cut = cotrip["properties.lastUpdated"].max() - timedelta(minutes=RECENT_MIN)
    mask = (dist != np.inf) & (cotrip["properties.lastUpdated"] >= recent_cut)

    rwis_match = pd.DataFrame(index=cotrip.index, columns=rwis_pts.columns)
    rwis_match.loc[dist != np.inf] = rwis_pts.iloc[idx[dist != np.inf]].values

    combined = pd.concat([cotrip.reset_index(drop=True), rwis_match.add_prefix("rwis_")], axis=1)

    # drop CoTrip fields you said you don't want
    drop = [
        "precipitation accumulation 12hr",
        "precipitation accumulation 1hr",
        "precipitation accumulation 24hr",
        "precipitation accumulation 3hr",
        "precipitation accumulation 6hr",
        "precipitation end time",
        "precipitation present",
        "precipitation rate",
        "precipitation situation",
        "precipitation start time",
        "geometry.type",
        "properties.nativeId",
        "road surface salinity",
        "road subsurface sensor error",
        "road subsurface temperature",
    ]
    combined = combined.drop(columns=[c for c in drop if c in combined.columns])

    return combined[mask].reset_index(drop=True)


def build_snapshot(api_key: str) -> xr.Dataset:
    """Full pipeline: RWIS meta → RTMA interp → CoTrip merge → xarray.Dataset."""
    rwis_meta = pd.read_csv(RWIS_META_CSV)

    grib = download_rtma_grib()
    rtma_df = interpolate_rtma_to_points(grib, rwis_meta)

    cotrip_df = fetch_cotrip(api_key)
    merged_df = pair_and_merge(rtma_df, cotrip_df)

    if merged_df.empty:
        raise ValueError("Merged dataframe is empty — no data to log.")

    # Add UTC time
    merged_df["time"] = pd.Timestamp.utcnow()

    # Set time + station ID as index for dimensions
    merged_df = merged_df.set_index(["time", "rwis_station_id"])

    # Convert to xarray Dataset
    ds = xr.Dataset.from_dataframe(merged_df)

    # Reset them to be dimensions, not coordinates
    ds = ds.reset_coords(["time", "station_id"])

    # Promote useful metadata as coordinates
    for coord in ["station_name", "lat", "lon"]:
        if coord in ds:
            ds = ds.set_coords(coord)

    return ds


def append_daily(ds: xr.Dataset) -> None:
    """Append snapshot to daily NetCDF; create new file if absent."""
    if "time" not in ds.dims and "time" not in ds.coords:
        raise ValueError("Dataset has no 'time' dimension or coordinate.")

    # Strip timezone if present
    if "time" in ds.indexes and hasattr(ds.indexes["time"], "tz"):
        ds = ds.assign_coords(time=ds.indexes["time"].tz_localize(None))

    fname = f"rwis_rtma_{pd.Timestamp.utcnow():%Y%m%d}.nc"

    if os.path.exists(fname):
        with xr.open_dataset(fname) as existing:
            ds = xr.concat([existing, ds], dim="time")

    ds.to_netcdf(fname, mode="w")


def main() -> None:
    api_key = os.getenv(API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(f"Set CoTrip API key in env var {API_KEY_ENV}")

    ds = build_snapshot(api_key)
    append_daily(ds)
    print(f"[{datetime.utcnow():%Y-%m-%d %H:%M}] snapshot appended.")


if __name__ == "__main__":
    main()
