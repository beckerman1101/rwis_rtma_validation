"""
Half‑hourly RWIS + RTMA snapshot logger
--------------------------------------
• Fetches RTMA 2.5 km analysis closest to now (–20 min safety lag)
• Interpolates RTMA fields to every station in rwis_metadata.csv
• Pulls CoTrip road‑weather JSON, pivots sensors → columns
• Spatially pairs CoTrip points to nearest RWIS station (≤0.005° ≈ 500 m)
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
PAIR_TOLERANCE_DEG = 0.005           # KD‑tree pairing radius (≈500 m)
RECENT_MIN = 20                      # CoTrip obs must be ≤ 20 min old
RTMA_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtma/prod"
API_KEY_ENV = "COTRIP_API_KEY"       # set as secret in GitHub Actions
# ---------------------------------------------------------------------------


def _clean_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decode bytes → str and coerce numeric‑looking columns to real numbers.

    The function works *in‑place* but returns *df* for chaining.
    """
    for col in df.select_dtypes(include=["object"]).columns:
        # First decode bytes/bytearray → str
        df[col] = df[col].apply(
            lambda x: x.decode("utf‑8") if isinstance(x, (bytes, bytearray)) else x
        )
        # Then try to coerce to numeric; if not numeric it stays object/str
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


# ---------------------------------------------------------------------------
# RTMA DOWNLOAD & INTERPOLATION
# ---------------------------------------------------------------------------

def download_rtma_grib() -> str:
    """Download the latest RTMA RU 2‑D var‑grid file; return local filename."""
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y%m%d")

    # choose previous 15‑min interval ≥ 20 min ago
    buffer = now - timedelta(minutes=20)
    cycle_stamp = f"{buffer:%H}{(buffer.minute // 15) * 15:02d}"

    url = (
        f"{RTMA_BASE}/rtma2p5_ru.{day}/rtma2p5_ru.t{cycle_stamp}z.2dvarges_ndfd.grb2"
    )
    fn = "rtma.bin"

    print(f"Downloading RTMA from: {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(fn, "wb") as fh:
        for chunk in r.iter_content(8192):
            fh.write(chunk)
    return fn


def interpolate_rtma_to_points(grib_file: str, rwis: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of RTMA values at each RWIS station."""

    try:
        datasets = cfgrib.open_datasets(grib_file, indexpath=None)
        print(f"Found {len(datasets)} datasets in GRIB file")

        # Identify datasets containing variables of interest
        tcc_ds = wind_ds = thermo_ds = None
        for ds in datasets:
            vars_in_ds = list(ds.data_vars.keys())
            if "tcc" in vars_in_ds:
                tcc_ds = ds
            if any(v in vars_in_ds for v in ["wdir10", "si10", "i10fg"]):
                wind_ds = ds
            if any(v in vars_in_ds for v in ["t2m", "d2m"]):
                thermo_ds = ds

        # Fallback assignment if not explicitly located
        tcc_ds = tcc_ds or datasets[0]
        wind_ds = wind_ds or datasets[-1]
        thermo_ds = thermo_ds or datasets[-1]

        def _safe_extract(ds_: xr.Dataset, var: str, fill: Any = np.nan) -> xr.DataArray:
            if var in ds_.data_vars:
                return ds_[var]
            print(f"Warning: {var} not found in GRIB; filling NaNs")
            first = list(ds_.data_vars.values())[0]
            return first * np.nan + fill

        tcc = _safe_extract(tcc_ds, "tcc")
        tmp_k = _safe_extract(thermo_ds, "t2m", 273.15)
        dpt_k = _safe_extract(thermo_ds, "d2m", 273.15)
        wdir = _safe_extract(wind_ds, "wdir10", 0)
        wgust = _safe_extract(wind_ds, "i10fg", 0)
        wspd = _safe_extract(wind_ds, "si10", 0)

        # Kelvin → Fahrenheit
        tmp_f = (tmp_k - 273.15) * 9 / 5 + 32
        dpt_f = (dpt_k - 273.15) * 9 / 5 + 32

        # Coordinate handling
        lat = tcc.latitude.values
        lon = tcc.longitude.values
        if lon.max() > 180:
            lon = lon - 360

        tree = cKDTree(np.column_stack((lat.ravel(), lon.ravel())))

        records = []
        for _, stn in rwis.iterrows():
            _, flat_idx = tree.query([stn.lat, stn.lon])
            iy, ix = np.unravel_index(flat_idx, lat.shape)

            def _val(arr: xr.DataArray, xi: int, yi: int):
                try:
                    return float(arr.values[yi, xi])
                except Exception:
                    return np.nan

            records.append(
                {
                    "station_id": stn.stid,
                    "station_name": stn.station_name,
                    "lat": stn.lat,
                    "lon": stn.lon,
                    "cloud_cover": _val(tcc, ix, iy),
                    "rtma_temps": _val(tmp_f, ix, iy),
                    "rtma_dp": _val(dpt_f, ix, iy),
                    "rtma_wind_direction": _val(wdir, ix, iy),
                    "rtma_wind_gust": _val(wgust, ix, iy),
                    "rtma_wind_speed": _val(wspd, ix, iy),
                }
            )

        return pd.DataFrame.from_records(records)

    except Exception as exc:
        print(f"Error processing GRIB file: {exc}")
        return pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "lat",
                "lon",
                "cloud_cover",
                "rtma_temps",
                "rtma_dp",
                "rtma_wind_direction",
                "rtma_wind_gust",
                "rtma_wind_speed",
            ]
        )


# ---------------------------------------------------------------------------
# COTRIP FETCH & CLEANING
# ---------------------------------------------------------------------------

def fetch_cotrip(api_key: str) -> pd.DataFrame:
    """Download CoTrip JSON, pivot sensors → columns, *clean* the result."""

    try:
        r = requests.get(f"{COTRIP_URL}?apiKey={api_key}", timeout=60)
        r.raise_for_status()
        data = r.json()

        if "features" not in data:
            print("Warning: no 'features' key in CoTrip response")
            return pd.DataFrame()

        df = pd.json_normalize(data["features"])
        if df.empty:
            print("Warning: CoTrip returned zero features")
            return pd.DataFrame()

        # explode sensors list
        if "properties.sensors" not in df.columns:
            print("Warning: no sensors field in CoTrip data")
            return pd.DataFrame()

        df = df.explode("properties.sensors").dropna(subset=["properties.sensors"])
        sensors = (
            pd.json_normalize(df["properties.sensors"])
            .rename(columns={"type": "sensor_type"})
            .drop(columns=[col for col in ["unit"] if col in df.columns])
        )
        meta = df.drop(columns=["properties.sensors"]).reset_index(drop=True)
        merged = pd.concat([meta, sensors], axis=1)

        # coordinates
        coords = merged["geometry.coordinates"].apply(
            lambda x: pd.Series(x) if isinstance(x, list) and len(x) >= 2 else pd.Series([np.nan, np.nan])
        )
        merged[["lon", "lat"]] = coords[[0, 1]]

        # pivot sensors into columns
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

        # timestamp parsing
        pivot["properties.lastUpdated"] = pd.to_datetime(
            pivot["properties.lastUpdated"], utc=True, errors="coerce"
        )

        # decode bytes + convert numerics
        return _clean_object_columns(pivot)

    except Exception as exc:
        print(f"Error fetching CoTrip data: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# PAIRING & MERGING
# ---------------------------------------------------------------------------

def pair_and_merge(rwis_pts: pd.DataFrame, cotrip: pd.DataFrame) -> pd.DataFrame:
    """Spatial/temporal join: nearest RWIS within tolerance + recent timestamp."""

    if rwis_pts.empty or cotrip.empty:
        print("Warning: empty inputs for pairing")
        return pd.DataFrame()

    # coordinates must be finite
    rwis_pts = rwis_pts.dropna(subset=["lat", "lon"]).copy()
    cotrip = cotrip.dropna(subset=["lat", "lon", "properties.lastUpdated"]).copy()
    if rwis_pts.empty or cotrip.empty:
        return pd.DataFrame()

    tree = cKDTree(rwis_pts[["lat", "lon"]].values)
    dist, idx = tree.query(cotrip[["lat", "lon"]].values, distance_upper_bound=PAIR_TOLERANCE_DEG)

    recent_cut = cotrip["properties.lastUpdated"].max() - timedelta(minutes=RECENT_MIN)
    mask = (dist != np.inf) & (cotrip["properties.lastUpdated"] >= recent_cut)

    rwis_match = rwis_pts.iloc[idx].reset_index(drop=True)
    rwis_match.loc[dist == np.inf, :] = np.nan
    combined = pd.concat([cotrip.reset_index(drop=True), rwis_match.add_prefix("rwis_")], axis=1)

    # drop sensor/metadata columns we don't care about
    drop_cols = [
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
    combined = combined.drop(columns=[c for c in drop_cols if c in combined.columns])

    paired = combined[mask].reset_index(drop=True)
    print(f"Paired {len(paired)} stations successfully")

    return _clean_object_columns(paired)


# ---------------------------------------------------------------------------
# SNAPSHOT BUILDING
# ---------------------------------------------------------------------------

def build_snapshot(api_key: str) -> xr.Dataset:
    if not os.path.exists(RWIS_META_CSV):
        raise FileNotFoundError(f"RWIS metadata file not found: {RWIS_META_CSV}")

    rwis_meta = pd.read_csv(RWIS_META_CSV)
    print(f"Loaded {len(rwis_meta)} RWIS stations")

    grib_file = download_rtma_grib()
    rtma_df = interpolate_rtma_to_points(grib_file, rwis_meta)
    cotrip_df = fetch_cotrip(api_key)
    merged_df = pair_and_merge(rtma_df, cotrip_df)

    if merged_df.empty:
        raise ValueError("Merged dataframe is empty — no data to log.")

    merged_df["time"] = pd.Timestamp.utcnow()
    station_id_col = next((c for c in merged_df.columns if "station_id" in c.lower()), None)
    if station_id_col is None:
        merged_df["station_index"] = range(len(merged_df))
        station_id_col = "station_index"

    merged_df = merged_df.set_index(["time", station_id_col])

    # final numeric/bytes cleaning before xarray conversion
    merged_df = _clean_object_columns(merged_df.reset_index()).set_index(["time", station_id_col])
    ds = xr.Dataset.from_dataframe(merged_df)

    # promote coords if present
    for coord in ["station_name", "lat", "lon"]:
        match = [v for v in ds.data_vars if coord in v.lower()]
        if match:
            ds = ds.set_coords(match[0])
    return ds


# ---------------------------------------------------------------------------
# DAILY APPEND
# ---------------------------------------------------------------------------

def append_daily(ds: xr.Dataset) -> None:
    if "time" not in ds.dims and "time" not in ds.coords:
        raise ValueError("Dataset lacks a 'time' dimension/coordinate")

    if "time" in ds.indexes and getattr(ds.indexes["time"], "tz", None):
        ds = ds.assign_coords(time=ds.indexes["time"].tz_localize(None))

    fname = f"rwis_rtma_{pd.Timestamp.utcnow():%Y%m%d}.nc"

    if os.path.exists(fname):
        with xr.open_dataset(fname) as existing:
            ds = xr.concat([existing, ds], dim="time")
        print(f"Appended to existing {fname}")

    # Define encoding only for *string* vars that remain object dtype
    encoding = {
        var: {"dtype": "S64"}
        for var in ds.data_vars
        if ds[var].dtype.kind == "O"
    }

    ds.to_netcdf(fname, mode="w", encoding=encoding)
    print(f"Saved snapshot to {fname}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv(API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(f"Set CoTrip API key in env var {API_KEY_ENV}")

    try:
        ds = build_snapshot(api_key)
        append_daily(ds)
        print(f"[{datetime.utcnow():%Y‑%m‑%d %H:%M}] snapshot appended successfully.")
    except Exception as exc:
        print(f"Error during execution: {exc}")
        raise


if __name__ == "__main__":
    main()
