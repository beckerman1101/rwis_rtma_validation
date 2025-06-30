"""
Half‑hourly RWIS + RTMA snapshot loggerAdd commentMore actions
--------------------------------------More actions
• Fetches RTMA 2.5 km analysis closest to now (–20 min safety lag)
Half‑hourly RWIS + RTMA snapshot logger
--------------------------------------
• Fetches RTMA 2.5 km analysis closest to now (–20 min safety lag)
• Interpolates RTMA fields to every station in rwis_metadata.csv
• Pulls CoTrip road‑weather JSON, pivots sensors → columns
• Spatially pairs CoTrip points to nearest RWIS station (≤0.005° ≈ 500 m)
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
PAIR_TOLERANCE_DEG = 0.005           # KD‑tree pairing radius (≈500 m)
RECENT_MIN = 20                      # CoTrip obs must be ≤ 20 min old
PAIR_TOLERANCE_DEG = 0.005           # KD‑tree pairing radius (≈500 m)
RECENT_MIN = 20                      # CoTrip obs must be ≤ 20 min old
RTMA_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtma/prod"
API_KEY_ENV = "COTRIP_API_KEY"       # set as secret in GitHub Actions
# ---------------------------------------------------------------------------


def download_rtma_grib() -> str:
    """Download the latest RTMA RU 2‑D var‑grid file; return local filename."""
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y%m%d")

    # choose previous 15‑min interval ≥ 20 min ago
    # choose previous 15‑min interval ≥ 20 min ago
    buffer = now - timedelta(minutes=20)
    cycle_stamp = f"{buffer:%H}{(buffer.minute // 15) * 15:02d}"

    url = f"{RTMA_BASE}/rtma2p5_ru.{day}/rtma2p5_ru.t{cycle_stamp}z.2dvarges_ndfd.grb2"
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
    try:
        datasets = cfgrib.open_datasets(grib_file, indexpath=None)
        print(f"Found {len(datasets)} datasets in GRIB file")
        
        # Check available datasets and their variables
        for i, ds in enumerate(datasets):
            print(f"Dataset {i} variables: {list(ds.data_vars.keys())}")
        
        # More flexible dataset selection - look for required variables
        tcc_ds = None
        wind_ds = None
        thermo_ds = None
        
        for i, ds in enumerate(datasets):
            vars_in_ds = list(ds.data_vars.keys())
            if 'tcc' in vars_in_ds:
                tcc_ds = ds
            if any(var in vars_in_ds for var in ['wdir10', 'si10', 'i10fg']):
                wind_ds = ds
            if any(var in vars_in_ds for var in ['t2m', 'd2m']):
                thermo_ds = ds
        
        # Use first dataset if specific ones not found
        if tcc_ds is None:
            tcc_ds = datasets[0]
        if wind_ds is None:
            wind_ds = datasets[-1] if len(datasets) > 1 else datasets[0]
        if thermo_ds is None:
            thermo_ds = datasets[-1] if len(datasets) > 1 else datasets[0]
        
        # Extract variables with error handling
        def safe_extract(ds, var_name, default_val=np.nan):
            if var_name in ds.data_vars:
                return ds[var_name]
            else:
                print(f"Warning: {var_name} not found in dataset")
                # Return array of NaNs with same shape as first available variable
                first_var = list(ds.data_vars.keys())[0]
                return ds[first_var] * np.nan
        
        tcc = safe_extract(tcc_ds, 'tcc', 0)
        tmp_k = safe_extract(thermo_ds, 't2m', 273.15)
        dpt_k = safe_extract(thermo_ds, 'd2m', 273.15)
        wdir = safe_extract(wind_ds, 'wdir10', 0)
        wgust = safe_extract(wind_ds, 'i10fg', 0)
        wspd = safe_extract(wind_ds, 'si10', 0)
        
        # Convert to Fahrenheit
        tmp_f = (tmp_k - 273.15) * 9 / 5 + 32
        dpt_f = (dpt_k - 273.15) * 9 / 5 + 32

        # Get coordinates - handle longitude adjustment for US
        lat = tcc.latitude.values
        lon = tcc.longitude.values
        if lon.max() > 180:
            lon = lon - 360
            
        tree = cKDTree(np.column_stack((lat.ravel(), lon.ravel())))

        records = []
        for _, stn in rwis.iterrows():
            _, flat = tree.query([stn.lat, stn.lon])
            iy, ix = np.unravel_index(flat, lat.shape)
            
            # Safe value extraction with bounds checking
            def safe_value(arr, ix, iy):
                try:
                    if hasattr(arr, 'values'):
                        return float(arr.values[iy, ix])
                    else:
                        return float(arr[iy, ix])
                except (IndexError, ValueError):
                    return np.nan
            
            records.append({
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
                "cloud_cover": safe_value(tcc, ix, iy),
                "rtma_temps": safe_value(tmp_f, ix, iy),
                "rtma_dp": safe_value(dpt_f, ix, iy),
                "rtma_wind_direction": safe_value(wdir, ix, iy),
                "rtma_wind_gust": safe_value(wgust, ix, iy),
                "rtma_wind_speed": safe_value(wspd, ix, iy),
            })
        
        return pd.DataFrame.from_records(records)
        
    except Exception as e:
        print(f"Error processing GRIB file: {e}")
        # Return empty DataFrame with expected structure
        return pd.DataFrame(columns=[
            "station_id", "station_name", "lat", "lon", "cloud_cover",
            "rtma_temps", "rtma_dp", "rtma_wind_direction", 
            "rtma_wind_gust", "rtma_wind_speed"
        ])


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
    try:
        r = requests.get(f"{COTRIP_URL}?apiKey={api_key}", timeout=60)
        r.raise_for_status()
        
        json_data = r.json()
        if "features" not in json_data:
            print("Warning: No 'features' key in CoTrip response")
            return pd.DataFrame()
            
        df = pd.json_normalize(json_data["features"])
        
        if df.empty:
            print("Warning: Empty features list from CoTrip")
            return pd.DataFrame()
            
        # Check if sensors data exists
        if "properties.sensors" not in df.columns:
            print("Warning: No sensor data in CoTrip response")
            return pd.DataFrame()
            
        df = df.explode("properties.sensors")
        
        # Handle case where sensors might be None
        df = df.dropna(subset=["properties.sensors"])
        
        if df.empty:
            print("Warning: No valid sensor data after explosion")
            return pd.DataFrame()
        
        sensors = pd.json_normalize(df["properties.sensors"]).rename(columns={"type": "sensor_type"})
        meta = df.drop(columns=["properties.sensors"]).reset_index(drop=True)
        merged = pd.concat([meta, sensors], axis=1)

        # Safe coordinate extraction
        if "geometry.coordinates" in merged.columns:
            coords = merged["geometry.coordinates"].apply(
                lambda x: pd.Series(x) if isinstance(x, list) and len(x) >= 2 else pd.Series([None, None])
            )
            merged[["lon", "lat"]] = coords[[0, 1]]
        else:
            print("Warning: No geometry coordinates in CoTrip data")
            merged[["lon", "lat"]] = [None, None]

        # Safe pivot with error handling
        try:
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
        except Exception as e:
            print(f"Error during pivot: {e}")
            return pd.DataFrame()

        # Safe timestamp parsing
        if "properties.lastUpdated" in pivot.columns:
            pivot["properties.lastUpdated"] = pd.to_datetime(
                pivot["properties.lastUpdated"], utc=True, errors="coerce"
            )
        
        return pivot
        
    except Exception as e:
        print(f"Error fetching CoTrip data: {e}")
        return pd.DataFrame()


def pair_and_merge(rwis_pts: pd.DataFrame, cotrip: pd.DataFrame) -> pd.DataFrame:
    """Spatial/temporal join: nearest RWIS within tolerance + recent timestamp."""
    rwis_xy = rwis_pts[["lat", "lon"]].to_numpy()
    cotrip_xy = cotrip[["lat", "lon"]].to_numpy()
    if rwis_pts.empty or cotrip.empty:
        print("Warning: Empty input DataFrames for pairing")
        return pd.DataFrame()
    
    # Check for required columns
    required_rwis_cols = ["lat", "lon"]
    required_cotrip_cols = ["lat", "lon", "properties.lastUpdated"]
    
    missing_rwis = [col for col in required_rwis_cols if col not in rwis_pts.columns]
    missing_cotrip = [col for col in required_cotrip_cols if col not in cotrip.columns]
    
    if missing_rwis:
        print(f"Missing RWIS columns: {missing_rwis}")
        return pd.DataFrame()
    if missing_cotrip:
        print(f"Missing CoTrip columns: {missing_cotrip}")
        return pd.DataFrame()
    
    # Remove rows with NaN coordinates
    rwis_clean = rwis_pts.dropna(subset=["lat", "lon"]).copy()
    cotrip_clean = cotrip.dropna(subset=["lat", "lon", "properties.lastUpdated"]).copy()
    
    if rwis_clean.empty or cotrip_clean.empty:
        print("Warning: No valid coordinates after cleaning")
        return pd.DataFrame()
    
    rwis_xy = rwis_clean[["lat", "lon"]].to_numpy()
    cotrip_xy = cotrip_clean[["lat", "lon"]].to_numpy()

    tree = cKDTree(rwis_xy)
    dist, idx = tree.query(cotrip_xy, distance_upper_bound=PAIR_TOLERANCE_DEG)

    recent_cut = cotrip["properties.lastUpdated"].max() - timedelta(minutes=RECENT_MIN)
    mask = (dist != np.inf) & (cotrip["properties.lastUpdated"] >= recent_cut)
    # Safe timestamp filtering
    try:
        recent_cut = cotrip_clean["properties.lastUpdated"].max() - timedelta(minutes=RECENT_MIN)
        mask = (dist != np.inf) & (cotrip_clean["properties.lastUpdated"] >= recent_cut)
    except Exception as e:
        print(f"Error in timestamp filtering: {e}")
        mask = dist != np.inf

    rwis_match = pd.DataFrame(index=cotrip.index, columns=rwis_pts.columns)
    rwis_match.loc[dist != np.inf] = rwis_pts.iloc[idx[dist != np.inf]].values
    rwis_match = pd.DataFrame(index=cotrip_clean.index, columns=rwis_clean.columns)
    valid_matches = dist != np.inf
    if valid_matches.any():
        rwis_match.loc[valid_matches] = rwis_clean.iloc[idx[valid_matches]].values

    combined = pd.concat([cotrip.reset_index(drop=True), rwis_match.add_prefix("rwis_")], axis=1)
    combined = pd.concat([cotrip_clean.reset_index(drop=True), rwis_match.add_prefix("rwis_")], axis=1)

    # drop CoTrip fields you said you don't want
    # Drop unwanted columns
    drop = [
        "precipitation accumulation 12hr",
        "precipitation accumulation 1hr",
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
    result = combined[mask].reset_index(drop=True)
    print(f"Paired {len(result)} stations successfully")
    return result


def build_snapshot(api_key: str) -> xr.Dataset:
    """Full pipeline: RWIS meta → RTMA interp → CoTrip merge → xarray.Dataset."""
    # Check if metadata file exists
    if not os.path.exists(RWIS_META_CSV):
        raise FileNotFoundError(f"RWIS metadata file not found: {RWIS_META_CSV}")
    
    rwis_meta = pd.read_csv(RWIS_META_CSV)
    print(f"Loaded {len(rwis_meta)} RWIS stations")

    grib = download_rtma_grib()
    rtma_df = interpolate_rtma_to_points(grib, rwis_meta)
    print(f"Interpolated RTMA to {len(rtma_df)} stations")

    cotrip_df = fetch_cotrip(api_key)
    print(f"Fetched {len(cotrip_df)} CoTrip observations")
    
    merged_df = pair_and_merge(rtma_df, cotrip_df)

    if merged_df.empty:
        raise ValueError("Merged dataframe is empty — no data to log.")

    # Add UTC time
    merged_df["time"] = pd.Timestamp.utcnow()

    # Ensure we have a valid station ID column
    station_id_col = None
    for col in ["rwis_station_id", "station_id", "rwis_stid"]:
        if col in merged_df.columns:
            station_id_col = col
            break
    
    if station_id_col is None:
        print("Warning: No station ID column found, using index")
        merged_df["station_index"] = range(len(merged_df))
        station_id_col = "station_index"

    # Set time + station ID as index for dimensions
    merged_df = merged_df.set_index(["time", "rwis_station_id"])
    merged_df = merged_df.set_index(["time", station_id_col])

    # Convert to xarray Dataset
    ds = xr.Dataset.from_dataframe(merged_df)

    # Reset them to be dimensions, not coordinates
    ds = ds.reset_coords(["time", "rwis_station_id"])

    # Promote useful metadata as coordinates
    # Promote useful metadata as coordinates if they exist
    for coord in ["station_name", "lat", "lon"]:
        if coord in ds:
            ds = ds.set_coords(coord)
        coord_cols = [c for c in ds.data_vars if coord in c.lower()]
        if coord_cols:
            # Use the first matching coordinate column
            ds = ds.set_coords(coord_cols[0])

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
        try:
            with xr.open_dataset(fname) as existing:
                ds = xr.concat([existing, ds], dim="time")
            print(f"Appended to existing file: {fname}")
        except Exception as e:
            print(f"Error reading existing file, creating new one: {e}")

    # Add encoding to prevent issues with string variables
    encoding = {}
    for var in ds.data_vars:
        if ds[var].dtype == 'object':
            encoding[var] = {'dtype': 'S64'}
    
    ds.to_netcdf(fname, mode="w", encoding=encoding)
    print(f"Saved to: {fname}")


def main() -> None:
    api_key = os.getenv(API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(f"Set CoTrip API key in env var {API_KEY_ENV}")

    ds = build_snapshot(api_key)
    append_daily(ds)
    print(f"[{datetime.utcnow():%Y-%m-%d %H:%M}] snapshot appended.")
    try:
        ds = build_snapshot(api_key)
        append_daily(ds)
        print(f"[{datetime.utcnow():%Y-%m-%d %H:%M}] snapshot appended successfully.")
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":Add commentMore actions
    main()
