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


def download_rtma_grib() -> str:
    """Download the latest RTMA RU 2‑D var‑grid file; return local filename."""
    now = datetime.now(timezone.utc)
    buffer = now - timedelta(minutes=20)
    cycle_hour = buffer.strftime("%H")
    cycle_min = f"{(buffer.minute // 15) * 15:02d}"
    cycle_stamp = f"{cycle_hour}{cycle_min}"
    day = buffer.strftime("%Y%m%d")

    # choose previous 15‑min interval ≥ 20 min ago
    buffer = now - timedelta(minutes=20)
    cycle_stamp = f"{buffer:%H}{(buffer.minute // 15) * 15:02d}"

    url = f"{RTMA_BASE}/rtma2p5_ru.{day}/rtma2p5_ru.t{cycle_stamp}z.2dvarges_ndfd.grb2"
    fn = "rtma.bin"

    timestamp = pd.Timestamp(f"{day}T{cycle_hour}:{cycle_min}:00Z")

    print(f"Downloading RTMA from: {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(fn, "wb") as fh:
        for chunk in r.iter_content(8192):
            fh.write(chunk)
    return fn, timestamp


def interpolate_rtma_to_points(grib_file: str, rwis: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of RTMA values at each RWIS station."""
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
            
            # Safe value extraction with bounds checking and proper type conversion
            def safe_value(arr, ix, iy):
                try:
                    if hasattr(arr, 'values'):
                        val = arr.values[iy, ix]
                    else:
                        val = arr[iy, ix]
                    
                    # Convert numpy types to native Python float
                    if isinstance(val, (np.floating, np.integer)):
                        return float(val)
                    elif np.isscalar(val):
                        return float(val)
                    else:
                        return np.nan
                except (IndexError, ValueError, TypeError):
                    return np.nan
            
            records.append({
                "station_id": str(stn.stid),  # Ensure string type
                "station_name": str(stn.station_name),  # Ensure string type
                "lat": float(stn.lat),  # Ensure float type
                "lon": float(stn.lon),  # Ensure float type  
                "cloud_cover": safe_value(tcc, ix, iy),
                "rtma_temps": safe_value(tmp_f, ix, iy),
                "rtma_dp": safe_value(dpt_f, ix, iy),
                "rtma_wind_direction": safe_value(wdir, ix, iy),
                "rtma_wind_gust": safe_value(wgust, ix, iy),
                "rtma_wind_speed": safe_value(wspd, ix, iy),
                "valid_time": datetime.now(timezone.utc)
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
                lambda x: pd.Series([float(x[0]), float(x[1])]) if isinstance(x, list) and len(x) >= 2 else pd.Series([np.nan, np.nan])
            )
            merged[["lon", "lat"]] = coords[[0, 1]]
        else:
            print("Warning: No geometry coordinates in CoTrip data")
            merged[["lon", "lat"]] = [np.nan, np.nan]

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

        # Safe timestamp parsing - keep as datetime
        if "properties.lastUpdated" in pivot.columns:
            pivot["properties.lastUpdated"] = pd.to_datetime(
                pivot["properties.lastUpdated"], utc=True, errors="coerce"
            )
        
        # Ensure string columns are properly typed
        string_cols = ["properties.name", "geometry.type", "properties.nativeId", "properties.direction"]
        for col in string_cols:
            if col in pivot.columns:
                pivot[col] = pivot[col].astype(str)
        
        # Ensure numeric columns are float
        numeric_cols = ["lon", "lat", "geometry.srid","visibility","min temperature","dew point","gust wind speed","max temperature","temperature","average wind speed","road surface friction index","humidity"]
        for col in numeric_cols:
            if col in pivot.columns:
                pivot[col] = pd.to_numeric(pivot[col], errors='coerce')
        
        return pivot
        
    except Exception as e:
        print(f"Error fetching CoTrip data: {e}")
        return pd.DataFrame()


def pair_and_merge(rwis_pts: pd.DataFrame, cotrip: pd.DataFrame) -> pd.DataFrame:
    """Spatial/temporal join: nearest RWIS within tolerance + recent timestamp."""
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

    # Safe timestamp filtering
    try:
        recent_cut = cotrip_clean["properties.lastUpdated"].max() - timedelta(minutes=RECENT_MIN)
        mask = (dist != np.inf) & (cotrip_clean["properties.lastUpdated"] >= recent_cut)
    except Exception as e:
        print(f"Error in timestamp filtering: {e}")
        mask = dist != np.inf

    rwis_match = pd.DataFrame(index=cotrip_clean.index, columns=rwis_clean.columns)
    valid_matches = dist != np.inf
    if valid_matches.any():
        rwis_match.loc[valid_matches] = rwis_clean.iloc[idx[valid_matches]].values

    combined = pd.concat([cotrip_clean.reset_index(drop=True), rwis_match.add_prefix("rwis_")], axis=1)

    # Drop unwanted columns
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

    result = combined[mask].reset_index(drop=True)
    print(f"Paired {len(result)} stations successfully")
    if "valid_time" in rwis_pts.columns:
        result["valid_time"] = rwis_pts["valid_time"].iloc[0]
    return result


def build_snapshot(api_key: str) -> xr.Dataset:
    """Full pipeline: RWIS meta → RTMA interp → CoTrip merge → xarray.Dataset."""
    # Check if metadata file exists
    if not os.path.exists(RWIS_META_CSV):
        raise FileNotFoundError(f"RWIS metadata file not found: {RWIS_META_CSV}")
    
    rwis_meta = pd.read_csv(RWIS_META_CSV)
    print(f"Loaded {len(rwis_meta)} RWIS stations")

    grib, snapshot_time = download_rtma_grib()
    rtma_df = interpolate_rtma_to_points(grib, rwis_meta)
    rtma_df["valid_time"] = pd.to_datetime(snapshot_time)
    print(f"Interpolated RTMA to {len(rtma_df)} stations")

    cotrip_df = fetch_cotrip(api_key)
    print(f"Fetched {len(cotrip_df)} CoTrip observations")
    
    merged_df = pair_and_merge(rtma_df, cotrip_df)

    if merged_df.empty:
        raise ValueError("Merged dataframe is empty — no data to log.")

    # Add UTC time as datetime (not string)
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
        merged_df[station_id_col] = merged_df[station_id_col].astype(str)

    # Ensure station IDs are strings
    merged_df[station_id_col] = merged_df[station_id_col].astype(str)
    # Ensure valid_time is timezone-naive (required for NetCDF)
    if "valid_time" in merged_df.columns:
        merged_df["valid_time"] = pd.to_datetime(merged_df["valid_time"], utc=True).dt.tz_convert(None)

    # Set time + station ID as index for dimensions
    merged_df = merged_df.set_index(["valid_time", station_id_col])
    merged_df.sort_values("valid_time")
    # Convert to xarray Dataset
    ds = xr.Dataset.from_dataframe(merged_df)

    # Promote useful metadata as coordinates if they exist
    coord_mappings = {
        "station_name": ["station_name", "properties.name", "rwis_station_name"],
        "lat": ["lat", "rwis_lat"],
        "lon": ["lon", "rwis_lon"]
    }
    
    for coord_name, possible_cols in coord_mappings.items():
        coord_cols = [c for c in ds.data_vars if c in possible_cols]
        if coord_cols:
            # Use the first matching coordinate column
            ds = ds.set_coords(coord_cols[0])

    return ds


def append_daily(ds: xr.Dataset) -> None:
    """Append snapshot to daily NetCDF; create new file if absent."""
    if "valid_time" not in ds.dims and "valid_time" not in ds.coords:
        raise ValueError("Dataset has no 'time' dimension or coordinate.")

    # Ensure time is datetime, not string
    # Ensure 'valid_time' is a coordinate
    if "valid_time" in ds.coords:
    # Convert to numpy datetime64 without timezone
        times = pd.to_datetime(ds["valid_time"].values).tz_localize(None).to_numpy()
        ds = ds.assign_coords(valid_time=times)

    elif "valid_time" in ds.dims:
    # If it's a dimension only
        ds["valid_time"] = ds["valid_time"].dt.tz_localize(None)

# Also assign to 'time' coordinate
    ds = ds.assign_coords(time=ds["valid_time"])


    fname = f"rwis_rtma_{pd.Timestamp.utcnow():%Y%m%d}.nc"

    if os.path.exists(fname):
        try:
            with xr.open_dataset(fname) as existing:
                # Combine datasets
                combined = xr.concat([existing, ds], dim="valid_time")
                
                # CRITICAL FIX: Sort by time to ensure chronological order
                combined = combined.sortby("valid_time")
                
                # Optional: Remove duplicate timestamps if they exist
                _, unique_indices = np.unique(combined.valid_time.values, return_index=True)
                if len(unique_indices) < len(combined.valid_time):
                    print(f"Removing {len(combined.valid_time) - len(unique_indices)} duplicate timestamps")
                    combined = combined.isel(time=unique_indices)
                
                ds = combined.sortby("valid_time")
                print(f"Appended to existing file: {fname}")
                print(f"Total time range: {ds.time.min().values} to {ds.time.max().values}")
                print(f"Total snapshots: {len(ds.time)}")
                
        except Exception as e:
            print(f"Error reading existing file, creating new one: {e}")
    else:
        print(f"Creating new daily file: {fname}")

    # Improved encoding - handle different data types properly
    encoding = {}
    
    # Process data variables
    for var in ds.data_vars:
        if ds[var].dtype == 'object':
            # Check if it's actually string data or datetime
            if ds[var].size > 0:
                sample_val = ds[var].values.flat[0]
                if isinstance(sample_val, str):
                    encoding[var] = {'dtype': 'U64'}  # Unicode string
                elif isinstance(sample_val, pd.Timestamp) or hasattr(sample_val, 'strftime'):
                    # Convert datetime to string for NetCDF storage
                    ds[var] = ds[var].astype(str)
                    encoding[var] = {'dtype': 'U32'}  # Datetime as string
                elif pd.isna(sample_val):
                    # Handle NaN values - convert to string
                    ds[var] = ds[var].astype(str)
                    encoding[var] = {'dtype': 'U32'}
    
    # Process coordinate variables
    for coord in ds.coords:
        if ds[coord].dtype == 'object':
            if ds[coord].size > 0:
                sample_val = ds[coord].values.flat[0]
                if isinstance(sample_val, str):
                    encoding[coord] = {'dtype': 'U64'}
                elif isinstance(sample_val, pd.Timestamp) or hasattr(sample_val, 'strftime'):
                    # Convert datetime to string for NetCDF storage
                    ds = ds.assign_coords({coord: ds[coord].astype(str)})
                    encoding[coord] = {'dtype': 'U32'}
                elif pd.isna(sample_val):
                    # Handle NaN values - convert to string
                    ds = ds.assign_coords({coord: ds[coord].astype(str)})
                    encoding[coord] = {'dtype': 'U32'}
    
    # Use netCDF4 backend for compression support, fallback to scipy without compression
    try:
        # Try netCDF4 backend with compression
        for var in ds.data_vars:
            if var not in encoding:
                encoding[var] = {}
            encoding[var].update({'zlib': True, 'complevel': 4})
        
        ds.to_netcdf(fname, mode="w", encoding=encoding, engine='netcdf4')
        print(f"Saved with netCDF4 backend and compression")
    except (ImportError, ValueError) as e:
        # Fallback to scipy backend without compression
        print(f"NetCDF4 not available, using scipy backend: {e}")
        # Remove compression from encoding
        for var in encoding:
            if isinstance(encoding[var], dict):
                encoding[var].pop('zlib', None)
                encoding[var].pop('complevel', None)
        
        ds.to_netcdf(fname, mode="w", encoding=encoding)
    print(f"Saved to: {fname}")


def main() -> None:
    api_key = os.getenv(API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(f"Set CoTrip API key in env var {API_KEY_ENV}")

    try:
        ds = build_snapshot(api_key)
        append_daily(ds)
        print(f"[{datetime.utcnow():%Y-%m-%d %H:%M}] snapshot appended successfully.")
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
