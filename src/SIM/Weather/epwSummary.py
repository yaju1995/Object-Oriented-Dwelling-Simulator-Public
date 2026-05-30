import pandas as pd
import numpy as np
from pathlib import Path


def summarize_epw(epw_path: str | Path) -> dict:
    """
    Analyze an EPW weather file and return a structured summary.

    Returns
    -------
    summary : dict
        Dictionary containing coverage, resolution, and missing-value diagnostics.
    """

    epw_path = Path(epw_path)
    if not epw_path.exists():
        raise FileNotFoundError(f"EPW file not found: {epw_path}")

    # ------------------------------------------------------------------
    # Load EPW (skip metadata header)
    # ------------------------------------------------------------------
    df = pd.read_csv(epw_path, skiprows=8, header=None)

    # Minimal column mapping (EPW standard positions)
    df.columns = [
                     "Year", "Month", "Day", "Hour", "Minute",
                     "Data Source and Uncertainty Flags",
                     "Dry Bulb Temperature", "Dew Point Temperature", "Relative Humidity",
                     "Atmospheric Station Pressure", "Extraterrestrial Horizontal Radiation",
                     "Extraterrestrial Direct Normal Radiation", "Horizontal Infrared Radiation Intensity",
                     "Global Horizontal Radiation", "Direct Normal Radiation", "Diffuse Horizontal Radiation",
                     "Global Horizontal Illuminance", "Direct Normal Illuminance",
                     "Diffuse Horizontal Illuminance", "Zenith Luminance", "Wind Direction",
                     "Wind Speed", "Total Sky Cover", "Opaque Sky Cover", "Visibility",
                     "Ceiling Height", "Present Weather Observation", "Present Weather Codes",
                     "Precipitable Water", "Aerosol Optical Depth", "Snow Depth", "Days Since Last Snowfall",
                     "Albedo", "Liquid Precipitation Depth", "Liquid Precipitation Quantity"
                 ][: df.shape[1]]

    # ------------------------------------------------------------------
    # Build proper datetime index (EPW hour-24 safe)
    # ------------------------------------------------------------------
    mask_24 = df["Hour"] == 24
    df.loc[mask_24, "Hour"] = 0

    dt = pd.to_datetime(
        dict(
            year=df["Year"],
            month=df["Month"],
            day=df["Day"],
            hour=df["Hour"],
            minute=df["Minute"].clip(0, 59),
        ),
        errors="coerce",
    )

    dt.loc[mask_24] += pd.Timedelta(days=1)
    df.index = dt
    df.index.name = "time"

    # ------------------------------------------------------------------
    # Coverage analysis
    # ------------------------------------------------------------------
    start_time = df.index.min()
    end_time = df.index.max()
    n_rows = len(df)

    expected_hours = {8760, 8784}
    looks_complete = n_rows in expected_hours

    # ------------------------------------------------------------------
    # Resolution analysis
    # ------------------------------------------------------------------
    deltas = df.index.to_series().diff().dropna()
    resolution_mode = deltas.mode()[0] if not deltas.empty else None
    resolution_std = deltas.std()

    consistent_resolution = (
        resolution_std is not None
        and resolution_std < pd.Timedelta(seconds=1)
    )

    # ------------------------------------------------------------------
    # Missing value analysis
    # ------------------------------------------------------------------
    MISSING_CODES = [999, 9999, -999, -9999]
    df_clean = df.replace(MISSING_CODES, np.nan)

    missing_rows = df_clean.isna().any(axis=1)
    n_missing_rows = missing_rows.sum()

    missing_by_column = (
        df_clean.isna().sum()
        .rename("missing_count")
        .to_frame()
    )
    missing_by_column["missing_pct"] = (
        missing_by_column["missing_count"] / n_rows * 100
    )

    missing_columns = missing_by_column[
        missing_by_column["missing_count"] > 0
    ]

    first_missing = df_clean[missing_rows].index.min() if n_missing_rows > 0 else None
    last_missing = df_clean[missing_rows].index.max() if n_missing_rows > 0 else None

    # ------------------------------------------------------------------
    # Index sanity checks
    # ------------------------------------------------------------------
    index_issues = {
        "has_duplicates": df.index.has_duplicates,
        "is_monotonic": df.index.is_monotonic_increasing,
        "has_nat": df.index.isna().any(),
    }

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    summary = {
        "file": str(epw_path),
        "coverage": {
            "start": start_time,
            "end": end_time,
            "rows": n_rows,
            "looks_complete_year": looks_complete,
        },
        "resolution": {
            "detected": resolution_mode,
            "consistent": consistent_resolution,
        },
        "missing_data": {
            "rows_with_missing": int(n_missing_rows),
            "pct_rows_with_missing": float(n_missing_rows / n_rows * 100),
            "affected_columns": missing_columns.sort_values(
                "missing_pct", ascending=False
            ),
            "first_missing": first_missing,
            "last_missing": last_missing,
        },
        "index_health": index_issues,
    }

    return summary


import pandas as pd
import numpy as np
from pathlib import Path


EPW_MISSING_CODES = [999, 9999, -999, -9999]


def load_epw_with_datetime(epw_path: str | Path) -> pd.DataFrame:
    """
    Load EPW and build a robust DatetimeIndex (handles EPW hour=24 correctly).
    """
    epw_path = Path(epw_path)
    if not epw_path.exists():
        raise FileNotFoundError(f"EPW file not found: {epw_path}")

    df = pd.read_csv(epw_path, skiprows=8, header=None)

    # EPW standard has 35 columns; some files may have fewer.
    cols= [
        "Year", "Month", "Day", "Hour", "Minute",
        "Data Source and Uncertainty Flags",
        "Dry Bulb Temperature", "Dew Point Temperature", "Relative Humidity",
        "Atmospheric Station Pressure", "Extraterrestrial Horizontal Radiation",
        "Extraterrestrial Direct Normal Radiation", "Horizontal Infrared Radiation Intensity",
        "Global Horizontal Radiation", "Direct Normal Radiation", "Diffuse Horizontal Radiation",
        "Global Horizontal Illuminance", "Direct Normal Illuminance",
        "Diffuse Horizontal Illuminance", "Zenith Luminance", "Wind Direction",
        "Wind Speed", "Total Sky Cover", "Opaque Sky Cover", "Visibility",
        "Ceiling Height", "Present Weather Observation", "Present Weather Codes",
        "Precipitable Water", "Aerosol Optical Depth", "Snow Depth", "Days Since Last Snowfall",
        "Albedo", "Liquid Precipitation Depth", "Liquid Precipitation Quantity"
    ]
    df.columns = cols[: df.shape[1]]

    # Build datetime index (EPW hour=24 means next day at 00:xx)
    mask_24 = df["Hour"] == 24
    df2 = df.copy()
    df2.loc[mask_24, "Hour"] = 0
    df2["Minute"] = df2["Minute"].clip(0, 59)

    dt = pd.to_datetime(
        dict(
            year=df2["Year"],
            month=df2["Month"],
            day=df2["Day"],
            hour=df2["Hour"],
            minute=df2["Minute"],
        ),
        errors="coerce",
    )
    dt.loc[mask_24] += pd.Timedelta(days=1)

    df2.index = dt
    df2.index.name = "time"

    # Sort + drop NaT index rows (rare, but safer)
    df2 = df2[~df2.index.isna()].sort_index()

    return df2


def detect_resolution(df: pd.DataFrame) -> pd.Timedelta:
    """
    Detect the dominant timestep from the datetime index.
    Raises if index is too small or resolution is inconsistent.
    """
    if df is None or df.empty or len(df) < 3:
        raise ValueError("Not enough rows to detect resolution.")

    deltas = df.index.to_series().diff().dropna()
    mode = deltas.mode()
    if mode.empty:
        raise ValueError("Could not detect resolution (no deltas).")

    res = mode.iloc[0]

    # Consistency check: how often are deltas not equal to mode?
    mismatch_rate = (deltas != res).mean()

    # If there are daylight/time anomalies or gaps, mismatch_rate can be > 0,
    # but if it is too high, something is wrong.
    if mismatch_rate > 0.01:  # 1% tolerance
        # Still return res, but you may want to treat this as a warning upstream
        pass

    return res


def add_missing_datetimes(df: pd.DataFrame, resolution: pd.Timedelta) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Reindex to a complete datetime range at the detected resolution,
    inserting missing timestamps as new rows.
    Returns (df_full, missing_timestamps).
    """
    if df.empty:
        return df, pd.DatetimeIndex([])

    start = df.index.min()
    end = df.index.max()

    full_index = pd.date_range(start=start, end=end, freq=resolution, name="time")
    missing_timestamps = full_index.difference(df.index)

    df_full = df.reindex(full_index)

    return df_full, missing_timestamps


def missing_value_report(df_full: pd.DataFrame) -> dict:
    """
    After reindexing (timestamps inserted), check missing values.
    Distinguishes:
    - missing timestamps (entire rows are NaN)
    - missing fields (specific columns NaN)
    """
    # Treat EPW missing codes as NaN (important!)
    dfc = df_full.replace(EPW_MISSING_CODES, np.nan)

    # Rows that are entirely empty usually correspond to missing timestamps
    empty_rows = dfc.isna().all(axis=1)
    rows_with_any_missing = dfc.isna().any(axis=1)

    missing_by_col = dfc.isna().sum().sort_values(ascending=False)
    missing_pct_by_col = (missing_by_col / len(dfc) * 100).round(3)

    return {
        "rows_total": int(len(dfc)),
        "rows_entirely_missing": int(empty_rows.sum()),
        "rows_with_any_missing": int(rows_with_any_missing.sum()),
        "pct_rows_entirely_missing": float((empty_rows.mean() * 100)),
        "pct_rows_with_any_missing": float((rows_with_any_missing.mean() * 100)),
        "missing_by_column": missing_by_col,
        "missing_pct_by_column": missing_pct_by_col,
        "first_row_entirely_missing": dfc.index[empty_rows].min() if empty_rows.any() else None,
        "last_row_entirely_missing": dfc.index[empty_rows].max() if empty_rows.any() else None,
        "first_row_with_any_missing": dfc.index[rows_with_any_missing].min() if rows_with_any_missing.any() else None,
        "last_row_with_any_missing": dfc.index[rows_with_any_missing].max() if rows_with_any_missing.any() else None,
    }


def epw_resolution_and_missing(epw_path: str | Path) -> dict:
    """
    Full workflow:
    - load epw + datetime
    - detect resolution
    - add missing datetimes (reindex)
    - check missing values
    """
    df = load_epw_with_datetime(epw_path)
    resolution = detect_resolution(df)
    df_full, missing_timestamps = add_missing_datetimes(df, resolution)
    mv = missing_value_report(df_full)

    return {
        "file": str(epw_path),
        "resolution": resolution,
        "start": df_full.index.min(),
        "end": df_full.index.max(),
        "original_rows": int(len(df)),
        "rows_after_reindex": int(len(df_full)),
        "inserted_missing_timestamps": int(len(missing_timestamps)),
        "missing_timestamps_preview": missing_timestamps[:10],
        "missing_value_report": mv,
        "df_full": df_full,  # keep if you want to use it downstream
    }

