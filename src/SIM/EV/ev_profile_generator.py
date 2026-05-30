
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_ev_sessions(
    csv_path: str,
    start_time: datetime,
    resolution: timedelta,
    duration: timedelta,
    oversample_factor: int = 5,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate EV plug-in / plug-out sessions from empirical CSV data.

    CSV format:
    day_id, start_time (minute-of-day), duration (minutes),
    start_soc, weekday (1/0), temperature
    """
    OUT_COLS = [
        "plug_in_time",
        "plug_out_time",
        "initial_soc",
        "temperature",
        "day_id",
        "weekday",
    ]
    rng = np.random.default_rng(seed)

    df_src = pd.read_csv(csv_path)

    required_cols = {
        "day_id",
        "start_time",
        "duration",
        "start_soc",
        "weekday",
        "temperature",
    }
    if not required_cols.issubset(df_src.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    end_time = start_time + duration

    start_day = datetime.combine(start_time.date(), datetime.min.time())
    end_day = datetime.combine(end_time.date(), datetime.min.time())

    days = pd.date_range(
        start=start_day,
        end=end_day,
        freq="D",
    )

    events = []
    last_plug_out = None
    # print(f'Days {days}')
    for day in days:
        # print(f'day {day}')
        day_dt = day.to_pydatetime()
        is_weekday = 1 if day_dt.weekday() < 5 else 0

        df_day = df_src[df_src["weekday"] == is_weekday]
        if df_day.empty:
            continue

        # -----------------------------
        # Oversample + random select
        # -----------------------------
        sample_pool = df_day.sample(
            n=min(len(df_day), oversample_factor * 10),
            replace=True,
            random_state=rng.integers(0, 1e9),
        )

        row = sample_pool.sample(
            n=1,
            random_state=rng.integers(0, 1e9),
        ).iloc[0]

        # -----------------------------
        # minute-of-day → datetime
        # -----------------------------
        plug_in_time = day_dt + timedelta(
            minutes=int(row["start_time"])
        )

        # Resolution alignment
        step = resolution.total_seconds()
        plug_in_time = datetime.fromtimestamp(
            (plug_in_time.timestamp() // step) * step
        )

        plug_out_time = plug_in_time + timedelta(
            minutes=float(row["duration"])
        )

        # -----------------------------
        # Overlap & bounds
        # -----------------------------
        if last_plug_out is not None and plug_in_time < last_plug_out:
            continue

        if plug_in_time < start_time or plug_out_time > end_time:
            continue

        events.append(
            {
                "plug_in_time": plug_in_time,
                "plug_out_time": plug_out_time,
                "initial_soc": float(row["start_soc"]),
                "temperature": float(row["temperature"]),
                "day_id": int(row["day_id"]),
                "weekday": int(row["weekday"])
            }
        )

        last_plug_out = plug_out_time
    # ✅ Always return a DataFrame with expected schema
    if not events:
        return pd.DataFrame(columns=OUT_COLS)
    return (
        pd.DataFrame(events, columns=OUT_COLS)
        .sort_values("plug_in_time")
        .reset_index(drop=True)
    )
