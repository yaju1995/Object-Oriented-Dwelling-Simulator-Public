from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
import pandas as pd


@dataclass
class DataStore:
    resolution: timedelta                    # ✅ base resolution as datetime.timedelta
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    lock: RLock = field(default_factory=RLock)

    # -------------------- write --------------------
    def append(self, t: datetime | pd.Timestamp, row: dict) -> None:
        t = pd.Timestamp(t)
        with self.lock:
            if self.df.empty:
                self.df = pd.DataFrame([row], index=pd.DatetimeIndex([t], name="time"))
            else:
                self.df.loc[t] = row
            self.df.sort_index(inplace=True)

    # -------------------- read: fixed-length history --------------------
    def get_past_period_df(
        self,
        now_time: datetime | pd.Timestamp,
        past_period: int = 24,
    ) -> pd.DataFrame | None:
        """
        Return exactly N samples ending at the latest available timestamp <= now_time,
        where:
            N = past_period(hours) / self.resolution
        """

        now_time = pd.Timestamp(now_time)

        # ---- convert ONCE for pandas ----
        res_pd = pd.Timedelta(self.resolution)
        period_pd = pd.Timedelta(hours=past_period)

        # ---- compute expected samples robustly ----
        expected_len = int(period_pd.total_seconds() / res_pd.total_seconds())
        if expected_len <= 0:
            raise ValueError("past_period too small for the configured resolution")

        with self.lock:
            if self.df.empty:
                return None

            idx = self.df.index
            if not isinstance(idx, pd.DatetimeIndex):
                raise ValueError("DataStore.df must have a DatetimeIndex")

            # Find last timestamp <= now_time (no assumption it exists)
            pos = idx.get_indexer([now_time], method="pad")[0]
            if pos == -1:
                return None

            end_time = idx[pos]
            start_pos = pos - (expected_len - 1)
            if start_pos < 0:
                return None

            window = self.df.iloc[start_pos:pos + 1].copy()

        # ---- strict continuity check ----
        if len(window) != expected_len:
            return None

        expected_index = pd.date_range(
            end=end_time,
            periods=expected_len,
            freq=res_pd,
        )
        if not window.index.equals(expected_index):
            return None

        return window

    # -------------------- read: resampling --------------------
    def get_resampled(
        self,
        df_sample: pd.DataFrame,
        resolution: timedelta,
        headers: list[str],
        agg: str = "mean",
    ) -> pd.DataFrame | None:
        if df_sample is None or df_sample.empty:
            return None

        if not isinstance(df_sample.index, pd.DatetimeIndex):
            raise ValueError("df_sample must have a DatetimeIndex")

        missing = set(headers) - set(df_sample.columns)
        if missing:
            raise KeyError(f"Missing columns in df_sample: {missing}")

        # ---- convert timedelta → pandas rule ----
        rule = pd.Timedelta(resolution)

        df = df_sample[headers].copy()
        r = df.resample(rule)

        if agg == "mean":
            return r.mean()
        if agg == "sum":
            return r.sum()
        if agg == "max":
            return r.max()
        if agg == "min":
            return r.min()
        raise ValueError(f"Unsupported aggregation: {agg}")

    # -------------------- convenience --------------------
    def past_period_resampled(
        self,
        now_time: datetime | pd.Timestamp,
        past_period: int,
        out_resolution: timedelta,
        headers: list[str],
        agg: str = "mean",
    ) -> pd.DataFrame | None:
        w = self.get_past_period_df(now_time, past_period)
        if w is None:
            return None
        return self.get_resampled(w, out_resolution, headers, agg)

    # Sample code ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # df15 = self.hems_database.past_period_resampled(
    #         now_time, past_period=3,
    #         out_resolution=timedelta(minutes=15),
    #         headers=["Consumption (kWh)", "Generation (kWh)"],
    #         agg="sum",
    #     )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_instant_state(
            self,
            keys: list[str],
            *,
            now_time: datetime | pd.Timestamp | None = None,
            strict: bool = True,
    ) -> pd.Series | None:
        """
        Return the latest values for selected keys.

        If now_time is provided, returns the latest row <= now_time.
        If strict=True, missing keys raise an error.
        """

        with self.lock:
            if self.df.empty:
                return None

            df = self.df

            # ----- choose row -----
            if now_time is None:
                row = df.iloc[-1]
            else:
                now_time = pd.Timestamp(now_time)
                pos = df.index.get_indexer([now_time], method="pad")[0]
                if pos == -1:
                    return None
                row = df.iloc[pos]

            # ----- key validation -----
            missing = set(keys) - set(df.columns)
            if missing:
                if strict:
                    raise KeyError(f"Missing instant keys: {missing}")
                else:
                    return row.reindex(keys)

            return row[keys].copy()

