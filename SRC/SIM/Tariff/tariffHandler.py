import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime
from typing import Optional
from support.lib_config import CustomLogger
from SIM.Tariff.TariffGenerator import BaseTariffGenerator

logger = CustomLogger(command=True)


def time_to_seconds(t) -> int:
    """
    Accepts datetime.time or datetime.datetime (or pandas Timestamp) and returns seconds since midnight.
    """
    if isinstance(t, pd.Timestamp):
        t = t.to_pydatetime()
    if isinstance(t, datetime):
        return t.hour * 3600 + t.minute * 60 + t.second
    if isinstance(t, dtime):
        return t.hour * 3600 + t.minute * 60 + t.second
    raise TypeError(f"Unsupported type for time_to_seconds: {type(t)}")


def get_active_period_value(df: pd.DataFrame, now_time) -> float:
    """
    df index: datetime.time values (00:00:00 ... 23:xx:xx)
    df must contain a column named 'value'
    now_time: datetime/datetime.time/pd.Timestamp
    """
    if df is None or df.empty:
        raise ValueError("Tariff DataFrame is None or empty.")
    if "value" not in df.columns:
        raise ValueError("Tariff DataFrame must contain a 'value' column.")

    idx_seconds = np.array([time_to_seconds(t) for t in df.index], dtype=np.int32)
    now_seconds = time_to_seconds(now_time)

    valid_idx = np.where(idx_seconds <= now_seconds)[0]
    if len(valid_idx) == 0:
        # Before first time → wrap to last value (previous day's last period)
        return float(df.iloc[-1]["value"])

    return float(df.iloc[valid_idx[-1]]["value"])


class tariffHandler:
    def __init__(
            self,
            tariff_model: Optional[BaseTariffGenerator] = None,  # fixed name
            feed_tariff_model: Optional[BaseTariffGenerator] = None,
            type: int = 2,
            tariff_resolution: timedelta = timedelta(minutes=60),
    ):
        """
        type:
          1 -> same tariff for import/export (not enforced here, but keep for your logic)
          2 -> separate import tariff + feed-in tariff
        """

        self.tariff_model = tariff_model
        self.feed_tariff_model = feed_tariff_model
        self.tariff_resolution = tariff_resolution
        self.type = type

        self.tariff: Optional[pd.DataFrame] = None
        self.feed_tariff: Optional[pd.DataFrame] = None

        self.max_tariff = None
        self.min_tariff = None
        self.max_feed_tariff = None
        self.min_feed_tariff = None
        self.avg_feed_tariff = None
        self.avg_tariff = None

        self.next_24hr_tariff: Optional[pd.DataFrame] = None
        self.next_24hr_feed_tariff: Optional[pd.DataFrame] = None

    def generate_tariff(self) -> None:
        """
        Generates next-day/next-24h tariffs into the staging buffers.
        """
        if self.tariff_model is not None:
            self.next_24hr_tariff = self.tariff_model.generate_tariff()
        if self.feed_tariff_model is not None:
            self.next_24hr_feed_tariff = self.feed_tariff_model.generate_tariff()

    def _load_tariff_csv(self, file: str) -> pd.DataFrame:
        df = pd.read_csv(file, index_col=0)

        if "value" not in df.columns:
            raise ValueError(f"Tariff file '{file}' must contain a 'value' column. Found: {list(df.columns)}")

        # Parse index as time-of-day
        idx = pd.to_datetime(df.index, format="%H:%M:%S", errors="raise").time
        df.index = idx

        # Ensure sorted by time
        df = df.sort_index()
        return df

    def upload_tariff(self, file: str) -> None:
        self.tariff = self._load_tariff_csv(file)
        self.update_min_max()

    def upload_feed_tariff(self, file: str) -> None:
        self.feed_tariff = self._load_tariff_csv(file)
        self.update_min_max()

    def _as_time(self, dt_like) -> dtime:
        """Return time-of-day from datetime/pd.Timestamp/time."""
        if isinstance(dt_like, pd.Timestamp):
            dt_like = dt_like.to_pydatetime()
        if isinstance(dt_like, datetime):
            return dt_like.time()
        if isinstance(dt_like, dtime):
            return dt_like
        raise TypeError(f"Unsupported time type: {type(dt_like)}")

    def _stamp_day_tariff(self, day_date, day_tariff_df: pd.DataFrame) -> pd.Series:
        """
        Convert a time-index tariff DF (index=python time) into a datetime-index series for a given date.
        """
        if day_tariff_df is None or day_tariff_df.empty:
            return pd.Series(dtype=float)

        if "value" not in day_tariff_df.columns:
            raise ValueError("Tariff DataFrame must have a 'value' column")

        # Build datetime index for that day using the existing time-of-day index
        dt_index = pd.to_datetime(
            [datetime.combine(day_date, t) for t in day_tariff_df.index]
        )
        s = pd.Series(day_tariff_df["value"].astype(float).values, index=dt_index)
        s = s.sort_index()
        return s

    def _build_48h_tariff_series(self, now_time: datetime, which: str) -> pd.Series:
        """
        Build 48h tariff series (datetime-indexed) from:
          - today: current tariff
          - tomorrow: next_24hr if available else current
        'which' in {"tariff", "feed_tariff"}
        """
        base_date = pd.Timestamp(now_time).date()
        tomorrow = (pd.Timestamp(now_time) + pd.Timedelta(days=1)).date()

        if which == "tariff":
            today_df = self.tariff
            tomorrow_df = self.next_24hr_tariff if self.next_24hr_tariff is not None else self.tariff
        elif which == "feed_tariff":
            today_df = self.feed_tariff
            tomorrow_df = self.next_24hr_feed_tariff if self.next_24hr_feed_tariff is not None else self.feed_tariff
        else:
            raise ValueError("which must be 'tariff' or 'feed_tariff'")

        s_today = self._stamp_day_tariff(base_date, today_df)
        s_tom = self._stamp_day_tariff(tomorrow, tomorrow_df)

        # Combine into one 48h series
        s_48 = pd.concat([s_today, s_tom]).sort_index()

        return s_48

    def get_tariff(self, now_time) -> tuple[float, float]:
        """
        Returns (import_tariff, feed_in_tariff) for the given timestamp/time.
        """
        tariff = 0.0
        feed_tariff = 0.0

        if self.tariff is not None and not self.tariff.empty:
            tariff = get_active_period_value(self.tariff, now_time)

        if self.feed_tariff is not None and not self.feed_tariff.empty:
            feed_tariff = get_active_period_value(self.feed_tariff, now_time)

        # If type==1 and you want same tariff both ways, enforce here if needed:
        # if self.type == 1:
        #     feed_tariff = tariff

        return float(tariff), float(feed_tariff)

    def get_tariff_range_df(
            self,
            now_time: datetime,
            period: int = 96,  # number of steps
            resolution: timedelta = timedelta(minutes=15),
    ) -> pd.DataFrame:
        """
        Returns tariff & feed tariff for next 'period' steps.
        Handles crossing midnight by constructing a 48h datetime-index tariff from
        (today=current tariff) + (tomorrow=next_24hr if available).
        """

        now_ts = pd.Timestamp(now_time)
        # print(self.next_24hr_tariff)
        # Build 48h datetime-indexed series
        s_tariff_48 = self._build_48h_tariff_series(now_time, which="tariff")
        s_feed_48 = self._build_48h_tariff_series(now_time, which="feed_tariff")

        # Create the requested timestamps
        idx = pd.date_range(start=now_ts, periods=period, freq=pd.Timedelta(resolution))

        # Reindex with forward-fill so each timestamp gets the active block value
        # (This assumes your time-index data points represent "period start" values.)
        tariff_vals = s_tariff_48.reindex(idx, method="ffill")
        feed_vals = s_feed_48.reindex(idx, method="ffill")

        # If a timestamp is before the first tariff entry of the day, ffill won't work.
        # Use bfill as a fallback (rare, but defensive).
        tariff_vals = tariff_vals.bfill().fillna(0.0)
        feed_vals = feed_vals.bfill().fillna(0.0)

        if self.type == 1:
            feed_vals = tariff_vals.copy()

        df = pd.DataFrame(
            {"tariff": tariff_vals.values.astype(float), "feed_tariff": feed_vals.values.astype(float)},
            index=idx
        )
        df.index.name = "time"
        return df

    def updated_tariff(self) -> None:
        """
        If the next-day tariffs are staged (e.g., Ireland day-ahead publish time),
        commit them and refresh min/max.
        """
        if self.next_24hr_tariff is not None:
            logger.commandline("Updating tariff value")
            self.tariff = self.next_24hr_tariff
            self.next_24hr_tariff = None

        if self.next_24hr_feed_tariff is not None:
            logger.commandline("Updating feed tariff value")
            self.feed_tariff = self.next_24hr_feed_tariff
            self.next_24hr_feed_tariff = None

        self.update_min_max()

    def update_min_max(self) -> None:
        if self.tariff is not None and not self.tariff.empty and "value" in self.tariff.columns:
            self.max_tariff = float(self.tariff["value"].max())
            self.min_tariff = float(self.tariff["value"].min())
            self.avg_tariff = float(self.tariff["value"].mean())

        if self.feed_tariff is not None and not self.feed_tariff.empty and "value" in self.feed_tariff.columns:
            self.max_feed_tariff = float(self.feed_tariff["value"].max())
            self.min_feed_tariff = float(self.feed_tariff["value"].min())
            self.avg_feed_tariff = float(self.feed_tariff["value"].mean())
