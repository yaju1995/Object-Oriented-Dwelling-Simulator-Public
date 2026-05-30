import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dtime, date
from typing import Optional
from support.lib_config import (CustomLogger)
from SIM.Tariff.TariffGenerator import (BaseTariffGenerator)

logger = CustomLogger(command=False)


def time_to_seconds(t) -> int:
    """
    Accepts datetime.time or datetime.datetime (or pandas Timestamp)
    and returns seconds since midnight.
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
        # Before first time -> wrap to last value (previous day's last period)
        return float(df.iloc[-1]["value"])

    return float(df.iloc[valid_idx[-1]]["value"])


class tariffHandler:
    def __init__(
        self,
        tariff_model: Optional[BaseTariffGenerator] = None,
        feed_tariff_model: Optional[BaseTariffGenerator] = None,
        type: int = 2,
        tariff_resolution: timedelta = timedelta(minutes=60),
    ):
        """
        type:
          1 -> same tariff for import/export
          2 -> separate import tariff + feed-in tariff
        """

        self.tariff_model = tariff_model
        self.feed_tariff_model = feed_tariff_model
        self.tariff_resolution = tariff_resolution
        self.type = type

        # active current-day tariff profiles (time-of-day indexed)
        self.tariff: Optional[pd.DataFrame] = None
        self.feed_tariff: Optional[pd.DataFrame] = None

        # staged next-day tariff profiles (time-of-day indexed)
        self.next_24hr_tariff: Optional[pd.DataFrame] = None
        self.next_24hr_feed_tariff: Optional[pd.DataFrame] = None

        # historic stores: {date: DataFrame(index=time, columns=["value"])}
        self.historic_tariff_by_day: dict[date, pd.DataFrame] = {}
        self.historic_feed_tariff_by_day: dict[date, pd.DataFrame] = {}
        self.historic_tariff_resolution: Optional[timedelta] = None
        self.historic_feed_tariff_resolution: Optional[timedelta] = None

        self.max_tariff = None
        self.min_tariff = None
        self.max_feed_tariff = None
        self.min_feed_tariff = None
        self.avg_feed_tariff = None
        self.avg_tariff = None

    # -------------------------------------------------------------------------
    # Generator-based tariff handling
    # -------------------------------------------------------------------------
    def generate_tariff(self) -> None:
        """
        Generates next-day/next-24h tariffs into the staging buffers.
        """
        if self.tariff_model is not None:
            self.next_24hr_tariff = self.tariff_model.generate_tariff()

        if self.feed_tariff_model is not None:
            self.next_24hr_feed_tariff = self.feed_tariff_model.generate_tariff()

        if self.type == 1 and self.next_24hr_tariff is not None:
            self.next_24hr_feed_tariff = self.next_24hr_tariff.copy()

    # -------------------------------------------------------------------------
    # Standard single-day tariff CSV upload
    # Expected format:
    #   index = HH:MM:SS
    #   columns include 'value'
    # -------------------------------------------------------------------------
    def _load_tariff_csv(self, file: str) -> pd.DataFrame:
        df = pd.read_csv(file, index_col=0)

        if "value" not in df.columns:
            raise ValueError(
                f"Tariff file '{file}' must contain a 'value' column. "
                f"Found: {list(df.columns)}"
            )

        idx = pd.to_datetime(df.index, format="%H:%M:%S", errors="raise").time
        df.index = idx
        df = df.sort_index()
        return df

    def upload_tariff(self, file: str) -> None:
        self.tariff = self._load_tariff_csv(file)
        if self.type == 1:
            self.feed_tariff = self.tariff.copy()
        self.update_min_max()

    def upload_feed_tariff(self, file: str) -> None:
        self.feed_tariff = self._load_tariff_csv(file)
        self.update_min_max()

    # -------------------------------------------------------------------------
    # Historic tariff CSV upload
    # Expected format like Irish_2026_Wholesale_price.csv:
    #   Timestamp, Euro/kWh
    # -------------------------------------------------------------------------
    def _load_historic_tariff_csv(
        self,
        file: str,
        timestamp_col: str = "Timestamp",
        value_col: str = "Euro/kWh",
        dayfirst: bool = True,
    ) -> tuple[dict[date, pd.DataFrame], Optional[timedelta]]:
        """
        Loads a timestamped historic tariff CSV and arranges it by date:
            {
                date(2026, 1, 1): DataFrame(index=time, columns=["value"]),
                date(2026, 1, 2): DataFrame(index=time, columns=["value"]),
                ...
            }
        """
        df = pd.read_csv(file)

        if timestamp_col not in df.columns:
            raise ValueError(
                f"Missing timestamp column '{timestamp_col}'. Found: {list(df.columns)}"
            )
        if value_col not in df.columns:
            raise ValueError(
                f"Missing value column '{value_col}'. Found: {list(df.columns)}"
            )

        df = df[[timestamp_col, value_col]].copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="raise")
        df = df.rename(columns={value_col: "value"})
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        diffs = df[timestamp_col].diff().dropna()
        resolution = None if diffs.empty else diffs.median().to_pytimedelta()

        df["date"] = df[timestamp_col].dt.date
        df["time"] = df[timestamp_col].dt.time

        by_day: dict[date, pd.DataFrame] = {}
        for day, g in df.groupby("date", sort=True):
            day_df = g[["time", "value"]].copy()

            # Keep last value if duplicate timestamps exist
            day_df = day_df.drop_duplicates(subset="time", keep="last")

            # Final per-day format matches existing handler format
            day_df = day_df.set_index("time").sort_index()

            by_day[day] = day_df

        return by_day, resolution

    def upload_historic_tariff(
        self,
        file: str,
        timestamp_col: str = "Timestamp",
        value_col: str = "Euro/kWh",
        dayfirst: bool = True,
    ) -> None:
        """
        Upload historic import tariff data and arrange it by day.
        """
        self.historic_tariff_by_day, self.historic_tariff_resolution = self._load_historic_tariff_csv(
            file=file,
            timestamp_col=timestamp_col,
            value_col=value_col,
            dayfirst=dayfirst,
        )

    def upload_historic_feed_tariff(
        self,
        file: str,
        timestamp_col: str = "Timestamp",
        value_col: str = "Euro/kWh",
        dayfirst: bool = True,
    ) -> None:
        """
        Upload historic feed tariff data and arrange it by day.
        """
        self.historic_feed_tariff_by_day, self.historic_feed_tariff_resolution = self._load_historic_tariff_csv(
            file=file,
            timestamp_col=timestamp_col,
            value_col=value_col,
            dayfirst=dayfirst,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _as_time(self, dt_like) -> dtime:
        """
        Return time-of-day from datetime/pd.Timestamp/time.
        """
        if isinstance(dt_like, pd.Timestamp):
            dt_like = dt_like.to_pydatetime()
        if isinstance(dt_like, datetime):
            return dt_like.time()
        if isinstance(dt_like, dtime):
            return dt_like
        raise TypeError(f"Unsupported time type: {type(dt_like)}")

    def _as_date(self, dt_like) -> date:
        """
        Return date from datetime/pd.Timestamp/date.
        """
        if isinstance(dt_like, pd.Timestamp):
            return dt_like.date()
        if isinstance(dt_like, datetime):
            return dt_like.date()
        if isinstance(dt_like, date):
            return dt_like
        raise TypeError(f"Unsupported date type: {type(dt_like)}")

    def _stamp_day_tariff(self, day_date, day_tariff_df: pd.DataFrame) -> pd.Series:
        """
        Convert a time-index tariff DF (index=python time)
        into a datetime-index series for a given date.
        """
        if day_tariff_df is None or day_tariff_df.empty:
            return pd.Series(dtype=float)

        if "value" not in day_tariff_df.columns:
            raise ValueError("Tariff DataFrame must have a 'value' column")

        dt_index = pd.to_datetime([datetime.combine(day_date, t) for t in day_tariff_df.index])
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
            tomorrow_df = (
                self.next_24hr_feed_tariff
                if self.next_24hr_feed_tariff is not None
                else self.feed_tariff
            )
        else:
            raise ValueError("which must be 'tariff' or 'feed_tariff'")

        s_today = self._stamp_day_tariff(base_date, today_df)
        s_tom = self._stamp_day_tariff(tomorrow, tomorrow_df)

        s_48 = pd.concat([s_today, s_tom]).sort_index()
        return s_48

    def validate_historic_day(
        self,
        target_date,
        which: str = "tariff",
        expected_steps: Optional[int] = None,
    ) -> bool:
        """
        Check if a historic day exists and optionally whether it has the expected
        number of timesteps.
        """
        target_date = self._as_date(target_date)

        if which == "tariff":
            store = self.historic_tariff_by_day
            resolution = self.historic_tariff_resolution
        elif which == "feed_tariff":
            store = self.historic_feed_tariff_by_day
            resolution = self.historic_feed_tariff_resolution
        else:
            raise ValueError("which must be 'tariff' or 'feed_tariff'")

        if target_date not in store:
            return False

        day_df = store[target_date]

        if expected_steps is None and resolution is not None:
            expected_steps = int(timedelta(days=1) / resolution)

        if expected_steps is None:
            return True

        return len(day_df) == expected_steps

    # -------------------------------------------------------------------------
    # Historic-data driven current/next day update
    # -------------------------------------------------------------------------
    def set_current_tariff_from_historic(self, target_date, update_feed: bool = True) -> None:
        """
        Set self.tariff (and optionally self.feed_tariff) from historic data
        for the given date.
        """
        target_date = self._as_date(target_date)

        if not self.historic_tariff_by_day:
            raise ValueError("No historic import tariff data uploaded.")
        if target_date not in self.historic_tariff_by_day:
            raise KeyError(f"No historic import tariff found for {target_date}")

        self.tariff = self.historic_tariff_by_day[target_date].copy()

        if update_feed:
            if self.type == 1:
                self.feed_tariff = self.tariff.copy()
            elif self.historic_feed_tariff_by_day:
                if target_date not in self.historic_feed_tariff_by_day:
                    raise KeyError(f"No historic feed tariff found for {target_date}")
                self.feed_tariff = self.historic_feed_tariff_by_day[target_date].copy()

        self.update_min_max()

    def update_next_24hr_from_historic(self, target_date, update_feed: bool = True) -> None:
        """
        Load a specific day's tariff from uploaded historic data into:
          - self.next_24hr_tariff
          - self.next_24hr_feed_tariff
        """
        target_date = self._as_date(target_date)

        if not self.historic_tariff_by_day:
            raise ValueError("No historic import tariff data uploaded.")
        if target_date not in self.historic_tariff_by_day:
            raise KeyError(f"No historic import tariff found for {target_date}")

        self.next_24hr_tariff = self.historic_tariff_by_day[target_date].copy()

        if update_feed:
            if self.type == 1:
                self.next_24hr_feed_tariff = self.next_24hr_tariff.copy()
            elif self.historic_feed_tariff_by_day:
                if target_date not in self.historic_feed_tariff_by_day:
                    raise KeyError(f"No historic feed tariff found for {target_date}")
                self.next_24hr_feed_tariff = self.historic_feed_tariff_by_day[target_date].copy()

    def prepare_day_ahead_tariffs(self, now_time, update_feed: bool = True) -> None:
        """
        For a given simulator timestamp:
          - self.tariff = tariff for current date
          - self.next_24hr_tariff = tariff for next date
        """
        now_ts = pd.Timestamp(now_time)
        today = now_ts.date()
        tomorrow = (now_ts + pd.Timedelta(days=1)).date()

        self.set_current_tariff_from_historic(today, update_feed=update_feed)
        self.update_next_24hr_from_historic(tomorrow, update_feed=update_feed)

    # -------------------------------------------------------------------------
    # Tariff access
    # -------------------------------------------------------------------------
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

        if self.type == 1:
            feed_tariff = tariff

        return float(tariff), float(feed_tariff)

    def get_tariff_range_df(
        self,
        now_time: datetime,
        period: int = 96,
        resolution: timedelta = timedelta(minutes=15),
    ) -> pd.DataFrame:
        """
        Returns tariff & feed tariff for next 'period' steps.
        Handles crossing midnight by constructing a 48h datetime-index tariff from
        (today=current tariff) + (tomorrow=next_24hr if available).
        """
        now_ts = pd.Timestamp(now_time)

        s_tariff_48 = self._build_48h_tariff_series(now_time, which="tariff")
        s_feed_48 = self._build_48h_tariff_series(now_time, which="feed_tariff")

        idx = pd.date_range(start=now_ts, periods=period, freq=pd.Timedelta(resolution))

        tariff_vals = s_tariff_48.reindex(idx, method="ffill")
        feed_vals = s_feed_48.reindex(idx, method="ffill")

        tariff_vals = tariff_vals.bfill().fillna(0.0)
        feed_vals = feed_vals.bfill().fillna(0.0)

        if self.type == 1:
            feed_vals = tariff_vals.copy()

        df = pd.DataFrame(
            {
                "tariff": tariff_vals.values.astype(float),
                "feed_tariff": feed_vals.values.astype(float),
            },
            index=idx,
        )
        df.index.name = "time"
        return df

    # -------------------------------------------------------------------------
    # Commit staged tariffs
    # -------------------------------------------------------------------------
    def updated_tariff(self) -> None:
        """
        If the next-day tariffs are staged, commit them and refresh min/max.
        """
        if self.next_24hr_tariff is not None:
            logger.commandline("Updating tariff value")
            self.tariff = self.next_24hr_tariff
            self.next_24hr_tariff = None

        if self.next_24hr_feed_tariff is not None:
            logger.commandline("Updating feed tariff value")
            self.feed_tariff = self.next_24hr_feed_tariff
            self.next_24hr_feed_tariff = None

        if self.type == 1 and self.tariff is not None:
            self.feed_tariff = self.tariff.copy()

        self.update_min_max()

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------
    def update_min_max(self) -> None:
        if self.tariff is not None and not self.tariff.empty and "value" in self.tariff.columns:
            self.max_tariff = float(self.tariff["value"].max())
            self.min_tariff = float(self.tariff["value"].min())
            self.avg_tariff = float(self.tariff["value"].mean())

        if self.feed_tariff is not None and not self.feed_tariff.empty and "value" in self.feed_tariff.columns:
            self.max_feed_tariff = float(self.feed_tariff["value"].max())
            self.min_feed_tariff = float(self.feed_tariff["value"].min())
            self.avg_feed_tariff = float(self.feed_tariff["value"].mean())