from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from support.lib_config import CustomLogger

logger = CustomLogger(command=True)


@dataclass(frozen=True)
class ContinuousSlot:
    """Represents a continuous run in the EPW index at a given frequency."""
    freq: pd.Timedelta
    start: pd.Timestamp
    end: pd.Timestamp  # inclusive timestamp of the last point in the run
    duration: pd.Timedelta  # (len(run)-1) * freq


class EPWWeatherHandler:
    """
    Handles .epw weather files and provides ambient temperature and PV generation data.

    Design goals:
    - Robust slice selection even when EPW has missing timestamps
    - Fast continuity checks using vectorized gap arrays and prefix sums
    - Optional "stitch + pack" fallback when no single continuous window exists
    - No print() calls; uses CustomLogger.commandline()
    """

    # EPW column names per EnergyPlus format (after skipping header rows)
    EPW_COLUMNS = [
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

    def __init__(self, epw_file: str):
        self.epw_file = epw_file
        self.weather_df = self._load_epw(epw_file)

        # native EPW resolution is detected in _load_epw
        self.epw_resolution: pd.Timedelta

        # Cache for max continuous slot per frequency
        self._max_slot_cache: dict[pd.Timedelta, ContinuousSlot] = {}

        # Updated on each get_weather_slice call
        self.max_continuous_slot: Optional[ContinuousSlot] = None

    # ---------------------------------------------------------
    #               LOAD EPW WEATHER FILE
    # ---------------------------------------------------------
    def _load_epw(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, skiprows=8, header=None)
        df.columns = self.EPW_COLUMNS

        # EPW hour starts at 1 → convert to 0–23
        df["Hour"] = df["Hour"] - 1

        df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
        df = df.set_index("datetime").sort_index()

        diffs = df.index.to_series().diff().dropna()
        # mode() may return Timedelta or numpy timedelta64; wrap with pd.Timedelta for safety
        self.epw_resolution = pd.Timedelta(diffs.mode().iloc[0])

        logger.commandline("EPW Data Loaded:")
        logger.commandline(f"  Start datetime: {df.index.min()}")
        logger.commandline(f"  End datetime:   {df.index.max()}")
        logger.commandline(f"  Detected timestep: {self.epw_resolution}")

        return df

    # ---------------------------------------------------------
    #      MAX CONTINUOUS SLOT (CACHED PER FREQ)
    # ---------------------------------------------------------
    def _get_max_continuous_slot(self, freq: pd.Timedelta) -> ContinuousSlot:
        """
        Returns the longest continuous run at `freq` in the EPW index.
        Cached so repeated calls are cheap.
        """
        freq = pd.Timedelta(freq)

        cached = self._max_slot_cache.get(freq)
        if cached is not None:
            return cached

        idx = self.weather_df.index
        if len(idx) == 0:
            slot = ContinuousSlot(freq=freq, start=pd.Timestamp.min, end=pd.Timestamp.min, duration=pd.Timedelta(0))
            self._max_slot_cache[freq] = slot
            return slot

        if len(idx) == 1:
            slot = ContinuousSlot(freq=freq, start=idx[0], end=idx[0], duration=pd.Timedelta(0))
            self._max_slot_cache[freq] = slot
            return slot

        diffs = idx.to_series().diff()
        # segment starts at positions where diff != freq (and at 0)
        break_pos = diffs.ne(freq).fillna(True).to_numpy().nonzero()[0]  # start indices of segments

        starts = break_pos
        ends = np.append(break_pos[1:], len(idx))  # exclusive ends

        lens = ends - starts
        best_i = int(np.argmax(lens))
        best_s = int(starts[best_i])
        best_e = int(ends[best_i])  # exclusive
        best_len = int(lens[best_i])

        duration = (best_len - 1) * freq if best_len >= 2 else pd.Timedelta(0)

        slot = ContinuousSlot(
            freq=freq,
            start=idx[best_s],
            end=idx[best_e - 1],
            duration=duration,
        )
        self._max_slot_cache[freq] = slot
        return slot

    # ---------------------------------------------------------
    #     STITCH MULTIPLE RUNS + PACK TO CONTINUOUS TIMELINE
    # ---------------------------------------------------------
    def stitch_packed_from(
            self,
            start_pos: int,
            n_steps: int,
            freq: pd.Timedelta,
            verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Stitch multiple continuous runs to reach n_steps rows, then PACK them
        onto a continuous timeline (freq) starting at the first chosen timestamp.

        NOTE: This ignores real-world time gaps between segments.
        """
        df = self.weather_df
        print(len(df))
        print(df.head(24))
        print(df.reset_index().tail(24))
        idx = df.index
        if len(idx) == 0:
            raise ValueError("No EPW data available.")

        start_pos = int(start_pos)
        if start_pos >= len(idx):
            start_pos = 0

        chunks = []
        p = start_pos
        collected = 0
        freq = pd.Timedelta(freq)

        while p < len(idx) and collected < n_steps:
            # Find end of continuous run starting at p
            end = p + 1
            while end < len(idx) and (idx[end] - idx[end - 1]) == freq:
                end += 1

            need = n_steps - collected
            take_end = min(end, p + need)

            chunk = df.iloc[p:take_end]
            if not chunk.empty:
                chunks.append(chunk)
                collected += len(chunk)

            p = end  # jump over the gap to next run

        if collected < n_steps:
            raise ValueError(
                f"Could only stitch {collected} points but need {n_steps} for resolution={freq}."
            )

        out = pd.concat(chunks, axis=0).iloc[:n_steps].copy()

        # Pack onto a continuous timeline
        t0 = out.index[0]
        out.index = pd.date_range(start=t0, periods=n_steps, freq=freq)
        out.index.name = idx.name

        if verbose:
            logger.commandline(f"🧩 Stitched {n_steps} points from multiple segments and PACKED.")
            logger.commandline(f"   Packed start: {out.index[0]}  Packed end: {out.index[-1]}")

        return out

    # ---------------------------------------------------------
    #          PUBLIC: GET SLICE (CONTINUOUS OR STITCHED)
    # ---------------------------------------------------------
    def get_weather_slice(
            self,
            start_date: datetime,
            duration: timedelta,
            resolution: Optional[timedelta] = None,
            search_forward: bool = True,
            allow_stitch: bool = True,
            verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a window of EPW data at *native index points* (no resampling here).
        This function focuses on:
        - picking a window of length `duration` at step `resolution` (or EPW native)
        - ensuring continuity, optionally searching forward and/or stitching.

        The returned DataFrame index corresponds to the packed/selected timestamps.
        """
        df = self.weather_df
        idx = df.index
        if df.empty:
            raise ValueError("EPW dataframe is empty.")

        freq = pd.Timedelta(resolution) if resolution is not None else pd.Timedelta(self.epw_resolution)
        if freq <= pd.Timedelta(0):
            raise ValueError("resolution must be a positive timedelta.")

        dur = pd.Timedelta(duration)
        if dur <= pd.Timedelta(0):
            raise ValueError("duration must be a positive timedelta.")
        if dur % freq != pd.Timedelta(0):
            raise ValueError(f"duration ({dur}) must be an integer multiple of resolution ({freq}).")

        n_steps = int(dur / freq)
        if n_steps <= 0:
            raise ValueError("duration too small for the chosen resolution.")

        start_ts = pd.Timestamp(start_date)

        # Warn if start isn't exactly present
        if verbose and start_ts not in idx:
            logger.commandline(f"⚠️ Requested start time {start_ts} is not an exact EPW timestamp.")
            pos_dbg = int(idx.searchsorted(start_ts))
            prev_ts = idx[pos_dbg - 1] if pos_dbg > 0 else None
            next_ts = idx[pos_dbg] if pos_dbg < len(idx) else None
            logger.commandline(f"   Nearest previous: {prev_ts}")
            logger.commandline(f"   Nearest next:     {next_ts}")

        # Report & store max continuous slot
        self.max_continuous_slot = self._get_max_continuous_slot(freq)
        if verbose and self.max_continuous_slot is not None:
            ms = self.max_continuous_slot
            logger.commandline(
                f"📏 Max continuous slot at resolution={freq}: {ms.duration} (from {ms.start} to {ms.end})"
            )
            if dur > ms.duration:
                logger.commandline(
                    f"⚠️ Requested duration {dur} exceeds max continuous {ms.duration} for resolution={freq}."
                )

        # Determine search start position (snap if out of range)
        start_pos = int(idx.searchsorted(start_ts))
        if start_pos >= len(idx):
            if verbose:
                logger.commandline(f"⚠️ Requested start {start_ts} is after EPW end {idx.max()}.")
                logger.commandline(f"👉 Falling back to EPW start {idx.min()} and searching from there.")
            start_pos = 0

        # Trivial case: one step => any point
        if n_steps == 1:
            t0 = idx[start_pos]
            return df.loc[t0:t0]

        # Fast continuity detection: gaps_ok[i] true if idx[i+1]-idx[i] == freq
        gaps_ok = ((idx[1:] - idx[:-1]) == freq)
        pref = gaps_ok.cumsum()  # numpy prefix sum

        def is_continuous_at(pos: int) -> bool:
            end_pos = pos + n_steps
            if end_pos > len(idx):
                return False
            left = pos
            right = end_pos - 2  # inclusive gap index
            got = pref[right] - (pref[left - 1] if left > 0 else 0)
            return got == (n_steps - 1)

        # 1) try at start_pos
        if is_continuous_at(start_pos):
            t0 = idx[start_pos]
            t1 = t0 + dur
            if verbose:
                logger.commandline(f"✅ Using continuous window start {t0} (resolution={freq}, duration={dur})")
            return df.loc[t0:(t1 - freq)]

        # If not searching forward, stop here
        if not search_forward:
            raise ValueError(
                f"Requested window is not continuous at {idx[start_pos]} "
                f"for duration={dur} and resolution={freq}."
            )

        # 2) vectorized search for any valid start >= start_pos
        w = n_steps - 1
        total_starts = len(idx) - n_steps + 1
        if total_starts > 0:
            # sums[p] = sum(gaps_ok[p:p+w]) using prefix sums
            sums = pref[w - 1: w - 1 + total_starts].copy()
            if total_starts > 1:
                sums[1:] = sums[1:] - pref[: total_starts - 1]

            candidates = np.nonzero(sums == w)[0]
            candidates = candidates[candidates >= start_pos]
            if candidates.size > 0:
                found_pos = int(candidates[0])
                t0 = idx[found_pos]
                t1 = t0 + dur
                if verbose:
                    logger.commandline("⚠️ Requested start had missing data; found next continuous slot.")
                    logger.commandline(f"✅ Using window start {t0} (resolution={freq}, duration={dur})")
                return df.loc[t0:(t1 - freq)]

        # 3) stitch fallback
        if allow_stitch:
            if verbose:
                logger.commandline(f"⚠️ No single continuous slot found for duration={dur} at resolution={freq}.")
                logger.commandline("👉 Stitching multiple segments and packing onto a continuous timeline...")
            return self.stitch_packed_from(start_pos=start_pos, n_steps=n_steps, freq=freq, verbose=verbose)

        # Hard fail
        ms = self.max_continuous_slot
        if ms is None:
            raise ValueError(f"❌ No continuous slot found for duration={dur} at resolution={freq}.")
        raise ValueError(
            f"❌ No continuous slot found for duration={dur} at resolution={freq} starting from {idx[start_pos]}.\n"
            f"Max continuous slot is {ms.duration} (from {ms.start} to {ms.end})."
        )

    # ---------------------------------------------------------
    #           GET WEATHER WINDOW FOR SIMULATION (RESAMPLED)
    # ---------------------------------------------------------
    def get_weather_window(
            self,
            start_date: datetime,
            duration: timedelta,
            resolution: Optional[timedelta] = None,
            *,
            search_forward: bool = True,
            allow_stitch: bool = True,
            verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a simulation-aligned weather dataframe of length:
            expected_len = duration / resolution + 1   (inclusive endpoints)

        Strategy:
        1) Take a native slice using get_weather_slice() at EPW native resolution (or `resolution` if coarser).
        2) If requested resolution is finer than EPW, reindex and forward-fill.
        3) If coarser, resample with mean.
        4) Rebase index to start_date (simulation timeline).
        """
        df = self.weather_df
        # epw_start = df.index.min()
        # epw_end = df.index.max()

        start_date = pd.Timestamp(start_date)
        # sim_end = start_date + pd.Timedelta(duration)

        # Choose slicing frequency:
        # - if resolution is None: slice at EPW native
        # - if resolution is coarser than EPW: slice at that coarser freq (still checks continuity)
        # - else (finer): slice at EPW native, then upsample
        req_res = pd.Timedelta(resolution) if resolution is not None else None
        slice_freq = self.epw_resolution if (req_res is None or req_res < self.epw_resolution) else req_res

        # if start_date not in df.index and verbose:
        #     logger.commandline(f"⚠️ Requested start date {start_date} not found in EPW index.")
        #     logger.commandline(f"👉 Will search from the nearest available timestamps (EPW starts at {epw_start}).")
        # if sim_end not in df.index and verbose:
        #     logger.commandline(f"⚠️ Requested end date {sim_end} not found in EPW index.")
        #     logger.commandline(f"👉 Will search from the nearest available timestamps (EPW starts at {epw_start}).")
        # Get the base slice (may search/stitch)
        weather_slice = self.get_weather_slice(
            start_date=start_date.to_pydatetime(),
            duration=duration,
            resolution=slice_freq,
            search_forward=search_forward,
            allow_stitch=allow_stitch,
            verbose=verbose,
        )

        # Now resample/upsample to requested resolution, and rebase to simulation timeline
        if req_res is not None:
            # Finer than slice_freq -> upsample with forward fill on a full grid
            if req_res <= slice_freq:
                step_minutes = int(req_res.total_seconds() // 60)
                if step_minutes <= 0:
                    raise ValueError("resolution must be at least 1 minute when upsampling.")

                # Build EPW-aligned target index from slice start (not necessarily start_date if stitched)
                t0 = weather_slice.index[0]
                target_epw_index = pd.date_range(
                    start=t0,
                    periods=int(pd.Timedelta(duration) / req_res) + 1,
                    freq=req_res,
                )

                weather_slice = weather_slice.reindex(target_epw_index).ffill()

            else:
                # Coarser -> downsample by mean
                weather_slice = weather_slice.resample(req_res).mean()

            # Rebase to simulation index
            sim_index = pd.date_range(
                start=start_date,
                periods=len(weather_slice),
                freq=req_res
            )
            weather_slice = weather_slice.copy()
            weather_slice.index = sim_index

        else:
            # No resolution requested: just rebase to EPW slice timeline step
            base_freq = pd.Timedelta(weather_slice.index[1] - weather_slice.index[0]) if len(
                weather_slice) > 1 else self.epw_resolution
            sim_index = pd.date_range(
                start=start_date,
                periods=len(weather_slice),
                freq=base_freq
            )
            weather_slice = weather_slice.copy()
            weather_slice.index = sim_index

        return weather_slice

    def get_temperature(
            self,
            start_date: datetime,
            duration: timedelta,
            resolution: timedelta,
    ) -> pd.Series:
        """Returns outdoor air temperature time series at requested resolution."""
        df_weather = self.get_weather_window(
            start_date=start_date,
            duration=duration,
            resolution=resolution,
        )
        return df_weather["Dry Bulb Temperature"].rename("Temperature - Outdoor (C)")

    def get_pv_generation(
            self,
            start_date: datetime,
            duration: timedelta,
            resolution: timedelta,
            pv_capacity_kw: float = 5.0,
            pv_efficiency: float = 0.95,
    ) -> pd.Series:
        """
        Returns PV electric power generation (kW).
        Simple scaling on GHI: pv_kw = (GHI/1000) * pv_capacity_kw * pv_efficiency
        """
        df_weather = self.get_weather_window(
            start_date=start_date,
            duration=duration,
            resolution=resolution,
        )

        ghi = df_weather["Global Horizontal Radiation"].fillna(0)
        pv_kw = (ghi / 1000.0) * pv_efficiency * pv_capacity_kw
        pv_kw = pv_kw.clip(lower=0)
        return pv_kw.rename("PV Electric Power (kW)")

    # ---------------------------------------------------------
    #      GET TEMPERATURE + PV GENERATION FOR SIMULATION
    # ---------------------------------------------------------
    def get_simulation_data(
            self,
            start_date: datetime,
            duration: timedelta,
            resolution: timedelta,
    ) -> pd.DataFrame:
        """
        Returns simulation-ready weather + PV data at the requested resolution.
        """
        df_weather = self.get_weather_window(
            start_date=start_date,
            duration=duration,
            resolution=resolution,
        )

        temperature = df_weather["Dry Bulb Temperature"]
        ghi = df_weather["Global Horizontal Radiation"].fillna(0)

        df_sim = pd.DataFrame(
            {
                "Temperature - Outdoor (C)": temperature,
                "Global Horizontal Radiation": ghi,
            },
            index=df_weather.index,
        )

        expected_len = int(pd.Timedelta(duration).total_seconds() // pd.Timedelta(resolution).total_seconds()) + 1
        if len(df_sim) != expected_len:
            raise RuntimeError(f"Simulation length mismatch: expected {expected_len}, got {len(df_sim)}")

        return df_sim
