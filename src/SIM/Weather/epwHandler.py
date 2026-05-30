import pandas as pd
from datetime import datetime, timedelta

class EPWWeatherHandler:
    """Handles .epw weather files and provides ambient temperature and PV generation data."""

    def __init__(self, epw_file: str):
        self.epw_file = epw_file
        self.weather_df = self._load_epw(epw_file)

        # Cache for max-continuous slot per frequency
        self._max_slot_cache: dict[pd.Timedelta, dict] = {}

    # ---------------------------------------------------------
    #               LOAD EPW WEATHER FILE
    # ---------------------------------------------------------
    def _load_epw(self, file_path) -> pd.DataFrame:
        df = pd.read_csv(file_path, skiprows=8, header=None)

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
        ]

        # EPW hour starts at 1 → convert to 0–23
        df["Hour"] = df["Hour"] - 1

        df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
        df = df.set_index("datetime").sort_index()

        diffs = df.index.to_series().diff().dropna()
        self.epw_resolution = diffs.mode()[0]

        print("EPW Data Loaded:")
        print(f"  Start datetime: {df.index.min()}")
        print(f"  End datetime:   {df.index.max()}")
        print(f"  Detected timestep: {self.epw_resolution}")

        return df

    # ---------------------------------------------------------
    #      MAX CONTINUOUS SLOT (CACHED PER FREQ)
    # ---------------------------------------------------------
    def _get_max_continuous_slot(self, freq: pd.Timedelta) -> dict:
        """
        Returns dict: {freq, start, end, duration} for the longest continuous run at `freq`.
        Cached so repeated calls are cheap.
        """
        if freq in self._max_slot_cache:
            return self._max_slot_cache[freq]

        idx = self.weather_df.index
        if len(idx) < 2:
            slot = {"freq": freq, "start": idx[0], "end": idx[0], "duration": pd.Timedelta(0)}
            self._max_slot_cache[freq] = slot
            return slot

        # breaks where diff != freq
        diffs = idx.to_series().diff()
        break_pos = diffs.ne(freq).fillna(True).to_numpy().nonzero()[0]  # positions where a new segment starts

        starts = break_pos
        ends = list(break_pos[1:]) + [len(idx)]

        best_s, best_e, best_len = starts[0], ends[0], ends[0] - starts[0]
        for s, e in zip(starts, ends):
            L = e - s
            if L > best_len:
                best_s, best_e, best_len = s, e, L

        duration = (best_len - 1) * freq if best_len >= 2 else pd.Timedelta(0)
        slot = {
            "freq": freq,
            "start": idx[best_s],
            "end": idx[best_e - 1],
            "duration": duration,
        }
        self._max_slot_cache[freq] = slot
        return slot

    # ---------------------------------------------------------
    #          PUBLIC: GET WINDOW (CONTINUOUS OR STITCHED)
    # ---------------------------------------------------------
    def get_weather_slice(
            self,
            start_date: datetime,
            duration: timedelta,
            resolution: timedelta | None = None,
            search_forward: bool = True,
            allow_stitch: bool = True,
            verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a window of EPW data.

        Preference order:
        1) continuous window starting at the first timestamp >= start_date
        2) if search_forward, next continuous window later in the file
        3) if allow_stitch, stitch multiple continuous runs and PACK onto a continuous timeline
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
            print(f"⚠️ Requested start time {start_ts} is not an exact EPW timestamp.")
            pos_dbg = idx.searchsorted(start_ts)
            prev_ts = idx[pos_dbg - 1] if pos_dbg > 0 else None
            next_ts = idx[pos_dbg] if pos_dbg < len(idx) else None
            print(f"   Nearest previous: {prev_ts}")
            print(f"   Nearest next:     {next_ts}")

        # Report & store max continuous slot
        self.max_continuous_slot = self._get_max_continuous_slot(freq)
        if verbose:
            ms = self.max_continuous_slot
            print(f"📏 Max continuous slot at resolution={freq}: {ms['duration']} (from {ms['start']} to {ms['end']})")
            if dur > ms["duration"]:
                print(f"⚠️ Requested duration {dur} exceeds max continuous {ms['duration']} for resolution={freq}.")

        # Determine search start position (snap if out of range)
        start_pos = idx.searchsorted(start_ts)
        if start_pos >= len(idx):
            if verbose:
                print(f"⚠️ Requested start {start_ts} is after EPW end {idx.max()}.")
                print(f"👉 Falling back to EPW start {idx.min()} and searching from there.")
            start_pos = 0

        # ---------------------------------------------------------
        # Fast continuity detection
        # ---------------------------------------------------------
        if n_steps == 1:
            t0 = idx[start_pos]
            return df.loc[t0:t0]

        # gaps_ok is already a NumPy boolean array
        gaps_ok = (idx[1:] - idx[:-1]) == freq

        # prefix sums for O(1) window checks
        pref = gaps_ok.cumsum()

        def is_continuous_at(pos: int) -> bool:
            end_pos = pos + n_steps
            if end_pos > len(idx):
                return False

            left = pos
            right = end_pos - 2  # gaps index
            got = pref[right] - (pref[left - 1] if left > 0 else 0)
            return got == (n_steps - 1)

        # 1) try at start_pos
        if is_continuous_at(start_pos):
            t0 = idx[start_pos]
            t1 = t0 + dur
            if verbose:
                print(f"✅ Using continuous window start {t0} (resolution={freq}, duration={dur})")
            return df.loc[t0:(t1 - freq)]

        # If not searching forward, stop here
        if not search_forward:
            raise ValueError(
                f"Requested window is not continuous at {idx[start_pos]} "
                f"for duration={dur} and resolution={freq}."
            )

        # 2) vectorized search for any valid start >= start_pos
        # We want positions where window_gaps_sum == n_steps-1
        # Compute rolling sums of gaps_ok over window size (n_steps-1)
        w = n_steps - 1
        # rolling sum using prefix sums
        # window sum at position p is sum(gaps_ok[p:p+w]) which corresponds to gaps indices p..p+w-1
        # valid starts range: 0 .. len(idx)-n_steps
        total_starts = len(idx) - n_steps + 1
        if total_starts > 0:
            # sums for p=0..total_starts-1
            # sum(p..p+w-1) = pref[p+w-1] - pref[p-1]
            sums = pref[w - 1: w - 1 + total_starts].copy()
            if total_starts > 1:
                prev = pref[: total_starts - 1]
                sums[1:] = sums[1:] - prev

            # find first valid start >= start_pos
            valid = (sums == w)
            candidates = valid.nonzero()[0]
            candidates = candidates[candidates >= start_pos]  # enforce starting from requested
            if len(candidates) > 0:
                found_pos = int(candidates[0])
                t0 = idx[found_pos]
                t1 = t0 + dur
                if verbose:
                    print("⚠️ Requested start had missing data; found next continuous slot.")
                    print(f"✅ Using window start {t0} (resolution={freq}, duration={dur})")
                return df.loc[t0:(t1 - freq)]

        # 3) stitch fallback
        if allow_stitch:
            if verbose:
                print(f"⚠️ No single continuous slot found for duration={dur} at resolution={freq}.")
                print("👉 Stitching multiple segments and packing onto a continuous timeline...")
            return self.stitch_packed_from(start_pos=start_pos, n_steps=n_steps, freq=freq, verbose=verbose)

        raise ValueError(
            f"❌ No continuous slot found for duration={dur} at resolution={freq} starting from {idx[start_pos]}.\n"
            f"Max continuous slot is {self.max_continuous_slot['duration']} (from {self.max_continuous_slot['start']} "
            f"to {self.max_continuous_slot['end']})."
        )

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

        This ignores real-world time gaps between segments.
        """
        df = self.weather_df
        idx = df.index
        if len(idx) == 0:
            raise ValueError("No EPW data available.")

        if start_pos >= len(idx):
            start_pos = 0

        chunks = []
        p = start_pos
        collected = 0

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
            print(f"🧩 Stitched {n_steps} points from multiple segments and PACKED.")
            print(f"   Packed start: {out.index[0]}  Packed end: {out.index[-1]}")

        return out

    # ---------------------------------------------------------
    #           GET WEATHER WINDOW FOR SIMULATION
    # ---------------------------------------------------------
    def get_weather_window(
            self,
            start_date: datetime,
            duration: timedelta,
            resolution: timedelta | None = None
    ) -> pd.DataFrame:

        df = self.weather_df
        epw_start = df.index.min()
        epw_end = df.index.max()

        if start_date not in df.index:
            print(f"⚠️ Requested start date {start_date} not found in EPW file.")
            print(f"👉 Using EPW start date {epw_start} to generate data.")
            data_start = epw_start
        else:
            data_start = start_date

        sim_end = start_date + duration
        data_end = data_start + duration

        if data_end > epw_end:
            raise ValueError(
                f"Requested simulation end date {sim_end} exceeds "
                f"EPW data range (max available: {epw_end})."
            )
        # weather_slice = df.loc[data_start:data_end]
        weather_slice = self.get_weather_slice(start_date=start_date,
                                               duration=duration,
                                               resolution=timedelta(minutes=60))
        if resolution is not None:

            # 🔽 finer → ffill
            if resolution <= self.epw_resolution:
                step_minutes = int(resolution.total_seconds() // 60)

                epw_index = pd.date_range(
                    start=data_start,
                    end=data_end,
                    freq=f"{step_minutes}min"
                )

                sim_index = pd.date_range(
                    start=start_date,
                    end=sim_end,
                    freq=f"{step_minutes}min"
                )

                weather_slice = weather_slice.reindex(epw_index).ffill()
                weather_slice.index = sim_index

            # 🔼 coarser → mean
            else:
                weather_slice = weather_slice.resample(resolution).mean()
                weather_slice.index = (
                        weather_slice.index - weather_slice.index[0] + start_date
                )

        else:
            weather_slice.index = (
                    weather_slice.index - weather_slice.index[0] + start_date
            )

        return weather_slice

    def get_temperature(
            self,
            start_date: datetime,
            duration: timedelta,
            resolution: timedelta
    ) -> pd.Series:
        """
        Returns outdoor air temperature time series at requested resolution.
        """

        df_weather = self.get_weather_window(
            start_date=start_date,
            duration=duration,
            resolution=resolution
        )

        return df_weather["Dry Bulb Temperature"].rename("Temperature - Outdoor (C)")

    def get_pv_generation(
            self,
            start_date: datetime,
            duration: timedelta,
            resolution: timedelta,
            pv_capacity_kw: float = 5.0,
            pv_efficiency: float = 0.95
    ) -> pd.Series:
        """
        Returns PV electric power generation (kW).
        """

        df_weather = self.get_weather_window(
            start_date=start_date,
            duration=duration,
            resolution=resolution
        )

        ghi = df_weather["Global Horizontal Radiation"].fillna(0)

        # Physically consistent area
        pv_kw = (ghi / 1000.0) * pv_efficiency * pv_capacity_kw
        pv_kw = (pv_kw).clip(lower=0)

        return pv_kw.rename("PV Electric Power (kW)")

    # ---------------------------------------------------------
    #      GET TEMPERATURE + PV GENERATION FOR SIMULATION
    # ---------------------------------------------------------
    def get_simulation_data(
            self,
            start_date: datetime,
            duration: timedelta,
            sim_resolution: timedelta,
            pv_capacity_kw: float = 5.0,
            pv_efficiency: float = 0.18,
            area_per_kw: float = 1,
    ) -> pd.DataFrame:
        """
        Returns simulation-ready weather + PV data at the requested resolution.

        Resolution is handled strictly as timedelta.
        """

        # -------------------------------------------------
        # 1. Get weather at simulation resolution
        # -------------------------------------------------
        df_weather = self.get_weather_window(
            start_date=start_date,
            duration=duration,
            resolution=sim_resolution
        )

        # -------------------------------------------------
        # 2. Compute PV generation
        # -------------------------------------------------
        temperature = df_weather["Dry Bulb Temperature"]
        ghi = df_weather["Global Horizontal Radiation"].fillna(0)

        pv_area = pv_capacity_kw * area_per_kw
        pv_kw = (ghi / 1000.0) * pv_area * pv_efficiency
        pv_kw = pv_kw.clip(lower=0)

        df_sim = pd.DataFrame(
            {
                "Temperature - Outdoor (C)": temperature,
                "PV Electric Power (kW)": pv_kw,
            },
            index=df_weather.index
        )

        # -------------------------------------------------
        # 3. Final length validation
        # -------------------------------------------------
        expected_len = int(duration.total_seconds() // sim_resolution.total_seconds()) + 1
        if len(df_sim) != expected_len:
            raise RuntimeError(
                f"Simulation length mismatch: expected {expected_len}, got {len(df_sim)}"
            )

        return df_sim