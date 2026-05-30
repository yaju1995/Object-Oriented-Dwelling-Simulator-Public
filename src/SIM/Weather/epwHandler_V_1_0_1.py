import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from support.lib_config import CustomLogger

logger = CustomLogger(command=True, color='yellow')


class EPWWeatherHandler:
    """
    Handles .epw weather files and provides ambient temperature and
    PV generation data over a requested time duration.
    """

    def __init__(self, epw_file: str):
        logger.commandline(f'Loading .epw File: {epw_file}')
        self.epw_file = epw_file
        self.weather_df = self._load_epw(epw_file)
        self.weather_window = None

    # ---------------------------------------------------------
    #               LOAD EPW WEATHER FILE
    # ---------------------------------------------------------
    def _load_epw(self, file_path) -> pd.DataFrame:
        """Loads EPW weather file and returns dataframe with parsed datetime index,
        and automatically detects time resolution (step size)."""

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

        # Create datetime index
        df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
        df = df.set_index("datetime")

        # ---------------------------------------------------------
        # Detect EPW resolution as timedelta
        # ---------------------------------------------------------
        diffs = df.index.to_series().diff().dropna()
        self.epw_resolution = diffs.mode()[0]

        print("EPW Data Loaded:")
        print(f"  Start datetime: {df.index.min()}")
        print(f"  End datetime:   {df.index.max()}")
        print(f"  Detected timestep: {self.epw_resolution}")

        return df

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
        weather_slice = df.loc[data_start:data_end]
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
