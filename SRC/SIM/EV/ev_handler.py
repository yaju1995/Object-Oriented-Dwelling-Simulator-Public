from __future__ import annotations

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from SIM.ESS.ess_handler import ESSHandler
from support.lib_config import CustomLogger
from SIM.EV.ev_profile_generator import generate_ev_sessions

logger = CustomLogger(command=True)


class EVHandler(ESSHandler):
    """
    EV model driven entirely by a CSV profile.
    """

    def __init__(
        self,
        name: str,
        ev_profile_csv: str,
        start_time:datetime,
        resolution:timedelta,
        duration:timedelta,
        total_capacity_Wh: float,
        charging_power_W: float,
        discharging_power_W: float,
        upper_limit_soc_pct: float,
        lower_limit_soc_pct: float,
        v2g_enabled: bool = False,



        in_eff: float = 1.0,
        out_eff: float = 1.0,
        seed:int= 42
    ):
        # -------------------------------------------------
        # Load EV profile
        # -------------------------------------------------
        self.ev_df = generate_ev_sessions(ev_profile_csv,
                                          start_time,resolution,duration,
                                          seed=seed)
        # logger.commandline(self.ev_df)
        self._validate_ev_profile()

        # -------------------------------------------------
        # Initialize ESS (SOC will be set on first plug-in)
        # -------------------------------------------------
        super().__init__(
            name=name,
            total_capacity_Wh=total_capacity_Wh,
            initial_soc_pct=0.0,  # placeholder, overwritten on plug-in
            charging_power_W=charging_power_W,
            discharging_power_W=discharging_power_W,
            resolution=resolution,
            upper_limit_soc_pct=upper_limit_soc_pct,
            lower_limit_soc_pct=lower_limit_soc_pct,
            in_eff=in_eff,
            out_eff=out_eff,
        )

        self.v2g_enabled = v2g_enabled

        # runtime state
        self.active_session_idx: Optional[int] = None
        self.previous_plugged: bool = False

        self.user_set_plugout = None
        self.expected_soc = 100

    # ==================================================================
    # Validation
    # ==================================================================

    def _validate_ev_profile(self):
        required = {
            "plug_in_time",
            "plug_out_time",
            "initial_soc",
            "day_id",
            "weekday",
        }
        missing = required - set(self.ev_df.columns)
        if missing:
            raise ValueError(f"EV profile missing columns: {missing}")

        # Validation: allow datetime dtype OR all-null column
        for col in ["plug_in_time", "plug_out_time"]:
            series = self.ev_df[col]
            if not pd.api.types.is_datetime64_any_dtype(series):
                # If dtype is not datetime, check if all values are None/NaT
                if not series.isna().all():
                    raise TypeError(f"{col} must be datetime or None")

        if (self.ev_df["plug_out_time"] <= self.ev_df["plug_in_time"]).any():
            raise ValueError("plug_out_time must be after plug_in_time")

    # ==================================================================
    # Session lookup
    # ==================================================================

    def _find_active_session(self, timestamp: datetime) -> Optional[int]:
        """
        Return index of active EV session, or None.
        """
        mask = (
            (self.ev_df["plug_in_time"] <= timestamp)
            & (timestamp < self.ev_df["plug_out_time"])
        )

        if not mask.any():
            return None

        return int(mask.idxmax())

    # ==================================================================
    # Step
    # ==================================================================

    def step(
            self,
            timestamp: datetime,
            control_power_W: Optional[float] = 0,
    ) -> dict:
        """
        Advance EV by one timestep.

        Returns
        -------
        dict with:
            - EV Electric Power (kW)
            - EV SOC (-)
            - EV Parked (0/1)
        """
        # logger.commandline('EV Step!!')
        session_idx = self._find_active_session(timestamp)
        plugged = session_idx is not None

        # -------------------------------------------------
        # Plug-in event
        # -------------------------------------------------
        if plugged and not self.previous_plugged:
            row = self.ev_df.loc[session_idx]
            init_soc = float(row["initial_soc"])
            self.user_set_plugout = row['plug_out_time']
            self.expected_soc = 100

            logger.commandline(
                f"[EV] Plug-in @ {timestamp} | "
                f"SOC reset → {init_soc:.2f}% | "
                f"day_id={row['day_id']} weekday={row['weekday']} |"
                f"Plug-out @ {self.user_set_plugout}"
            )

            self.set_soc(init_soc, normalized=False, clip=False) # initalized soc
            self.active_session_idx = session_idx # set session to active

        # -------------------------------------------------
        # Plug-out event
        # -------------------------------------------------
        if not plugged and self.previous_plugged:
            logger.commandline(
                f"[EV] Plug-out @ {timestamp} | "
                f"SOC={self.getStateOfCharge():.2f}%"
            )
            self.active_session_idx = None

        self.previous_plugged = plugged

        # -------------------------------------------------
        # Power decision
        # -------------------------------------------------
        if not plugged:
            power_setpoint_W = 0.0
            self.user_set_plugout = None
        else:
            if control_power_W is None:
                power_setpoint_W = 0 # charge with 0 power
            else:
                power_setpoint_W = float(control_power_W)

            if not self.v2g_enabled:
                power_setpoint_W = max(0.0, power_setpoint_W)

        # -------------------------------------------------
        # ESS update (power-based)
        # -------------------------------------------------
        out = self.update(
            power_setpoint_W=power_setpoint_W,
            timestamp=timestamp,
        )

        ev_power_kw = out["Battery Electric Power (kW)"]
        ev_set_power_kw = out["Battery Set Power (kW)"]
        ev_soc = out["Battery SOC (-)"] if plugged else 0.0

        return {
            "EV Electric Power (kW)": round(ev_power_kw, 6),
            "EV Set Power (kW)": round(ev_set_power_kw,6),
            "EV SOC (-)": round(ev_soc, 3),
            "EV Parked": int(plugged),
            "User Plug Out Time": self.user_set_plugout,
            "Expected SOC": self.expected_soc
        }

    # ==================================================================
    def getEVStateOfCharge(self, timestamp: datetime) -> float:
        return self.getStateOfCharge() if self._find_active_session(timestamp) else 0.0

    # ==================================================================
    def upload_ev_df_from_csv(self, csv_path: str):
        """
        Directly load EV session DataFrame from CSV.

        Expected columns:
            plug_in_time, plug_out_time, initial_soc, day_id, weekday
        """

        logger.commandline(f"[EV] Loading EV profile from CSV: {csv_path}")

        df = pd.read_csv(
            csv_path,
            parse_dates=["plug_in_time", "plug_out_time"],
        )

        # sort & reset index
        df = df.sort_values("plug_in_time").reset_index(drop=True)

        self.ev_df = df
        self._validate_ev_profile()

        # reset runtime state
        self.active_session_idx = None
        self.previous_plugged = False

        logger.commandline(
            f"[EV] Loaded {len(self.ev_df)} EV sessions from CSV"
        )
