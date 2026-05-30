from __future__ import annotations

from datetime import timedelta, datetime
from typing import Optional

import pandas as pd

from support.lib_config import CustomLogger

logger = CustomLogger(command=False)


class ESSHandler:
    """
    Energy Storage System (ESS)

    Sign convention:
      + power_setpoint_W : charge battery
      - power_setpoint_W : discharge battery

    update() advances the ESS by exactly `resolution`.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        name: str,
        total_capacity_Wh: float,
        initial_soc_pct: float,
        charging_power_W: float,
        discharging_power_W: float,
        resolution: timedelta,

        upper_limit_soc_pct: float = 100.0,
        lower_limit_soc_pct: float = 0.0,

        in_eff: float = 1.0,
        out_eff: float = 1.0,

        enable_self_discharge: bool = False,
        self_discharge_per_day_pct: float = 0.0,

        init_battery_cycle: float = 0.0,
        enable_battery_degradation: bool = False,
        max_cycle: Optional[float] = None,
        max_cycle_dod_pct: Optional[float] = None,
    ):
        # ---------------- Validation ----------------
        if total_capacity_Wh <= 0:
            raise ValueError("total_capacity_Wh must be > 0")
        if charging_power_W < 0 or discharging_power_W < 0:
            raise ValueError("Power ratings must be >= 0")
        if not (0 <= initial_soc_pct <= 100):
            raise ValueError("initial_soc_pct must be in [0, 100]")
        if not (0 <= lower_limit_soc_pct <= upper_limit_soc_pct <= 100):
            raise ValueError("Invalid SoC limits")
        if in_eff <= 0 or in_eff > 1:
            raise ValueError("in_eff must be in (0,1]")
        if out_eff <= 0 or out_eff > 1:
            raise ValueError("out_eff must be in (0,1]")
        if resolution.total_seconds() <= 0:
            raise ValueError("resolution must be positive timedelta")

        # ---------------- Configuration ----------------
        self.name = name
        self.resolution = resolution
        self.dt_hours = resolution.total_seconds() / 3600.0

        self.rated_capacity_Wh = float(total_capacity_Wh)
        self.charging_power_W = float(charging_power_W)
        self.discharging_power_W = float(discharging_power_W)
        self.upper_limit_soc_pct = float(upper_limit_soc_pct)
        self.lower_limit_soc_pct = float(lower_limit_soc_pct)

        self.in_eff = float(in_eff)
        self.out_eff = float(out_eff)

        self.enable_self_discharge = enable_self_discharge
        self.self_discharge_per_day_pct = float(self_discharge_per_day_pct)

        self.enable_battery_degradation = enable_battery_degradation
        self.max_cycle = max_cycle
        self.max_cycle_dod_pct = max_cycle_dod_pct

        # ---------------- Capacity & SOH ----------------
        self.current_battery_cycle = float(init_battery_cycle)
        self.soh_pct = 100.0
        self.total_capacity_Wh = self.rated_capacity_Wh

        if self.enable_battery_degradation:
            if max_cycle is None or max_cycle_dod_pct is None:
                raise ValueError("max_cycle and max_cycle_dod_pct required")
            self._update_soh()

        self.upper_limit_Wh = self.upper_limit_soc_pct * self.total_capacity_Wh / 100
        self.lower_limit_Wh = self.lower_limit_soc_pct * self.total_capacity_Wh / 100

        self.inital_soc_pct = initial_soc_pct
        self.avai_capacity_Wh = self.inital_soc_pct * self.total_capacity_Wh / 100

        # cycle throughput
        self.Wh_cycle = self.current_battery_cycle * self.total_capacity_Wh

        # ---------------- Self-discharge ----------------
        self.self_discharge_Wh_step = 0.0
        if self.enable_self_discharge and self.self_discharge_per_day_pct > 0:
            daily_loss = self.total_capacity_Wh * self.self_discharge_per_day_pct / 100
            self.self_discharge_Wh_step = daily_loss * (self.dt_hours / 24.0)

        # ---------------- History ----------------
        self.history = pd.DataFrame(
            columns=[
                "timestamp",
                "power_setpoint_W",
                "actual_power_W",
                "battery_energy_delta_Wh",
                "soc_pct",
                "available_capacity_Wh",
                "battery_cycle",
                "soh_pct",
            ]
        )

    # ------------------------------------------------------------------
    def update(
        self,
        power_setpoint_W: float,
        timestamp: Optional[datetime] = None
    ) -> dict:
        """
        Advance ESS by one resolution step.

        Input:
            power_setpoint_W : float
                +ve charge, -ve discharge (W)

        Returns:
            dict with:
            {
                'Battery SOC (-)': float (0–1),
                'Battery Electric Power (kW)': float
            }
        """
        timestamp = timestamp or datetime.utcnow()
        # logger.commandline(power_setpoint_W)
        # ---------- Self-discharge ----------
        if self.self_discharge_Wh_step > 0:
            self._apply_battery_energy(-self.self_discharge_Wh_step)

        # ---------- Clamp power ----------

        req_power_W = max(
            -self.discharging_power_W,
            min(self.charging_power_W, power_setpoint_W),
        )
        # ---------- Power → terminal energy ----------
        terminal_Wh = req_power_W * self.dt_hours

        # ---------- Terminal → battery energy ----------
        if terminal_Wh > 0:
            batt_Wh = terminal_Wh * self.in_eff
        elif terminal_Wh < 0:
            batt_Wh = terminal_Wh / self.out_eff
        else:
            batt_Wh = 0.0

        # ---------- SoC limits ----------
        batt_Wh = self._limit_by_soc(batt_Wh)

        # ---------- Apply battery-side change ----------
        self._apply_battery_energy(batt_Wh)

        # ---------- Battery → terminal ----------
        if batt_Wh > 0:
            actual_terminal_Wh = batt_Wh / self.in_eff
        elif batt_Wh < 0:
            actual_terminal_Wh = batt_Wh * self.out_eff
        else:
            actual_terminal_Wh = 0.0

        actual_power_W = actual_terminal_Wh / self.dt_hours

        # ---------- Record history ----------
        self._record_step(
            timestamp,
            power_setpoint_W=req_power_W,
            actual_power_W=actual_power_W,
            batt_Wh=batt_Wh,
        )
        # kWh = round(actual_power_W / 1000.0, 6)
        # ---------- Return contract ----------
        return {
            'Battery SOC (-)': round(self.getStateOfCharge() / 100.0, 6),
            'Battery Electric Power (kW)': round(actual_power_W / 1000.0, 6),
            'Battery Set Power (kW)': round(power_setpoint_W/1000, 6)
        }

    # ------------------------------------------------------------------
    def _limit_by_soc(self, batt_Wh: float) -> float:

        if batt_Wh > 0:
            # if self.avai_capacity_Wh>self.upper_limit_Wh:
                # logger.commandline(f'Battery over charger to {self.getStateOfCharge()}%')
            return min(batt_Wh, self.upper_limit_Wh - self.avai_capacity_Wh)
        elif batt_Wh < 0:
            if self.avai_capacity_Wh<self.lower_limit_Wh:
                # logger.commandline(f'Battery below lower limit to {self.getStateOfCharge()}%')
                return 0.0
            return max(batt_Wh, self.lower_limit_Wh - self.avai_capacity_Wh)
        return 0.0

    def _apply_battery_energy(self, batt_Wh: float):
        if batt_Wh == 0:
            return

        self.avai_capacity_Wh += batt_Wh

        # cycle counting
        self.Wh_cycle += abs(batt_Wh) / 2
        self.current_battery_cycle = self.Wh_cycle / self.total_capacity_Wh

        if self.enable_battery_degradation:
            self._update_soh()
            self.upper_limit_Wh = self.upper_limit_soc_pct * self.total_capacity_Wh / 100
            self.lower_limit_Wh = self.lower_limit_soc_pct * self.total_capacity_Wh / 100

    def _update_soh(self):
        self.soh_pct = 100 - (self.current_battery_cycle / self.max_cycle) * (
            100 - self.max_cycle_dod_pct
        )
        self.soh_pct = max(self.max_cycle_dod_pct, self.soh_pct)
        self.total_capacity_Wh = self.rated_capacity_Wh * self.soh_pct / 100

    # ------------------------------------------------------------------
    def _record_step(
        self,
        timestamp,
        power_setpoint_W,
        actual_power_W,
        batt_Wh,
    ):
        self.history.loc[len(self.history)] = {
            "timestamp": timestamp,
            "power_setpoint_W": round(power_setpoint_W, 3),
            "actual_power_W": round(actual_power_W, 3),
            "battery_energy_delta_Wh": round(batt_Wh, 3),
            "soc_pct": round(self.getStateOfCharge(), 2),
            "available_capacity_Wh": round(self.avai_capacity_Wh, 3),
            "battery_cycle": round(self.current_battery_cycle, 6),
            "soh_pct": round(self.soh_pct, 3),
        }

    # ------------------------------------------------------------------
    def getStateOfCharge(self) -> float:
        return (self.avai_capacity_Wh / self.total_capacity_Wh) * 100

    def resetESSParameters(self):
        logger.commandline("------- ESS Reset ----------")
        self.current_battery_cycle = 0.0
        self.Wh_cycle = 0.0
        self.soh_pct = 100.0
        self.total_capacity_Wh = self.rated_capacity_Wh
        self.avai_capacity_Wh = self.upper_limit_Wh
        self.history = self.history.iloc[0:0]

    # ------------------------------------------------------------------
    def set_soc(
        self,
        soc: float,
        normalized: bool = True,
        clip: bool = True,
    ):
        """
        Force-update ESS SOC from an external system.

        Parameters
        ----------
        soc : float
            If normalized=True  -> SOC in [0, 1]
            If normalized=False -> SOC in [0, 100]

        normalized : bool
            Whether soc is normalized (0–1) or percent (0–100)

        clip : bool
            If True, SOC is clipped to [lower_limit, upper_limit]
            If False, invalid SOC raises ValueError

        Notes
        -----
        - This does NOT affect battery cycle count
        - This does NOT affect SOH
        - This does NOT record history
        """
        # -------- convert SOC to percentage --------
        if normalized:
            soc_pct = soc * 100.0
        else:
            soc_pct = soc

        # -------- validate / clip --------
        min_soc = self.lower_limit_soc_pct
        max_soc = self.upper_limit_soc_pct

        if clip:
            soc_pct = max(min_soc, min(max_soc, soc_pct))
        else:
            if not (0 <= soc_pct <=100):
                raise Warning(
                    f"SOC {soc_pct}% outside limits of 0 and 100"
                    f"[{min_soc}, {max_soc}]"
                )

        # -------- update energy state --------
        self.avai_capacity_Wh = (soc_pct / 100.0) * self.total_capacity_Wh

        logger.commandline(
            f"[ESS] {self.name}:SOC externally set to {soc_pct:.2f}% "
            f"({self.avai_capacity_Wh:.2f} Wh)"
        )

