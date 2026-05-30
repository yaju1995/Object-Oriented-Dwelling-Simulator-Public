import pytest
from datetime import timedelta, datetime

import numpy as np
import pandas as pd

from ess_handler import ESSHandler   # adjust import path


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def base_ess():
    """
    10 kWh battery
    5 kW charge / discharge
    15-min resolution
    """
    return ESSHandler(
        name="TestESS",
        total_capacity_Wh=10_000,
        initial_soc_pct=50,
        charging_power_W=5_000,
        discharging_power_W=5_000,
        resolution=timedelta(minutes=15),
        upper_limit_soc_pct=100,
        lower_limit_soc_pct=0,
        in_eff=1.0,
        out_eff=1.0,
    )


# ---------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------

def test_initialization(base_ess):
    assert base_ess.getStateOfCharge() == pytest.approx(50.0)
    assert base_ess.avai_capacity_Wh == pytest.approx(5000.0)
    assert base_ess.dt_hours == pytest.approx(0.25)


# ---------------------------------------------------------------------
# 2. Charge step
# ---------------------------------------------------------------------

def test_charge_step(base_ess):
    result = base_ess.update(2000)  # 2 kW charge

    # 2 kW * 0.25 h = 0.5 kWh
    assert result["Battery Electric Power (kW)"] == pytest.approx(2.0)
    assert result["Battery SOC (-)"] == pytest.approx(0.55, abs=1e-4)


# ---------------------------------------------------------------------
# 3. Discharge step
# ---------------------------------------------------------------------

def test_discharge_step(base_ess):
    result = base_ess.update(-3000)  # 3 kW discharge

    # 3 kW * 0.25 h = 0.75 kWh
    assert result["Battery Electric Power (kW)"] == pytest.approx(-3.0)
    assert result["Battery SOC (-)"] == pytest.approx(0.425, abs=1e-4)


# ---------------------------------------------------------------------
# 4. Power limit enforcement
# ---------------------------------------------------------------------

def test_power_limit(base_ess):
    result = base_ess.update(10_000)  # request > max charge

    # clipped to 5 kW
    assert result["Battery Electric Power (kW)"] == pytest.approx(5.0)


# ---------------------------------------------------------------------
# 5. SOC upper limit enforcement
# ---------------------------------------------------------------------

def test_soc_upper_limit():
    ess = ESSHandler(
        name="FullESS",
        total_capacity_Wh=10_000,
        initial_soc_pct=99,
        charging_power_W=5_000,
        discharging_power_W=5_000,
        resolution=timedelta(hours=1),
    )

    result = ess.update(5000)

    # only 1% SOC space left → 100 Wh max
    assert ess.avai_capacity_Wh <= ess.total_capacity_Wh
    assert result["Battery SOC (-)"] <= 1.0


# ---------------------------------------------------------------------
# 6. Resolution scaling correctness
# ---------------------------------------------------------------------

def test_resolution_scaling():
    ess_1h = ESSHandler(
        name="ESS_1h",
        total_capacity_Wh=10_000,
        initial_soc_pct=50,
        charging_power_W=1_000,
        discharging_power_W=1_000,
        resolution=timedelta(hours=1),
    )

    ess_15m = ESSHandler(
        name="ESS_15m",
        total_capacity_Wh=10_000,
        initial_soc_pct=50,
        charging_power_W=1_000,
        discharging_power_W=1_000,
        resolution=timedelta(minutes=15),
    )

    ess_1h.update(1000)
    for _ in range(4):
        ess_15m.update(1000)

    assert ess_1h.getStateOfCharge() == pytest.approx(
        ess_15m.getStateOfCharge(), abs=1e-6
    )


# ---------------------------------------------------------------------
# 7. Self-discharge correctness
# ---------------------------------------------------------------------

def test_self_discharge():
    ess = ESSHandler(
        name="SelfDischargeESS",
        total_capacity_Wh=10_000,
        initial_soc_pct=100,
        charging_power_W=0,
        discharging_power_W=0,
        resolution=timedelta(hours=1),
        enable_self_discharge=True,
        self_discharge_per_day_pct=2.4,  # 2.4% per day → 0.1% per hour
    )

    ess.update(0)

    expected_loss = 10_000 * 0.001
    assert ess.avai_capacity_Wh == pytest.approx(10_000 - expected_loss)


# ---------------------------------------------------------------------
# 8. Return contract validation
# ---------------------------------------------------------------------

def test_return_contract(base_ess):
    result = base_ess.update(1000)

    assert isinstance(result, dict)
    assert "Battery SOC (-)" in result
    assert "Battery Electric Power (kW)" in result
    assert 0 <= result["Battery SOC (-)"] <= 1


# ---------------------------------------------------------------------
# 9. History DataFrame integrity
# ---------------------------------------------------------------------

def test_history_recording(base_ess):
    ts = datetime(2025, 1, 1, 0, 0)
    base_ess.update(1000, timestamp=ts)
    base_ess.update(-1000, timestamp=ts + timedelta(minutes=15))

    hist = base_ess.history

    assert isinstance(hist, pd.DataFrame)
    assert len(hist) == 2
    assert "timestamp" in hist.columns
    assert hist.iloc[0]["timestamp"] == ts
    assert hist.iloc[1]["timestamp"] > hist.iloc[0]["timestamp"]


if __name__ == '__main__':
    ess = base_ess()
    test_initialization(base_ess=ess)