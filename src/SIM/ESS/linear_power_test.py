from datetime import datetime, timedelta

from ess_handler import ESSHandler   # adjust path if needed


def main():
    # ------------------------------------------------------------
    # 1. Create ESS
    # ------------------------------------------------------------
    ess = ESSHandler(
        name="LinearTestESS",
        total_capacity_Wh=5_000,            # 5 kWh
        initial_soc_pct=100,                # start full
        charging_power_W=2_000,             # 2 kW max charge
        discharging_power_W=1_000,           # 1 kW max discharge
        resolution=timedelta(minutes=1),      # 1 hour per step
        in_eff=1.0,
        out_eff=1.0,
        upper_limit_soc_pct=100,
        lower_limit_soc_pct=0,
        max_cycle=10,
        max_cycle_dod_pct=80,
        enable_battery_degradation=True
    )

    start_time = datetime(2025, 1, 1, 0, 0)
    current_time = start_time

    print("\n--- DISCHARGE PHASE (1 kW) ---")

    # ------------------------------------------------------------
    # 2. Discharge from 100% → 0%
    # ------------------------------------------------------------
    for step in range(10):
        result = ess.update(
            power_setpoint_W=-1_000,   # discharge 1 kW
            timestamp=current_time
        )

        print(
            f"Step {step+1:02d} | "
            f"Power: {result['Battery Electric Power (kW)']:+.2f} kW | "
            f"SOC: {result['Battery SOC (-)']*100:6.2f} %"
        )

        current_time += ess.resolution

    # ------------------------------------------------------------
    # 3. Extra discharge steps (SOC limit test)
    # ------------------------------------------------------------
    print("\n--- EXTRA DISCHARGE (SOC SHOULD STAY AT 0%) ---")

    for step in range(3):
        result = ess.update(
            power_setpoint_W=-1_000,
            timestamp=current_time
        )

        print(
            f"Extra {step+1} | "
            f"Power: {result['Battery Electric Power (kW)']:+.2f} kW | "
            f"SOC: {result['Battery SOC (-)']*100:6.2f} %"
        )

        current_time += ess.resolution

    # ess.resetESSParameters()
    ess.set_soc(0.5)
    # ------------------------------------------------------------
    # 4. Charging with excessive power (limit test)
    # ------------------------------------------------------------
    print("\n--- CHARGE PHASE (REQUEST 5 kW, LIMIT = 2 kW) ---")

    for step in range(4):
        result = ess.update(
            power_setpoint_W=5_000,   # request > charger limit
            timestamp=current_time
        )

        print(
            f"Charge {step+1} | "
            f"Power: {result['Battery Electric Power (kW)']:+.2f} kW | "
            f"SOC: {result['Battery SOC (-)']*100:6.2f} %"
        )

        current_time += ess.resolution

    # ------------------------------------------------------------
    # 5. Export history
    # ------------------------------------------------------------
    ess.history.to_csv("linear_charge_discharge_test.csv", index=False)

    print("\nTest complete.")
    print("History exported to: linear_charge_discharge_test.csv")


if __name__ == "__main__":
    main()
