from datetime import datetime, timedelta
import pandas as pd

from ev_handler import EVHandler   # adjust import path if needed


def main():
    # -------------------------------------------------
    # Simulation config
    # -------------------------------------------------
    start_time = datetime(2018, 1, 1, 0, 0)
    resolution = timedelta(minutes=15)
    duration = timedelta(days=7)

    # -------------------------------------------------
    # EV / battery parameters
    # -------------------------------------------------
    ev = EVHandler(
        name="EV_1",
        ev_profile_csv="pdf_Veh1_Level0.csv",   # raw CSV used by generator
        start_time=start_time,
        resolution=resolution,
        duration=duration,
        total_capacity_Wh=60_000,          # 60 kWh
        charging_power_W=7_400,            # 7.4 kW charger
        discharging_power_W=7_400,          # V2G capable
        v2g_enabled=False,
        seed=100,
    )
    ev.upload_ev_df_from_csv('test_data.csv')
    # -------------------------------------------------
    # Simulation loop
    # -------------------------------------------------
    t = start_time
    end_time = start_time + duration

    records = []

    while t < end_time:
        # Example control (charge whenever parked)
        out = ev.step(
            timestamp=t,
            control_power_W=-500,  # W
        )

        records.append({
            "timestamp": t,
            **out,
        })

        t += resolution

    # -------------------------------------------------
    # Results
    # -------------------------------------------------
    df = pd.DataFrame(records).set_index("timestamp")

    df.to_csv("ev_simulation_output.csv")
    ev.history.to_csv('EV_battery_output.csv')
    print("Saved ev_simulation_output.csv")


if __name__ == "__main__":
    main()
