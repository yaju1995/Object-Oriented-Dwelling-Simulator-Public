import numpy as np
from datetime import datetime, timedelta

from ess_handler import ESSHandler   # adjust path if needed


def main():
    # ------------------------------------------------------------
    # 1. Create ESS
    # ------------------------------------------------------------
    ess = ESSHandler(
        name="DemoESS",
        total_capacity_Wh=10_000,          # 10 kWh battery
        initial_soc_pct=50,                # start at 50%
        charging_power_W=5_000,             # 5 kW charge limit
        discharging_power_W=5_000,          # 5 kW discharge limit
        resolution=timedelta(minutes=15),  # 15-minute time step
        in_eff=1,
        out_eff=1,
    )

    # ------------------------------------------------------------
    # 2. Simulation timeline
    # ------------------------------------------------------------
    start_time = datetime(2025, 1, 1, 0, 0)
    n_steps = 96   # 96 × 15 min = 24 hours

    timestamps = [
        start_time + i * ess.resolution
        for i in range(n_steps)
    ]

    # ------------------------------------------------------------
    # 3. Random power demand (export / import request)
    # ------------------------------------------------------------
    # Positive → charge battery
    # Negative → discharge battery
    rng = np.random.default_rng(seed=42)

    power_requests_W = rng.uniform(
        low=-6_000,   # may exceed limits on purpose
        high=6_000,
        size=n_steps
    )

    # ------------------------------------------------------------
    # 4. Run simulation
    # ------------------------------------------------------------
    print("Running ESS simulation...\n")

    for ts, p_req in zip(timestamps, power_requests_W):
        result = ess.update(p_req, timestamp=ts)

        print(
            f"{ts} | "
            f"Requested: {p_req:7.0f} W | "
            f"Actual: {result['Battery Electric Power (kW)']:+6.2f} kW | "
            f"SOC: {result['Battery SOC (-)']*100:6.2f} %"
        )

    # ------------------------------------------------------------
    # 5. Export history to CSV
    # ------------------------------------------------------------
    output_file = "ess_history.csv"
    ess.history.to_csv(output_file, index=False)

    print("\nSimulation complete.")
    print(f"History exported to: {output_file}")


if __name__ == "__main__":
    main()
