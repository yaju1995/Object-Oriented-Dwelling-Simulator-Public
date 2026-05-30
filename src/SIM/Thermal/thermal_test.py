import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from thermal_handler import ThermalHandler

# ---------------- Simulation parameters ----------------
resolution = timedelta(minutes=5)
n_steps = 96  # 24 hours @ 15 min

# External temperature (daily sine wave)
T_mean = 10.0   # °C
T_amp = 6.0     # °C
time_idx = np.arange(n_steps)
T_out = T_mean + T_amp * np.sin(2 * np.pi * time_idx / n_steps)

# Heater
heater_power_W = 2000.0  # constant heating

# ---------------- Thermal model ----------------
thermal = ThermalHandler(
    resolution=resolution,
    initial_internal_temperature=21.0,   # °C
    tau=12 * 3600,                # τ = 6 hours
    n=0.95,                              # heater efficiency
    W=500.0,                             # thermal conductivity
)

# ---------------- Run loop ----------------
T_in = []
heating = False
for t in range(n_steps):
    if heating:
        heater_power_W = -5000
    temp = thermal.step(
            power_W=heater_power_W,
            external_temperature=T_out[t],
        )
    T_in.append(temp)
    if temp<20:
        heating=True
    elif temp>25:
        heating = False

# ---------------- Plot ----------------
plt.figure()
plt.plot(T_out, label="External Temp (°C)")
plt.plot(T_in, label="Internal Temp (°C)")
plt.xlabel("Timestep (15 min)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.show()
