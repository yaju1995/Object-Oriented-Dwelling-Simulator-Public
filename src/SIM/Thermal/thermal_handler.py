import math
from datetime import timedelta
from support.lib_config import CustomLogger

class ThermalHandler:
    """
    First-order RC thermal model with time-dependent discretization.

    Continuous model:
        dT/dt = (T_eq - T) / tau

    Discrete model:
        T(k+1) = E * T(k) + (1 - E) * T_eq
        E = exp(-dt / tau)
    """

    def __init__(
        self,
        resolution: timedelta,
        initial_internal_temperature: float,
        tau: float,     # thermal time constant [seconds]
        n: float,       # heater efficiency
        W: float,       # thermal conductivity
    ):
        self.resolution = resolution
        self.tau = tau
        self.n = n
        self.W = W

        self.internal_temperature = float(initial_internal_temperature)

        # compute E from resolution
        self._update_E()

    # --------------------------------------------------
    def _update_E(self, tau= None):
        if tau:
            self.tau = tau
        dt = self.resolution.total_seconds()
        self.E = math.exp(-dt / self.tau)

    # --------------------------------------------------
    def set_resolution(self, resolution: timedelta):
        """
        Change timestep during simulation (rolling horizon safe).
        """
        self.resolution = resolution
        self._update_E()

    # --------------------------------------------------
    def update(self, power_W: float, external_temperature: float) -> float:
        """
        Advance one timestep.
        """
        # print(external_temperature, power_W, self.n, self.W)
        T_eq = external_temperature - (power_W * self.n) / self.W

        self.internal_temperature = (
            self.E * self.internal_temperature
            + (1.0 - self.E) * T_eq
        )

        return round(self.internal_temperature,2)
