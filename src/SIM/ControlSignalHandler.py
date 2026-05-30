from typing import Optional
from pydantic import BaseModel


class ControlSignal(BaseModel):
    HVAC_Heating_Setpoint: Optional[float] = None # Degree Celcius
    HVAC_Heating_Power: Optional[float] = None # Watt
    Battery_P_Setpoint: Optional[float] = None # Watt
    EV_Max_Power: Optional[float] = None # Watt
    Lighting_P_Setpoint: Optional[float] = None # Percentage
    Water_Heating_Setpoint: Optional[float] = None # Watt

    def generate_control_signal(self) -> dict:
        control_signal = {}
        if self.HVAC_Heating_Setpoint is not None:
            control_signal["HVAC Heating"] = {"Setpoint": self.HVAC_Heating_Setpoint}
        if self.HVAC_Heating_Power is not None:
            control_signal.setdefault("HVAC Heating", {})["P Setpoint"] = self.HVAC_Heating_Power
        if self.Battery_P_Setpoint is not None:
            control_signal["Battery"] = {"P Setpoint": self.Battery_P_Setpoint}
        if self.EV_Max_Power is not None:
            control_signal["EV"] = {"Max Power": self.EV_Max_Power}
        if self.Lighting_P_Setpoint is not None:
            control_signal["Lighting"] = {"P Setpoint": self.Lighting_P_Setpoint}
        if self.Water_Heating_Setpoint is not None:
            control_signal["Water Heating"] = {"Setpoint": self.Water_Heating_Setpoint}

        return control_signal
