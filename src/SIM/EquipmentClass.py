from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


# defining equipment from OCHER
class battery(BaseModel):
    capacity: int = 5
    capacity_kwh: int = 10


class ev(BaseModel):
    vehicle_type: str = "BEV"
    charging_level: str = "Level 1"
    range: int = 150


class pv(BaseModel):
    capacity: int = 5
    tilt: int = 20
    azimuth: int = 0


class AdditionalEquipment(BaseModel):
    Battery: battery = Field(default_factory=battery)
    EV: ev = Field(default_factory=ev)
    PV: pv = Field(default_factory=pv)


def dict_to_equipment(data: dict) -> AdditionalEquipment:
    return AdditionalEquipment(**data)


# House IOT Devices Instantaneous information storage
class MeterModel(BaseModel):
    time: datetime | int | float=0.0
    active_power: float=0.0
    reactive_power: float=0.0
    tariff: float=0.0
    feed_tariff: float=0.0
    total_energy: float = 0.0  # cumulative kWh
    total_cost: float = 0.0  # cumulative cost
    tariff_24hrs: Optional[list] = None
    feed_tariff_24hrs: Optional[list] = None

    def add_period(self, resolution_minutes: float):
        """
        Adds energy and cost for one period.
        active_power is in kW.
        resolution_minutes is e.g. 1, 5, 15...
        """
        # print(f'{self.active_power}*{resolution_minutes}/{60}')
        energy_kwh = self.active_power * (resolution_minutes / 60)
        self.total_energy += energy_kwh
        self.total_cost += energy_kwh * self.tariff


class InverterModel(BaseModel):
    time: datetime | int | float = 0.0
    pv_power: float= 0.0
    battery_power: float= 0.0
    battery_soc: float= 0.0
    forecast_demand:float = 0.0
    forecast_generation: float= 0.0


class EVModel(BaseModel):
    time: datetime | int | float= 0.0
    ev_status: bool= 0.0
    ev_soc: float= 0.0
    ev_power: float= 0.0
    user_ev_dc_time:datetime | int | float = 0.0 # EV disconnect time
    expected_soc:int = 100 # in percentage


class HVACModel(BaseModel):
    time: datetime | int | float= 0.0
    ti: float= 0.0
    hvac_power: float= 0.0
    temp_ref:int = 22.5


if __name__ == '__main__':
    equipment = {
        "Battery": {
            "capacity": 5,  # in kW
            "capacity_kwh": 10,
        },
        "EV": {
            "vehicle_type": "BEV",
            "charging_level": "Level 1",
            "range": 150,
        },
        "PV": {
            "capacity": 5,
            "tilt": 20,
            "azimuth": 0,
        },
    }
    # equipment = AdditionalEquipment()
    # equipment.pv.capacity = 5
    # print(type(equipment.model_dump()))

    equipment_obj = dict_to_equipment(equipment)

    print(equipment_obj.Battery.capacity_kwh)
    print(type(equipment_obj))
