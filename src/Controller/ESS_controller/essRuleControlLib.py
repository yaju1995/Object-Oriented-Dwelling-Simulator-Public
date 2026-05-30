import pandas as pd
import numpy as np
from datetime import timedelta
from support.lib_config import CustomLogger
from Controller.Database.PandasDatabase import DataStore
from SIM.EquipmentClass import InverterModel, MeterModel

logger = CustomLogger(command=True, color='red')


class essController:
    def __init__(self, resolution: timedelta = timedelta(minutes=1),
                 update_period: timedelta = timedelta(minutes=60),
                 global_database: DataStore = None,
                 max_charging_kw=7,
                 max_discharging_kw=7,
                 look_ahead=1):
        self.resolution: timedelta = resolution
        self.update_period = update_period
        self.global_databased = global_database
        self.look_ahead = look_ahead
        #############################################################
        self.max_charging_power = max_charging_kw
        self.max_discharging_power = max_discharging_kw
        self.ESS_charge = False
        self.set_battery_power = 0

        #############################################################
        self.tariff_handler = None
        #############################################################
        self.total_ess_charging_cost = 0
        self.total_ess_charging_energy = 0

        self.total_ess_discharging_cost = 0
        self.total_ess_discharging_energy = 0

    def update_status(self, meter_info: MeterModel, inverter_info: InverterModel):

        # ===============================================================
        #   CONTROL UPDATE CHECK
        # ===============================================================
        # to do update in 14-min
        # Update only when time aligns with resolution (e.g., 15 min)
        now_time = meter_info.time
        next_time = now_time + self.resolution
        do_update = (next_time.minute % (self.update_period.total_seconds() // 60) == 0)

        instant_power = inverter_info.battery_power
        step_energy = round(instant_power * (self.resolution.total_seconds() / 3600), 6)
        tariff, feed_tariff = self.tariff_handler.get_tariff(meter_info.time)
        if step_energy > 0:
            self.total_ess_charging_energy += abs(step_energy)
            self.total_ess_charging_cost += round(abs(step_energy) * tariff, 6)
        else:
            self.total_ess_discharging_energy += abs(step_energy)
            self.total_ess_discharging_cost += round(abs(step_energy) * tariff, 6)

        if do_update:  # every 15 or 30 mins update

            consumption = round(meter_info.active_power - inverter_info.battery_power + inverter_info.pv_power, 3)
            generation = round(inverter_info.pv_power, 3)
            surplus = generation - consumption
            next_tariff, next_feed_tariff = self.tariff_handler.get_tariff(next_time)

            if next_feed_tariff>next_tariff:
                self.set_battery_power = self.max_charging_power
            else:
                self.set_battery_power = surplus




        return self.set_battery_power

    def save(self):
        return f'No model to save!!!'

    def load(self):
        return f'No model to load!!!'
