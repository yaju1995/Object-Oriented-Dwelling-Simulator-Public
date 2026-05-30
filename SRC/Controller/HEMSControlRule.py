import pandas as pd
from datetime import timedelta
import os

from SIM.Tariff.tariffHandler import tariffHandler
from SIM.EquipmentClass import InverterModel, EVModel, HVACModel, MeterModel
from SIM.ControlSignalHandler import ControlSignal
from Controller.Database.PandasDatabase import DataStore
from Controller.Constants import COLUMNS_KEYS
from support.lib_config import CustomLogger

logger = CustomLogger(command=True)

# EV Import and Definition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rule based EV controller
from Controller.EV_controller.evRuleControlLib import evController

# ESS Import and definition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rule based ESS controller
from Controller.ESS_controller.essRuleControlLib import essController

# HVAC Import and definition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rule based HVAC controller
from Controller.HVAC_controller.hvacRuleControlLib import hvacController


# Currently direct definition for early training and testing

class HEMSController:
    def __init__(self, name: str,
                 data_resolution: timedelta,
                 meter_tariff: tariffHandler,
                 ev_tariff: tariffHandler = None,  # pass ESS, EV and hvac control config from the simulator
                 ess_update_period: timedelta = timedelta(minutes=15),
                 ess_config: dict = None,
                 ev_update_period: timedelta = timedelta(minutes=15),
                 ev_config: dict = None,
                 havc_update_period: timedelta = timedelta(minutes=5),
                 hvac_config: dict = None,
                 mode='Train',
                 controller='RL'):
        """

        :param name: name for the controller
        :param data_resolution: timedelta (data storage resolution(1 min in most case))
        :param meter_tariff: global tariff for dwelling simulation
        :param ev_tariff: [optional], provide when ev tariff is separate from the meter_tariff
        """
        self.name = name
        self.resolution: timedelta = data_resolution
        self.ev_update_period = ev_update_period
        self.ess_update_period = ess_update_period
        self.hvac_update_period = havc_update_period
        # self.mode = mode
        # Databased definition
        self.hems_database = DataStore(resolution=data_resolution)
        self.hems_logs = pd.DataFrame(columns=COLUMNS_KEYS)
        self.hems_logs.index.name = "timestamp"
        self.control_signals = ControlSignal()
        self.controller = controller
        self.ev_controller = None
        self.ess_controller = None
        self.hvac_controller = None
        self.ev_config = ev_config
        self.ess_config = ess_config
        self.hvac_config = hvac_config
        # Ev Rule controller
        self.ev_controller = evController(resolution=self.resolution,
                                           update_period=self.ev_update_period,
                                           global_database=self.hems_database, mode=mode,
                                           max_charging_power=ev_config.get('charging power W', 7_000) / 1000)
        #
        if self.ev_controller is not None:
            if ev_tariff is None:
                self.ev_controller.tariff_handler = meter_tariff
            else:
                self.ev_controller.tariff_handler = ev_tariff
        # RULE based Controller ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.ess_controller = essController(resolution=self.resolution,
                                            update_period=self.ess_update_period,
                                            global_database=self.hems_database,
                                            max_charging_kw=ess_config.get('charging power W', 1000) / 1000,
                                            max_discharging_kw=ess_config.get("discharging power W", 1000) / 1000,
                                            )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.ess_controller.tariff_handler = meter_tariff

        # RULE based HVAC ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.hvac_controller = hvacController(resolution=self.resolution,
                                              update_period=self.hvac_update_period,
                                              global_database=self.hems_database)
        self.load_forecasting_model = None
        self.generation_forecasting_model = None

        # Tariff handler
        self.tariff_handler = None

    def update(self, ev_info: EVModel, inverter_info: InverterModel, hvac_info: HVACModel, meter_info: MeterModel):

        now_time = meter_info.time
        # logger.commandline(f'Now time:{now_time}')
        # Consumption power
        consumption = round(meter_info.active_power - inverter_info.battery_power + inverter_info.pv_power, 3)
        hours = self.resolution.total_seconds() / 3600
        minute = self.resolution.total_seconds() / 60

        # Cost of consumed power
        if meter_info.active_power > 0:
            instant_cost = round(meter_info.active_power * hours * meter_info.tariff, 4)
        else:
            instant_cost = round(meter_info.active_power * hours * meter_info.feed_tariff, 4)

        # Storing information in HVAC
        row = {  # Base on Constants COLUMNS_KEYS
            'Consumption (kW)': consumption,
            'Consumption (kWh)': consumption * hours,
            'Generation (kW)': inverter_info.pv_power,
            'Generation (kWh)': inverter_info.pv_power * hours,
            'Total Electric Power (kW)': meter_info.active_power,
            'Total Electric Power (kWh)': meter_info.active_power * hours,

            'tariff': meter_info.tariff,
            'feed tariff': meter_info.feed_tariff,
            'Instant Cost': instant_cost,

            'Battery SOC (-)': inverter_info.battery_soc,
            'Battery Set Power (W)': self.control_signals.Battery_P_Setpoint or 0,
            'Battery Electric Power (kW)': inverter_info.battery_power,
            'Battery Electric Energy (kWh)': inverter_info.battery_power * hours,

            'EV Parked': ev_info.ev_status,
            'EV SOC (-)': ev_info.ev_soc,
            'EV Set Point (kW)': self.control_signals.EV_Max_Power or 0,
            'EV Electric Power (kW)': ev_info.ev_power,
            'EV Electric Energy (kWh)': ev_info.ev_power * hours,

            'Temperature - Indoor (C)': hvac_info.ti,
            'Heating Electric Power (kW)': hvac_info.hvac_power,
            'Heating Electric Energy (kWh)': hvac_info.hvac_power * hours,
        }

        # Database test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.hems_database.append(now_time, row)  # First update row then collect information

        # EV control ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.control_signals.EV_Max_Power = self.ev_controller.update_status(ev_info=ev_info)

        if self.control_signals.EV_Max_Power is not None:
            self.control_signals.EV_Max_Power = float(self.control_signals.EV_Max_Power) * 1000

        # Battery control ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.control_signals.Battery_P_Setpoint = self.ess_controller.update_status(meter_info=meter_info,
                                                                                    inverter_info=inverter_info)
        if self.control_signals.Battery_P_Setpoint is not None:
            self.control_signals.Battery_P_Setpoint = float(self.control_signals.Battery_P_Setpoint) * 1000

        # HVAC control ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.control_signals.HVAC_Heating_Power = self.hvac_controller.update_status(hvac_info=hvac_info)

        if self.control_signals.HVAC_Heating_Power is not None:
            self.control_signals.HVAC_Heating_Power = float(self.control_signals.HVAC_Heating_Power) * 1000

        # generate controller signals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        control_signal = self.control_signals.generate_control_signal()
        # logger.commandline(control_signal)
        # return the generated signal
        return control_signal

    def save_models(self, episode=None):
        logger.commandline('No model to save !!!')


    def load_models(self,episode=None):
        logger.commandline('No model to Load')
