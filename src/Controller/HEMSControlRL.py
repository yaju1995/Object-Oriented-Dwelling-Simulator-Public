import pandas as pd
from datetime import timedelta, datetime, time

from SIM.Tariff.tariffHandler import tariffHandler
from SIM.EquipmentClass import InverterModel, EVModel, HVACModel, MeterModel
from SIM.ControlSignalHandler import ControlSignal
from Controller.Database.PandasDatabase import DataStore
from Controller.Constants import COLUMNS_KEYS
from support.lib_config import CustomLogger

import os

logger = CustomLogger(command=True)

# RL based EV controller import
from Controller.EV_controller.evControlLib_General import evController
from Controller.EV_controller.EV_RL_CONFIG import (EV_RL_AGENT, EV_LOOK_AHEAD,
                                         EV_INPUT_DIM, EV_OUT_DIM, EV_MODEL_DIR,
                                         EV_MODEL_NAME)

# RL agent ESS controller import
from Controller.ESS_controller.ESS_RL_CONFIG import (ESS_RL_AGENT, ESS_LOOK_AHEAD,
                                           ESS_INPUT_DIM, ESS_OUT_DIM,
                                           ESS_MODEL_NAME, ESS_MODEL_DIR)
from Controller.ESS_controller.essRLControlLib import essController


# RL agent HVAC controller import
from Controller.HVAC_controller.hvacRLControlLib import hvacController
from Controller.HVAC_controller.HVAC_RL_CONFIG import (HVAC_RL_AGENT,
                                             HVAC_INPUT_DIM, HVAC_LOOK_AHEAD,
                                             HVAC_MODEL_DIR, HVAC_MODEL_NAME)


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
                 havc_update_period: timedelta = timedelta(minutes=15),
                 hvac_config: dict = None,
                 mode='Train',
                 plotter= False):
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
        self.ev_controller = None
        self.ess_controller = None
        self.hvac_controller = None
        self.ev_config = ev_config
        self.ess_config = ess_config
        self.hvac_config = hvac_config
        # EV RL controller
        if EV_RL_AGENT is not None:
            self.ev_controller = evController(rl_agent=EV_RL_AGENT, resolution=self.resolution,
                                              update_period=self.ev_update_period,
                                              global_database=self.hems_database, mode=mode,
                                              max_charging_power=ev_config.get('charging power W', 7_000) / 1000,
                                              look_ahead=EV_LOOK_AHEAD,
                                              enable_plotter=plotter)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if ev_tariff is None:
                self.ev_controller.tariff_handler = meter_tariff
            else:
                self.ev_controller.tariff_handler = ev_tariff
        # ESS RL controller ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if ESS_RL_AGENT is not None:
            self.ess_controller = essController(rl_agent=ESS_RL_AGENT,
                                                mode=mode,
                                                resolution=self.resolution,
                                                update_period=self.ess_update_period,
                                                global_database=self.hems_database,
                                                max_charging_kw=self.ess_config.get('charging power W', 1000) / 1000,
                                                max_discharging_kw=self.ess_config.get("discharging power W",
                                                                                       1000) / 1000,
                                                look_ahead=ESS_LOOK_AHEAD,
                                                energy_normalizer = self.ess_config.get('capacity Wh') / 1000,
                                                enable_plotter=plotter,
                                                trigger_time=[
                                                    time(0, 0),
                                                    # time(6, 0),
                                                    # time(12, 0),
                                                    # time(18, 0),
                                                ])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.ess_controller.tariff_handler = meter_tariff

        if HVAC_RL_AGENT is not None:
            # HVAC RL controller ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.hvac_controller = hvacController(rl_agent=HVAC_RL_AGENT,
                                                  mode=mode,
                                                  resolution=self.resolution,
                                                  update_period=self.hvac_update_period,
                                                  global_database=self.hems_database,
                                                  hvac_power_kw=self.hvac_config.get('HVAC electric power Wh',
                                                                                     8_000) / 1000,
                                                  energy_normalizer=self.hvac_config.get('HVAC electric power Wh',
                                                                                         8_000) / 1000,
                                                  temp_ref=22.5,
                                                  temp_deviation=2,
                                                  look_ahead=HVAC_LOOK_AHEAD,
                                                  enable_plotter=False
                                                  )
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.hvac_controller.tariff_handler = meter_tariff

        self.load_forecasting_model = None
        self.generation_forecasting_model = None

        # Load models

        self.tariff_handler = None

        self.ESS_charge = False
        self.HVAC_ON = False

    def update(self, ev_info: EVModel, inverter_info: InverterModel, hvac_info: HVACModel, meter_info: MeterModel):

        now_time = meter_info.time
        # logger.commandline(f'Now time:{now_time}')
        # Consumption power
        consumption = round(meter_info.active_power - inverter_info.battery_power + inverter_info.pv_power, 3)
        hours = self.resolution.total_seconds() / 3600
        minute = self.resolution.total_seconds() / 60

        # Cost of consumed power
        if meter_info.active_power>0:
            instant_cost = round(meter_info.active_power * hours * meter_info.tariff, 4)
        else:
            instant_cost = round(meter_info.active_power * hours * meter_info.feed_tariff, 4)
        # getting attribute from controller
        user_exp_soc = 0
        user_dc_set_time = datetime(1, 1, 1, 0, 0, 0)
        if self.ev_controller is not None:
            user_exp_soc = self.ev_controller.user_exp_soc,
            user_dc_set_time = self.ev_controller.user_dc_set_time
        temp_ref = 0
        if self.hvac_controller is not None:
            temp_ref = self.hvac_controller.temp_ref

        # Storing information in HVAC
        row = {  # Base on Constants COLUMNS_KEYS
            'Consumption (kW)': consumption,  #Demand only (Demand + EV+ HVAC)
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

            'User Expected SOC (-)': user_exp_soc,
            'User Expected Plugout Time': user_dc_set_time,

            'Temperature - Indoor (C)': hvac_info.ti,
            'Heating Electric Power (kW)': hvac_info.hvac_power,
            'Heating Electric Energy (kWh)': hvac_info.hvac_power * hours,
            'Temperature - Ref (C)': temp_ref
        }

        # Database test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.hems_database.append(now_time, row)  # First update row then collect information

        # EV control ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.ev_controller:
            self.control_signals.EV_Max_Power = self.ev_controller.update_status(ev_info=ev_info)

        if self.control_signals.EV_Max_Power is not None:
            self.control_signals.EV_Max_Power = float(self.control_signals.EV_Max_Power) * 1000

        # HVAC control ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.hvac_controller:
            self.control_signals.HVAC_Heating_Power = self.hvac_controller.update_status(hvac_info=hvac_info)
        if self.control_signals.HVAC_Heating_Power is not None:
            self.control_signals.HVAC_Heating_Power = float(self.control_signals.HVAC_Heating_Power) * 1000

        # # Battery control ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.ess_controller:
            self.control_signals.Battery_P_Setpoint = self.ess_controller.update_status(meter_info=meter_info,
                                                                                        inverter_info=inverter_info)
        if self.control_signals.Battery_P_Setpoint is not None:
            self.control_signals.Battery_P_Setpoint = float(self.control_signals.Battery_P_Setpoint) * 1000

        # # generate controller signals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        control_signal = self.control_signals.generate_control_signal()
        # logger.commandline(control_signal)
        # return the generated signal
        return control_signal

    def save_models(self, episode=None):
        if episode is not None:
            EV_MODEL_NAME = f'states_{EV_INPUT_DIM}_delay_{EV_LOOK_AHEAD}_{episode}eps.pth'
            ESS_MODEL_NAME = f'states_{ESS_INPUT_DIM}_delay_{ESS_LOOK_AHEAD}_{episode}eps.pth'
            HVAC_MODEL_NAME = f'states_{HVAC_INPUT_DIM}_delay_{HVAC_LOOK_AHEAD}_{episode}eps.pth'

        if self.ev_controller:
            os.makedirs(EV_MODEL_DIR, exist_ok=True)
            EV_PATH = os.path.join(EV_MODEL_DIR, EV_MODEL_NAME)
            logger.commandline(self.ev_controller.rl_agent.save(EV_PATH))

        if self.ess_controller:
            os.makedirs(ESS_MODEL_DIR, exist_ok=True)
            ESS_PATH = os.path.join(ESS_MODEL_DIR, ESS_MODEL_NAME)
            logger.commandline(self.ess_controller.rl_agent.save(ESS_PATH))
            # logger.commandline(agent.save(ESS_PATH))
        # save command for other models

        if self.hvac_controller:
            os.makedirs(HVAC_MODEL_DIR, exist_ok=True)
            HVAC_PATH = os.path.join(HVAC_MODEL_DIR, HVAC_MODEL_NAME)
            logger.commandline(self.hvac_controller.rl_agent.save(HVAC_PATH))

    def load_models(self, episode=None):
        print('Loading Model!!')
        # Default names (from config)
        EV_name = EV_MODEL_NAME
        ESS_name = ESS_MODEL_NAME
        HVAC_name = HVAC_MODEL_NAME

        # Override if episode-specific models are requested
        if episode is not None:
            EV_name = f"states_{EV_INPUT_DIM}_delay_{EV_LOOK_AHEAD}_{episode}eps.pth"
            ESS_name = f"states_{ESS_INPUT_DIM}_delay_{ESS_LOOK_AHEAD}_{episode}eps.pth"
            HVAC_name = f"states_{HVAC_INPUT_DIM}_delay_{HVAC_LOOK_AHEAD}_{episode}eps.pth"

        # EV
        if self.ev_controller:
            logger.commandline("Load EV Controller")
            EV_PATH = os.path.join(EV_MODEL_DIR, EV_name)
            logger.commandline(self.ev_controller.load_model(EV_PATH))

        # ESS
        if self.ess_controller:
            logger.commandline("Load ESS Controller")
            ESS_PATH = os.path.join(ESS_MODEL_DIR, ESS_name)
            logger.commandline(self.ess_controller.load_model(ESS_PATH))

        # HVAC
        if self.hvac_controller:
            logger.commandline("Load HVAC Controller")
            HVAC_PATH = os.path.join(HVAC_MODEL_DIR, HVAC_name)
            logger.commandline(self.hvac_controller.rl_agent.load(HVAC_PATH))

