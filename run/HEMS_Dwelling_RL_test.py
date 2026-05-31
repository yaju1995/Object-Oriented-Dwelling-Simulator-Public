import time
import pandas as pd
import random
from datetime import datetime, timedelta, time
from SIM.Simulator import dwelling
from SIM.Simulator_Config.config_list import (pv_config,
                                                  ev_config,
                                                  thermal_config,
                                                  weather_file,
                                                  demand_config,
                                                  battery_config)
from Controller.HEMSControlRL import HEMSController
# from SIM.ControlSignalHandler import ControlSignal

RESOLUTION = timedelta(minutes=60)  # 1 min resolution info
DURATION = timedelta(days=30)
START_TIME = datetime(2020, 1, 1)
TARIFF_TYPE = 'TOU'

# defining dwelling simulator
House = dwelling(name='Dwelling_1',
                 start_time=START_TIME,
                 resolution=RESOLUTION,
                 duration=DURATION,
                 demand_config=None,
                 weather_file=None,
                 pv_config=None,
                 battery_config=battery_config,
                 ev_config=None,
                 thermal_config=None,
                 seed=1)


# Defning Tariff type [use the ]
if TARIFF_TYPE == 'TOU':
    House.tariff.upload_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example-TOU.csv')
    House.tariff.upload_feed_tariff('./SRC/SIM/Defaults/Tariff/hourly_feed_tariff_example-TOU_0_2.csv')
else:

    House.tariff.upload_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example-Dynamic.csv')
    House.tariff.upload_feed_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example-Dynamic.csv')

# TOU tariff
House.tariff.updated_tariff()

# Initialized Dwelling
House.initialized_df()

# Upload House demand and generation [uploading real data for analysis- else follow demand_config and pv_config and weather_config]
House.upload_data('Results/Test_Data/house2_consumption_dwell.csv',
                  columns= ["Demand Electric Power (kW)", "PV Electric Power (kW)"])


# Defining controller
Controller = HEMSController(name='Dwelling_1', data_resolution=RESOLUTION, meter_tariff=House.tariff,
                            ev_update_period=RESOLUTION,
                            ess_update_period=RESOLUTION,
                            havc_update_period=RESOLUTION,
                            mode='Test',
                            ev_config=ev_config,
                            ess_config=battery_config,
                            hvac_config=thermal_config)

#########################################################################
current_time = START_TIME
end_time = START_TIME + DURATION
control_signal = {}

TEST_EPS = 500
Controller.load_models(episode=5000) # Load model defined at the direcoty at Agent config

# Running a training loop
while current_time <= end_time-RESOLUTION:
    inverter, meter, ev, hvac, status = House.step(control_signal)

    Demand = House.simulation_df.loc[current_time + RESOLUTION, "Demand Electric Power (kW)"]  # get next period info
    Generation = House.simulation_df.loc[current_time + RESOLUTION, "PV Electric Power (kW)"]  # get next period info
    inverter.forecast_demand = Demand
    inverter.forecast_generation = Generation

    # Get controller action
    control_signal = Controller.update(ev_info=ev, inverter_info=inverter, hvac_info=hvac, meter_info=meter)
    if control_signal:
        # print(control_signal)
        pass

    # Updating time
    current_time += RESOLUTION


print(f'Final House Cost: {Controller.hems_database.df["Instant Cost"].sum()}')
Controller.hems_database.df.to_csv(f'./Results/controller_ESS-{TARIFF_TYPE}_{TEST_EPS}_change.csv')
House.simulation_df.to_csv(f'./Results/simulation_ESS-{TARIFF_TYPE}_{TEST_EPS}_change.csv')
