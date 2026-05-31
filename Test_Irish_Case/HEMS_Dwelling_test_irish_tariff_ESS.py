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

from Controller.HEMSControlRule import HEMSController
from SIM.ControlSignalHandler import ControlSignal


RESOLUTION = timedelta(minutes=60)  # 1 min resolution info
DURATION = timedelta(hours=8683)
# DURATION = timedelta(hours=48)
START_TIME = datetime(2020, 1, 1,0,0)
TARIFF_TYPE = 'TOU'

House = dwelling(name='Dwelling_1',
                 start_time=START_TIME,
                 resolution=RESOLUTION,
                 duration=DURATION,
                 demand_config=demand_config,
                 weather_file=weather_file,
                 pv_config=pv_config,
                 battery_config=battery_config,
                 ev_config=ev_config,
                 thermal_config=thermal_config,
                 seed=1)

# to enable step to get inverter, meter, Hvac, ev information separately
if TARIFF_TYPE == 'TOU':
    House.tariff.upload_tariff('./src/SIM/Defaults/Tariff/hourly_tariff_example-TOU.csv')
    House.tariff.upload_feed_tariff('./src/SIM/Defaults/Tariff/hourly_feed_tariff_example-TOU_0_2.csv')
elif TARIFF_TYPE == 'Irish':
    House.tariff.upload_historic_tariff('./src/SIM/Defaults/Tariff/Irish_2020/Irish_2020_Wholesale_tariff_price.csv')
    House.tariff.upload_historic_feed_tariff('./src/SIM/Defaults/Tariff/Irish_2020/Irish_2020_Wholesale_feed_price.csv')
    House.tariff.prepare_day_ahead_tariffs(START_TIME)
elif TARIFF_TYPE == 'Dynamic_old_fw':
    House.tariff.upload_historic_tariff('./src/SIM/Defaults/Tariff/Irish_2020/Irish_2020_Wholesale_tariff_price.csv')
    House.tariff.upload_historic_feed_tariff(
        './src/SIM/Defaults/Tariff/Irish_2020/Irish_2020_Wholesale_feed_price.csv')
    House.tariff.prepare_day_ahead_tariffs(START_TIME)
else:
    House.tariff.upload_tariff('./src/SIM/Defaults/Tariff/hourly_tariff_example-Dynamic.csv')
    House.tariff.upload_feed_tariff('./src/SIM/Defaults/Tariff/hourly_tariff_example-Dynamic.csv')

# TOU tariff

# add TOU tariff day night and peak tariff
# Initialized House
House.initialized_df()


# # Upload House demand and generation 
# House.upload_data('./Results/Test_Data/house2_consumption_dwell.csv',
#                   columns= ["Demand Electric Power (kW)", "PV Electric Power (kW)"])

# # Defining controller
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

start = datetime.now()

### Train the moodels - 300 days
## Save the models - Properly name them
# Test the models - test them
# load the model before running
SEED = 0
TEST_EPS = 500

# Controller.load_models(episode=7000)
random.seed(SEED)
day = 0
control = ControlSignal()
# Running a training loop
while current_time <= end_time-RESOLUTION:
    inverter, meter, ev, hvac, status = House.step(control_signal)

    Demand = House.simulation_df.loc[current_time + RESOLUTION, "Demand Electric Power (kW)"]  # next period
    Generation = House.simulation_df.loc[current_time + RESOLUTION, "PV Electric Power (kW)"]  # next period
    inverter.forecast_demand = Demand
    inverter.forecast_generation = Generation

    control_signal = Controller.update(ev_info=ev, inverter_info=inverter, hvac_info=hvac, meter_info=meter)
    if control_signal:
        # print(control_signal)
        pass
    
    if current_time.time() == time(0, 0):
        if TARIFF_TYPE in ('Dynamic', 'Irish'):
            House.tariff.prepare_day_ahead_tariffs(current_time)
        # Controller.hvac_controller.temp_ref = random.randrange(15, 26)  # ref is set

    # Updating time
    current_time += RESOLUTION

end = datetime.now()

print(f'Final House Cost: {Controller.hems_database.df["Instant Cost"].sum()}')

Controller.hems_database.df.to_csv(f'./Results/controller_Irish_ESS-{TARIFF_TYPE}_{TEST_EPS}_change.csv')
House.simulation_df.to_csv(f'./Results/simulation_Irish_HVAC-ESS{TARIFF_TYPE}_{TEST_EPS}_change.csv')



