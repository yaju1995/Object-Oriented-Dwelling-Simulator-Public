import time
import pandas as pd
import random
from datetime import datetime, timedelta, time
from SRC.SIM.Simulator import dwelling
from SRC.SIM.Simulator_Config.config_list import (pv_config,
                                                  ev_config,
                                                  thermal_config,
                                                  weather_file,
                                                  demand_config,
                                                  battery_config)
# ~~~~~~~~~~ Switch Controller ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from SRC.Controller.HEMSControlRL import HEMSController
# from SRC.Controller.HEMSControlRule import HEMSController


from SRC.SIM.ControlSignalHandler import ControlSignal

# from SRC.Controller.HVAC_controller.HVAC_RL_CONFIG import  HVAC_MODEL_DIR
# from SRC.Controller.HEMSControlRule import HEMSController


RESOLUTION = timedelta(minutes=60)  # 1 min resolution info
DURATION = timedelta(days=30)
START_TIME = datetime(2020, 1, 1)
TARIFF_TYPE = 'TOU'


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


# to enable step to get inverter, meter, Hvac, ev information separately
if TARIFF_TYPE == 'TOU':
    House.tariff.upload_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example-TOU.csv')
    House.tariff.upload_feed_tariff('./SRC/SIM/Defaults/Tariff/hourly_feed_tariff_example-TOU_0_2.csv')
else:

    House.tariff.upload_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example-Dynamic.csv')
    House.tariff.upload_feed_tariff('./SRC/SIM/Defaults/Tariff/hourly_tariff_example-Dynamic.csv')

# TOU tariff
House.tariff.updated_tariff()

# add TOU tariff day night and peak tariff
# Initialized House
House.initialized_df()

# Upload House demand and generation
House.upload_data('Results/Test_Data/house2_consumption_dwell.csv',
                  columns= ["Demand Electric Power (kW)", "PV Electric Power (kW)"])
# print(df)
# House.simulation_df.to_csv(f'./Results/simulation.csv')

# Extracting EV profile
# House.EV.ev_df.to_csv(f'./Results/ev_profile.csv')

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
Controller.load_models(episode=5000)
# Controller.load_models()
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

    # control.EV_Max_Power = 7_000
    # control_signal = control.generate_control_signal()
    control_signal = Controller.update(ev_info=ev, inverter_info=inverter, hvac_info=hvac, meter_info=meter)
    if control_signal:
        # print(control_signal)
        pass
    # if current_time.time() == time(0, 0):
    #     Controller.hvac_controller.temp_ref = random.randrange(15, 26)  # ref is set

    # Updating time

    current_time += RESOLUTION

end = datetime.now()
# plt.figure(1)
# plt.savefig('Cost per kwh.png')  # Save as PNG, PDF, SVG, etc.
# plt.show()
# plt.figure(2)
# plt.hist(Controller.ev_controller.final_soc_list)
# plt.title('Final SOC Distribution')
# plt.savefig('final_soc.png')  # Save as PNG, PDF, SVG, etc.
# plt.show()
# plt.figure(3)
# plt.hist(Controller.ev_controller.initial_soc_list)
# plt.title('Initial SOC Distribution')
# plt.savefig('initial_soc.png')  # Save as PNG, PDF, SVG, etc.
# plt.show()
# print(f"Simulation took {end - start:.4f} seconds")
# print(f'UnSatisfied SOC : {Controller.ev_controller.unsatified_energy}')
# print(f'Satisfied SOC : {Controller.ev_controller.satisfied_energy}')
# ev_soc = ev_config.get("capacity Wh")
# print(f'UnSatisfied SOC Wh: {Controller.ev_controller.unsatified_energy * ev_config.get("capacity Wh")}')
# print(f'Satisfied SOC Wh: {Controller.ev_controller.satisfied_energy * ev_config.get("capacity Wh")}')
# print(f'Not fill charge count: {Controller.ev_controller.not_full_count}')
# print(f'EV only charging cost : {Controller.ev_controller.total_ev_charging_cost}')
# print(f'EV only charging energy : {Controller.ev_controller.total_ev_charging_energy}')
# print(f'EV only total $/kwh : {Controller.ev_controller.total_ev_charging_cost / Controller.ev_controller.total_ev_charging_energy}')
# #
print(f'Final House Cost: {Controller.hems_database.df["Instant Cost"].sum()}')
#
# import pyperclip
#
# unsat = Controller.ev_controller.unsatified_energy
# not_full = Controller.ev_controller.not_full_count
# cost = Controller.ev_controller.total_ev_charging_cost
# energy = Controller.ev_controller.total_ev_charging_energy
# ratio = cost / energy if energy != 0 else 0
#
# column = "\n".join([
#     str(unsat),
#     str(not_full),
#     str(cost),
#     str(energy),
#     str(ratio)
# ])
#
# pyperclip.copy(column)
# print("Copied column-wise to clipboard:")
# print(column)

Controller.hems_database.df.to_csv(f'./Results/controller_ESS-{TARIFF_TYPE}_{TEST_EPS}_change.csv')
House.simulation_df.to_csv(f'./Results/simulation_ESS-{TARIFF_TYPE}_{TEST_EPS}_change.csv')
