import random
from datetime import datetime, timedelta, time

from SIM.Simulator import dwelling
from SIM.Simulator_Config.config_list_train import (pv_config,
                                                        ev_config,
                                                        thermal_config,
                                                        weather_file,
                                                        demand_config,
                                                        battery_config)

from Controller.HEMSControlRule import HEMSController
from SIM.Tariff.TariffGenerator import RandomTariffGenerator
from SIM.ControlSignalHandler import ControlSignal
manual_control = ControlSignal()
RES = 60
RESOLUTION = timedelta(minutes=RES)  # 1 min resolution info
DURATION = timedelta(days=10)
START_TIME = datetime(2018, 1, 1)

SEED = 0
House = dwelling(name='Dwelling_1',
                 start_time=START_TIME,
                 resolution=RESOLUTION,
                 duration=DURATION,
                 demand_config=demand_config,
                 weather_file=None,
                 pv_config=pv_config,
                 battery_config=battery_config,
                 ev_config=None,
                 thermal_config=None,
                 seed=SEED)

Tariff_gen = RandomTariffGenerator(low=0.1, high=0.4, resolution=timedelta(minutes=RES), seed=SEED)
House.tariff.tariff_model = Tariff_gen
House.tariff.feed_tariff_model = Tariff_gen
House.tariff.generate_tariff()  # First Generate
House.tariff.updated_tariff()  # Then Update
House.initialized_df()


Controller = HEMSController(name='Dwelling_1', data_resolution=RESOLUTION, meter_tariff=House.tariff,
                            ev_update_period=timedelta(minutes=RES),
                            ess_update_period=timedelta(minutes=RES),
                            havc_update_period=timedelta(minutes=RES),
                            ev_config=ev_config,
                            ess_config=battery_config,
                            hvac_config=thermal_config)

current_time = START_TIME
end_time = START_TIME + DURATION
control_signal = {}

start = datetime.now()

day = 0

# Running a training loop
while current_time <= end_time-RESOLUTION:

    inverter, meter, ev, hvac, status = House.step(control_signal)
    
    # geting perfect future value
    Demand = House.simulation_df.loc[current_time+RESOLUTION, "Demand Electric Power (kW)"] # next period
    Generation = House.simulation_df.loc[current_time+RESOLUTION, "PV Electric Power (kW)"] # next period

    inverter.forecast_demand = Demand
    inverter.forecast_generation = Generation

    control_signal = Controller.update(ev_info=ev, inverter_info=inverter, hvac_info=hvac, meter_info=meter)

    if control_signal: # check control signal
        # print(control_signal)

        # # update manually -> control signal update
        # manual_control.Battery_P_Setpoint = 1000 #in watts
        # manual_control.EV_Max_Power = 0 # in watts
        # manual_control.HVAC_Heating_Power = 0 # in watts
        # control_signal = manual_control.generate_control_signal() # will give a dictonary use in the simulator
        pass

    current_time += RESOLUTION
    if current_time.time() == time(12, 00):
        House.tariff.generate_tariff()
    elif current_time.time() == time(0, 0):
        House.tariff.updated_tariff()
        # update the SOC external for training
        next_soc = House.Battery.set_soc(random.uniform(0.05, 1)) # reset that will occur
        day +=1
    # only for RL approch wi
    if (day + 1) % 5 == 0 or day == 1000:
        percent = day / 1000 * 100
        if day == 1000:  # force 100% at the end
            percent = 100
        bar = '█' * int(percent / 5) + '-' * (20 - int(percent / 5))
        # print(f"\rSeed {SEED} |{bar}| {percent:.1f}% completed ::{day}:: {Controller.ess_controller.avg_reward}", end="")



end = datetime.now()
duration = (end - start).total_seconds()
print(f"Simulation took {duration:.4f} seconds")

print(f'Final House Cost: $ {Controller.hems_database.df["Instant Cost"].sum()}')

# check the dir 
Controller.hems_database.df.to_csv('./Results/controller_RULE.csv')
House.simulation_df.to_csv('./Results/simulation_RULE.csv')