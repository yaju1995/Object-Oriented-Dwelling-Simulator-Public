
import random
from datetime import datetime, timedelta, time
from SIM.Simulator import dwelling
from SIM.Simulator_Config.config_list_train import (pv_config,
                                                    ev_config,
                                                        thermal_config,
                                                        weather_file,
                                                        demand_config,
                                                        battery_config)

from Controller.HEMSControlRL import HEMSController
from SIM.Tariff.TariffGenerator import RandomTariffGenerator

RES = 60
RESOLUTION = timedelta(minutes=RES)  # 1 min resolution info
DURATION = timedelta(days=10)
START_TIME = datetime(2018, 1, 1)



SEED = 0
random.seed(SEED)

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

# initialize house
House.initialized_df()


# # Defining controller
Controller = HEMSController(name='Dwelling_1', data_resolution=RESOLUTION, meter_tariff=House.tariff,
                            ev_update_period=timedelta(minutes=RES),
                            ess_update_period=timedelta(minutes=RES),
                            havc_update_period=timedelta(minutes=RES),
                            ev_config=ev_config,
                            ess_config=battery_config,
                            hvac_config=thermal_config)
#
#########################################################################
current_time = START_TIME
end_time = START_TIME + DURATION
control_signal = {}

start = datetime.now()



day = 0
# Running a training loop
while current_time <= end_time-RESOLUTION:

    inverter, meter, ev, hvac, status = House.step(control_signal) # step return house info
    
    # geting perfect future value
    Demand = House.simulation_df.loc[current_time+RESOLUTION, "Demand Electric Power (kW)"] # next period
    Generation = House.simulation_df.loc[current_time+RESOLUTION, "PV Electric Power (kW)"] # next period

    inverter.forecast_demand = Demand
    inverter.forecast_generation = Generation

    control_signal = Controller.update(ev_info=ev, inverter_info=inverter, hvac_info=hvac, meter_info=meter)

    if control_signal: # check control signal
        # print(control_signal)
        pass

    current_time += RESOLUTION
    if current_time.time() == time(12, 00):
        House.tariff.generate_tariff() # generating new tariff for next day at 12.00 pm 
    
    elif current_time.time() == time(0, 0):
        House.tariff.updated_tariff() # updating next day tariff
        # update the SOC external for training
        next_soc = House.Battery.set_soc(random.uniform(0.05, 1)) # reset that will occur
        day +=1

    if (day + 1) % 5 == 0 or day == 1000:
        percent = day / 1000 * 100
        if day == 1000:  # force 100% at the end
            percent = 100
        bar = '█' * int(percent / 5) + '-' * (20 - int(percent / 5))
        print(f"\rSeed {SEED} |{bar}| {percent:.1f}% completed ::{day}:: {Controller.ess_controller.avg_reward}", end="")

    # saving all policy [model] at required eps 
    if day in (500, 1000, 2000, 3000, 4000, 5000,6000, 7000, 8000, 9000,10000):
        Controller.save_models(day)

end = datetime.now()
duration = (end - start).total_seconds()

print(f"Simulation took {duration:.4f} seconds")
# print(f'Final House Cost: {Controller.hems_database.df["Instant Cost"].sum()}')

Controller.hems_database.df.to_csv('../Results/controller_train_EV_V2G.csv')
House.simulation_df.to_csv('../Results/simulation_train_EV_V2G.csv')
