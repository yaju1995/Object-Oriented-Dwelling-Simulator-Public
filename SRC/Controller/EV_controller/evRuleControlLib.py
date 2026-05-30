import pandas as pd
import numpy as np
from datetime import timedelta, datetime


from support.lib_config import CustomLogger
from support.live_plotter import LivePlotter4

from SIM.EquipmentClass import EVModel
from Controller.Database.PandasDatabase import DataStore
from SIM.Tariff.tariffHandler import tariffHandler

##############################
logger = CustomLogger(command=False, color='green')


##############################

def safe_div(a, b):
    return a / b if b else 0.0


class evController:
    def __init__(self, resolution: timedelta(minutes=1),
                 update_period: timedelta = timedelta(minutes=30),
                 global_database: DataStore = None, mode='Train', max_charging_power=7, look_ahead=1,enable_plotter = True):
        self.user_exp_soc = None
        self.user_dc_set_time = None
        self.ev_status = 0
        self.global_database = global_database
        self.mode = mode
        self.max_charging_power = max_charging_power
        self.look_ahead = look_ahead
        ########################################################
        self.connection_status = False
        self.connect_time = None
        self.leave_time = None
        self.full_charge_time = None
        self.connect_period = None
        self.resolution: timedelta = resolution
        self.update_period: timedelta = update_period
        self.initial_soc = None
        self.final_soc = None
        self.full_charge_status = False
        ######################################################
        self.energy_state_info = None
        self.instant_state_info = None
        ######################################################
        self.cumulative_reward = 0
        self.nom_period_charging_cost = 0  # reward variable
        self.step_count = 0
        ######################################################
        self.update_time = None
        # self.now_time = None
        self.instant_power = 0
        self.instant_soc = 0
        self.instant_tariff = None
        self.period_charging_energy = 0
        self.period_charging_cost = 0
        self.ev_sessions_charging_energy = 0
        self.ev_sessions_charging_cost = 0
        self.ev_df = pd.DataFrame()
        self.set_charging_power = 1.5  # charge with minimal power
        self.ev_sessions = 0
        self.not_full_count = 0
        ######################################################
        self.tariff_handler = tariffHandler()
        self.data_storage = None
        ######################################################
        self.total_ev_charging_cost = 0
        self.total_ev_charging_energy = 0
        self.unsatified_energy = 0
        self.satisfied_energy = 0
        self.initial_soc_list = []
        self.final_soc_list = []
        self.duration_list = []
        self.multiPlotter = LivePlotter4(['Cost per kWh', 'change SoC', 'Final SOC', 'Overall $/kwh'],
                                         xlabels=['Episode', 'Episode', 'Episode', 'Episode'],
                                         ylabels=['$/kWh', 'SoC/min', 'SOC%', '$/kwh'])
        self.enable_plotter = enable_plotter

    def update_status(self, ev_info: EVModel):

        # --- Update EV dataframe ---
        self.ev_df = pd.concat(
            [self.ev_df, pd.DataFrame([ev_info.model_dump()]).set_index("time")]
        )

        tariff, feed_tariff = self.tariff_handler.get_tariff(ev_info.time)
        self.instant_tariff = tariff
        # ===============================================================
        #   HANDLE CONNECTION / DISCONNECTION EVENTS
        # ===============================================================
        self.ev_status = ev_info.ev_status
        if ev_info.ev_status:  # EV is connected
            if not self.connection_status:  # just now connected initialize session
                logger.commandline(f"Car connected {ev_info.time}")
                self.connection_status = True
                self.ev_sessions += 1
                self.connect_time = ev_info.time
                self.update_time = None  # reset control timer
                self.period_charging_energy = 0
                self.period_charging_cost = 0
                self.ev_sessions_charging_energy = 0
                self.ev_sessions_charging_cost = 0
                self.initial_soc = ev_info.ev_soc

        # EV is not connected
        else:
            if self.connection_status:  # just disconnected end session
                # Print disconnection header
                logger.commandline(f"Car disconnected at {ev_info.time}")
                logger.commandline(f"Final reading → Power: {ev_info.ev_power:.2f} kW, SOC: {ev_info.ev_soc:.1f}%")

                self.connection_status = False
                self.leave_time = ev_info.time
                self.final_soc = self.instant_soc
                self.set_charging_power = 1.5  # reset the set_charging power to 1.5

                # --- FORCE FINAL REWARD FOR LAST ACTION ---
                now_time = ev_info.time
                next_time = now_time + self.resolution
                self.control_logic(next_time, ev_info)
                # --- PRINT SESSION SUMMARY ---
                session_duration = self.leave_time - self.connect_time
                session_minutes = int(session_duration.total_seconds() // 60)
                soc_change_rate = (self.final_soc * 100 - self.initial_soc * 100) / session_minutes
                if self.full_charge_status:
                    full_charge_duration = self.full_charge_time - self.connect_time
                    full_charge_minutes = int(full_charge_duration.total_seconds() // 60)
                    soc_change_rate = (self.final_soc * 100 - self.initial_soc * 100) / full_charge_minutes
                else:
                    self.not_full_count += 1
                    full_charge_minutes = 'Not fully charged'
                self.initial_soc_list.append(self.initial_soc)
                self.final_soc_list.append(self.final_soc)
                self.unsatified_energy += (1 - self.final_soc)
                self.satisfied_energy += self.final_soc
                self.duration_list.append(session_minutes)
                if self.enable_plotter:
                    self.multiPlotter.update(
                        [safe_div(self.ev_sessions_charging_cost, self.ev_sessions_charging_energy),
                         soc_change_rate, self.final_soc * 100,
                         self.total_ev_charging_cost / self.total_ev_charging_energy])

                summary = (
                    "\n===== EV Charging Session Summary =====\n"
                    f"Arrival time:    {self.connect_time}\n"
                    f"Departure time:  {self.leave_time}\n"
                    f"Session length:  {session_minutes} minutes\n"
                    f"Initial SOC:     {self.initial_soc * 100:.1f}%\n"
                    f"Final SOC:       {self.final_soc * 100:.1f}%\n"
                    f"Energy charged:  {self.ev_sessions_charging_energy:.3f} kWh\n"
                    f"Total cost:      €{self.ev_sessions_charging_cost:.3f}\n"
                    f"Time to full:    {full_charge_minutes} minutes\n"
                    f"Data points:     {len(self.ev_df)}\n"
                    f"SOC Rate change: {soc_change_rate}\n"
                    f"Overall cost : {self.total_ev_charging_cost}\n"
                    f"Overall energy : {self.total_ev_charging_energy}\n"
                    "=======================================\n"
                )

                logger.commandline(summary)

            return None  # If EV not connected return None
        # ===============================================================
        #   EV IS CONNECTED → UPDATE METRICS
        # ===============================================================
        self.step_count += 1
        # Update instantaneous state
        self.instant_power = ev_info.ev_power
        self.instant_soc = ev_info.ev_soc
        self.connect_period = ev_info.time - self.connect_time

        if self.instant_soc >= 1:
            self.full_charge_status = True  # if full charged complete a cycle
            self.full_charge_time = ev_info.time
            # logger.commandline('EV Full charged !')
        else:
            self.full_charge_status = False

        # --- Compute incremental energy PER STEP ---
        step_energy = round(self.instant_power * (self.resolution.total_seconds() / 3600), 6)

        self.period_charging_energy += step_energy
        self.ev_sessions_charging_energy += step_energy
        self.total_ev_charging_energy += step_energy

        # Cost ONLY for this step:
        self.period_charging_cost += round(step_energy * tariff, 6)
        self.ev_sessions_charging_cost += round(step_energy * tariff, 6)
        self.total_ev_charging_cost += round(step_energy * tariff, 6)

        # ===============================================================
        #   CONTROL UPDATE CHECK
        # ===============================================================
        # to do update in 14-min
        # Update only when time aligns with resolution (e.g., 15 min)
        now_time = ev_info.time
        next_time = now_time + self.resolution
        do_update = (next_time.minute % (self.update_period.total_seconds() // 60) == 0)

        if do_update:  # every 15 or 30 mins update
            # Apply control logic
            self.update_time = ev_info.time

            self.control_logic(next_time, ev_info=ev_info)  # Update charging power using controller

            # Reset period accumulators
            # logger.commandline(f'period charging cost: {self.period_charging_cost}')
            self.period_charging_energy = 0.0
            self.period_charging_cost = 0.0
            self.step_count = 0

        return self.set_charging_power  # return in kW

    def control_logic(self, control_time: datetime, ev_info):  # ddpg logic

        if ev_info.ev_status:
            self.set_charging_power = self.max_charging_power
        else:
            self.set_charging_power = 0
        return
    
    def update_user_dc_time(self, dc_time: datetime):
        # Need to add noise to this
        self.user_dc_set_time = dc_time

    def update_user_exp_soc(self, soc: int):
        self.user_exp_soc = soc / 100
