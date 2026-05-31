import pandas as pd
import numpy as np
from datetime import timedelta, datetime

from Controller.DDPGmodel.DDPG_Agent_multistep import DDPGAgent
from support.lib_config import CustomLogger
from SIM.EquipmentClass import EVModel
from Controller.Database.PandasDatabase import DataStore
from support.live_plotter import LivePlotter4
from SIM.Tariff.tariffHandler import tariffHandler

##############################
logger = CustomLogger(command=False, color='green')


##############################

def safe_div(a, b):
    return a / b if b else 0.0


class evController:
    def __init__(self, rl_agent: DDPGAgent, resolution: timedelta(minutes=1),
                 update_period: timedelta = timedelta(minutes=30),
                 global_database: DataStore = None, mode='Train',
                 max_charging_power=7, ev_battery_capacity=60,
                 look_ahead=1,
                 enable_plotter=False, time_constant=None):

        self.agent_name = rl_agent.name
        self.ev_status = 0
        self.global_database = global_database
        self.mode = mode
        self.max_charging_power = max_charging_power
        self.ev_battery_capacity = ev_battery_capacity
        self.chg_time_constant = (ev_battery_capacity / max_charging_power) *60 # in mins

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
        self.rl_agent = rl_agent
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = False

        self.cumulative_reward = 0
        self.eps_count = 0
        self.avg_reward = 0

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
        ######### user expectation ############################
        self.user_dc_set_time = None
        self.user_exp_soc = None
        self.remain_time_dc = None
        self.min_time_exp_soc = None
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
        self.enable_plotter = enable_plotter
        if self.enable_plotter:
            self.multiPlotter = LivePlotter4(['Cost per kWh', 'Overall $/kwh', 'Final SOC', 'Reward'],
                                             xlabels=['Episode', 'Episode', 'Episode', 'Episode'],
                                             ylabels=['$/kWh', '$/kwh', 'SOC%', 'Reward'])

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
                self.update_user_dc_time(ev_info.user_ev_dc_time)
                self.update_user_exp_soc(ev_info.expected_soc)


        else:  # EV is not connected
            if self.connection_status:  # just disconnected end session
                # Print disconnection header
                logger.commandline(f"Car disconnected at {ev_info.time}")
                logger.commandline(f"Final reading → Power: {ev_info.ev_power:.2f} kW, SOC: {ev_info.ev_soc:.1f}%")

                self.connection_status = False
                self.done = False
                self.leave_time = ev_info.time
                self.final_soc = self.instant_soc
                self.set_charging_power = 1.5  # reset the set_charging power to 1.5

                # --- FORCE FINAL REWARD FOR LAST ACTION ---
                now_time = ev_info.time
                next_time = now_time + self.resolution
                self.control_logic(next_time, True)
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
                         self.total_ev_charging_cost / self.total_ev_charging_energy,
                         self.final_soc * 100,
                         self.avg_reward])

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
        # Energy [kWh] = kW * (minutes / 60)
        step_energy = round(self.instant_power * (self.resolution.total_seconds() / 3600), 6)
        # logger.commandline(self.resolution, self.resolution.total_seconds()/ 3600)

        self.period_charging_energy += step_energy
        self.ev_sessions_charging_energy += step_energy
        self.total_ev_charging_energy += step_energy
        # Cost ONLY for this step:
        self.period_charging_cost += round(step_energy * tariff, 6)
        self.ev_sessions_charging_cost += round(step_energy * tariff, 6)
        self.total_ev_charging_cost += round(step_energy * tariff, 6)

        sim_cont_ration = self.update_period.total_seconds() / self.resolution.total_seconds()
        tariff_normal = ((tariff - self.tariff_handler.min_tariff) /
                         (self.tariff_handler.max_tariff - self.tariff_handler.min_tariff))
        period_energy_normal = (step_energy / (self.max_charging_power * (self.resolution.total_seconds() / 3600)))

        self.nom_period_charging_cost += (round(period_energy_normal * tariff_normal,
                                                6)) / sim_cont_ration

        # User based info extraction on disconnect time and min time to expected soc at max charging
        self.remain_time_dc = self.user_dc_set_time - ev_info.time

        # minimum time to SOC
        remain_soc = max(self.user_exp_soc - self.instant_soc, 0)
        self.min_time_exp_soc = remain_soc * self.chg_time_constant
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
            # logger.commandline(f'Updating control signal using RL: \n'
            #                    f'at time  : {now_time} \n'
            #                    f'from time: {next_time}')
            logger.commandline(f'Now Time: {now_time}')
            self.control_logic(next_time, False)  # Update charging power using controller
            # if fully charged it return, no action and reset all

            self.set_charging_power = self.action.item() * self.max_charging_power

            # Reset period accumulators
            # logger.commandline(f'period charging cost: {self.period_charging_cost}')
            self.period_charging_energy = 0.0
            self.period_charging_cost = 0.0
            self.nom_period_charging_cost = 0
            self.step_count = 0

        return self.set_charging_power  # return in kW

    def control_logic(self, control_time: datetime, done):  # ddpg logic control_time is now time
        '''
        Upadate self.action,
        :param control_time:
        :param now_time:
        :param done: if True then self.action = 0 no charging
        Else get action using rl_agent and update,
        :return: NA
        '''
        # ==========================================================
        # If we have a previous state-action pair, compute reward and update RL
        # ==========================================================
        if self.state is not None and self.action is not None:
            # Get reward
            reward = self.compute_reward()
            self.eps_count +=1
            self.cumulative_reward += reward

            self.avg_reward = self.cumulative_reward/self.eps_count
            # if self.enable_plotter:
            #     self.plotter.update(reward)

            # Get next state #
            self.next_state = self.get_state(control_time)

            # Save transition (terminal or non-terminal)
            # pass to deque for delay reward update, update reward then add to agent store
            self.rl_agent.store_transition(self.state, self.action, reward, self.next_state, done)

            logger.commandline(f'Control Time {control_time},{self.state}, {self.action}, {reward}, {self.next_state}, {done}')

            # Train agent
            # if update ready then only at the end of the session
            if self.mode == 'Train':  # updating every time
                self.rl_agent.train()

        # Check is the charge is disconnected : if so the set action = 0
        if done:
            # compute reward and end
            # reset all state, reward, and action to None and set action to 0
            self.state = None
            self.action = None
            self.reward = None
            self.next_state = None
            self.action = 0
            return
        # ==========================================================
        # Update state for next action
        # ==========================================================
        if self.next_state is not None:
            self.state = self.next_state
        else:
            self.state = self.get_state(control_time)
        # logger.commandline(f'Getting States {self.state}')

        # ==========================================================
        # Choose next action [update self.action]
        # ==========================================================
        if self.mode == 'Train':
            noise = 0.1
        else:
            noise = 0.0
        self.action = self.rl_agent.choose_action(self.state, noise_std=noise)
        if self.full_charge_status:
            self.action = np.array([0])
        # logger.commandline(f'Getting action {self.action}')
        return

    def get_state(self, control_time: datetime):
        """
        Get info from storage
        control_time is the next time period (15.59 now time the control time 15.59 + timedelta(resolution)
        => control time = 14.00 for resolution = 1 min
        State for RL agent.
        Must be normalized and numeric.
        soc : -(1-soc) handle the weight of ev charge [safely mapping for better exploration ]
        surplus with ev : reward on savings [net cost of energy during ev charging]
        surplus over next hour?

        tariff: tariff over next hour ?

        """
        soc = self.instant_soc

        # next period oven period tariff
        tariff_df = self.tariff_handler.get_tariff_range_df(control_time, period=self.look_ahead,
                                                            resolution=self.update_period)

        # print(tariff_df)
        tariff_states = tariff_df['tariff'].tolist()
        # feed_tariff_states = tariff_df['feed_tariff'].tolist()
        # print(control_time, soc, tariff_states)

        tariff_states = np.array(tariff_states)
        tariff_max = self.tariff_handler.max_tariff
        tariff_min = self.tariff_handler.min_tariff
        tariff = (tariff_states - tariff_min) / (tariff_max - tariff_min)  # Normalized over max and min

        W1 = self.weight_info()
        return np.array([soc, self.ev_status, W1, *tariff], dtype=float)

    def compute_reward(self):
        """
        Compute reward for the current step.
        Reward must be independent, reusable, and not change state.
        """
        W1 = self.weight_info()

        SoC_penalty = -(self.user_exp_soc - self.instant_soc)
        Cost_penalty = -round(self.nom_period_charging_cost, 3)

        reward = W1 * SoC_penalty + (1 - W1) * Cost_penalty

        return reward

    def weight_info(self):

        remain_time = int(self.remain_time_dc.total_seconds() // 60) # To hours

        min_time_exp_soc = int(self.min_time_exp_soc)
        # logger.commandline(self.remain_time_dc, remain_time, min_time_exp_soc)
        if remain_time > min_time_exp_soc:
            W1 = min_time_exp_soc / remain_time
        else:
            W1 = 1

        return W1

    def update_user_dc_time(self, dc_time: datetime):
        # Need to add noise to this
        self.user_dc_set_time = dc_time

    def update_user_exp_soc(self, soc: int):
        self.user_exp_soc = soc / 100

    def save_model(self, path):
        # Ensure the directory exists
        return self.rl_agent.save(path)

    def load_model(self, path):
        return self.rl_agent.load(path)
