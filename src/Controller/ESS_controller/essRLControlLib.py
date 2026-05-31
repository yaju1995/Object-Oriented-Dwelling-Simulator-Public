import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
from support.lib_config import CustomLogger
from Controller.Database.PandasDatabase import DataStore
from SIM.EquipmentClass import InverterModel, MeterModel
from support.live_plotter import LivePlotter, LivePlotter4

from Controller.DDPGmodel.bounded_DDPG_Agent_multistep import Bound_DDPGAgent

logger = CustomLogger(command=False, color='cyan')

class essController:
    def __init__(self, rl_agent: Bound_DDPGAgent, mode='Train', resolution: timedelta = timedelta(minutes=1),
                 update_period: timedelta = timedelta(minutes=15),
                 global_database: DataStore = None,
                 max_charging_kw=0,
                 max_discharging_kw=0,
                 look_ahead=1,
                 energy_normalizer=1,
                 enable_plotter=False,
                 trigger_time=None):

        '''

        :param rl_agent:
        :param mode:
        :param resolution:
        :param update_period:
        :param global_database:
        :param max_charging_kw:
        :param max_discharging_kw:
        :param look_ahead: look ahead period for n step return
        :param energy_normalizer: Battery max capacity
        '''
        #############################################################
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
        #############################################################
        self.rl_agent = rl_agent
        self.agent_name = rl_agent.name
        self.rl_agent.bound_fn = self.soc_charge_limit
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = False
        self.mode = mode
        self.cumulative_reward = 0
        #############################################################
        self.energy_normalizer = energy_normalizer
        ############################################################
        self.enable_plotter = enable_plotter
        # self.live_plotter = LivePlotter(title='Reward',
        #                                xlabel='episode',
        #                                ylabel='reward')
        self.no_sim = 0
        self.sum_reward = 0
        self.avg_reward = 0
        if enable_plotter:
            self.live_plotter = LivePlotter4(titles=['Cumulated Reward', 'Avg Reward', 'Critic loss', 'Actor loss', ],
                                             xlabels=['Eps', 'Eps', 'Eps', 'Eps'],
                                             ylabels=['Reward', 'Avg Reward', 'Critic loss', 'Actor loss'])

        self.trigger_time = trigger_time

        self.forecast_demands = None
        self.forecast_generations = None

    def __str__(self):
        return (
            "ESSController(\n"
            f"  agent='{self.agent_name}',\n"
            f"  mode='{self.mode}',\n"
            f"  resolution={self.resolution},\n"
            f"  update_period={self.update_period},\n"
            f"  max_charge_kw={self.max_charging_power},\n"
            f"  max_discharge_kw={self.max_discharging_power},\n"
            f"  look_ahead={self.look_ahead},\n"
            f"  energy_normalizer={self.energy_normalizer},\n"
            f"  ESS_charge={self.ESS_charge},\n"
            f"  set_battery_power={self.set_battery_power},\n"
            f"  cumulative_reward={self.cumulative_reward}\n"
            ")"
        )

    def soc_charge_limit(self, state):
        soc = state[0]

        dt = (self.resolution.total_seconds() / 60) / 60
        max_c_rate = 0.5

        charge_limit = min(max_c_rate, (1 - soc) / dt)
        discharge_limit = min(max_c_rate, (soc - 0.05) / dt)

        charge_limit = max(0.0, charge_limit)
        discharge_limit = max(0.0, discharge_limit)

        # charge_limit = min(0.5, 1 - soc)
        # discharge_limit = min(0.5, soc - 0.05)

        return np.array([-discharge_limit]), np.array([charge_limit])

    def update_status(self, meter_info: MeterModel, inverter_info: InverterModel):
        done = False
        # consumption = round(meter_info.active_power -
        #                     inverter_info.battery_power +
        #                     inverter_info.pv_power, 3)

        # ===============================================================
        #   CONTROL UPDATE CHECK
        # ===============================================================
        # to do update in 14-min
        # Update only when time aligns with resolution (e.g., 15 min)
        now_time = meter_info.time
        next_time = now_time + self.resolution
        self.forecast_demands = inverter_info.forecast_demand
        self.forecast_generations = inverter_info.forecast_generation

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

        if next_time.time() == time(0, 0, 0):
            done = True
        if do_update:  # every 15 or 30 mins update 60 mins
            self.control_logic(next_time, done)  # next time period is the control time period
            if self.action is not None:
                # print(self.action)
                self.set_battery_power = round(float(self.action.item()) * self.energy_normalizer, 3)
            else:
                self.set_battery_power = self.action
            # states = self.get_state(now_time)
            # logger.commandline(states)

        return self.set_battery_power

    def control_logic(self, control_time: datetime, done):  # ddpg logic
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
            reward = self.compute_reward(control_time)
            self.cumulative_reward += reward

            # print(reward, self.cumulative_reward)

            # Get next state #
            self.next_state = self.get_state(control_time)  # get latest state

            # Save transition (terminal or non-terminal)
            # pass to deque for delay reward update, update reward then add to agent store
            self.rl_agent.store_transition(self.state, self.action, reward, self.next_state, False)
            # self.rl_agent.store_transition(self.state, self.action, reward, self.next_state)

            logger.commandline(control_time, self.state, self.action, reward, self.next_state, False)
            # print(control_time, self.state, self.action, reward, self.next_state, False)

            # Train agent
            # if update ready then only at the end of the session
            if self.mode == 'Train':

                if control_time.time() in self.trigger_time:
                    # print(f'Updating Agent : {control_time}')
                    losses = self.rl_agent.train()
                    if losses:
                        critic = losses.get('critic_loss')
                        actor = losses.get('actor_loss')
                    else:
                        critic = 0
                        actor = 0
                    self.no_sim += 1
                    self.sum_reward += self.cumulative_reward
                    self.avg_reward = self.sum_reward / self.no_sim
                    if self.enable_plotter:  # only plot after 24 hrs or something

                        self.live_plotter.update([self.cumulative_reward,
                                                  self.avg_reward,
                                                  critic,
                                                  actor,
                                                  ])

                        # print(f'{self.cumulative_reward, avg_reward, critic, actor}')
                    self.cumulative_reward = 0

                    # update model every 24 hrs

        # Check is the charge is disconnected : if so the set action = 0
        if done:
            # compute reward and end
            # reset all state, reward, and action to None and set action to 0
            self.state = None
            self.action = None
            self.reward = None
            self.next_state = None
            return
        # ==========================================================
        # Update state for next action
        # ==========================================================
        if self.next_state is not None:
            self.state = self.next_state
        else:
            self.state = self.get_state(control_time)

        # ==========================================================
        # Choose next action [update self.action]
        # ==========================================================
        if self.mode == 'Train':
            noise = 0.1
        else:
            noise = 0.0

        # safe layer implemented
        self.action = self.rl_agent.choose_action(self.state, noise_std=noise, bound_fn=self.soc_charge_limit)

        return

    def get_state(self, control_time: datetime):
        value = self.global_databased.get_instant_state(now_time=control_time, keys=['Consumption (kW)',
                                                                                     'Battery SOC (-)',
                                                                                     'Generation (kW)'])
        # getting current time information
        # print(value)
        consumption = value.get('Consumption (kW)')
        soc = value.get('Battery SOC (-)')
        generation = value.get('Generation (kW)')

        forecast_surplus = round((self.forecast_generations - self.forecast_demands) / self.energy_normalizer, 6)
        surplus = round((generation - consumption)/self.energy_normalizer, 6)

        # print(f'now {surplus}')
        # print(f'forecast {forecast_surplus}')
        # print(consumption, generation, self.energy_normalizer, surplus)

        # getting next tariff
        tariff_df = self.tariff_handler.get_tariff_range_df(control_time, period=self.look_ahead,
                                                            resolution=self.update_period)
        tariff_states = tariff_df['tariff'].tolist()
        feed_tariff_states = tariff_df['feed_tariff'].tolist()
        tariff_states = np.array(tariff_states)
        im_tariff_max = self.tariff_handler.max_tariff
        im_tariff_min = self.tariff_handler.min_tariff

        feed_tariff_states = np.array(feed_tariff_states)
        exp_tariff_max = self.tariff_handler.max_feed_tariff
        exp_tariff_min = self.tariff_handler.min_feed_tariff

        tariff_max = round(max(exp_tariff_max, im_tariff_max),2)
        tariff_min = round(min(exp_tariff_min, im_tariff_min),2)
        # print(tariff_max, tariff_min)

        tariff = np.round((tariff_states - tariff_min) / (tariff_max - tariff_min), 3)  # Normalized over max and min
        feed_tariff = np.round((feed_tariff_states - tariff_min) / (tariff_max - tariff_min), 3)
        # print(value)
        logger.commandline(
            f'state:{control_time}:::{soc}, {surplus}, {tariff_states},->{tariff},{feed_tariff_states}->{feed_tariff}')
        # pass
        return np.array([ soc,forecast_surplus, *tariff, *feed_tariff], dtype=float)

    def compute_reward(self, control_time: datetime):  # replace control time wiht self time??

        value = self.global_databased.get_instant_state(now_time=control_time, keys=['Instant Cost',
                                                                                     'tariff',
                                                                                     'feed tariff',
                                                                                     'Total Electric Power (kW)',
                                                                                     'Battery Set Power (W)',
                                                                                     'Battery Electric Power (kW)'])

        # print(value)
        # tariff normalization
        tariff_states = value.get('tariff')
        im_tariff_max = self.tariff_handler.max_tariff
        im_tariff_min = self.tariff_handler.min_tariff

        feed_tariff_states = value.get('feed tariff')
        feed_tariff_max = self.tariff_handler.max_feed_tariff
        feed_tariff_min = self.tariff_handler.min_feed_tariff

        tariff_max = max(feed_tariff_max, im_tariff_max)
        tariff_min = min(feed_tariff_min, im_tariff_min)

        normalized_tariff = (tariff_states - tariff_min) / (tariff_max - tariff_min)  # Normalized over max and min
        normalized_feed_tariff = (feed_tariff_states - tariff_min) / (
                tariff_max - tariff_min)  # Normalized over max and min

        # Normalized energy
        period_power = value.get('Total Electric Power (kW)')
        period_energy = period_power * int(self.resolution.total_seconds() // 60) / 60
        normalized_period_energy = round(period_power / self.energy_normalizer, 6)

        # Normalized reward
        if period_power >= 0:  # importing
            cost = -round(normalized_period_energy * normalized_tariff, 6)
        else:  # exporting
            cost = -round(normalized_period_energy * normalized_feed_tariff, 6)
        # cost = value.get('Instant Cost')
        logger.commandline(f'reward:{control_time}:\n'
                           f'{tariff_states}->{normalized_tariff}\n'
                           f'{feed_tariff_states}->{normalized_feed_tariff}\n'
                           f'{period_power},{normalized_period_energy},{cost}')
        # pass
        # check error is action does not match the reward
        set_power = value.get('Battery Set Power (W)') / 1000
        actual_power = value.get('Battery Electric Power (kW)')
        error_Reward = 0
        # print(set_power, actual_power)
        if (round(set_power, 3) - round(actual_power, 3))>0.001:
            # print(f'Unbalance: { round(set_power, 3)}!={round(actual_power, 3)} ')
            # logger.commandline(f'{set_power}!={actual_power} ')
            error_Reward = -5
        # print(f'{period_power} {set_power}: Cost Reward: {cost} Error Reward {error_Reward} ')
        reward = cost + error_Reward

        # pass
        return reward

    def save_model(self, path):
        # Ensure the directory exists
        return self.rl_agent.save(path)

    def load_model(self, path):
        # logger.commandline(self.rl_agent.load_old(path))
        return self.rl_agent.load(path)
