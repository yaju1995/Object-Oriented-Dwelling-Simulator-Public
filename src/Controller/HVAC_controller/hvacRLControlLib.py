import numpy as np
from datetime import timedelta, datetime, time
from Controller.ControlLib import controller
from SIM.EquipmentClass import InverterModel, EVModel, MeterModel, HVACModel
from support.lib_config import CustomLogger
from support.live_plotter import LivePlotter, LivePlotter4
from Controller.DQNmodel.DQN_Agent import DQNAgent

logger = CustomLogger(command=False, color='green')


class hvacController(controller):
    def __init__(self, resolution: timedelta, global_database, update_period,look_ahead,
                 rl_agent: DQNAgent,
                 mode='Train', hvac_power_kw=8,
                 energy_normalizer=1,
                 action_map=None,
                 temp_ref=22.5,
                 temp_deviation=1,
                 hvac_type = 'heating',
                 enable_plotter = False):
        super().__init__(resolution, global_database, update_period, look_ahead)
        self.hvac_type = hvac_type
        self.set_HVAC_Power = None
        self.HVAC_ON = False
        self.hvac_power_kw = hvac_power_kw
        self.temp_ref = temp_ref  # can be changed based on user requirement
        self.temp_deviation = temp_deviation  # can be changed based on user requirement
        #############################################################
        self.tariff_handler = None
        ############################################################
        self.rl_agent = rl_agent
        self.agent_name = rl_agent.name
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = None
        self.mode = mode
        self.cumulative_reward = 0
        self.action_map = action_map
        ############################################################
        self.energy_normalizer = energy_normalizer
        ############################################################
        self.enable_plotter = enable_plotter
        self.no_sim = 0
        self.sum_reward = 0
        self.avg_reward = 0
        if self.enable_plotter:
            self.live_plotter = LivePlotter4(titles=['Cum Reward', 'Avg Reward', 'Loss', 'Epsilon', ],
                                             xlabels=['Eps', 'Eps', 'Eps', 'Eps'],
                                             ylabels=['Reward', 'Avg Reward', 'Loss', 'Eps'])

    def __str__(self):
        return (
            "HVAC Controller("
            ")"
        )

    def update_status(self, hvac_info: HVACModel):
        now_time = hvac_info.time
        next_time = now_time + self.resolution
        do_update = (next_time.minute % (self.update_period.total_seconds() // 60) == 0)
        if do_update:
            # print(f'Update statue time {now_time}')
            self.control_logic(next_time, False)
            if self.action is not None:
                # mapping action to value
                if self.action_map is not None:
                    action = self.action_map.get(self.action, 0)
                else:
                    action = self.action
                self.set_HVAC_Power = round(float(action) * self.hvac_power_kw)
        if self.hvac_type == 'heating':
            self.set_HVAC_Power = - self.set_HVAC_Power

        return self.set_HVAC_Power

    def control_logic(self, control_time: datetime, done):
        # print(f'In control logic: {control_time}')
        if self.state is not None and self.action is not None:
            # Get reward
            reward = self.compute_reward(control_time)
            self.cumulative_reward += reward
            self.no_sim += 1
            # print(reward, self.cumulative_reward)

            # Get next state #
            self.next_state = self.get_state(control_time)

            # Save transition (terminal or non-terminal)
            # pass to deque for delay reward update, update reward then add to agent store
            self.rl_agent.store_transition(self.state, self.action, reward, self.next_state, False)
            # self.rl_agent.store_transition(self.state, self.action, reward, self.next_state)

            logger.commandline(control_time, self.state, self.action, reward, self.next_state, False)

            # Train agent
            # if update ready then only at the end of the session
            if self.mode == 'Train':
                time_list = [
                    time(0, 0, 0),
                    time(6, 0, 0),
                    time(12, 0, 0),
                    time(18, 0, 0),
                ]

                # Update NN once a day
                if control_time.time() in time_list:
                    # print(f'Updating Agent : {control_time}')
                    losses = self.rl_agent.train(batch_size=500)
                    if losses:
                        loss = losses.get('loss')
                        eps = losses.get('eps')
                        self.sum_reward += self.cumulative_reward
                        self.avg_reward = self.sum_reward / self.no_sim
                        if self.enable_plotter:  # only plot after 24 hrs or something

                            
                            self.live_plotter.update([self.cumulative_reward,
                                                      self.avg_reward,
                                                      loss,
                                                      eps,
                                                      ])

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
            greedy = False
        else:
            greedy = True

        if self.state is not None:
            # safe layer implemented
            logger.commandline(f'state: {self.state}')
            self.action = self.rl_agent.choose_action(self.state, greedy=greedy)

        return

    def get_state(self, control_time: datetime):
        '''
        State:
        - Rate of change of temp
        - Instantaneous Temperature measurement
        - Boundary reference info
        - Next Tariff
        '''
        temp_t = None
        temp_t_1 = None
        control_time_1 = control_time - self.resolution
        value_t = self.global_databased.get_instant_state(now_time=control_time_1, keys=['Temperature - Indoor (C)'])
        control_time_2 = control_time_1 - self.resolution
        value_t_1 = self.global_databased.get_instant_state(now_time=control_time_2, keys=['Temperature - Indoor (C)'])
        # print(control_time_1,value_t)
        # print(control_time_2,value_t_1)
        res_mins = int(self.resolution.total_seconds() // 60)
        temp_t = value_t.get('Temperature - Indoor (C)')
        if value_t_1 is not None:
            temp_t_1 = value_t_1.get('Temperature - Indoor (C)')

        if temp_t and temp_t_1:
            diff = (temp_t - temp_t_1)/self.temp_deviation
            rate = diff/res_mins
        else:
            rate = 0
        # logger.commandline(value_t)
        # logger.commandline(value_t_1)
        t_up_limit = self.temp_ref + self.temp_deviation
        t_down_limit = self.temp_ref - self.temp_deviation
        t_in_norm = (temp_t - self.temp_ref)/self.temp_deviation
        t_dev_norm = self.temp_deviation / self.temp_ref

        # getting next tariff
        tariff_time = control_time
        tariff_df = self.tariff_handler.get_tariff_range_df(tariff_time, period=self.look_ahead,
                                                            resolution=self.update_period)
        tariff_states = tariff_df['tariff'].tolist()
        feed_tariff_states = tariff_df['feed_tariff'].tolist()

        # print(f'tariff : {control_time}: {tariff_states}, {feed_tariff_states}')
        tariff_states = np.array(tariff_states)
        im_tariff_max = self.tariff_handler.max_tariff
        im_tariff_min = self.tariff_handler.min_tariff

        feed_tariff_states = np.array(feed_tariff_states)
        exp_tariff_max = self.tariff_handler.max_feed_tariff
        exp_tariff_min = self.tariff_handler.min_feed_tariff

        tariff_max = max(exp_tariff_max, im_tariff_max)
        tariff_min = min(exp_tariff_min, im_tariff_min)

        tariff = np.round((tariff_states - tariff_min) / (tariff_max - tariff_min), 3)  # Normalized over max and min
        feed_tariff = np.round((feed_tariff_states - tariff_min) / (tariff_max - tariff_min), 3)

        state = np.array([rate, t_in_norm, *tariff, *feed_tariff], dtype=float)

        return state


    def compute_reward(self, control_time: datetime):
        # Reward for past data hence
        reward_time = control_time- self.resolution
        value = self.global_databased.get_instant_state(now_time=reward_time, keys=['Instant Cost',
                                                                                     'tariff',
                                                                                     'feed tariff',
                                                                                     'Heating Electric Power (kW)',
                                                                                     'Temperature - Indoor (C)'])

        tariff_states = value.get('tariff')

        im_tariff_max = self.tariff_handler.max_tariff
        im_tariff_min = self.tariff_handler.min_tariff

        feed_tariff_states = value.get('feed tariff')
        # print(f'tariff : {reward_time}: {tariff_states}, {feed_tariff_states}')
        feed_tariff_max = self.tariff_handler.max_feed_tariff
        feed_tariff_min = self.tariff_handler.min_feed_tariff

        tariff_max = max(feed_tariff_max, im_tariff_max)
        tariff_min = min(feed_tariff_min, im_tariff_min)

        normalized_tariff = (tariff_states - tariff_min) / (tariff_max - tariff_min)  # Normalized over max and min
        normalized_feed_tariff = (feed_tariff_states - tariff_min) / (
                tariff_max - tariff_min)  # Normalized over max and min

        # Normalized energy
        hvac_period_power = abs(value.get('Heating Electric Power (kW)'))
        normalized_period_energy = hvac_period_power / self.energy_normalizer

        # Normalized reward

        r_cost = -(normalized_period_energy * normalized_tariff)
        ########################################################################
        # thermal reward
        t_in = value.get('Temperature - Indoor (C)')
        t_up_limit = self.temp_ref + self.temp_deviation
        t_down_limit = self.temp_ref - self.temp_deviation
        if t_down_limit<=t_in<=t_up_limit:
            r_thermal = 1
        else:
            r_thermal = -5
        # r_thermal = 1 - ((t_in - self.temp_ref) / self.temp_deviation) ** 2

        logger.commandline(f'power reward: {hvac_period_power, tariff_states, normalized_tariff, r_cost}')
        logger.commandline(f'temp reward: {t_in, self.temp_ref, self.temp_deviation, r_thermal}')
        logger.commandline(f'total reward: {t_in, self.temp_ref, self.temp_deviation, r_thermal}')

        reward = r_cost + r_thermal

        return reward
