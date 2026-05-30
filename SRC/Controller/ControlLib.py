import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from support.lib_config import CustomLogger
from SIM.EquipmentClass import InverterModel, MeterModel, HVACModel, EVModel

from Controller.DDPGmodel.DDPG_Agent import DDPGAgent, DDPGConfig
from Controller.Database.PandasDatabase import DataStore

logger = CustomLogger(command=True, color='green')


class controller:
    def __init__(self, resolution:timedelta,global_database:DataStore,
                 update_period:timedelta= timedelta(minutes=30),
                 look_ahead= 1):
        '''

        :param agent:
        :param resolution:
        :param control_period:
        '''

        self. resolution = resolution
        self.update_period = update_period
        self.ess_controller_df = pd.DataFrame()
        self.global_databased = global_database
        self.look_ahead = look_ahead

        # if self.agent = None in this condition use Rule based

    def update_status(self, now_time: datetime):
        # Here I am get 1 min data need to convert that to
        # get state vs individual data
        # estimate load, pv, battery soc, tariffs

        #Update EV

        """

        if control_time >= control_period:
        evaluate the state over last period
        demand[next period], generation[next period], tariff[next period], feed_tariff[next period], soc[instant]

        Then
        Priority control
        EV, [15 min resolution ]
        HVAC, [5 min resolution ]
        ESS , [10 min resolution ] , minimum to support all the loads
        """
        now_time = now_time
        next_time = now_time + self.resolution
        do_update = (next_time.minute % (self.update_period.total_seconds() // 60) == 0)
        if do_update:
            pass
        pass



    def get_state(self,control_time: datetime):
        # based on the time we collect the state
        # 00.00.00 collect information instants wh with respect to control_period
        """
        :param time:
        :return:
        """
        '''
        Forecasted Load 
        Forecast generation 
        get next period tariff
        get next period feed tariff
        get instantaneous soc
        '''
        pass
