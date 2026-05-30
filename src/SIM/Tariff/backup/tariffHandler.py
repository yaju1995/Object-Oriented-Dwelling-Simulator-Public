import pandas as pd
import numpy as np
from datetime import datetime
from SRC.support.lib_config import CustomLogger
from SRC.support.probability_lib import ProbabilityDistributions
from datetime import timedelta
from SRC.SIM.Tariff.TariffGenerator import BaseTariffGenerator
from typing import Optional

logger = CustomLogger(command=True)


# --- Convert to seconds ---
def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second


def get_active_period_value(df, now_time):
    idx_seconds = np.array([time_to_seconds(t) for t in df.index])
    now_seconds = time_to_seconds(now_time)
    valid_idx = np.where(idx_seconds <= now_seconds)[0]
    if len(valid_idx) == 0:
        # before first time → wrap around to last value of previous day
        return df.iloc[-1]['value']
    return df.iloc[valid_idx[-1]]['value']


class tariffHandler:
    def __init__(self, traiff_model: Optional[BaseTariffGenerator] = None,
                 feed_tariff_model: Optional[BaseTariffGenerator] = None,
                 type=2, tariff_resolution: timedelta = timedelta(minutes=60)):
        ''''
        tariff_model = Based on some sort of tariff model from loacl grid ore DSO
        type if 1 only one tariff, sane for feed in and export tariff
        type if 2 have different feed in or export tariff
        '''

        self.tariff_model = traiff_model  # might be as gaussin or based on something
        self.feed_tariff_model = feed_tariff_model
        self.tariff_resolution = tariff_resolution
        self.tariff = None
        self.max_tariff = None
        self.min_tariff = None
        self.feed_tariff = None
        self.max_feed_tariff = None
        self.min_feed_tariff = None
        self.avg_tariff = None
        self.avg_feed_tariff = None

        self.next_24hr_tariff = None
        self.next_24hr_feed_tariff = None

    def generate_tariff(self):
        '''
        :param model: any model to generate 24 hrs data
        :return: df with
        time        tariff
        00:00:00    97.33
        00:30:00   120.11
        if generated update in self.tariff and self.feed_tariff
        self.tariff = generated tariff
        self.feed_tariff = generated tariff
        '''
        if self.tariff_model:
            self.next_24hr_tariff = self.tariff_model.generate_tariff()
        if self.feed_tariff_model:
            self.next_24hr_feed_tariff = self.feed_tariff_model.generate_tariff()

    def upload_tariff(self, file: str):
        tariff = pd.read_csv(file, index_col=0)  #
        tariff.index = pd.to_datetime(tariff.index, format="%H:%M:%S").time
        # check if the resolution matches the self.tariff resolution
        self.tariff = tariff
        self.update_min_max()

    def upload_feed_tariff(self, file: str):
        tariff = pd.read_csv(file, index_col=0)  #
        tariff.index = pd.to_datetime(tariff.index, format="%H:%M:%S").time
        self.feed_tariff = tariff
        self.update_min_max()

    def get_tariff(self, now_time):
        # print(type(now_time))
        tariff = 0.0
        feed_tariff = 0.0
        if self.tariff.empty is not None:
            tariff = get_active_period_value(self.tariff, now_time)
        if self.feed_tariff is not None:
            feed_tariff = get_active_period_value(self.feed_tariff, now_time)
        return float(tariff), float(feed_tariff)

    def get_tariff_range_df(self, now_time, period=4, resolution: timedelta = timedelta(minutes=15)):
        """
        Returns a DataFrame of tariff & feed-in tariff for the next N steps.

        :param now_time: starting datetime
        :param period: number of steps to return (NOT hours)
        :param resolution: minutes per step
        :return: DataFrame indexed by timestamp
        """
        timestamps = []
        tariff_list = []
        feed_tariff_list = []

        current_time = now_time

        for _ in range(period):
            timestamps.append(current_time)

            t, f = self.get_tariff(current_time)
            tariff_list.append(t)
            feed_tariff_list.append(f)

            current_time += resolution

        df = pd.DataFrame({
            "tariff": tariff_list,
            "feed_tariff": feed_tariff_list,
        }, index=pd.to_datetime(timestamps))

        df.index.name = "time"

        return tariff_list, feed_tariff_list

    def updated_tariff(self):
        ''' In case of irish gird next day tariff is revived at the 12 hour of the day'''
        if self.next_24hr_tariff is not None:
            logger.commandline('Updating tariff value')
            self.tariff = self.next_24hr_tariff
            self.next_24hr_tariff = None
        if self.next_24hr_feed_tariff is not None:
            logger.commandline('Updating feed tariff value')
            self.feed_tariff = self.next_24hr_feed_tariff
            self.next_24hr_feed_tariff = None
        self.update_min_max()

    def update_min_max(self):
        if self.tariff is not None:
            self.max_tariff = self.tariff['value'].max()
            self.min_tariff = self.tariff['value'].min()
            self.avg_tariff = self.tariff['value'].mean()
        if self.feed_tariff is not None:
            self.max_feed_tariff = self.feed_tariff['value'].max()
            self.min_feed_tariff = self.feed_tariff['value'].min()
            self.avg_feed_tariff = self.tariff['value'].mean()
