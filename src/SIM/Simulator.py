import numpy as np
import datetime as dt
import pandas as pd

from SIM.ESS.ess_handler import ESSHandler
from SIM.EV.ev_handler import EVHandler
from SIM.Thermal.thermal_handler import ThermalHandler
from SIM.Weather.epwhandler_V_1_0_3 import EPWWeatherHandler
from SIM.Tariff.tariffHandler_V_2 import tariffHandler
from SIM.DataGenerator.data_generators import PatternGenerationHandler
from support.lib_config import CustomLogger
from SIM.EquipmentClass import InverterModel, EVModel, MeterModel, HVACModel

np.set_printoptions(suppress=True, precision=2)

logger = CustomLogger(command=True)

# Information collection
NET_POWER = ['Total Electric Power (kW)', 'Total Reactive Power (kVAR)']
INVERTER = ['PV Electric Power (kW)', 'Battery Electric Power (kW)', 'Battery SOC (-)']
EV = ['EV Parked', 'EV SOC (-)', 'EV Electric Power (kW)', 'User Plug Out Time', 'Expected SOC']
HVAC = ["Temperature - Indoor (C)", "HVAC Heating Electric Power (kW)", "Temperature - Outdoor (C)"]


class SimulationEnded(Exception):
    pass


class dwelling:
    def __init__(self, name: str, start_time: dt.datetime, duration: dt.timedelta, resolution: dt.timedelta,
                 demand_config: dict = None, weather_file: str = None, pv_config: dict = None,
                 battery_config: dict = None, ev_config: dict = None, thermal_config=None, PV=0,
                 tariff: tariffHandler = None, seed: int = 42):
        self.name = name
        self.start_time: dt.datetime = start_time  # dt.datetime(2018,1,1)
        self.duration = duration  # dt.timedelta(days=7)
        self.resolution: dt.timedelta = resolution  # dt.timedelta(minutes=resolution) # all data is generated based on
        self.end_time = self.start_time + self.duration
        self.now_time = None
        # the resolution
        logger.commandline(f'Start time: {self.start_time}: End time:{self.end_time}')
        self.weather_file = weather_file  # if none then random information, .ewp file
        self.pv_config = pv_config
        self.demand_config = demand_config
        self.battery_config: dict = battery_config
        self.Battery: ESSHandler | None = None  # pass ESS from external [take delta energy to estimated value]
        self.ev_config: dict = ev_config  # Need to change for multiple ev
        self.EV: EVHandler | None = None
        self.thermal_config = thermal_config
        self.Thermal: ThermalHandler | None = None  # a mode which take in temperature and power and return thermal, will have it setpoint

        # self.tariff = tariffHandler()

        # time range
        time_index = pd.date_range(start=self.start_time, end=self.end_time, freq=resolution)
        self.simulation_df = pd.DataFrame(index=time_index)  # generate data from day 1 0: to day 20
        self.simulation_df.index.name = "Time"

        self.tariff = tariffHandler()
        self.seed = seed

        ## IoT Devices Definition
        self.MeterModel = MeterModel()
        self.InverterModel = InverterModel()
        self.EVModel = EVModel()
        self.HVACModel = HVACModel()

        # seed setup
        self.rng = np.random.default_rng(seed)

    def initialized_df(self):
        meter_df = pd.DataFrame(
            0.0,
            index=self.simulation_df.index,
            columns=['Total Electric Power (kW)', 'Total Reactive Power (kVAR)']
        )
        self.simulation_df = self.simulation_df.join(meter_df)
        # for demand
        if self.demand_config:
            logger.commandline('Setup up the demand profile for simulation')
            # Validate that self.demand is a dict
            if not isinstance(self.demand_config, dict):
                logger.raise_error("self.demand must be a dictionary")
                return

            # Validate required keys
            required_keys = {"model", "file"}
            missing = required_keys - self.demand_config.keys()
            if missing:
                logger.raise_error(f"Missing required keys in self.demand: {missing}")
                return

            # Safe access
            model = self.demand_config.get('model')
            if model == 'default':
                model = 'normal'
                file_path = './src/SIM/Defaults/Demand/15_min_normal_test.csv'
            else:
                file_path = self.demand_config.get('file')

            # Proceed only if validation passed
            demandHandler = PatternGenerationHandler(
                model_name=model,
                csv_path=file_path
            )
            demand_df = demandHandler.get_simulation_data(
                start_date=self.start_time,
                duration_days=self.duration,
                resolution=self.resolution,
                seed=self.seed + 1, # different seed to aviod same value from generation
                column_name="Demand Electric Power (kW)",
            )
            # plot value
            # demandHandler.visualize_csv_pattern()

        else:
            logger.commandline("No demand model detected self.demand = None, setting value to 0")

            # Create a zero-filled demand DataFrame with same index length
            demand_df = pd.DataFrame(
                0.0,
                index=self.simulation_df.index,
                columns=["Demand Electric Power (kW)"]
            )
        self.simulation_df = self.simulation_df.join(demand_df)

        # For weather and PV
        if self.weather_file:
            logger.commandline('weather conf detected setting up weather infor (Outdoor Temp C)')
            weatherHandler = EPWWeatherHandler(self.weather_file)
            weather_df = weatherHandler.get_simulation_data(start_date=self.start_time,
                                                            resolution=self.resolution,
                                                            duration=self.duration)
            self.simulation_df = self.simulation_df.join(weather_df)
        else:
            logger.commandline('No weather file detected: self.weather = None')

        # For PV profile
        if self.pv_config:
            logger.commandline(self.pv_config)
            required_keys = {"type", "capacity W", "efficiency", "area per W", "tilt", "azimuth"}
            missing = required_keys - self.pv_config.keys()
            pv_rating = self.pv_config.get('capacity W', 0) / 1000
            pv_efficiency = self.pv_config.get('efficiency', 1)

            if missing:
                logger.raise_error(f"Missing required keys in self.demand: {missing}")
                return
            # check the type :
            pv_config_type = self.pv_config.get('type')
            if pv_config_type == 'train':
                logger.commandline('PV conf detected setting up PV generation for training setup')
                # len from the weather file fill the colum with random 0 to 1000 value
                self.simulation_df["Global Horizontal Radiation"] = self.rng.uniform(low=0,
                                                                                     high=self.pv_config.get('max_irradiance'),
                                                                                     size=len(self.simulation_df)
                                                                                     )
            else:  # use the
                logger.commandline('PV conf detected setting up PV generation using weather file')

            ghi = self.simulation_df["Global Horizontal Radiation"].fillna(0)
            pv_df = np.round((ghi / 1000.0) * pv_efficiency * pv_rating, 3)
            pv_df = pv_df.clip(lower=0)

            pv_df.name = "PV Electric Power (kW)"

            # print(pv_df.head())
            # pv_df = weatherHandler.get_pv_generation(start_date=self.start_time,
            #                                          duration=self.duration,
            #                                          resolution=self.resolution,
            #                                          pv_capacity_kw=pv_rating,
            #                                          pv_efficiency=pv_efficiency,
            #                                          )
            self.simulation_df = self.simulation_df.join(pv_df)

        ### For EV
        if self.ev_config or self.EV:  # Multi EV
            logger.commandline('==== Initializing EV ====')
            EV_KEYS = ["EV Electric Power (kW)", "EV Set Power (kW)", "EV SOC (-)", "EV Parked"]
            ev_df = pd.DataFrame(
                0.0,
                index=self.simulation_df.index,
                columns=EV_KEYS
            )
            # if ev config is provided then setup self.EV handler
            if self.ev_config:
                # checking config
                # required_keys = {"capacity Wh", "initial soc", "charging power W", "discharging power W",
                #                  "charging eff", "discharging eff"}
                required_keys = {"capacity Wh", "charging power W", "discharging power W",
                                 "charging eff", "discharging eff", "v2g_enabled", "profile_file"}
                missing = required_keys - self.ev_config.keys()
                if missing:
                    logger.raise_error(f"Missing required keys in self.demand: {missing}")
                    return
                # setup battery
                total_capacity_Wh = self.ev_config.get('capacity Wh')
                charging_power_W = self.ev_config.get('charging power W')
                discharging_power_W = self.ev_config.get('discharging power W')

                in_eff = self.ev_config.get('charging eff')
                out_eff = self.ev_config.get('discharging eff')
                v2g_enable = self.ev_config.get('v2g_enabled')
                if v2g_enable:
                    discharging_power_W = self.ev_config.get('discharging power W', 0)

                profile = self.ev_config.get('profile_file',
                                             "./SRC/SIM/Defaults/EV/pdf_Veh1_Level0.csv")  ############## defaults

                self.EV = EVHandler(
                    name="EV_1",
                    ev_profile_csv=profile,  # raw CSV used by generator
                    start_time=self.start_time,
                    resolution=self.resolution,
                    duration=self.duration,
                    total_capacity_Wh=total_capacity_Wh,  # 60 kWh
                    charging_power_W=charging_power_W,  # 7. kW charger
                    discharging_power_W=discharging_power_W, # 7. kW charger
                    v2g_enabled=v2g_enable, # V2G capable
                    upper_limit_soc_pct = 100,
                    lower_limit_soc_pct = 20,
                    seed=self.seed,
                    in_eff=in_eff,
                    out_eff=out_eff
                )

            elif isinstance(self.EV, EVHandler):
                logger.commandline("EV is a EVHandler")
            else:
                logger.commandline('EV must be EVHandler, has different format recommended to use EVHandler format')
            self.simulation_df = self.simulation_df.join(ev_df)
        else:
            logger.commandline('NO EVs found: Ignoring EV setup')

        ### FOR ESS
        if self.battery_config or self.Battery:  # multi ESS
            logger.commandline('==== Initializing ESS ====')
            # setting up the df for battery
            BATTERY_KEYS = ["Battery Electric Power (kW)", "Battery Set Power (kW)", "Battery SOC (-)"]
            ess_df = pd.DataFrame(
                0.0,
                index=self.simulation_df.index,
                columns=BATTERY_KEYS
            )
            self.simulation_df = self.simulation_df.join(ess_df)
            if self.battery_config:
                # setting up battery
                required_keys = {"capacity Wh", "initial soc", "charging power W", "discharging power W",
                                 "charging eff", "discharging eff"}
                missing = required_keys - self.battery_config.keys()
                if missing:
                    logger.raise_error(f"Missing required keys in self.demand: {missing}")
                    return
                self.Battery = ESSHandler(
                    name=f"HouseESS_{self.name}",
                    total_capacity_Wh=self.battery_config.get('capacity Wh'),  # 10 kWh battery
                    initial_soc_pct=self.battery_config.get('initial soc'),  # start at 50%
                    charging_power_W=self.battery_config.get('charging power W'),  # 5 kW charge limit
                    discharging_power_W=self.battery_config.get('discharging power W'),  # 5 kW discharge limit
                    resolution=self.resolution,  # 15-minute time step
                    in_eff=self.battery_config.get('charging eff'),
                    out_eff=self.battery_config.get('discharging eff'),
                )
            elif isinstance(self.Battery, ESSHandler):
                logger.commandline("ESS is a ESSHandler")
            else:
                logger.commandline('ESS must be ESSHandler, has different format recommended to use ESSHandler format')
        else:
            logger.commandline('NO ESS found: Ignoring ESS setup')

        ### For Thermal
        if self.thermal_config or self.Thermal:
            logger.commandline('==== Initializing Thermal Loads ====')
            THERMAL_KEYS = ["HVAC Heating Electric Power (kW)", "HVAC Heating Setpoint (C)", "Temperature - Indoor (C)"]
            # outdoor temperature obtained from the
            thermal_df = pd.DataFrame(
                0.0,
                index=self.simulation_df.index,
                columns=THERMAL_KEYS
            )
            self.simulation_df = self.simulation_df.join(thermal_df)

            # initialized Thermal model
            if self.thermal_config:
                required_keys = {"initial temperature C", "tau", "W", "n"}
                missing = required_keys - self.thermal_config.keys()
                if missing:
                    logger.raise_error(f"Missing required keys in self.demand: {missing}")
                    return
                self.Thermal = ThermalHandler(resolution=self.resolution,
                                              initial_internal_temperature=self.thermal_config.get(
                                                  'initial temperature C'),
                                              tau=self.thermal_config.get('tau'),
                                              W=self.thermal_config.get('W'),
                                              n=self.thermal_config.get('n'))
            elif isinstance(self.Thermal, ThermalHandler):
                logger.commandline('Thermal model is ThermalHandler instance')
            else:
                logger.commandline('Thermal model is not ThermalHandler; recommended to use ThermalHandler')

        else:
            logger.commandline('No thermal Load detected: Ignoring Thermal Load demand')
        if self.tariff:
            pass
        else:
            logger.commandline('No tariff is defined: Ignoring Cost and tariff in the simulation')

    EXPECTED_KEYS = {
        "timestamp",
        "Demand Electric Power (kW)",
        "PV Electric Power (kW)",
    }

    def upload_data(self, file, columns = None):  # Upload for Demand and Generation only
        """
        Upload CSV data and update self.simulation_df.

        Requirements:
        - 'timestamp' column REQUIRED → becomes index
        - Resolution, start time, and duration must match
        - Updates only available columns
        """
        logger.commandline('Uploading data for simulation')
        # ----------------------------
        # Load CSV
        # ----------------------------
        df = pd.read_csv(file)

        # ----------------------------
        # Enforce timestamp column
        # ----------------------------
        if "timestamp" not in df.columns:
            raise ValueError("CSV must contain a 'timestamp' column.")

        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            raise ValueError("Failed to parse 'timestamp' column as datetime.")

        df = df.set_index("timestamp").sort_index()

        if not df.index.is_monotonic_increasing:
            raise ValueError("Timestamp index must be strictly increasing.")

        # ----------------------------
        # Handle column selection
        # ----------------------------
        if isinstance(columns, str):
            columns = [columns]

        data_columns = set(df.columns)
        requested_columns = set(columns)

        missing = requested_columns - data_columns
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        df = df[columns]

        # if not used_keys:
        #     raise ValueError("None of the expected data columns are present.")
        #
        # # Keep only usable columns
        # df = df[list(used_keys)]
        # ----------------------------
        # Infer temporal properties (ROBUST)
        # ----------------------------
        diffs = df.index.to_series().diff().dropna()
        if diffs.empty:
            raise ValueError("Cannot infer resolution from data.")

        resolution = diffs.median()  # ← IMPORTANT FIX
        start_time = df.index[0]
        duration = df.index[-1] - df.index[0] + resolution

        # ----------------------------
        # Validate against simulation_df
        # ----------------------------
        if self.simulation_df is None:
            raise ValueError("simulation_df must exist before uploading external data.")

        sim_index = self.simulation_df.index
        sim_diffs = sim_index.to_series().diff().dropna()

        sim_resolution = sim_diffs.median()
        sim_start = sim_index[0]
        sim_duration = sim_index[-1] - sim_index[0] + sim_resolution

        if resolution != sim_resolution:
            raise ValueError(f"Resolution mismatch: {resolution} vs {sim_resolution}")
        logger.commandline("✔ Verified resolution match")

        if start_time != sim_start:
            raise ValueError(f"Start time mismatch: {start_time} vs {sim_start}")
        logger.commandline("✔ Verified start time match")

        # ----------------------------
        # Handle duration
        # ----------------------------
        if duration < sim_duration:
            raise ValueError(
                f"Data too short: {duration} < required {sim_duration}"
            )

        elif duration > sim_duration:
            logger.commandline("⚠ Data longer than required → clipping to simulation window")

            # Clip exactly to simulation range
            df = df.loc[sim_start: sim_index[-1]]

        # If equal → nothing to do
        logger.commandline("✔ Duration validated (clipped if necessary)")

        if not df.index.equals(sim_index):
            raise ValueError("Timestamp index mismatch with simulation_df.")
        logger.commandline("✔ Verified timestamp index match")

        # ----------------------------
        # Update simulation_df
        # ----------------------------
        updated_columns = []

        for col in columns:
            self.simulation_df[col] = df[col]
            updated_columns.append(col)

        print(f"✅ Updated columns: {sorted(updated_columns)}")

    def upload_time_series(self, df: pd.DataFrame, timestamp_col: str = "timestamp",
                             agg: str = "mean") -> pd.DataFrame:
        """
        Validate + clip + resample dataframe based on class time settings
        """

        df = df.copy()

        if timestamp_col not in df.columns:
            raise ValueError(f"Missing timestamp column: {timestamp_col}")

        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).set_index(timestamp_col)

        # --- Filter from start_time ---
        df = df.loc[self.start_time:]

        if df.empty:
            raise ValueError("No data available after start_time.")

        # --- Check duration ---
        actual_duration = df.index.max() - df.index.min()

        if actual_duration < self.duration:
            raise ValueError(
                f"Data too short: {actual_duration} < required {self.duration}"
            )

        # --- Clip to exact window ---
        df = df.loc[self.start_time:self.end_time]

        # --- Detect resolution ---
        diffs = df.index.to_series().diff().dropna()
        if diffs.empty:
            raise ValueError("Cannot infer resolution.")

        current_resolution = diffs.mode().iloc[0]

        # --- Resample if needed ---
        if current_resolution != self.resolution:
            rule = f"{int(self.resolution.total_seconds() // 60)}min"

            if agg == "mean":
                df = df.resample(rule).mean()
            elif agg == "sum":
                df = df.resample(rule).sum()
            elif agg == "first":
                df = df.resample(rule).first()
            else:
                raise ValueError("agg must be 'mean', 'sum', or 'first'")

        return df

    def _advance_time(self):
        if self.now_time is None:
            self.now_time = self.start_time
        else:
            self.now_time += self.resolution

        if self.now_time > self.end_time:
            raise StopIteration("Simulation finished")

    def update(self, control: dict) -> dict:

        self._advance_time()
        t = self.now_time

        row = self.simulation_df.loc[t]

        demand_kw = row.get("Demand Electric Power (kW)", 0)
        pv_kw = row.get("PV Electric Power (kW)", 0)

        # ---------------- EV ----------------
        ev_kw = 0.0
        battery_control = control.get('EV', {})
        power_P_set = battery_control.get('Max Power', 0)  # if control is not provided set to None
        if self.EV:
            ev_status = self.EV.step(
                control_power_W=power_P_set,
                timestamp=t
            )
            # logger.commandline(ev_status)
            ev_kw = ev_status.get('EV Electric Power (kW)', 0)
            self.simulation_df.loc[t, "EV Electric Power (kW)"] = ev_kw
            set_ev_kw = ev_status.get('EV Set Power (kW)', 0)
            self.simulation_df.loc[t, "EV Set Power (kW)"] = set_ev_kw
            self.simulation_df.loc[t, "EV SOC (-)"] = ev_status.get('EV SOC (-)', 0)
            self.simulation_df.loc[t, "EV Parked"] = ev_status.get('EV Parked', 0)
            self.simulation_df.loc[t, "User Plug Out Time"] = ev_status.get('User Plug Out Time')
            self.simulation_df.loc[t, "Expected SOC"] = ev_status.get('Expected SOC')

        # ---------------- Thermal ----------------
        thermal_kw = 0
        if self.Thermal:
            HVAC_control = control.get('HVAC Heating', {})
            power_p_set = HVAC_control.get('P Setpoint', 0)
            thermal_kw = abs(power_p_set / 1000)
            outdoor_temp = row.get("Temperature - Outdoor (C)", None)  # only used for the Thermal System
            indoor_temp = self.Thermal.update(
                power_W=control.get("thermal_kw", power_p_set),
                external_temperature=outdoor_temp,
            )
            self.simulation_df.loc[t, "HVAC Heating Electric Power (kW)"] = thermal_kw
            self.simulation_df.loc[t, "Temperature - Indoor (C)"] = indoor_temp

        # ---------------- Battery ----------------
        battery_kw = 0.0

        if self.Battery:
            battery_control = control.get('Battery', {})
            power_P_set = battery_control.get('P Setpoint', 0)
            # # Checking control
            # logger.commandline(battery_control, power_P_set)
            battery_status = self.Battery.update(power_setpoint_W=power_P_set, timestamp=self.now_time)
            battery_kw = battery_status.get('Battery Electric Power (kW)')
            battery_set_kw = battery_status.get('Battery Set Power (kW)')
            battery_soc = battery_status.get('Battery SOC (-)')
            # # checking status
            # logger.commandline(battery_status)
            self.simulation_df.loc[t, "Battery Electric Power (kW)"] = battery_kw
            self.simulation_df.loc[t, "Battery Set Power (kW)"] = battery_set_kw
            self.simulation_df.loc[t, "Battery SOC (-)"] = battery_soc

        # ---------------- Power balance ----------------
        total_load_kw = demand_kw + ev_kw + thermal_kw
        total_generation_kw = pv_kw - battery_kw
        net_active_kw = total_load_kw - total_generation_kw
        # Net total active power
        self.simulation_df.loc[t, "Total Electric Power (kW)"] = round(net_active_kw, 3)
        # Net total reactive power
        self.simulation_df.loc[t, "Total Reactive Power (kVAR)"] = round(net_active_kw,
                                                                         3)  #for now it same *by powerfactor

        # ✅ RETURN THE UPDATED ROW
        row_return = self.simulation_df.reset_index().loc[self.simulation_df.index.get_loc(t)].to_dict()

        return row_return

    def step(self, control_signal: dict) -> tuple[InverterModel, MeterModel, EVModel, HVACModel, dict]:

        if control_signal is None:
            control_signal = {}
        try:
            house_status = self.update(control=control_signal)
        except StopIteration:
            raise SimulationEnded('Simulation Ended-> add days to sim')
        # print(house_status)
        time_value = house_status['Time']

        self.InverterModel.time = time_value
        self.InverterModel.pv_power = float(house_status.get(INVERTER[0], 0))
        self.InverterModel.battery_power = float(house_status.get(INVERTER[1], 0))
        self.InverterModel.battery_soc = float(house_status.get(INVERTER[2], 0))
        # print(self.InverterModel.model_dump())
        tariff, feed_tariff = self.tariff.get_tariff(time_value.time())

        self.MeterModel.time = time_value
        self.MeterModel.active_power = float(house_status.get(NET_POWER[0], 0))
        self.MeterModel.reactive_power = float(house_status.get(NET_POWER[1], 0))
        self.MeterModel.tariff = float(tariff)
        self.MeterModel.feed_tariff = float(feed_tariff)
        self.MeterModel.tariff_24hrs, self.MeterModel.feed_tariff_24hrs = (self.tariff.get_tariff_range_df
                                                                           (now_time=time_value, period=24,
                                                                            resolution=dt.timedelta(minutes=60)))
        # print(self.MeterModel.model_dump())
        self.MeterModel.add_period(self.resolution.total_seconds() / 60)

        self.EVModel.time = time_value
        self.EVModel.ev_status = bool(house_status.get(EV[0], 0))
        if self.EVModel.ev_status:
            self.EVModel.ev_soc = float(house_status.get(EV[1], 0))
            self.EVModel.user_ev_dc_time = house_status.get(EV[3], 0)
            self.EVModel.expected_soc = int(house_status.get(EV[4], 0))

        else:
            self.EVModel.ev_soc = float(0)
        self.EVModel.ev_power = float(house_status.get(EV[2], 0))
        # print(self.EVModel.model_dump())



        # Need to add user connect and disconnect time by smart system


        self.HVACModel.time = time_value
        self.HVACModel.ti = float(house_status.get(HVAC[0], 0))
        self.HVACModel.hvac_power = float(house_status.get(HVAC[1], 0))
        # print(self.HVACModel.model_dump())

        # print(f'[TIME]: {house_status["Time"]} ')
        return (self.InverterModel,
                self.MeterModel,
                self.EVModel,
                self.HVACModel,
                house_status)
