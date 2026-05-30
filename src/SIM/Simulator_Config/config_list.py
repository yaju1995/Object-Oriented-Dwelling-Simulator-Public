BATTERY_CAPACITY_WH = 10000
C_RATING = 0.5
battery_config = {
    "capacity Wh": BATTERY_CAPACITY_WH,
    "initial soc": 100,
    "charging power W": BATTERY_CAPACITY_WH * C_RATING,
    "discharging power W": BATTERY_CAPACITY_WH * C_RATING,
    "charging eff": 1,
    "discharging eff": 1,
}

ev_config = {
    "capacity Wh": 33_200,  # 60 kWh
    "charging power W": 7_700,  # 7 kW charger
    "discharging power W": 6_000,  # V2G capable
    "charging eff": 1,
    "discharging eff": 1,
    "v2g_enabled": False,
    "profile_file": "./SRC/SIM/Defaults/EV/pdf_Veh1_Level1.csv"
}

thermal_config = {
    'initial temperature C': 10,
    'tau': 30 * 3600,  # need to get reference for these values
    'W': 200,  # need to get reference for these values
    'n': 0.95,  # need to get reference for these values
    'heating_power': 8_000, # 8kw power
    'efficiency':0
}

# model type with list
# file path to the data for the model
demand_config = {
    'model': 'normal',
    'file': './SRC/SIM/Defaults/Demand/15_min_normal_test.csv'
}

# Weather file is the just Path ot name of the CSV with .epw
weather_file = './SRC/SIM/Defaults/Weather/IRL_Dublin.039690_IWEC.epw'
# weather_file = './SRC/SIM/Defaults/Weather/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3.epw'


pv_config = {
    "type":'test',
    "capacity W": 5000,  # only used
    "efficiency": 0.95,  # only used
    "area per W": 1,  # pending
    "tilt": 20,  # pending
    "azimuth": 0,  # pending
}

'''
Manual handle vs Automatic Handle
Manual: Better as it allow external handling with in the simulator
type: TOU only use 1 type of tariff
    : Dynamic has next 24 hrs tariff impact seen when getting next 24hrs tariff
Update of tariff handled externally.
'''
tariff_config = {
    "type": 'TOU',  # TOU, Dynamic, Result use of next day 24hr generation,
    "Resolution": 60,  # minutes
    'seed' : 0 # only needed when you hand random point in the generator
}
