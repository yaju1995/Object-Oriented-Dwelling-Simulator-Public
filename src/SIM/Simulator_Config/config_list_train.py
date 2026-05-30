from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BATTERY_CAPACITY_WH = 5000
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
    "capacity Wh": 24_000,  # 60 kWh
    "charging power W": 6_000,  # 7 kW charger
    "discharging power W": 6_000,  # V2G capable
    "charging eff": 1,
    "discharging eff": 1,
    "v2g_enabled": False,
    "profile_file": BASE_DIR / "Defaults/EV/pdf_Veh1_Level1.csv"
}

thermal_config = {
    'initial temperature C': 21,
    'tau': 30 * 3600,  # need to get reference for these values
    'W': 200,  # need to get reference for these values
    'n': 0.95, # need to get reference for these values
    'HVAC electric power Wh': 8_000
}

# model type with list
# file path to the data for the model
demand_config = {
    'model': 'uniform',
    'file': BASE_DIR / 'Defaults/Demand/15_min_uniform_10kva.csv'
}

# Weather file is the just Path ot name of the CSV with .epw
weather_file = BASE_DIR / 'Defaults/Weather/IRL_Dublin.039690_IWEC.epw'
# weather_file = './SRC/SIM/Defaults/Weather/USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3.epw'


pv_config = {
    'type':'train',
    'seed':1,
    "capacity W": 10_000,  # only used
    "efficiency": 1,  # only used
    "area per W": 1,  # pending
    "tilt": 20,  # pending
    "azimuth": 0,  # pending
    "max_irradiance":1000,
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
