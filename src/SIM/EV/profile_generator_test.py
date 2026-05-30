from datetime import datetime, timedelta
from ev_profile_generator import generate_ev_sessions

df_ev = generate_ev_sessions(
    csv_path="pdf_Veh1_Level0.csv",
    start_time=datetime(2018, 1, 1),
    resolution=timedelta(minutes=15),
    duration=timedelta(days=7),
    seed=42,
)
datetime(2018, 1, 1)
print(df_ev)
df_ev.to_csv('test_data.csv')


