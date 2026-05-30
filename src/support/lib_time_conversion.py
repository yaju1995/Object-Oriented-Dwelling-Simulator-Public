
def FloatToTime(float_val):
    min_convert = 0
    sec_convert = 0

    hour_convert = int(float_val)
    time_float = round(float_val, 2)
    decimal_part = (time_float % 1) * 100
    if decimal_part != 0:
        min_convert = int(decimal_part * 60 / 100)
        time_float = round(decimal_part * 60 / 100, 2)
        decimal_part = (time_float % 1) * 100
        if decimal_part != 0:
            sec_convert = int(decimal_part * 60 / 100)

    return hour_convert, min_convert, sec_convert
