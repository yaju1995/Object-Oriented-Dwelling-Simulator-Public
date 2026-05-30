from datetime import datetime, timedelta
from Controller.ControlLib import controller
from SIM.EquipmentClass import InverterModel, EVModel, MeterModel, HVACModel
from support.lib_config import CustomLogger

logger = CustomLogger(command=False, color='green')


class hvacController(controller):
    def __init__(self, resolution, global_database, update_period):
        super().__init__(resolution, global_database, update_period)
        self.set_HVAC_Power = None
        self.HVAC_ON = False
        self.temp_ref = 22.5
        self.temp_deviation = 2

    def update_status(self, hvac_info: HVACModel):
        now_time = hvac_info.time
        next_time = now_time + self.resolution
        do_update = (next_time.minute % (self.update_period.total_seconds() // 60) == 0)

        t_upper = self.temp_ref + self.temp_deviation
        t_lower = self.temp_ref - self.temp_deviation
        if do_update:

            # HVAC control ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if hvac_info.ti <= t_lower:
                logger.commandline(f'{next_time}: Turning on Heating at temp:{hvac_info.ti}')
                self.HVAC_ON = True
            elif hvac_info.ti >= t_upper:
                logger.commandline(f'{next_time}: Turning off Heating at temp:{hvac_info.ti}')
                self.HVAC_ON = False

            if self.HVAC_ON:
                self.set_HVAC_Power = -8
            else:
                self.set_HVAC_Power = None

            return self.set_HVAC_Power
