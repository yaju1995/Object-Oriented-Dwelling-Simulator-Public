import os
import inspect
import warnings

class CustomLogger:
    COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }

    def __init__(self, command=True, color="red"):
        self.command = command
        self.color_code = self.COLORS.get(color, self.COLORS["red"])  # default red

    def get_file_name_without_extension(self, file_path):
        file_name = os.path.basename(file_path)
        return os.path.splitext(file_name)[0]

    def commandline(self, *args, **kwargs):
        if self.command:
            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            file_name_without_extension = self.get_file_name_without_extension(caller_file)

            reset_code = self.COLORS["reset"]
            print(f"{self.color_code}{file_name_without_extension}:{reset_code}", *args, **kwargs)

    def raise_warning(self, message):
        if self.command:
            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            file_name_without_extension = self.get_file_name_without_extension(caller_file)

            def custom_formatwarning(msg, *args, **kwargs):
                return f"{msg}\n"

            warnings.formatwarning = custom_formatwarning
            warnings.warn(f"{file_name_without_extension}: {message}", UserWarning)

    def raise_error(self, message):
        if self.command:
            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            file_name_without_extension = self.get_file_name_without_extension(caller_file)
            raise Exception(f"{file_name_without_extension}: {message}")