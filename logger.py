import inspect
from datetime import datetime

class Logger(object):
    def __init__(self):
        print(f"Logger started at {self._current_time()}")

    def _current_time(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def print(self, message, *args):
        caller_function = inspect.stack()[1].function
        print(f"[{self._current_time()}] - {caller_function} - {message}", *args)
        
