from typing import Union
import logging
import sys
import os


class DummyLogger:
    def debug(self, *args):
        pass

    def info(self, *args):
        pass

    def warning(self, *args):
        pass

    def error(self, *args):
        pass

    def critical(self, *args):
        pass


def setup_logger(name: str = 'root', debug: bool = False,
                 console_logging: bool = False, dummy=False) -> Union[logging.Logger, DummyLogger]:
    if dummy:
        return DummyLogger()

    logger = logging.getLogger(name=name)

    # setup handlers
    # write to package log if running in VirtualDrugBuffet directory else write to CWD
    if "VirtualDrugBuffet" in os.getcwd():
        file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), "vdb.log"))
    else:
        file_handler = logging.FileHandler(os.path.join(os.getcwd(), "vdb.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))

    # set levels
    if debug:
        logger.setLevel("DEBUG")
        file_handler.setLevel("DEBUG")
        console_handler.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")
        file_handler.setLevel("INFO")
        console_handler.setLevel("INFO")

    # add handlers to logger
    logger.addHandler(file_handler)
    if console_logging:
        logger.addHandler(console_handler)

    # TODO work on this here
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception

    return logger
