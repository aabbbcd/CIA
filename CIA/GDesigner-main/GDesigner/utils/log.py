import logging
import os
import sys
from datetime import datetime
from typing import Optional

class LoggerUtils:

    
    def __init__(self, name: str = "GDesigner", level: int = logging.INFO, 
                 log_file: Optional[str] = None, console_output: bool = True):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if self.logger.handlers:
            return
            

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        

        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        

        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
    
    def exception(self, message: str):
        self.logger.exception(message)


def get_logger(name: str = "GDesigner", level: int = logging.INFO, 
               log_file: Optional[str] = None, console_output: bool = True) -> LoggerUtils:

    return LoggerUtils(name, level, log_file, console_output)


logger = get_logger()


def debug(message: str):
    logger.debug(message)

def info(message: str):
    logger.info(message)

def warning(message: str):
    logger.warning(message)

def error(message: str):
    logger.error(message)

def critical(message: str):
    logger.critical(message)

def exception(message: str):
    logger.exception(message)


def set_level(level: int):
    logger.logger.setLevel(level)

def set_log_file(log_file: str):

    for handler in logger.logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.logger.removeHandler(handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logger.logger.level)
        file_handler.setFormatter(formatter)
        logger.logger.addHandler(file_handler)


    
