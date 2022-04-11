# -*-coding:utf-8-*-
import os
import logging
from logging.handlers import RotatingFileHandler

_loggers = {}

# init logger here


class Logger(object):

    def __init__(self, log_type, log_file='log/service.log', log_level=logging.INFO):
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.logger = logging.getLogger(log_type)
        if log_type == "common":
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        elif log_type == "biz":
            formatter = logging.Formatter(
                "%(asctime)s|%(name)s|%(levelname)s|%(message)s", "%Y-%m-%d %H:%M:%S")
        elif log_type == "graylog":
            formatter = logging.Formatter(
                "%(asctime)s - %(message)s")
        else:
            raise Exception("invalid log_type")

        rt = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=10)
        rt.setLevel(log_level)
        rt.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setLevel(log_level)
        sh.setFormatter(formatter)

        self.logger.setLevel(log_level)
        self.logger.addHandler(rt)
        # self.logger.addHandler(sh)


def init_logger(log_type, log_path):
    global _loggers
    assert log_type in ["common", "biz", "graylog"]
    _loggers[log_type] = Logger(log_type, log_file=log_path).logger
    return _loggers[log_type]


def get_logger(log_type):
    return _loggers[log_type]
