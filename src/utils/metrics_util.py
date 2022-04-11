#-*-coding:utf-8-*-
from src.utils.log_util import init_logger, get_logger

class MetricUtil(object):
    __instance = None

    @staticmethod
    def getInstance():
        if MetricUtil.__instance == None:
            raise Exception("MetricUtil has not inited yet!")
        return MetricUtil.__instance

    def __init__(self, path):
        if MetricUtil.__instance != None:
            raise Exception("MetricUtil class is a singleton!")
        self.logger = init_logger("biz", path)
        MetricUtil.__instance = self

    def record(self, api, http_code, cost):
        """Record the api access log.
        Args:
            api: string, api name
            http_code: int, http return code
            cost: float, time cost in millisecond
        Returns:
        Raises:
        """
        if api is None or api == "":
            api = "-"
        if http_code is None:
            http_code = "-"
        if cost is None:
            cost = "-"
        self.logger.info("%s|%s|%s" % (api, http_code, cost))

def init_metrics(path):
    return MetricUtil(path)

def get_metrics():
    return MetricUtil.getInstance()
