from collections import OrderedDict


class LogRecord(object):
    def __init__(self):
        self.image_info = {}
        self.profile_info = {}
        self.result_info = {}
        self.error_msg = {}

    def record_image_info(self, key, value):
        self.image_info.update({key: value})

    def record_profile_info(self, key, value):
        self.profile_info.update({key: f'{value:.4f}ms'})

    def record_result_info(self, key, value):
        self.result_info.update({key: value})

    def record_result_info(self, info):
        self.result_info.update(info)

    def record_error_info(self, key, value):
        self.error_msg.update({key: value})

    def gather(self):
        altogether = {}
        altogether['image_info'] = self.image_info
        altogether['profile_info'] = self.profile_info
        altogether['result_info'] = self.result_info
        altogether['error_msg'] = self.error_msg
        return altogether
