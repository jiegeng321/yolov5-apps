import json
import time
import os
from io import BytesIO
from PIL import Image
import falcon
import base64
import numpy as np

from src.utils.logo_service_errors import ImageFormatUnspportedError
from src.utils.log_util import init_logger
from src.logo_detector_trt import LogoDetector
import logging
from src.utils.log_record import LogRecord
from src.utils.utils import Timer
import model.config as config
from prometheus_client import (
    Counter,
    Histogram,
    CollectorRegistry,
    multiprocess,
    generate_latest,
)

request_counter = Counter(
    "xdcv_http_requests", "reqeust num counter", ["method", "endpoint"]
)
status_counter = Counter(
    "xdcv_http_response_stat", "request status counter", ["endpoint", "status"]
)
histogram = Histogram("xdcv_request_latency_seconds", "request latency histogram")

def count_request(req, resp, resource, params):
    request_counter.labels(req.method, req.path).inc()
def count_response_status(req, resp, resource):
    status_counter.labels(req.path, resp.status).inc()

APPNAME = os.environ.get("APPNAME", "ai-brand-logo")

# 按容器平台需要，设置应用的日志输出路径，请勿随意修改
log_path = "/home/admin/output/%s/logs/common.log" % (APPNAME)
graylog_path = "/home/admin/output/%s/logs/gray.log" % (APPNAME)
logger = init_logger("common", log_path)
gray_logger = init_logger("graylog", graylog_path)


logo_detector = LogoDetector(logger)
logo_detector.warmup()


def imread_binary(img_buffer):
    img = Image.open(BytesIO(img_buffer))
    if img.format == "GIF":
        raise ImageFormatUnspportedError("GIF")
    else:
        return img


class HealthCheck(object):
    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.body = "ok"

    def on_head(self, req, resp):
        resp.status = falcon.HTTP_200


class BrandRec(object):
    def on_head(self, req, resp):
        resp.status = falcon.HTTP_200

    @falcon.before(count_request)
    @falcon.after(count_response_status)
    @histogram.time()
    def on_post(self, req, resp):
        post_start = time.time()
        res = None
        log_recorder = LogRecord()
        image_id = None
        image_binary = None
        try:
            if falcon.MEDIA_JSON in req.content_type:
                data = req.get_media()
                image_id = data.get("imageId")
                image_base64 = data.get("img")
                image_binary = base64.b64decode(image_base64)
            elif falcon.MEDIA_MULTIPART in req.content_type:
                form = req.get_media()
                for p in form:
                    if p.name == "imageId":
                        image_id = p.text
                    if p.name == "img":
                        image_binary = p.stream.read()
            else:
                raise Exception("unsupported content type.")

            if image_id is None or image_binary is None:
                raise Exception("parameter invalid!")

        except Exception as e:
            resp.status = falcon.HTTP_400
            resp.media = {
                "err_msg": str(e)
                + " please refer to http://wiki.tongdun.me/pages/viewpage.action?pageId=39305668"
            }
            log_recorder.record_error_info("err_msg", str(e))
            return

        data_done = time.time()
        download_time = (data_done - post_start) * 1000
        log_recorder.record_profile_info("download", download_time)
        log_recorder.record_image_info("image_id", image_id)
        #print(len(image_binary))
        # 统计解码时间
        with Timer() as decode_timer:
            image = imread_binary(image_binary)
        log_recorder.record_profile_info("decode", decode_timer.elapse)
        log_recorder.record_image_info("height", image.size[0])
        log_recorder.record_image_info("width", image.size[1])

        try:
            with Timer() as algo_timer:
                res = logo_detector.detect(image, log_recorder)
            total_time = download_time + decode_timer.elapse + algo_timer.elapse
            log_recorder.record_profile_info("total", total_time)
        except Exception as e:
            resp.status = falcon.HTTP_500
            resp.media = {"err_msg": str(e)}
            log_recorder.record_error_info("err_msg", str(e))
            return
        else:
            res.update({"alg_cost": f"{algo_timer.elapse:.2f}ms"})
            resp.media = res
            resp.status = falcon.HTTP_200
            log_recorder.record_result_info(res)
        finally:
            gray_logger.info(json.dumps(log_recorder.gather()))
class MetaData(object):
    def on_head(self, req, resp):
        resp.status = falcon.HTTP_200
    def on_get(self, req, resp):
        meta_data = {}
        meta_data["version"] = config.model_version
        meta_data["model_name"] = config.app_name
        meta_data["class_names"] = config.logo_id_to_name
        resp.status = falcon.HTTP_200
        resp.media = meta_data
class MetricsResource(object):
    def on_get(self, req, resp):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
        resp.status = falcon.HTTP_200
        resp.content_type = falcon.MEDIA_TEXT
        resp.text = data
app = falcon.App()
api_service = BrandRec()
validate_service = HealthCheck()
Metrics_Resource = MetricsResource()
meta_data = MetaData()
app.add_route("/logo_rec_base64", api_service)
app.add_route("/v2/logo_brand_rec", api_service)
app.add_route("/ok.htm", validate_service)
app.add_route("/actuator/prometheus", Metrics_Resource)
app.add_route("/get_logodet_metadata", meta_data)
# if __name__ == "__main__":
#     from wsgiref import simple_server
#
#     httpd = simple_server.make_server("0.0.0.0", 8900, app)
#     httpd.serve_forever()
