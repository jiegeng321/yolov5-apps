# encoding=utf-8
import numpy as np
from PIL import Image, ImageFile
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import model.config as config
import time,os
import ctypes
from src.utils.log_record import LogRecord
ImageFile.LOAD_TRUNCATED_IMAGES = True
libpath = os.path.join(os.path.dirname(__file__), config.so_name)
ctypes.CDLL(libpath)
NEW_IMG = Image.new("RGB", (config.max_size, config.max_size), color=(114, 114, 114))
def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2-x1+1)*(y2-y1+1)
    orders = scores.argsort()[::-1]

    keep = []
    while orders.size > 0:
        i = orders[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[orders[1:]])
        yy1 = np.maximum(y1[i], y1[orders[1:]])
        xx2 = np.minimum(x2[i], x2[orders[1:]])
        yy2 = np.minimum(y2[i], y2[orders[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w*h
        iou = inter / (areas[i] + areas[orders[1:]] - inter)

        inds = np.where(iou <= thresh)[0]
        orders = orders[inds + 1]

    return keep


def pad_image(im, max_size=config.max_size):
    try:
        width, height = im.size
        new_img_start = time.time()
        #new_im = Image.new("RGB", (max_size, max_size), color=(114, 114, 114))
        new_im = NEW_IMG
        new_img_cost = 1000*(time.time() - new_img_start)
        if width > height:
            new_width = max_size
            new_height = int(new_width*height/width)
            top_x = 0
            top_y = (new_width - new_height)//2
        else:
            new_height = max_size
            new_width = int(new_height*width/height)
            top_x = (new_height - new_width)//2
            top_y = 0
        resize_start = time.time()
        #print(np.mean(im))
        im_resized = im.resize(
            (new_width, new_height), resample=Image.BILINEAR)
        #print(np.mean(im_resized))
        resize_cost = 1000*(time.time() - resize_start)
        paste_start = time.time()
        new_im.paste(im_resized, (top_x, top_y))
        paste_cost = 1000*(time.time() - paste_start)
        return new_im, new_img_cost, resize_cost, paste_cost
    except Exception as e:
        print(e.message)
        return None

def project_coor_back(bboxes, original_shape, input_size=config.max_size):
    ratio = min(input_size/original_shape[0], input_size/original_shape[1])
    nw, nh = int(ratio*original_shape[0]), int(ratio*original_shape[1])
    pw, ph = input_size - nw, input_size - nh
    bboxes[:, [0, 2]] -= pw/2
    bboxes[:, [1, 3]] -= ph/2
    bboxes /= ratio
    return bboxes


class LogoDetector(object):
    def __init__(self, logger, model_name=config.model_name):
        # load tensorrt engine
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        TRTbin = '{0}'.format(model_name)
        print ("TRTbin",TRTbin)
        with open(TRTbin, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()
        stream = cuda.Stream()

        # allocate memory
        inputs, outputs, bindings = [], [], []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem })
            else:
                outputs.append({'host': host_mem, 'device': device_mem })
        # save to class
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream
        self.logger = logger

    def inference(self, img,log_recorder):
        #self.inputs[0]['host'] = np.ravel(img)
        np.copyto(self.inputs[0]['host'], np.ravel(img))
        copy_cpu2gpu_start = time.time()
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        copy_cpu2gpu_cost = 1000*(time.time() - copy_cpu2gpu_start)
        log_recorder.record_profile_info('copy_cpu2gpu', copy_cpu2gpu_cost)
        inference_start = time.time()
        self.context.execute_async(
                bindings=self.bindings,
                stream_handle=self.stream.handle)
        inference_cost = 1000*(time.time() - inference_start)
        log_recorder.record_profile_info('inference', inference_cost)
        copy_gpu2cpu_start = time.time()
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        copy_gpu2cpu_cost = 1000*(time.time() - copy_gpu2cpu_start)
        log_recorder.record_profile_info('copy_gpu2cpu', copy_gpu2cpu_cost)
        synchronize_start = time.time()
        self.stream.synchronize()
        synchronize_cost = 1000*(time.time() - synchronize_start)
        log_recorder.record_profile_info('synchronize', synchronize_cost)
        output = self.outputs[0]['host']
        return output

    def postprocess(self, output, original_shape):
        num = int(output[0])
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        boxes = pred[:, :4]
        scores = pred[:, 4]
        class_ids = pred[:, 5]

        si = scores > config.conf_thresh
        boxes = boxes[si, :]
        scores = scores[si]
        class_ids = class_ids[si]
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        indices = nms(dets, config.nms_thresh)

        boxes = boxes[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]

        boxes = project_coor_back(boxes,original_shape)

        return boxes, scores, class_ids

    def detect(self, image, log_recorder):
        preprocess_start = time.time()
        pad_start = time.time()
        padded_image, new_img_cost, resize_cost, paste_cost = pad_image(image)
        pad_cost = 1000*(time.time() - pad_start)
        log_recorder.record_profile_info('new_img_cost', new_img_cost)
        log_recorder.record_profile_info('resize_cost', resize_cost)
        log_recorder.record_profile_info('paste_cost', paste_cost)
        asarray_start = time.time()
        input_image = np.asarray(padded_image)
        asarray_cost = 1000*(time.time() - asarray_start)
        normal_start = time.time()
        input_im = input_image.transpose(2, 0, 1)
        #input_im = np.ascontiguousarray(input_im)
        normal_cost = 1000*(time.time() - normal_start)
        preprocess_cost = 1000*(time.time() - preprocess_start)
        log_recorder.record_profile_info('pad_cost', pad_cost)
        log_recorder.record_profile_info('asarray_cost', asarray_cost)
        log_recorder.record_profile_info('normal_cost', normal_cost)
        log_recorder.record_profile_info('preprocess', preprocess_cost)

        inference_start = time.time()
        raw_pred = self.inference(input_im,log_recorder)
        inference_cost = 1000*(time.time() - inference_start)
        self.logger.info(
            "inference cost time: {:.2f}ms".format(inference_cost))
        log_recorder.record_profile_info('inference_total', inference_cost)
        postprocess_start = time.time()
        boxes, scores, class_ids = self.postprocess(raw_pred,image.size)
        postprocess_cost = 1000*(time.time() - postprocess_start)
        log_recorder.record_profile_info('postprocess', postprocess_cost)
        logo_list = []
        for i in range(len(boxes)):
            score = float(scores[i])
            logo_name = config.logo_id_to_name[int(class_ids[i])].split("-")
            if logo_name[0] in config.brand_filter:
                if score < config.brand_filter[logo_name[0]]:
                    continue
            if "w" in logo_name and score < config.word_conf_thresh:
                continue
            elif score < config.pic_conf_thresh:
                continue
            logo_object = {}
            logo_object['score'] = score
            logo_object['logo_name'] = logo_name[0]
            box_object = {}
            box_prediction = boxes[i]
            box_object['x1'] = int(box_prediction[0])
            box_object['y1'] = int(box_prediction[1])
            box_object['x2'] = int(box_prediction[2])
            box_object['y2'] = int(box_prediction[3])
            logo_object['box'] = box_object
            logo_list.append(logo_object)
        result = {}
        result['res'] = logo_list
        return result

    def warmup(self):
        warm_log_record = LogRecord()
        self.logger.info('warming up...')
        for i in range(5):
            image = np.random.uniform(0, 1, size=(3, config.max_size, config.max_size)).astype(np.float32)
            res = self.inference(image,warm_log_record)
            self.logger.info('i={}, warm res={}'.format(i, res))
        self.logger.info('finish warm up.')



# if __name__=="__main__":
#     import sys
#     sys.path.append('.')
#     log_path = "./dist/log"
#     from src.utils.log_util import init_logger
#     from log_record import LogRecord
#     logger = init_logger("common", log_path)
#     terror_detector = TerrorDetector(logger)
#     img_file = Image.open("./test/1920_1088_1_gun.jpg")
#     logrecord = LogRecord()
#     res = terror_detector.detect(img_file, logrecord)
#     print(res)

