# encoding=utf-8
import numpy as np
from PIL import Image, ImageFile
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import time,os,sys
sys.path.append('..')
import model.config as config
import ctypes
from src.utils.log_record import LogRecord
ImageFile.LOAD_TRUNCATED_IMAGES = True
NEW_IMG = Image.new("RGB", (config.max_size, config.max_size), color=(114, 114, 114))

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
        #print(raw_pred)
        inference_cost = 1000*(time.time() - inference_start)
        self.logger.info(
            "inference cost time: {:.2f}ms".format(inference_cost))
        result = {}
        result['res'] = raw_pred
        return result

    def warmup(self):
        warm_log_record = LogRecord()
        self.logger.info('warming up...')
        for i in range(5):
            image = np.random.uniform(0, 1, size=(3, config.max_size, config.max_size)).astype(np.float32)
            res = self.inference(image,warm_log_record)
            self.logger.info('i={}, warm res={}'.format(i, res))
        self.logger.info('finish warm up.')



if __name__=="__main__":
    import sys
    sys.path.append('.')
    log_path = "./tmp_log.txt"
    from src.utils.log_util import init_logger
    from src.utils.log_record import LogRecord
    logger = init_logger("common", log_path)
    terror_detector = LogoDetector(logger)
    img_file = Image.open("/home/tdops/xu.fx/777bs_1377ks_test_img/checked_ZumbaFitness_ZumbaFitness472.jpg")
    logrecord = LogRecord()
    res = terror_detector.detect(img_file, logrecord)
    print(res)

