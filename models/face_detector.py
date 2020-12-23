import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time

TRT_PATH = '/home/tuan/FRCheckInWeights/face_detector_320_192.trt'

class FaceDetector(object):
    def _load_plugins(self):
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self, model_path):
        with open(model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            # print("szie:", size)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append({ 'host': host_mem, 'device': device_mem })
            else:
                outputs.append({ 'host': host_mem, 'device': device_mem })
    
        return inputs, outputs, bindings

    def __init__(self, landmarks=False, batch=1):
        self.landmarks = landmarks
        self.threshold = 0.4
        self.batch_size = batch
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 192, 320, 1, 1
        self.striped_h, self.striped_w =  int(self.img_h_new / 4), int(self.img_w_new / 4)
        # self.shape_of_output = [(1, 1, int(self.img_h_new / 4), int(self.img_w_new / 4)),
        #                    (1, 2, int(self.img_h_new / 4), int(self.img_w_new / 4)),
        #                    (1, 2, int(self.img_h_new / 4), int(self.img_w_new / 4)),
        #                    (1, 10, int(self.img_h_new / 4), int(self.img_w_new / 4))]
        self.shape_of_output = [(self.striped_h * self.striped_w),
                                (2, self.striped_h * self.striped_w),
                                (2, self.striped_h * self.striped_w),
                                (10, self.striped_h * self.striped_w)]

        self.trt_logger = trt.Logger(trt.Logger.INFO)

        try:
            self._load_plugins()
            self.engine = self._load_engine(TRT_PATH)
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.inputs, self.outputs, self.bindings = self._allocate_buffers()
            print("[FaceDetector] Model loaded")
            dummy_inp = np.random.normal(loc=100, scale=50, size=(self.img_h_new, self.img_w_new, 3)).astype(np.uint8)
            self.inference_tensorrt(dummy_inp)
        except Exception as e:
            raise RuntimeError('Fail to allocate CUDA resources in FaceDetector') from e

        

    def __call__(self, img):
        height, width = img.shape[:2]
        self.scale_h, self.scale_w = self.img_h_new / height, self.img_w_new / width
        # self.scale_h, self.scale_w 
        return self.inference_tensorrt(img)

    def preprocess(self, img):
        img = cv2.resize(img, (self.img_w_new, self.img_h_new))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2,0,1).astype(np.float32)
        img = img.reshape(-1)
        return img
    
    
    def inference_tensorrt(self, img):
        # t = time.time()
        self.inputs[0]['host'] = self.preprocess(img)

        # start = time.time()
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        
        self.context.execute_async(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream

        self.stream.synchronize()
        
        # end = time.time()
        # print('Inference time:', end-start)
        trt_outputs = [out['host'] for out in self.outputs]
        
        heatmap, scale, offset, lms = [output.reshape(shape) for output, shape in zip(trt_outputs, self.shape_of_output)]
        # print("Infer time: ", time.time() - t)
        return self.postprocess(heatmap, lms, offset, scale)

    def postprocess(self, heatmap, lms, offset, scale):
        if self.landmarks:
            dets, lms = self.decode(heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new))
        else:
            # self.decode(heatmap, scale, offset, None, (self.img_h_new, self.img_w_new))
            dets = self.decode(heatmap, scale, offset, None, (self.img_h_new, self.img_w_new))
            
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / self.scale_w, dets[:, 1:4:2] / self.scale_h
            if self.landmarks:
                lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2] / self.scale_w, lms[:, 1:10:2] / self.scale_h
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            if self.landmarks:
                lms = np.empty(shape=[0, 10], dtype=np.float32)
        if self.landmarks:
            return dets, lms
        else:
            return dets

    def decode(self, heatmap, scale, offset, landmark, size):
        boxes = []
        # lm = []
        inds = np.where(heatmap > self.threshold)[0]
        c0, c1 = np.unravel_index(inds, (self.striped_h, self.striped_w))
        if c0.shape[0] > 0:
            sc = np.take(scale, inds, 1)
            
            sc = np.exp(sc) * 4
            o = np.take(offset, inds, 1)
            score = np.take(heatmap, inds, 0)

            x1 = (c1 +o[1] + 0.5) * 4 - sc[1] / 2
            y1 = (c0 +o[0] + 0.5) * 4 - sc[0] / 2
            x1 = np.clip(x1, 0, self.img_w_new)
            y1 = np.clip(y1, 0, self.img_h_new)

            # lm = np.take(landmark, inds, 1)
            # print("shape1: ", lm.shape)
            # lm[::2, :] = lm[::2, :]*sc[1] + x1
            # lm[1::2, :] = lm[1::2, :]*sc[0] + y1
            

            x2 = x1 + sc[1]
            y2 = y1 + sc[0]
            x2 = np.clip(x2, 0, self.img_w_new)
            y2 = np.clip(y2, 0, self.img_h_new)
            boxes = np.stack((x1,y1,x2,y2, score), axis=-1)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]

            # lm = np.asarray(lm, dtype=np.float32).transpose()
            # lm = lm[keep, :]
            # print("shape3: ", lm.shape)
        return boxes
        # return boxes, lm


    def nms(self, dets, scores, thresh):
        '''
        dets is a numpy array : num_dets, 4
        scores ia  nump array : num_dets,
        '''
        # t = time.time()
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1] # get boxes with more ious first

        keep = []
        while order.size > 0:
            i = order[0] # pick maxmum iou box
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
            h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        # print("NMS score: ", time.time() - t)
        return keep