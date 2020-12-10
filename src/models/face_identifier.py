import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time
import os
import pickle

TRT_PATH = '/home/tuan/FRCheckInWeights/face_identifier.trt'
EMBED_PATH = '/home/tuan/FRCheckInWeights/test_db.npy'

NAME_FACE = '/home/tuan/FRCheckInWeights/name_face'
EMBEDDED_FACE = '/home/tuan/FRCheckInWeights/embedded_face'



class FaceIdentifier(object):
    def _load_plugins(self):
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self, model_path):
        with open(model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
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
                inputs = { 'host': host_mem, 'device': device_mem }
            else:
                outputs = { 'host': host_mem, 'device': device_mem }
    
        return inputs, outputs, bindings

    def _load_data(self):
        if not os.path.exists(NAME_FACE) or not os.path.exists(EMBEDDED_FACE):
            print("There is no data to load")

            return None, None
        with open (NAME_FACE, 'rb') as fp_1:
            face_IDs = pickle.load(fp_1)

        with open (EMBEDDED_FACE, 'rb') as fp_2:
            face_encodings = pickle.load(fp_2)
            face_encodings = np.stack(face_encodings, axis = 0)
            num_embed = face_encodings.shape[0]
            face_encodings = np.reshape(face_encodings, (num_embed, -1))

        return face_IDs, face_encodings

        

    def __init__(self):
        self.img_size = 224
        self.mean = np.stack((np.ones((224,224))*0.6068*255, np.ones((224,224))*0.4517*255, np.ones((224,224))*0.38*255))
        self.std = np.stack((np.ones((224,224))*0.2492*255, np.ones((224,224))*0.2173*255, np.ones((224,224))*0.2082*255))
        
        # self.db = torch.load(DB_PATH)
        # self.label = ['Hieu', 'Khuong']
        # self.embed = np.load(EMBED_PATH)

        self.face_id, self.face_embed = self._load_data()
        # print(self.face_id, self.face_embed)
        # print(len(self.face_id))
        # print(self.face_embed.shape)
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        try:
            self._load_plugins()
            self.engine = self._load_engine(TRT_PATH)
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.inputs, self.outputs, self.bindings = self._allocate_buffers()
            print("[FaceIdentifier] Model loaded")
            dummy_inp = np.random.normal(loc=100, scale=50, size=(self.img_size, self.img_size, 3)).astype(np.uint8)
            self.inference_tensorrt(dummy_inp)
        except Exception as e:
            raise RuntimeError('Fail to allocate CUDA resources in FaceIdentifier') from e

    def kNearest(self, inp, k=1):
        dists = np.sqrt(np.sum((inp - self.face_embed) ** 2, axis=1))
        idx = np.argmin(dists)
        return idx, dists[idx]

    def __call__(self, img):
        # results = []
        # if isinstance(imgs, list):
        #     for img in imgs:
        #         embed = self.inference_tensorrt(imgs)[0]
        #         results.append(self.inference_tensorrt(img)[0])
        # else:
        embed = self.inference_tensorrt(img)
        idx, dist = self.kNearest(embed)
        if dist < 0.65:
            result = self.face_id[idx]
        else:
            result = 'Unknown'
        return result, dist

    def preprocess(self, img):
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2,0,1).astype(np.float32)
        img = (img - self.mean) / self.std
        return np.array(img, dtype=np.float32, order='C')
    
    
    def inference_tensorrt(self, img):
        # t = time.time()
        self.inputs['host'] = self.preprocess(img)
        cuda.memcpy_htod_async(self.inputs['device'], self.inputs['host'], self.stream)
        # run inference
        
        self.context.execute_async(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        
        cuda.memcpy_dtoh_async(self.outputs['host'], self.outputs['device'], self.stream)
        # synchronize stream

        self.stream.synchronize()
        
        final_output = self.outputs['host'].reshape(1,256).astype(np.float32)
        
        return final_output


if __name__ == '__main__':
    fi = FaceIdentifier()
