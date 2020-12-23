import threading
import pycuda.driver as cuda

from models.face_detector import FaceDetector
from models.face_identifier import FaceIdentifier

import queue
import time

class IdentifierThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cuda_ctx = None  # to be created when run
        self.trt_model = None   # to be created when run

        self.input = None

        self.eventStop = threading.Event()
        self.input_queue = queue.Queue(10)
        self.output_queue0 = queue.Queue(10)
        self.output_queue1 = queue.Queue(10)

    def run(self):

        print('TrtThread: loading the TRT model...')
        cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_model = FaceIdentifier()
        print('TrtThread: start running...')
        while not self.eventStop.is_set():

            obj = self.input_queue.get()
            stt, crop, track_id = obj['stt'], obj['crop'], obj['id']
            label, dist = self.trt_model(crop)

            if stt == 0:
                self.output_queue0.put({'stt': stt, 'id': track_id, 'label': label, 'dist': dist})
            else:
                self.output_queue1.put({'stt': stt, 'id': track_id, 'label': label, 'dist': dist})

        del self.trt_model
        cuda_ctx.pop()
        del cuda_ctx
        print('TrtThread: stopped...')

    def stop(self):
        self.eventStop.set()
        time.sleep(1)
        self.join()