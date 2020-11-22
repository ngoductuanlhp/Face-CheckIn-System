import threading
import pycuda.driver as cuda

from models.face_detector import FaceDetector
from models.face_identifier import FaceIdentifier

import queue
import time


def parse_model(mod):
    if mod == 'detector':
        return FaceDetector()
    elif mod == 'identifier':
        return FaceIdentifier()


class TrtThread(threading.Thread):
    def __init__(self, eventStart, eventEnd, mod = 'detector'):
        threading.Thread.__init__(self)
        
        self.eventStart = eventStart
        self.eventEnd = eventEnd

        self.cuda_ctx = None  # to be created when run
        self.trt_model = None   # to be created when run

        self.input = None

        self.result = None

        self.mod = mod

        self.eventStop = threading.Event()
        self.input_queue = queue.Queue(10)
        self.output_queue = queue.Queue(10)

    def run(self):

        print('TrtThread: loading the TRT model...')
        cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_model = parse_model(self.mod)
        print('TrtThread: start running...')
        while not self.eventStop.is_set():
            # with self.conditionStart as cond:
            # self.eventStart.wait()
            # self.eventStart.clear()

            obj = self.input_queue.get()
            crop, track_id = obj['crop'], obj['id']
            label, dist = self.trt_model(crop)

            self.output_queue.put({'id': track_id, 'label': label, 'dist': dist})
            # self.eventEnd.set()

        del self.trt_model
        cuda_ctx.pop()
        del cuda_ctx
        print('TrtThread: stopped...')

    def stop(self):
        self.eventStop.set()
        time.sleep(1)
        self.join()