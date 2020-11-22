import threading
import pycuda.driver as cuda

from models.face_detector import FaceDetector
from models.face_identifier import FaceIdentifier


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


    def run(self):

        print('TrtThread: loading the TRT model...')
        cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_model = parse_model(self.mod)
        print('TrtThread: start running...')
        while True:
            # with self.conditionStart as cond:
            self.eventStart.wait()
            self.eventStart.clear()
            if self.input is not None:
                self.result = self.trt_model(self.input)
                self.input = None
            self.eventEnd.set()

        del self.trt_model
        cuda_ctx.pop()
        del cuda_ctx
        print('TrtThread: stopped...')

    def stop(self):
        self.join()