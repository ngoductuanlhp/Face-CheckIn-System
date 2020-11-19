import threading
import pycuda.driver as cuda

from models.face_detector import FaceDetector



class TrtThread(threading.Thread):
    def __init__(self, eventStart, eventEnd):
        threading.Thread.__init__(self)
        
        self.eventStart = eventStart
        self.eventEnd = eventEnd

        self.cuda_ctx = None  # to be created when run
        self.trt_model = None   # to be created when run

        self.img = None

        self.dets = None
        self.landmarks = None




    def run(self):

        print('TrtThread: loading the TRT model...')
        cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_model = FaceDetector(landmarks=False)
        print('TrtThread: start running...')
        while True:
            # with self.conditionStart as cond:
            self.eventStart.wait()
            self.eventStart.clear()
            if self.img is not None:
                h, w = self.img.shape[:2]
                self.dets = self.trt_model(self.img, h, w)
                self.img = None
            self.eventEnd.set()

        del self.trt_model
        cuda_ctx.pop()
        del cuda_ctx
        print('TrtThread: stopped...')

    def stop(self):
        self.join()