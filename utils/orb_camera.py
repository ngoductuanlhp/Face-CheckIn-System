
from primesense import openni2
from primesense import _openni2 as c_api
import numpy as np
import cv2
import threading
import time

OPENNI2_PATH = './utils'

class OrbCamera:
    def __init__(self):
        self.rgb_image = None
        self.lock_rgb = threading.Lock()
        openni2.initialize(OPENNI2_PATH)
        self.dev = openni2.Device.open_any()

        self.rgb_stream = self.dev.create_color_stream()
        self.rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))
        self.rgb_stream.start()

        self.read_thread = threading.Thread(name="read_thread", target=self.getRGB)
        self.read_thread.daemon = True

    def start(self):
        
        self.read_thread.start()

    def getRGB(self):
        while True:
            try:
                bgr             = np.fromstring(self.rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(480,640,3)
                rgb             = cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB), (480,360))
                self.lock_rgb.acquire()
                self.rgb_image   = cv2.flip(rgb, 1)
                self.lock_rgb.release()
            except:
                self.rgb_image   = None

    def read(self):
        if self.rgb_image is None:
            return None
        self.lock_rgb.acquire()
        img = self.rgb_image.copy()
        self.lock_rgb.release()
        return img
        
    def shutdown(self):
        self.rgb_stream.stop()
        openni2.unload()