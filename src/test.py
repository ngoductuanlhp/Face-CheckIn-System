import cv2
import os
from models.trt_thread import TrtThread
import time
import threading
from utils.camera import Camera
from models.face_identifier import FaceIdentifier
import copy
import pycuda.driver as cuda
import numpy as np


fi = FaceIdentifier()

def infer(img_name):
    img = cv2.imread(img_name)
    embed = fi(img)
    return embed

if __name__ == '__main__':
    e1 = infer('/home/tuan/sample_img/ew1.png')
    e2 = infer('/home/tuan/sample_img/ew3.png')
    e3 = infer('/home/tuan/sample_img/ws1.png')
    e4 = infer('/home/tuan/sample_img/ws2.png')
    e5 = infer('/home/tuan/sample_img/ts2.png')
    e6 = infer('/home/tuan/sample_img/ts1.png')

    # print(e2)
    print(np.linalg.norm(e1 - e2))
    print(np.linalg.norm(e1 - e3))
    print(np.linalg.norm(e1 - e4))
    print(np.linalg.norm(e1 - e5))
    print(np.linalg.norm(e1 - e6))

    print(np.linalg.norm(e3 - e4))
    print(np.linalg.norm(e3 - e6))

    print(np.linalg.norm(e5 - e6))

