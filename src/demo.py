import cv2
import os
from models.trt_thread import TrtThread
import time
import threading
from utils.camera import Camera
from models.face_detector import FaceDetector
import copy
import pycuda.driver as cuda

font = cv2.FONT_HERSHEY_SIMPLEX

e1 = threading.Event()
e2 = threading.Event()
trtThread = TrtThread(e1, e2)

def draw_box(img, dets, fps = 0):
    if len(dets) > 0:
        for det in dets:
            boxes, score = det[:4], det[4]
            cv2.rectangle(img, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 2)
    # if landmarks:
    #     for lm in lms:
    #         for i in range(0, 5):
    #             cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
    cv2.putText(img, "FPS: " + str(fps), (10,10), font, 0.6, (0,0,255), 2, cv2.LINE_AA)
    return img


def test_image_tensorrt():
    global trtThread
    frame = cv2.imread('/home/tuan/face_sample1.jpg')
    frame = cv2.resize(frame, (320,256))

    # cuda_ctx = cuda.Device(0).make_context()  # GPU 0
    # trt_model = FaceDetector(landmarks=False)
    # while True:
    #     t = time.time()
    #     img = frame.copy()
    #     h, w = img.shape[:2]
        
    #     dets = trt_model(img, h, w)

    #     interval = time.time() - t
    #     print("time: ", interval)
    #     fps = int(1.0/interval)
    #     img2 = frame.copy()
    #     img2 = draw_box(img2, dets, fps)
    #     cv2.imshow('out', img2)
    #     k = cv2.waitKey(20)
    #     if k & 0xFF == 27:
    #         trtThread.stop()
    #         break

    while True:
        trtThread.img = frame.copy()
        trtThread.eventStart.set()

        trtThread.eventEnd.wait()
        trtThread.eventEnd.clear()

        interval = time.time() - t
        print("Total time: ", interval)
        dets = copy.deepcopy(trtThread.dets)
        
        fps = int(1.0/interval)
        img2 = frame.copy()
        img2 = draw_box(img2, dets, fps)
        cv2.imshow('out', img2)
        k = cv2.waitKey(25)
        # break
        # if k & 0xFF == 27:
        #     trtThread.stop()
        #     break


if __name__ == '__main__':
    trtThread.start()
    test_image_tensorrt()
