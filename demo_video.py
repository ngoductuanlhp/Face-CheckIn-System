import cv2
import os
import time
import threading
import pycuda.driver as cuda
import numpy as np
import queue
import copy
import argparse

from modules.identifier_thread import IdentifierThread
from modules.face_detector import FaceDetector

font = cv2.FONT_HERSHEY_SIMPLEX

save_id = 0

stop_flag = False
    
inp_img = None
img1 = None
sync_img1, sync_img2 = False, False
dets1 = []
fps = 30
lock = threading.Lock()

def draw_box(img, dets, fps = 0):
    draw = img.copy()
    if len(dets) > 0:
        for det in dets:
            boxes = [int(i) for i in det[:4]]
            cv2.rectangle(draw, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (2, 255, 0), 2)
    cv2.putText(draw, "FPS: " + str(fps), (15,15), font, 0.6, (0,0,255), 2, cv2.LINE_AA)
    return draw

def readVid(vid_path):
    global inp_img, lock, sync_img2
    cap = cv2.VideoCapture(vid_path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640,360))
        lock.acquire()
        inp_img = frame.copy()
        lock.release()
        sync_img2 = True


def main_process():
    global stop_flag, img1, sync_img1, dets1, fps, inp_img, lock, sync_img2
    
    
    cuda_ctx = cuda.Device(0).make_context()  # GPU 0
    detector = FaceDetector()
    
    while not stop_flag:
        if not sync_img2:
            print("No frame")
            time.sleep(0.001)
            continue
        fps_t = time.time()
        lock.acquire()
        frame = inp_img.copy()
        lock.release()
        
        dets = detector(frame.copy())
        # print(dets)
        interval = time.time() - fps_t
        fps_t = time.time()
        
        fps = int(1.0/interval)
        # debug = draw_box(frame, dets, fps)
        # output = np.hstack((frame, debug))
        # cv2.imshow("Output", output)
        # k = cv2.waitKey(1)
        img1 = frame.copy()
        dets1 = copy.deepcopy(dets)
        sync_img1 = True
        print("Detect: %d faces" % (len(dets)))


def main():
    global stop_flag, img1, sync_img1, dets1, fps

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--vid", required=True,
        help="video name")
    args = vars(ap.parse_args())

    vid_path = os.path.join('./videos/', str(args["vid"]))

    vid_thread = threading.Thread(name="vid_thread", target=readVid, args=[vid_path])
    vid_thread.start()
    time.sleep(0.01)
    in_thread = threading.Thread(name="in_thread", target=main_process)
    in_thread.start()

    
    while True:
        if sync_img1:
            sync_img1 = False
            debug = draw_box(img1, dets1, fps)
            output = np.hstack((img1, debug))
            cv2.imshow("Output", output)
        k = cv2.waitKey(1)
        if k & 0xFF == 27:
            stop_flag = True
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()

    
