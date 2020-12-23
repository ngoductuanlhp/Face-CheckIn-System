import cv2
import os
import time
import threading
import pycuda.driver as cuda
import numpy as np
import queue

from utils.camera import Camera
from models.identifier_thread import IdentifierThread
from models.face_detector import FaceDetector
from models.tracker import Tracker

font = cv2.FONT_HERSHEY_SIMPLEX

save_id = 0

def draw_box_with_lm(img, dets, lms):
    print(lms.shape)
    if len(dets) > 0:
        for det in dets:
            boxes = det[:4]
            cv2.rectangle(img, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 2)
        for lm in lms:
            for i in range(0, 5):
                # cv2.circle(img, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
                cv2.putText(img, str(i+1), (int(lm[i * 2]), int(lm[i * 2 + 1])), font, 0.3, (0,0,255), 2, cv2.LINE_AA)
    return img

def draw_box(img, dets, track_dict, fps = 0):
    global save_id
    # print(dets)
    if len(dets) > 0:
        for det in dets:
            boxes = det[:4]
            
            cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (2, 255, 0), 2)

            det_id = det[4]
            track = track_dict.get(det_id)
            name = track['label']
            cv2.putText(img, "TrackId: " + str(det_id), (boxes[0],boxes[1] - 20), font, 0.6, (255,0,0), 2, cv2.LINE_AA)
            if name != None:
                cv2.putText(img, "Name: " + name, (boxes[0], boxes[1]), font, 0.6, (255,0,0), 2, cv2.LINE_AA)
    # if landmarks:
    #     for lm in lms:
    #         for i in range(0, 5):
    #             cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
    cv2.putText(img, "FPS: " + str(fps), (15,15), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
    # save_path = '/home/tuan/debug/img' + str(save_id) + '.png'
    # cv2.imwrite(save_path, img)
    # save_id += 1
    return img



if __name__ == '__main__':

    cam = Camera()
    cam.start()

    identifierThread = IdentifierThread()
    identifierThread.start()
    time.sleep(0.1)

    cuda_ctx = cuda.Device(0).make_context()  # GPU 0
    detector = FaceDetector()
    detector2 = FaceDetector()
    tracker_arr = [Tracker(0), Tracker(1)]
    # tracker = Tracker()
    
    # vid = cv2.VideoCapture('/home/tuan/test1.avi')
    fps_t = time.time()
    while True:
        count_inp = 0
        count_res = 0
        final_dets = [[None] for i in range(2)]
        # _, frame = vid.read()
        # frame = cv2.imread('/home/tuan/face_sample1.jpg')
        
        frame = cam.read()
        if frame is None:
            print("No frame")
            time.sleep(0.01)
            continue
        # frame = cv2.imread('/home/tuan/sample_img/ew_detection.png')
        frame = cv2.resize(frame, (640,360))
        frame2 = cv2.flip(frame, 1)

        dets1 = detector(frame.copy())
        dets2 = detector2(frame.copy())
        # print("Dets", dets)
        # if len(dets) > 0:

        dets_arr = [dets1, dets2]

        for idx, dets in enumerate(dets_arr):
            final_dets[idx], inp_queue = tracker_arr[idx].process(dets, frame)
            count_inp += len(inp_queue)
            for inp in inp_queue:
                identifierThread.input_queue.put(inp)

        # for i in range(2):
        try:
            while(True):
                res = identifierThread.output_queue.get_nowait()
                stt = res['stt']
                tracker_arr[stt].updateTrackDict(res)
                count_res += 1
        except queue.Empty:
            pass

        interval = time.time() - fps_t
        fps_t = time.time()
        # print("Total time: ", interval)
        fps = int(1.0/interval)
        img = frame.copy()
        img = draw_box(img, final_dets[0], tracker_arr[0].track_dict, fps)
        img2 = frame2.copy()
        img2 = draw_box(img2, final_dets[1], tracker_arr[1].track_dict, fps)

        debug = np.hstack((img, img2))
        cv2.imshow('out', debug)

        k = cv2.waitKey(1)
        if k & 0xFF == 27:
            cam.stop()
            cv2.destroyAllWindows()
            identifierThread.stop()
            break
