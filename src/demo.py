import cv2
import os
from models.trt_thread import TrtThread
import time
import threading
from utils.camera import Camera
from models.face_detector import FaceDetector
from utils.sort import *
import copy
import pycuda.driver as cuda
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

USE_TRACKING = False

# e1 = threading.Event()
# e2 = threading.Event()
# trtThread_detector = TrtThread(e1, e2, 'detector')

e3 = threading.Event()
e4 = threading.Event()
trtThread_identifier = TrtThread(e3, e4, 'identifier')

track_dict = {}

def draw_box(img, dets, fps = 0):
    global track_dict
    if len(dets) > 0:
        for det in dets:
            boxes = det[:4]
            
            cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (2, 255, 0), 2)
            if USE_TRACKING:
                det_id = det[4]
                name = track_dict.get(det_id)
                cv2.putText(img, "TrackId: " + str(det_id), (boxes[0],boxes[1] - 20), font, 0.6, (255,0,0), 2, cv2.LINE_AA)
                if name != None:
                    cv2.putText(img, "Name: " + str(name), (boxes[0], boxes[1]), font, 0.6, (255,0,0), 2, cv2.LINE_AA)
            else:
                name = det[4]
                cv2.putText(img, "Name: " + str(name), (boxes[0], boxes[1]), font, 0.6, (255,0,0), 2, cv2.LINE_AA)

    # if landmarks:
    #     for lm in lms:
    #         for i in range(0, 5):
    #             cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
    cv2.putText(img, "FPS: " + str(fps), (15,15), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
    return img


def test_image_tensorrt():
    global trtThread, track_dict, USE_TRACKING
    cam = Camera()
    cam.start()
    time.sleep(0.1)

    cuda_ctx = cuda.Device(0).make_context()  # GPU 0
    detector = FaceDetector()
    tracker = Sort(min_hits=1, max_age=10)
    

    while True:
        frame = cam.read()
        # frame = cv2.imread('/home/tuan/sample_img/ew_detection.png')
        # frame = cv2.resize(frame, (640,360))
        if frame is None:
            print("No frame")
            time.sleep(0.01)
            continue
        # trtThread_detector.input = frame.copy()
        t = time.time()
        # trtThread_detector.eventStart.set()

        # trtThread_detector.eventEnd.wait()
        # trtThread_detector.eventEnd.clear()

        dets = detector(frame.copy())

        final_det = []
        new_iden_inputs = []
        if USE_TRACKING:
            if dets.shape[0] > 0: 
                track_dets = tracker.update(dets)
            else:
                track_dets = tracker.update()

            new_id = []
            
            for xmin, ymin, xmax, ymax, track_id in track_dets:
                final_det.append([int(i) for i in [xmin, ymin, xmax, ymax, track_id]])
                track_id = int(track_id)
                if track_id not in track_dict.keys():
                    new_id.append(track_id)
                    bbox = [int(i) for i in [xmin, ymin, xmax, ymax]]
                    new_iden_inputs.append(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                    
            if len(new_iden_inputs) > 0:
                print("Identifying ... ")
                trtThread_identifier.input = new_iden_inputs
                trtThread_identifier.eventStart.set()

                trtThread_identifier.eventEnd.wait()
                trtThread_identifier.eventEnd.clear()
                embed = copy.deepcopy(trtThread_identifier.result)

                for idx, track_id in enumerate(new_id):
                    print("Identified: ", track_id)
                    bbox = track_dets[idx][0:4]
                    track_dict[track_id] = embed[idx][0]
        else:
            for xmin, ymin, xmax, ymax, score in dets:
                bbox = [int(i) for i in [xmin, ymin, xmax, ymax]]
                final_det.append(bbox)
                new_iden_inputs.append(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])

            if len(new_iden_inputs) > 0:
                print("Identifying ... ")
                trtThread_identifier.input = new_iden_inputs
                trtThread_identifier.eventStart.set()

                trtThread_identifier.eventEnd.wait()
                trtThread_identifier.eventEnd.clear()
                embed = copy.deepcopy(trtThread_identifier.result)
            
                for idx in range(len(embed)):
                    final_det[idx].append(embed[idx][0])

        interval = time.time() - t
        # print("Total time: ", interval)
        fps = int(1.0/interval)
        img2 = frame.copy()
        img2 = draw_box(img2, final_det, fps)
        cv2.imshow('out', img2)
        k = cv2.waitKey(1)
        # break
        if k & 0xFF == 116:
            USE_TRACKING = ~USE_TRACKING
        elif k & 0xFF == 27:
            # trtThread_detector.stop()
            trtThread_identifier.stop()
            break


if __name__ == '__main__':
    # trtThread_detector.start()
    trtThread_identifier.start()
    test_image_tensorrt()
