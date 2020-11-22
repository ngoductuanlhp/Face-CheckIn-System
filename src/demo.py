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

import queue

font = cv2.FONT_HERSHEY_SIMPLEX

USE_TRACKING = True
IMG_H = 360
IMG_W = 640

# e1 = threading.Event()
# e2 = threading.Event()
# trtThread_detector = TrtThread(e1, e2, 'detector')

e3 = threading.Event()
e4 = threading.Event()
trtThread_identifier = TrtThread(e3, e4, 'identifier')

track_dict = {}
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

def draw_box(img, dets, fps = 0):
    global track_dict, save_id
    if len(dets) > 0:
        for det in dets:
            boxes = det[:4]
            
            cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (2, 255, 0), 2)
            if USE_TRACKING:
                det_id = det[4]
                track = track_dict.get(det_id)
                name = track['label']
                cv2.putText(img, "TrackId: " + str(det_id), (boxes[0],boxes[1] - 20), font, 0.6, (255,0,0), 2, cv2.LINE_AA)
                if name != None:
                    cv2.putText(img, "Name: " + name, (boxes[0], boxes[1]), font, 0.6, (255,0,0), 2, cv2.LINE_AA)
            else:
                name = det[4]
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

def checkbox(xmin, ymin, xmax, ymax):
    if xmin == xmax or ymin == ymax:
        return -1,-1,-1,-1
    xmin = max(0, min(xmin, IMG_W))
    ymin = max(0, min(ymin, IMG_H))
    xmax = max(0, min(xmax, IMG_W))
    ymax = max(0, min(ymax, IMG_H))
    return xmin, ymin, xmax, ymax


def test_image_tensorrt():
    global trtThread, track_dict, USE_TRACKING
    # cam = Camera()
    # cam.start()
    time.sleep(0.1)

    cuda_ctx = cuda.Device(0).make_context()  # GPU 0
    detector = FaceDetector()
    tracker = Sort(min_hits=1, max_age=10)
    
    vid = cv2.VideoCapture('/home/tuan/test1.avi')
    fps_t = time.time()
    while True:
        count_inp = 0
        count_res = 0
        
        _, frame = vid.read()
        # frame = cv2.imread('/home/tuan/face_sample1.jpg')
        
        # frame = cam.read()
        # frame = cv2.imread('/home/tuan/sample_img/ew_detection.png')
        # frame = cv2.resize(frame, (640,360))
        if frame is None:
            print("No frame")
            time.sleep(0.01)
            continue
        # trtThread_detector.input = frame.copy()
        
        # trtThread_detector.eventStart.set()

        # trtThread_detector.eventEnd.wait()
        # trtThread_detector.eventEnd.clear()

        dets, lms = detector(frame.copy())
        # img = draw_box_with_lm(frame.copy(), dets, lms)
        # cv2.imshow('out', img)
        # k = cv2.waitKey(10000)
        # # break
        # if k & 0xFF == 116:
        #     USE_TRACKING = ~USE_TRACKING
        # elif k & 0xFF == 27:
        #     # trtThread_detector.stop()
        #     trtThread_identifier.stop()
        #     break
        # break
        
        final_det = []
        new_iden_inputs = []
        if USE_TRACKING:
            if dets.shape[0] > 0: 
                track_dets = tracker.update(dets)
            else:
                track_dets = tracker.update()

            new_id = []
            
            for xmin, ymin, xmax, ymax, track_id in track_dets:
                xmin, ymin, xmax, ymax = checkbox(xmin, ymin, xmax, ymax)
                if xmin == -1:
                    continue
                final_det.append([int(i) for i in [xmin, ymin, xmax, ymax, track_id]])
                track_id = int(track_id)
                
                # t1 = time.time()
                if track_id not in track_dict.keys():
                    # new_id.append(track_id)
                    track_dict[track_id] = {'label': 'Unknown', 'dist': 100, 'try': 0}
                    bbox = [int(i) for i in [xmin, ymin, xmax, ymax]]
                    crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    trtThread_identifier.input_queue.put({'id': track_id, 'crop': crop})
                    count_inp += 1

                else:
                    track = track_dict[track_id]
                    if (track['label'] == 'Unknown' and track['try'] <  5) or (track['dist'] > 0.6 and track['try'] <  10):
                        bbox = [int(i) for i in [xmin, ymin, xmax, ymax]]
                        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        trtThread_identifier.input_queue.put({'id': track_id, 'crop': crop})

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
                    final_det[idx].append(embed[idx])

        try:
            while(True):
                res = trtThread_identifier.output_queue.get_nowait()
                tried = track_dict[res['id']]['try']
                track_dict[res['id']] = {'label': res['label'], 'dist': res['dist'], 'try': tried+1}
                count_res += 1
        except queue.Empty:
            pass

        interval = time.time() - fps_t
        fps_t = time.time()
        print("Total time: ", interval)
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
    trtThread_identifier.start()
    test_image_tensorrt()
