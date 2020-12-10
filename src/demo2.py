import cv2
import os
import time
import threading
import pycuda.driver as cuda
import numpy as np
import queue

from utils.orb_camera import OrbCamera
from utils.csi_camera import CSICamera
from models.identifier_thread import IdentifierThread
from models.face_detector import FaceDetector
from models.tracker import Tracker

font = cv2.FONT_HERSHEY_SIMPLEX

save_id = 0

stop_flag = False
    
img1, img2 = None, None
sync_img1, sync_img2 = False, False

def draw_box(img, dets, track_dict, fps = 0, proc_id=0):
    global save_id
    if len(dets) > 0:
        for det in dets:
            boxes = det[:4]
            
            cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (2, 255, 0), 2)

            det_id = det[4]
            track = track_dict.get(det_id)
            name = track['label']
            # cv2.putText(img, "TrackId: " + str(det_id), (boxes[0],boxes[1] - 20), font, 0.6, (255,0,0), 2, cv2.LINE_AA)
            if name != None:
                cv2.putText(img, "Name: " + name, (boxes[0], boxes[1]), font, 0.6, (255,0,0), 2, cv2.LINE_AA)
    if proc_id == 0:
        cv2.putText(img, "IN_CAM", (15,15), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "OUT_CAM", (15,15), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(img, "FPS: " + str(fps), (15,70), font, 0.8, (0,0,255), 2, cv2.LINE_AA)
    return img


def main_process(proc_id, cam, iden_thread):
    global stop_flag, img1, img2, sync_img1, sync_img2
    
    cuda_ctx = cuda.Device(0).make_context()  # GPU 0
    detector = FaceDetector()
    tracker = Tracker(proc_id)
    fps_t = time.time()
    while not stop_flag:
        count_inp = 0
        count_res = 0
        final_dets = []

        frame = cam.read()
        if frame is None:
            print("No frame")
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (640,360))

        dets = detector(frame.copy())
        # print("Dets", dets)
        # if len(dets) > 0:


        final_dets, inp_queue = tracker.process(dets, frame)
        count_inp += len(inp_queue)
        for inp in inp_queue:
            iden_thread.input_queue.put(inp)

        try:
            if proc_id == 0:
                out_queue = iden_thread.output_queue0
            else:
                out_queue = iden_thread.output_queue1
            while(True):
                res = out_queue.get_nowait()
                stt = res['stt']
                tracker.updateTrackDict(res)
                count_res += 1
        except queue.Empty:
            pass

        interval = time.time() - fps_t
        fps_t = time.time()
        
        fps = int(1.0/interval)
        if proc_id == 0:
            img1 = draw_box(frame, final_dets, tracker.track_dict, fps, proc_id=proc_id)
            sync_img1 = True
        else:
            img2 = draw_box(frame, final_dets, tracker.track_dict, fps, proc_id=proc_id)
            sync_img2 = True


def main():
    global stop_flag, img1, img2, sync_img1, sync_img2

    cam1 = CSICamera()
    cam1.start()

    cam2 = OrbCamera()
    cam2.start()

    identifierThread = IdentifierThread()
    identifierThread.start()
    time.sleep(0.1)

    in_thread = threading.Thread(name="in_thread", target=main_process, args=[0, cam1, identifierThread])
    in_thread.start()

    out_thread = threading.Thread(name="out_thread", target=main_process, args=[1, cam2, identifierThread])
    out_thread.start()

    while True:
        if sync_img1:
            sync_img1 = False
            img1_small = cv2.resize(img1, (640,360))
            cv2.imshow("IMG1", img1_small)
        if sync_img2:
            sync_img2 = False
            img2_small = cv2.resize(img2, (640,480))
            cv2.imshow('IMG2', img2_small)
        k = cv2.waitKey(1)
        if k & 0xFF == 27:
            stop_flag = True
            cam.stop()
            identifierThread.stop()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()

    
