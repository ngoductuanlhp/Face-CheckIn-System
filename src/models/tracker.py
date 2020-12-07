from utils.sort import Sort
import copy
import numpy as np
import cv2
import os
import time
import threading

IMG_H = 360
IMG_W = 640

def checkbox(xmin, ymin, xmax, ymax):
    if xmin == xmax or ymin == ymax:
        return -1,-1,-1,-1
    xmin = max(0, min(xmin, IMG_W))
    ymin = max(0, min(ymin, IMG_H))
    xmax = max(0, min(xmax, IMG_W))
    ymax = max(0, min(ymax, IMG_H))
    return xmin, ymin, xmax, ymax

class Tracker():
    def __init__(self):
        self.sort = Sort(min_hits=1, max_age=10)
        self.track_dict = {}

    def updateSort(self, dets):
        if dets.shape[0] > 0: 
            return self.sort.update(dets)
        else:
            self.sort.update()

    def process(self, dets, frame):

        track_dets = self.updateSort(dets)

        final_det = []
        inp_queue = []
        if track_dets is None or len(track_dets) == 0:
            return final_det, inp_queue

        for xmin, ymin, xmax, ymax, track_id in track_dets:
            xmin, ymin, xmax, ymax = checkbox(xmin, ymin, xmax, ymax)
            if xmin == -1:
                continue
            final_det.append([int(i) for i in [xmin, ymin, xmax, ymax, track_id]])
            track_id = int(track_id)
            
            # t1 = time.time()
            if track_id not in self.track_dict.keys():
                # new_id.append(track_id)
                self.track_dict[track_id] = {'label': 'Unknown', 'dist': 100, 'try': 0}
                bbox = [int(i) for i in [xmin, ymin, xmax, ymax]]
                crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                inp_queue.append({'id': track_id, 'crop': crop})
            else:
                track = self.track_dict[track_id]
                if (track['label'] == 'Unknown' and track['try'] <  5) or (track['dist'] > 0.6 and track['try'] <  10):
                    track['try'] += 1
                    bbox = [int(i) for i in [xmin, ymin, xmax, ymax]]
                    crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    inp_queue.append({'id': track_id, 'crop': crop})

        return final_det, inp_queue

    def updateTrackDict(self, res):
        tracklet = self.track_dict.get(res['id'])
        if tracklet is not None:
            tried = tracklet['try']
            print(res['id'], tried)
            self.track_dict[res['id']]['label'] = res['label']
            self.track_dict[res['id']]['dist'] = res['dist']
            # print('update trackdict', self.track_dict['id'])
            # self.track_dict['id']= {'label': res['label'], 'dist': res['dist'], 'try': tried+1}


    
    