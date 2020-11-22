import cv2
from utils.camera import Camera
import time

if __name__ == '__main__':
    cam = Camera()
    cam.start()
    time.sleep(0.1)

    result = cv2.VideoWriter('/home/tuan/test1.avi',  
                         cv2.VideoWriter_fourcc(*'XVID'), 
                         30, (640,360)) 

    while(True):
        frame = cam.read()
        if frame is None:
            print("No frame")
            time.sleep(0.01)
            continue

        result.write(frame)
        cv2.imshow('out', frame)
        k = cv2.waitKey(25)
        if k & 0xFF == 27:
            print("Saving video ...")
            result.release()
            cam.stop()
            break


