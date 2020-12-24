import cv2
import time
from utils.orb_camera import OrbCamera


if __name__ == "__main__":
    cam = OrbCamera()
    cam.start()

    while True:
        rgb = cam.read()
        if rgb is None:
            time.sleep(0.001)
            continue
        cv2.imshow('RGB', rgb)
        cv2.waitKey(30)
