import cv2
import threading
import time

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class Camera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(gstreamer_pipeline(),cv2.CAP_GSTREAMER)
        # Start the thread to read frames from the video stream
        self.rlock = threading.RLock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.status = False
        self.frame = None
        

    def start(self):
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                with self.rlock:
                    self.status = status
                    self.frame = frame
            time.sleep(0.025)

    def read(self):
        with self.rlock:
            status = self.status
            if self.frame is None:
                frame = None
            else:
                frame = self.frame.copy()
        return frame

    def stop(self):
        cv2.destroyAllWindows()
        self.thread.join()
        self.capture.release()

if __name__ == '__main__':
    cam = Camera()
    cam.start()
    cv2.waitKey(100)
    t = time.time()
    while(True):
        print("Interval time: ", time.time() - t)
        t = time.time()
        img = cam.read()
        # if img is not None:
        cv2.imshow("Img", img)
        k = cv2.waitKey(1)
        if k & 0xFF == 27:
            break
    cam.stop()