import cv2

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
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
    def __init__(self):
        self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # set buffer size
        if self.cap is None:
            print("Open camera failed !!!")
        if not self.cap.isOpened():
            self.cap.open()
        
    def __call__(self):
        ret_val, img = self.cap.read()
        return img if ret_val else None

    def __del__(self):
        self.cap.release()
