import threading
import time
import cv2
from logger import Logger


class RunCamera:
    def __init__(self, src=0, name="Camera_1") -> None:
        try:
            self.name = name
            self.src = src
            self.ret = None
            self.frame = None
            self.stopped = False
            self.loggerReport = Logger(name)
            self.loggerReport.logger.info("Init constructor RunCamera")
            self.lock = threading.Lock()
        except Exception as e:
            self.loggerReport.logger.error("Error in RunCamera: " + str(e))

    def start(self):
        try:
            self.stream = cv2.VideoCapture(self.src)

            self.fps = self.stream.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30
                self.loggerReport.logger.warning(
                    f"Could not determine FPS for {self.src}. Defaulting to {self.fps}."
                )
            else:
                self.loggerReport.logger.info(f"Video FPS: {self.fps}")

            time.sleep(0.5)
            self.ret, self.frame = self.stream.read()

            if self.stream.isOpened():
                self.loggerReport.logger.info("Creating Camera thread")
                self.cam_thread = threading.Thread(
                    target=self._capture_loop, name=self.name, daemon=True
                )
                self.cam_thread.start()
            else:
                self.loggerReport.logger.warning("Camera not initialized")
        except Exception as e:
            self.loggerReport.logger.error("Error starting camera thread: " + str(e))

    def stop(self):
        self.loggerReport.logger.info("Stopping camera thread and releasing resources")
        self.stopped = True
        if hasattr(self, "cam_thread") and self.cam_thread.is_alive():
            self.cam_thread.join(timeout=2)
        if hasattr(self, "stream") and self.stream.isOpened():
            self.stream.release()

    def _capture_loop(self):
        delay_between_frames = 1.0 / self.fps
        self.loggerReport.logger.info(
            "Starting thread, time between frames is %f" % (delay_between_frames)
        )
        while not self.stopped:
            start_time = time.time()
            ret, frame = self.stream.read()

            if not ret:
                self.loggerReport.logger.warning("Failed to read frame")
                break

            with self.lock:
                self.ret = ret
                self.frame = frame

            processing_time = time.time() - start_time
            sleep_time = delay_between_frames - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def isOpened(self):
        return hasattr(self, "stream") and self.stream.isOpened()


def main():
    video_source = "./Vid_Piezas/Zetas/Zetas_Buenas.mp4"
    camera = RunCamera(src=video_source)
    camera.start()

    # Wait for first frame
    time.sleep(1)

    cv2.namedWindow("Video Classifier", cv2.WINDOW_NORMAL)

    try:
        while camera.isOpened():
            ret, frame = camera.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            cv2.imshow("Video Classifier", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
