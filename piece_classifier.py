import cv2
from camera import RunCamera
import time

import binarization as bn


EDGE_PADDING = 150


class Classifier(RunCamera):
    def __init__(self, src=0, name="Camera_1"):
        super(Classifier, self).__init__(src, name)
        self.piece_in_frame = False
        self.new_piece_incoming = True
        self.current_frame = None

    def separate_frame(self) -> bool:
        """Se asegura de que la pieza está completa en el frame y es válida para el procesamiento.
        La pieza debe estar alejada de los bordes y registrarse como 'Una sola' (cnt_num == 1).
        """

        with self.lock:
            if self.frame is None:
                # If no frame, set the tracking flag to False and exit.
                self.piece_in_frame = False
                return False
            frame_copy = self.frame.copy()

        binary = bn.binarize(frame_copy)
        ext_contour, cnt_num = bn.get_external(binary)

        x, y, w, h = cv2.boundingRect(ext_contour)
        frame_width = binary.shape[1]
        if not (x < EDGE_PADDING or (x + w) > (frame_width - EDGE_PADDING)):
            self.piece_in_frame = True
        else:
            self.piece_in_frame = False
            self.new_piece_incoming = True

        if self.new_piece_incoming and self.piece_in_frame and cnt_num == 1:
            with self.lock:
                self.current_frame = frame_copy
            self.loggerReport.logger.info("nueva pieza en frame y válida")
            self.new_piece_incoming = False
            return True
        return False

    def read_current(self):
        with self.lock:
            frame_copy = (
                self.current_frame.copy() if self.current_frame is not None else None
            )
        return frame_copy


def main() -> None:
    video_source = "./Vid_Piezas/Zetas/Zetas_Buenas.mp4"
    camera = Classifier(src=video_source)
    camera.start()

    # Wait for first frame
    time.sleep(1)
    try:
        while camera.isOpened():
            ret, frame = camera.read()
            if camera.separate_frame():
                current_frame = camera.read_current()
                cv2.imshow("current_frame", current_frame)

            if not ret or frame is None:
                time.sleep(0.01)
                continue
            binary = bn.binarize(frame)
            binaryBGR = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            contours = bn.get_contours(binary)
            ext_contour, _ = bn.get_external(binary)
            # for cnt in contours:
            #     x, y, w, h = cv2.boundingRect(cnt)
            #     cv2.rectangle(binaryBGR, (x, y), (x + w, y + h), (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(ext_contour)
            cv2.rectangle(binaryBGR, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Video", binaryBGR)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
