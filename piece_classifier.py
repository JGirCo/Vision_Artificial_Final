import cv2
from camera import RunCamera
import time
import numpy as np

import binarization as bn
import features

EDGE_PADDING = 600


def get_piece(frame: np.ndarray) -> np.ndarray:
    ext_contour, _ = bn.get_external(frame)
    rot_rect = cv2.minAreaRect(ext_contour)

    center, size, angle = rot_rect
    w, h = size

    if w < h:
        # If the width is smaller than the height, swap them and adjust the angle.
        w, h = h, w
        angle += 90.0

    piece_center = center
    rot_mat = cv2.getRotationMatrix2D(piece_center, angle, 1.0)
    result_rotated = cv2.warpAffine(
        frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR
    )
    cnt_rotated, _ = bn.get_external(result_rotated)
    x, y, w, h = cv2.boundingRect(cnt_rotated)
    piece = result_rotated[y : y + h, x : x + w]
    return piece

class Classifier(RunCamera):
    def __init__(self, src=0, name="Camera_1"):
        super(Classifier, self).__init__(src, name)
        self.piece_in_frame = False
        self.new_piece_incoming = True
        self.current_frame = None
        self.get_piece = None

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
                self.current_frame = binary
                self.current_piece = get_piece(binary)
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

    def read_piece(self):
        with self.lock:
            frame_copy = (
                self.current_piece.copy() if self.current_piece is not None else None
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
                if current_frame is not None:
                    # Binarizar y obtener contorno externo
                    binary_cf = bn.binarize(current_frame)
                    ext_contour_cf, cnt_num_cf = bn.get_external(binary_cf)

                    if cnt_num_cf > 0 and ext_contour_cf is not None and len(ext_contour_cf) >= 3:
                        # Extraer features y clasificar
                        features_dict_cf = features.extract_features(binary_cf, ext_contour_cf)
                        piece_type_cf, condition_cf = features.classify_piece(features_dict_cf)

                        # Dibujar texto en el current_frame
                        cv2.putText(current_frame, f"{piece_type_cf}-{condition_cf}",
                                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 255, 0), 2)
                    else:
                        cv2.putText(current_frame, "Esperando pieza...",
                                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 255, 255), 2)

                cv2.imshow("current_frame", current_frame)

            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            # Binarización y obtención de contornos
            binary = bn.binarize(frame)
            binaryBGR = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            contours = bn.get_contours(binary)
            ext_contour, cnt_num = bn.get_external(binary)

            # for cnt in contours:
            #     x, y, w, h = cv2.boundingRect(cnt)
            #     cv2.rectangle(binaryBGR, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if cnt_num > 0 and ext_contour is not None and len(ext_contour) >= 3:
                # Bounding box
                x, y, w, h = cv2.boundingRect(ext_contour)
                cv2.rectangle(binaryBGR, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Clasificación usando features
                features_dict = features.extract_features(binary, ext_contour)
                piece_type, condition = features.classify_piece(features_dict)

                # Mostrar resultado en pantalla
                cv2.putText(binaryBGR, f"{piece_type}-{condition}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"Tipo: {piece_type}, Condición: {condition}, "
                      f"Circularidad={features_dict['circularity']:.2f}, "
                      f"Solidity={features_dict['solidity']:.2f}, "
                      f"Agujeros={features_dict['num_holes']}")
            else:
                # No hay pieza válida en este frame
                cv2.putText(binaryBGR, "Esperando pieza...", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Video", binaryBGR)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
