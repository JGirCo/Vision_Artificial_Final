import cv2
import numpy as np


KERNEL = np.ones((5, 5), np.uint8)
KERNEL_CLOSE = np.ones((10, 10), np.uint8)
MIN_AREA_THRESHOLD = 500
THRESHOLD_VALUE = 150  # The value used in cv2.threshold

Frame = np.ndarray
Contour = np.ndarray


def binarize(frame: Frame) -> Frame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Binarización
    _, binary = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    # Filtro para eliminar punticos (apertura morfológica)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    return binary_clean


def get_contours(frame: Frame) -> list[Contour]:
    contours, _ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []
    valid_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) >= MIN_AREA_THRESHOLD
    ]
    return valid_contours


def get_external(frame: Frame) -> (Contour, int):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) >= MIN_AREA_THRESHOLD
    ]
    if not valid_contours:
        return (None, 0)
    return (valid_contours[0], len(valid_contours))
