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


def get_external(frame: Frame) -> tuple[Contour, int]: #Si no encuentra contornos devuelve None, 0
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) >= MIN_AREA_THRESHOLD
    ]

    if not valid_contours:
        # Retorna un contorno vacío válido para evitar errores en OpenCV
        return (np.zeros((0, 1, 2), dtype=np.int32), 0)

    # Ordenar por área y devolver el más grande
    largest = max(valid_contours, key=cv2.contourArea)
    return (largest, len(valid_contours))

