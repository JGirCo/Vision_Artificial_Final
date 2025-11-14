import cv2
import numpy as np
from typing import Dict, Tuple, Any


def extract_features(
    binary: np.ndarray, ext_contour: np.ndarray | None
) -> Dict[str, Any]:
    if ext_contour is None or len(ext_contour) < 3:
        return {
            "area": 0,
            "perimeter": 0,
            "circularity": 0,
            "solidity": 0,
            "num_holes": 0,
            "hole_areas": [],
            "hu_moments": np.zeros(7),
            "excentricidad": 0,
            "aspect_ratio": 0,
            "compactness": 0,
            "hole_offset": 0,
            "fourier_desc": np.zeros(10),
        }

    # Área y perímetro externos
    area_ext = cv2.contourArea(ext_contour)
    peri_ext = cv2.arcLength(ext_contour, True)
    circularity = 4 * np.pi * area_ext / (peri_ext**2 + 1e-9)

    # Compactness
    compactness = (peri_ext**2) / (area_ext + 1e-9)

    # Convex hull y solidity
    if len(ext_contour) >= 3:
        hull = cv2.convexHull(ext_contour)
        hull_area = cv2.contourArea(hull) if len(hull) > 0 else area_ext
        solidity = area_ext / (hull_area + 1e-9)
    else:
        solidity = 0

    # Buscar agujeros internos
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    holes = []
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1:
                holes.append(contours[i])

    num_holes = len(holes)
    hole_areas = [cv2.contourArea(h) for h in holes]

    # Hu Moments
    moments = cv2.moments(ext_contour)
    hu = cv2.HuMoments(moments).flatten()

    # Excentricidad
    if len(ext_contour) >= 5:
        ellipse = cv2.fitEllipse(ext_contour)
        (_, axes, _) = ellipse
        major, minor = axes
        excentricidad = major / minor if minor > 0 else 0
    else:
        excentricidad = 0

    # Aspect ratio
    rect = cv2.minAreaRect(ext_contour)
    (_, (w, h), _) = rect
    aspect_ratio = max(w, h) / (min(w, h) + 1e-9)

    # Hole offset (si hay agujero principal)
    hole_offset = 0
    if num_holes > 0:
        # Centroide pieza
        cx_piece = int(moments["m10"] / (moments["m00"] + 1e-9))
        cy_piece = int(moments["m01"] / (moments["m00"] + 1e-9))
        # Centroide agujero principal
        m_hole = cv2.moments(holes[0])
        cx_hole = int(m_hole["m10"] / (m_hole["m00"] + 1e-9))
        cy_hole = int(m_hole["m01"] / (m_hole["m00"] + 1e-9))
        hole_offset = np.sqrt((cx_piece - cx_hole)**2 + (cy_piece - cy_hole)**2)

    # Fourier descriptors (primeros 10 coeficientes normalizados)
    contour_complex = np.empty(len(ext_contour), dtype=complex)
    contour_complex.real = ext_contour[:, 0, 0]
    contour_complex.imag = ext_contour[:, 0, 1]
    fourier_result = np.fft.fft(contour_complex)
    fourier_desc = np.abs(fourier_result[:10]) / (np.abs(fourier_result[0]) + 1e-9)

    return {
        "area": area_ext,
        "perimeter": peri_ext,
        "circularity": circularity,
        "solidity": solidity,
        "num_holes": num_holes,
        "hole_areas": hole_areas,
        "hu_moments": hu,
        "excentricidad": excentricidad,
        "aspect_ratio": aspect_ratio,
        "compactness": compactness,
        "hole_offset": hole_offset,
        "fourier_desc": fourier_desc,
    }

def classify_piece(features: Dict[str, Any]) -> Tuple[str, str]:
    piece_type = "Unknown"
    condition = "Bueno"

    num_holes = features["num_holes"]
    circularity = features["circularity"]
    solidity = features["solidity"]
    excentricidad = features["excentricidad"]
    aspect_ratio = features["aspect_ratio"]
    compactness = features["compactness"]
    hole_offset = features["hole_offset"]

    if num_holes == 1 and features["area"] > 0:
        ratio = features["hole_areas"][0] / features["area"]
        print(f"{ratio=}")
        if ratio < 0.4 and circularity > 0.7:
            if (1.0 <= aspect_ratio <= 1.08):
                piece_type = "Arandela"
        else:
            if circularity > 0.8 and excentricidad >= 0.94 and aspect_ratio <= 1.08:
                piece_type = "Anillo"
            else:
                piece_type = "Unknown"

    elif num_holes == 2:
        piece_type = "Tensor"

    # Zeta: 1 agujero pero forma irregular
    if num_holes == 1 and circularity < 0.3:
        if (
            0.7 <= solidity <= 0.9 and
            0.50 <= excentricidad <= 0.90 and
            1.20 <= aspect_ratio <= 1.85 and
            45 <= compactness <= 70 and
            30 <= hole_offset <= 43
        ):
            piece_type = "Zeta"
        else:
            piece_type = "Unknown"

    # Defectos: baja circularidad o baja solidez

    if piece_type == "Unknown":
        condition = "Defectuosa"
    else:
        condition = "Bueno"

    return piece_type, condition

# solidity 0.7 0.9, excentricudad 0.50 0.90, aspect_ratio 1.20 1.85, compactness 45 70, hole_offset 30 43