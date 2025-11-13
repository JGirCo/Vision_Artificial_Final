import cv2
import numpy as np

def extract_features(binary, ext_contour):
    if ext_contour is None or len(ext_contour) < 3:
        # Retorna métricas vacías si el contorno no es válido
        return {
            "area": 0,
            "perimeter": 0,
            "circularity": 0,
            "solidity": 0,
            "num_holes": 0,
            "hole_areas": [],
            "hu_moments": np.zeros(7)
        }

    # Área y perímetro externos
    area_ext = cv2.contourArea(ext_contour)
    peri_ext = cv2.arcLength(ext_contour, True)
    circularity = 4 * np.pi * area_ext / (peri_ext**2 + 1e-9)

    # Convex hull y solidity (solo si hay puntos suficientes) Me ayuda a encontrar roturas o irregularidades (Defectos)
    if len(ext_contour) >= 3:
        hull = cv2.convexHull(ext_contour)
        hull_area = cv2.contourArea(hull) if len(hull) > 0 else area_ext
        solidity = area_ext / (hull_area + 1e-9)
    else:
        solidity = 0

    # Buscar agujeros internos
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1:  # contorno con padre → agujero interno
                holes.append(contours[i])

    num_holes = len(holes)
    hole_areas = [cv2.contourArea(h) for h in holes]

    # Hu Moments
    moments = cv2.moments(ext_contour)
    hu = cv2.HuMoments(moments).flatten()

    return {
        "area": area_ext,
        "perimeter": peri_ext,
        "circularity": circularity,
        "solidity": solidity,
        "num_holes": num_holes,
        "hole_areas": hole_areas,
        "hu_moments": hu
    }

def classify_piece(features):

    #Clasifico la pieza en Anillo, Arandela, Tensor o Zeta y determino si está Buena o Defectuosa.
    
    #(Hay que mejorar de acuerdo a las formas de las piezas y tal vez agregar más features :))
    piece_type = "Unknown"
    condition = "Bueno"

    num_holes = features["num_holes"]
    circularity = features["circularity"]
    solidity = features["solidity"]

    if num_holes == 1:
        ratio = features["hole_areas"][0] / features["area"]
        if ratio < 0.2:
            piece_type = "Anillo"
        else:
            piece_type = "Arandela"
    elif num_holes == 2:
        piece_type = "Tensor"
    elif num_holes == 1 and circularity < 0.7:
        piece_type = "Zeta"

    # Defectos: baja circularidad o baja solidez (Hay que mejorar esto)
    if solidity < 0.9 or circularity < 0.6:
        condition = "Defectuosa"

    return piece_type, condition