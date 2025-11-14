import cv2
import numpy as np
from typing import Dict, Tuple, Any

import joblib
import pandas as pd
import numpy as np


def load_and_predict_zeta(new_data):
    # 1. Define file names (must match the saving file names)
    model_filename = "svc_model_zetas.joblib"
    scaler_filename = "scaler_zetas.joblib"
    pca_filename = "pca_zetas.joblib"

    # 2. Load the trained objects
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)
    loaded_pca = joblib.load(pca_filename)

    # 3. Apply the transformations in the same order as training
    new_data_scaled = loaded_scaler.transform(new_data)
    new_data_transformed = loaded_pca.transform(new_data_scaled)

    # 4. Make a prediction
    prediction = loaded_model.predict(new_data_transformed)
    return prediction


def load_and_predict_anillo(new_data):
    # 1. Define file names (must match the saving file names)
    model_filename = "svc_model_anillos.joblib"
    scaler_filename = "scaler_anillos.joblib"
    pca_filename = "pca_anillos.joblib"

    # 2. Load the trained objects
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)
    loaded_pca = joblib.load(pca_filename)

    # 3. Apply the transformations in the same order as training
    new_data_scaled = loaded_scaler.transform(new_data)
    new_data_transformed = loaded_pca.transform(new_data_scaled)

    # 4. Make a prediction
    prediction = loaded_model.predict(new_data_transformed)
    return prediction


def extract_features(
    binary: np.ndarray, ext_contour: np.ndarray | None
) -> Dict[str, Any]:
    if ext_contour is None or len(ext_contour) < 3:
        # Retorna métricas vacías si el contorno no es válido
        return {
            "area": 0,
            "perimeter": 0,
            "circularity": 0,
            "solidity": 0,
            "num_holes": 0,
            "hole_areas": [],
            "hu_moments": np.zeros(7),
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
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
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
        "hu_moments": hu,
    }


def classify_piece(features: Dict[str, Any]) -> Tuple[str, bool]:
    # Clasifico la pieza en Anillo, Arandela, Tensor o Zeta y determino si está Buena o Defectuosa.

    # (Hay que mejorar de acuerdo a las formas de las piezas y tal vez agregar más features :))
    piece_type = "Unknown"
    condition = 1

    num_holes = features["num_holes"]
    circularity = features["circularity"]
    solidity = features["solidity"]

    if num_holes == 1:
        ratio = features["hole_areas"][0] / features["area"]
        if ratio < 0.5:
            piece_type = "Arandela"
        else:
            piece_type = "Anillo"
            prediction = load_and_predict_anillo(
                np.array(
                    [
                        [
                            features["area"],
                            features["perimeter"],
                            features["circularity"],
                            features["solidity"],
                            features["num_holes"],
                            features["hole_areas"][0],
                            features["hu_moments"][0],
                            features["hu_moments"][1],
                            features["hu_moments"][2],
                            features["hu_moments"][3],
                            features["hu_moments"][4],
                            features["hu_moments"][5],
                            features["hu_moments"][6],
                        ]
                    ]
                )
            )

    elif num_holes == 2:
        piece_type = "Tensor"
    if num_holes == 1 and circularity < 0.7:
        piece_type = "Zeta"
        prediction = load_and_predict_zeta(
            np.array(
                [
                    [
                        features["area"],
                        features["perimeter"],
                        features["circularity"],
                        features["solidity"],
                        features["num_holes"],
                        features["hole_areas"][0],
                        features["hu_moments"][0],
                        features["hu_moments"][1],
                        features["hu_moments"][2],
                        features["hu_moments"][3],
                        features["hu_moments"][4],
                        features["hu_moments"][5],
                        features["hu_moments"][6],
                    ]
                ]
            )
        )
    condition = prediction
    if piece_type == "Unknown":
        condition = 0

    return piece_type, condition
