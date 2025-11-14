import cv2
import time
import numpy as np
import csv
import os
import glob

from camera import RunCamera
import binarization as bn
import features

from typing import Dict, Tuple
from collections import OrderedDict
import pandas as pd

EDGE_PADDING = 600


def save_features_to_csv(piece_index, features_dict, piece_type, piece_condition):
    """
    Guarda los datos de la pieza en un archivo CSV específico para el piece_type usando Pandas.
    Aclana el diccionario de características, incluyendo hu_moments en 7 columnas
    y hole_areas (solo el primer valor).

    Args:
        piece_index (int): El número incremental que identifica la pieza.
        features_dict (dict): El diccionario de características de la pieza.
        piece_type (str): El tipo de pieza, usado para nombrar la carpeta y el archivo CSV.
        save_base_dir (str): El directorio base para guardar los archivos.

    Returns:
        str: La ruta completa del archivo CSV, o None si falla.
    """
    # 1. Definir el directorio y la ruta del archivo CSV
    target_dir = os.path.join("imagenes_piezas", piece_type)
    csv_file_path = os.path.join(target_dir, f"{piece_type}_features.csv")

    # Aseguramos que el directorio exista
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as e:
        print(f"Error al crear el directorio '{target_dir}': {e}")
        return None

    hole_area_value = (
        features_dict["hole_areas"][0] if features_dict["hole_areas"] else 0
    )

    hu_moments_list = features_dict["hu_moments"].flatten().tolist()
    data_row = [
        piece_index,  # Columna 0: piece_id (vínculo con la imagen)
        piece_condition,
        features_dict["area"],  # Columna 1
        features_dict["perimeter"],  # Columna 2
        features_dict["circularity"],  # Columna 3
        features_dict["solidity"],  # Columna 4
        features_dict["num_holes"],  # Columna 5
        hole_area_value,  # Columna 6: Primera área del agujero
    ] + hu_moments_list  # Columna 7-13: Los 7 Hu Moments

    column_names = [
        "piece_id",
        "Roto",
        "col_1",
        "col_2",
        "col_3",
        "col_4",
        "col_5",
        "col_6",
        "Hu_1",
        "Hu_2",
        "Hu_3",
        "Hu_4",
        "Hu_5",
        "Hu_6",
        "Hu_7",
    ]

    df_new_row = pd.DataFrame([data_row], columns=column_names)

    try:
        df_new_row.to_csv(csv_file_path, index=False, header=False, mode="a")

        print(f"📝 Datos de la pieza {piece_index} guardados en: {csv_file_path}")
        return csv_file_path

    except Exception as e:
        print(f"❌ Error al escribir en el CSV con Pandas '{csv_file_path}': {e}")
        return None


def save_from_type(frame: np.ndarray, piece_type: str) -> int:
    target_dir = os.path.join("imagenes_piezas", piece_type)
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as e:
        print(f"Error al crear el directorio '{target_dir}': {e}")
        return None
    search_pattern = os.path.join(target_dir, "*.png")
    existing_files = glob.glob(search_pattern)
    max_index = 0
    for file_path in existing_files:
        # Extraer el nombre base (ej: '0001.png')
        file_name = os.path.basename(file_path)
        # Extraer solo la parte numérica (ej: '0001')
        try:
            # Quitamos la extensión y convertimos a entero
            current_index = int(file_name.split(".")[0])
            if current_index > max_index:
                max_index = current_index
        except ValueError:
            # Ignorar archivos que no tengan un nombre numérico (ej: 'otra_cosa.png')
            continue

    new_index = max_index + 1

    file_name = f"{new_index:04d}.png"  # Usamos formato 04d para 0001, 0002, etc.
    file_path = os.path.join(target_dir, file_name)

    success = cv2.imwrite(file_path, frame)

    if success:
        print(f"✅ Pieza guardada: {file_path}")
        return new_index
    else:
        print(f"❌ Error al guardar la imagen en: {file_path}")
        return None


def save_to_csv(piece_index, features_dict, piece_type):
    """
    Guarda los datos de la pieza en un archivo CSV específico para el piece_type.
    Asegura que la columna 'piece_id' esté al principio.

    Args:
        piece_index (int): El número incremental que identifica la pieza.
        features_dict (dict): El diccionario de características de la pieza.
        piece_type (str): El tipo de pieza, usado para nombrar la carpeta y el archivo CSV.

    Returns:
        str: La ruta completa del archivo CSV, o None si falla.
    """
    # 1. Definir el directorio y la ruta del archivo CSV
    target_dir = os.path.join("imagenes_piezas", piece_type)
    csv_file_path = os.path.join(target_dir, f"{piece_type}_features.csv")

    # 2. Preparar el registro de datos
    # Convertimos features_dict a OrderedDict para garantizar el orden de las columnas
    row_data = OrderedDict([("piece_id", piece_index)])
    row_data.update(features_dict)

    fieldnames = list(row_data.keys())

    # 3. Guardar en el CSV
    try:
        # Si el archivo no existe, lo abrimos en modo 'w' para escribir el encabezado.
        # Si ya existe, lo abrimos en modo 'a' para anexar.
        file_exists = os.path.exists(csv_file_path)

        with open(csv_file_path, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()  # Escribe el encabezado solo si el archivo es nuevo

            writer.writerow(row_data)

        print(f"📝 Datos de la pieza {piece_index} guardados en: {csv_file_path}")
        return csv_file_path

    except Exception as e:
        print(f"❌ Error al escribir en el CSV '{csv_file_path}': {e}")
        return None


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

    def classify_piece(self) -> Tuple[str, str, Dict]:
        piece = self.current_piece.copy()

        ext_contour, cnt_num_cf = bn.get_external(piece)

        features_dict = features.extract_features(piece, ext_contour)
        piece_type, condition = features.classify_piece(features_dict)
        return (piece_type, condition, features_dict)


def process_videos() -> None:
    for root, _, files in os.walk("./Vid_Piezas"):
        for file in files:
            if file.lower().endswith(
                (".mp4", ".avi", ".mov")
            ):  # Se puede expandir a otras extensiones
                video_source = os.path.join(root, file)
                print("-" * 50)
                print(f"🎥 Procesando video: {video_source}")
                print("-" * 50)
                main(video_source)


def main(video_source) -> None:
    camera = Classifier(src=video_source)
    piece_condition = 0 if "Mal" in video_source else 1
    print(f"{piece_condition=}")
    camera.start()

    # Wait for first frame
    time.sleep(1)
    try:
        while camera.isOpened():
            ret, frame = camera.read()
            if camera.separate_frame():
                piece = camera.read_piece()
                piece_type, condition, features_dict = camera.classify_piece()
                index = save_from_type(piece, piece_type)
                save_features_to_csv(index, features_dict, piece_type, piece_condition)

                print(
                    f"Tipo: {piece_type}, Condición: {condition}, "
                    f"Circularidad={features_dict['circularity']:.2f}, "
                    f"Solidity={features_dict['solidity']:.2f}, "
                    f"Agujeros={features_dict['num_holes']}"
                )
                # Dibujar texto en el current_frame
                frame_text = f"{piece_type}-{condition}"
                frame_color = (0, 255, 0)

            elif not camera.piece_in_frame:
                frame_text = "Esperando pieza..."
                frame_color = (0, 255, 255)

            if not ret or frame is None:
                time.sleep(0.01)
                break

            binary = bn.binarize(frame)
            binaryBGR = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            cv2.putText(
                binaryBGR,
                frame_text,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                frame_color,
                2,
            )
            # cv2.imshow("Video", bn.binarize(binaryBGR))

            # if cv2.waitKey(30) & 0xFF == ord("q"):
            #     return
    except Exception as e:
        print(f"⚠️ Error al procesar {video_source}: {e}")
    finally:
        if not camera.stopped:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_videos()
