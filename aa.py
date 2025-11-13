import cv2
import numpy as np

# Selección de fuente
video_path = r'Vid_Piezas\Tensores\Tensores_Buenos.mp4'  # None para cámara en vivo

if video_path is None:
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Cámara externa
else:
    cap = cv2.VideoCapture(video_path)        # Archivo de video

# Kernel para operaciones morfológicas
kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Binarización
    _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

    # Filtro para eliminar punticos (apertura morfológica)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel)

    #Find contours in the mask 
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Analizar Contornos
    if(len(contours) > 0):
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # print(f"Bounding box of the first contour: x={x}, y={y}, w={w}, h={h}")
            Area = cv2.contourArea(cnt)
            rect_area = w * h
            fill_ratio = Area / (rect_area + 1e-6)
            # print(f"Area of contour: {Area}")
     
                # Filtros combinados
            if Area < 20000:   # descarta áreas pequeñas
                continue
            if w < 150 or h < 150:  # descarta rectángulos muy chicos
                continue
            if fill_ratio < 0.5:  # descarta rectángulos grandes con poco relleno
                continue
            

            print(f"Area of contour: {Area}")
            #Drawn Rectangle
            cv2.rectangle(binary_clean, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Contours", binary_clean)
            #Draw contours on the original image
            cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
            cv2.waitKey(1)


    # Mostrar resultados
    cv2.imshow('Original', frame)
    # cv2.imshow('Binarized Frame', binqqqqary)
    # cv2.imshow('Binarized Clean', binary_clean)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()