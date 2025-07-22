import os
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Carga el modelo entrenado
best_model = YOLO('model/best.pt')

def process_image(file_path):
    """Procesar imagen con el modelo YOLOv8"""
    results = best_model.predict(source=file_path, imgsz=640, conf=0.25)
    annotated_img = results[0].plot()
    output_path = file_path.replace('uploads', 'results')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated_img)
    return output_path.replace("\\", "/")  # Compatible con Windows y web

def process_video(file_path):
    """Procesar video con el modelo YOLOv8 y generar .mp4 compatible"""
    cap = cv2.VideoCapture(file_path)

    # Preparar ruta de salida
    base_name = os.path.basename(file_path)
    name_wo_ext = os.path.splitext(base_name)[0]
    output_name = name_wo_ext + "_processed.mp4"
    output_path = os.path.join("static", "results", output_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Códec compatible web
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (30, 50)
    font_scale = 1
    font_color = (255, 255, 255)
    bar_color = (0, 0, 255)
    damage_deque = deque(maxlen=10)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = best_model.predict(source=frame, imgsz=640, conf=0.25)
        processed_frame = results[0].plot(boxes=False)

        percentage_damage = 0
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            image_area = frame.shape[0] * frame.shape[1]
            total_area = 0
            for mask in masks:
                binary_mask = (mask > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    total_area += cv2.contourArea(contours[0])
            percentage_damage = (total_area / image_area) * 100

        damage_deque.append(percentage_damage)
        smoothed = sum(damage_deque) / len(damage_deque)

        # Dibujar barra y texto
        cv2.line(processed_frame, (text_position[0], text_position[1] - 10),
                 (text_position[0] + 350, text_position[1] - 10), bar_color, 40)

        cv2.putText(processed_frame, f'Daño estimado: {smoothed:.2f}%',
                    text_position, font, font_scale, font_color, 2, cv2.LINE_AA)

        out.write(processed_frame)

    cap.release()
    out.release()

    return output_path.replace("\\", "/")  # Ruta compatible con HTML
