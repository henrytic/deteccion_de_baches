
import warnings

warnings.filterwarnings('ignore')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import deque
from ultralytics import YOLO
import argparse
import time
from pathlib import Path


def select_yolo_model(model_size='auto'):

    # Detectar hardware disponible
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # GB
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detectada: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        gpu_memory = 0
        print("No se detectó GPU, usando CPU")

    # Modelos disponibles (YOLOv11 preferido sobre YOLOv8)
    models = {
        'nano': ['yolo11n-seg.pt', 'yolov8n-seg.pt'],
        'small': ['yolo11s-seg.pt', 'yolov8s-seg.pt'],
        'medium': ['yolo11m-seg.pt', 'yolov8m-seg.pt'],
        'large': ['yolo11l-seg.pt', 'yolov8l-seg.pt'],
        'extra_large': ['yolo11x-seg.pt', 'yolov8x-seg.pt']
    }

    if model_size == 'auto':
        # Selección automática basada en hardware
        if not has_gpu or gpu_memory < 4:
            recommended_size = 'small'  # Para procesamiento de video en tiempo real
            print("Hardware limitado detectado. Recomendación: small")
        elif gpu_memory < 6:
            recommended_size = 'medium'
            print("GPU moderada detectada. Recomendación: medium")
        elif gpu_memory < 8:
            recommended_size = 'medium'
            print("GPU buena detectada. Recomendación: medium")
        else:
            recommended_size = 'large'
            print("GPU potente detectada. Recomendación: large")
    else:
        recommended_size = model_size.lower()

    # Obtener lista de modelos para el tamaño recomendado
    model_options = models.get(recommended_size, models['small'])

    # Intentar cargar YOLOv11 primero, luego YOLOv8
    for model_name in model_options:
        try:
            print(f"Intentando cargar: {model_name}")
            model = YOLO(model_name)
            print(f"Modelo cargado exitosamente: {model_name}")

            # Mostrar información del modelo
            if 'yolo11' in model_name:
                print("Usando YOLOv11 - ¡La versión más reciente!")
            else:
                print("Usando YOLOv8 - Versión estable")

            return model, model_name

        except Exception as e:
            print(f"Error cargando {model_name}: {e}")
            continue

    # Si no se pudo cargar ningún modelo
    raise Exception("No se pudo cargar ningún modelo YOLO")


def load_model(model_path=None):
    """Cargar el modelo entrenado o preentrenado"""
    if model_path and os.path.exists(model_path):
        print(f"Cargando modelo entrenado desde: {model_path}")
        try:
            model = YOLO(model_path)
            print("Modelo personalizado cargado exitosamente")
            return model
        except Exception as e:
            print(f"Error cargando modelo personalizado: {e}")
            print("Intentando cargar modelo preentrenado...")

    print("Cargando modelo preentrenado...")
    model, model_name = select_yolo_model('auto')
    print(
        "Nota: Usando modelo preentrenado. Para mejores resultados con baches, usa un modelo entrenado específicamente.")
    return model


def analyze_single_image(model, image_path, conf_threshold=0.5):

    if not os.path.exists(image_path):
        print(f"Error: No se encontró la imagen en {image_path}")
        return None, None

    print(f"Analizando imagen: {image_path}")

    # Realizar predicción
    results = model.predict(source=image_path, imgsz=640, conf=conf_threshold, verbose=False)

    # Procesar resultados
    annotated_image = results[0].plot()
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Calcular área de daño
    damage_info = calculate_damage_area(results[0])

    return annotated_image_rgb, damage_info


def calculate_damage_area(result):
    damage_info = {
        'total_area': 0,
        'individual_areas': [],
        'percentage_damage': 0,
        'num_potholes': 0
    }

    if result.masks is None:
        return damage_info

    masks = result.masks.data.cpu().numpy()
    image_area = masks.shape[1] * masks.shape[2]
    total_area = 0

    for i, mask in enumerate(masks):
        binary_mask = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = contours[0]
            area = cv2.contourArea(contour)
            damage_info['individual_areas'].append(area)
            total_area += area

    damage_info['total_area'] = total_area
    damage_info['percentage_damage'] = (total_area / image_area) * 100 if image_area > 0 else 0
    damage_info['num_potholes'] = len(damage_info['individual_areas'])

    return damage_info


def process_video_with_damage_assessment(model, video_path, output_path, conf_threshold=0.25, show_preview=False):
    print(f"Procesando video: {video_path}")
    print(f"Archivo de salida: {output_path}")
    print(f"Umbral de confianza: {conf_threshold}")

    # Configuración de texto mejorada
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    text_position = (30, 70)
    font_color = (255, 255, 255)  # Blanco
    background_color = (0, 0, 255)  # Rojo
    shadow_color = (0, 0, 0)  # Negro para sombra

    # Deque para promediar daños
    damage_deque = deque(maxlen=15)  # Aumentado para mejor suavizado

    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Propiedades del video:")
    print(f"- FPS: {fps}")
    print(f"- Resolución: {width}x{height}")
    print(f"- Total de frames: {total_frames}")
    print(f"- Duración: {total_frames / fps:.2f} segundos")

    # Configurar writer de video con mejor calidad
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()
    damage_history = []
    pothole_count_history = []
    severe_damage_frames = []  # Frames con daño severo (>5%)

    print("\nIniciando procesamiento...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Mostrar progreso
        if frame_count % 30 == 0 or frame_count == 1:
            elapsed = time.time() - start_time
            progress = (frame_count / total_frames) * 100
            eta = (elapsed / frame_count) * (total_frames - frame_count)
            fps_processing = frame_count / elapsed if elapsed > 0 else 0
            print(
                f"Progreso: {progress:.1f}% - Frame {frame_count}/{total_frames} - ETA: {eta:.1f}s - FPS: {fps_processing:.1f}")

        # Realizar predicción
        results = model.predict(source=frame, imgsz=640, conf=conf_threshold, verbose=False)
        processed_frame = results[0].plot(boxes=False, masks=True, probs=False)

        # Calcular daño
        damage_info = calculate_damage_area(results[0])
        percentage_damage = damage_info['percentage_damage']
        num_potholes = damage_info['num_potholes']

        # Registrar frames con daño severo
        if percentage_damage > 5.0:
            severe_damage_frames.append({
                'frame': frame_count,
                'time': frame_count / fps,
                'damage': percentage_damage,
                'potholes': num_potholes
            })

        damage_deque.append(percentage_damage)
        smoothed_percentage_damage = sum(damage_deque) / len(damage_deque)
        damage_history.append(smoothed_percentage_damage)
        pothole_count_history.append(num_potholes)

        overlay = processed_frame.copy()

        cv2.rectangle(overlay, (20, 20), (500, 150), background_color, -1)
        cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0, processed_frame)

        main_text = f'Dano en Carretera: {smoothed_percentage_damage:.2f}%'
        # Sombra
        cv2.putText(processed_frame, main_text, (text_position[0] + 2, text_position[1] + 2),
                    font, font_scale, shadow_color, 3, cv2.LINE_AA)
        # Texto principal
        cv2.putText(processed_frame, main_text, text_position,
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Información adicional
        bache_position = (text_position[0], text_position[1] + 40)
        bache_text = f'Baches detectados: {num_potholes}'
        cv2.putText(processed_frame, bache_text, (bache_position[0] + 2, bache_position[1] + 2),
                    font, 0.8, shadow_color, 3, cv2.LINE_AA)
        cv2.putText(processed_frame, bache_text, bache_position,
                    font, 0.8, font_color, 2, cv2.LINE_AA)

        # Tiempo del video
        time_position = (text_position[0], text_position[1] + 80)
        time_text = f'Tiempo: {frame_count / fps:.1f}s'
        cv2.putText(processed_frame, time_text, (time_position[0] + 2, time_position[1] + 2),
                    font, 0.6, shadow_color, 2, cv2.LINE_AA)
        cv2.putText(processed_frame, time_text, time_position,
                    font, 0.6, font_color, 1, cv2.LINE_AA)

        # Indicador de severidad
        if smoothed_percentage_damage > 10:
            severity_color = (0, 0, 255)  # Rojo
            severity_text = "DAnO SEVERO"
        elif smoothed_percentage_damage > 5:
            severity_color = (0, 165, 255)  # Naranja
            severity_text = "DAnO MODERADO"
        elif smoothed_percentage_damage > 1:
            severity_color = (0, 255, 255)  # Amarillo
            severity_text = "DAnO LEVE"
        else:
            severity_color = (0, 255, 0)  # Verde
            severity_text = "BUENO"

        # Círculo indicador en la esquina superior derecha
        cv2.circle(processed_frame, (width - 50, 50), 25, severity_color, -1)
        cv2.circle(processed_frame, (width - 50, 50), 25, (255, 255, 255), 2)

        # Escribir frame procesado
        out.write(processed_frame)

        # Mostrar preview si está habilitado
        if show_preview:
            cv2.imshow('Procesamiento en Tiempo Real', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Procesamiento interrumpido por el usuario.")
                break

    # Limpiar recursos
    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()

    # Estadísticas finales mejoradas
    total_time = time.time() - start_time
    avg_damage = np.mean(damage_history) if damage_history else 0
    max_damage = max(damage_history) if damage_history else 0
    avg_potholes = np.mean(pothole_count_history) if pothole_count_history else 0
    max_potholes = max(pothole_count_history) if pothole_count_history else 0

    # Calcular tiempo total con daño significativo
    significant_damage_frames = sum(1 for d in damage_history if d > 1.0)
    significant_damage_time = (significant_damage_frames / fps) if fps > 0 else 0

    print(f"\n Procesamiento completado!")
    print(f"  Tiempo total: {total_time:.2f} segundos")
    print(f"  Frames procesados: {frame_count}")
    print(f" FPS de procesamiento: {frame_count / total_time:.2f}")
    print(f" Estadísticas de daño:")
    print(f"   - Daño promedio: {avg_damage:.2f}%")
    print(f"   - Daño máximo: {max_damage:.2f}%")
    print(f"   - Promedio de baches por frame: {avg_potholes:.2f}")
    print(f"   - Máximo baches detectados: {max_potholes}")
    print(f"   - Tiempo con daño significativo (>1%): {significant_damage_time:.1f}s")
    print(f"   - Frames con daño severo (>5%): {len(severe_damage_frames)}")

    return {
        'damage_history': damage_history,
        'pothole_count_history': pothole_count_history,
        'avg_damage': avg_damage,
        'max_damage': max_damage,
        'avg_potholes': avg_potholes,
        'max_potholes': max_potholes,
        'total_frames': frame_count,
        'processing_time': total_time,
        'severe_damage_frames': severe_damage_frames,
        'significant_damage_time': significant_damage_time
    }


def plot_damage_analysis(stats, output_name="damage_analysis.png"):
    damage_history = stats['damage_history']
    pothole_history = stats['pothole_count_history']

    if not damage_history:
        print("No hay datos de daño para graficar.")
        return

    plt.figure(figsize=(16, 12))

    # Convertir frames a tiempo
    fps = 30  # Asumir 30 fps si no se especifica
    time_axis = np.array(range(len(damage_history))) / fps

    # Gráfico 1: Evolución del daño
    plt.subplot(3, 2, 1)
    plt.plot(time_axis, damage_history, 'b-', linewidth=1, alpha=0.7, label='Daño por frame')

    # Línea de promedio móvil
    if len(damage_history) > 50:
        window_size = min(50, len(damage_history) // 10)
        moving_avg = []
        for i in range(len(damage_history)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(damage_history), i + window_size // 2)
            moving_avg.append(np.mean(damage_history[start_idx:end_idx]))
        plt.plot(time_axis, moving_avg, 'r-', linewidth=2, label=f'Promedio móvil')

    # Líneas de referencia
    plt.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Daño leve (1%)')
    plt.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Daño moderado (5%)')
    plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Daño severo (10%)')

    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Porcentaje de Daño (%)')
    plt.title('Evolución del Daño en Carretera')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfico 2: Conteo de baches
    plt.subplot(3, 2, 2)
    plt.plot(time_axis, pothole_history, 'g-', linewidth=1, alpha=0.8, label='Baches detectados')
    plt.fill_between(time_axis, pothole_history, alpha=0.3, color='green')
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Número de Baches')
    plt.title('Evolución del Conteo de Baches')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfico 3: Histograma de daño
    plt.subplot(3, 2, 3)
    plt.hist(damage_history, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Porcentaje de Daño (%)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución del Porcentaje de Daño')

    # Estadísticas en el histograma
    avg_damage = np.mean(damage_history)
    max_damage = max(damage_history)
    plt.axvline(avg_damage, color='red', linestyle='--', linewidth=2, label=f'Promedio: {avg_damage:.2f}%')
    plt.axvline(max_damage, color='orange', linestyle='--', linewidth=2, label=f'Máximo: {max_damage:.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfico 4: Histograma de baches
    plt.subplot(3, 2, 4)
    plt.hist(pothole_history, bins=max(pothole_history) + 1 if pothole_history else 1,
             alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Número de Baches')
    plt.ylabel('Frecuencia')
    plt.title('Distribución del Conteo de Baches')
    plt.grid(True, alpha=0.3)

    # Gráfico 5: Correlación daño vs baches
    plt.subplot(3, 2, 5)
    plt.scatter(pothole_history, damage_history, alpha=0.5, s=10)
    plt.xlabel('Número de Baches')
    plt.ylabel('Porcentaje de Daño (%)')
    plt.title('Correlación: Baches vs Daño')
    plt.grid(True, alpha=0.3)

    # Gráfico 6: Resumen estadístico
    plt.subplot(3, 2, 6)
    plt.axis('off')

    # Crear tabla de estadísticas
    stats_text = f"""
    RESUMEN ESTADÍSTICO

    Daño en Carretera:
    • Promedio: {stats['avg_damage']:.2f}%
    • Máximo: {stats['max_damage']:.2f}%
    • Mínimo: {min(damage_history):.2f}%
    • Desviación: {np.std(damage_history):.2f}%

    Baches Detectados:
    • Promedio por frame: {stats['avg_potholes']:.2f}
    • Máximo: {stats['max_potholes']}
    • Total acumulado: {sum(pothole_history)}

    Video:
    • Duración: {len(damage_history) / 30:.1f}s
    • Tiempo con daño >1%: {stats.get('significant_damage_time', 0):.1f}s
    • Frames con daño >5%: {len(stats.get('severe_damage_frames', []))}
    """

    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))

    plt.tight_layout()
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"📊 Análisis gráfico guardado en: {output_name}")


def batch_process_images(model, images_folder, output_folder, conf_threshold=0.5):
    print(f"Procesando imágenes en lote desde: {images_folder}")

    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Buscar imágenes
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(images_folder).glob(f"*{ext}"))
        image_files.extend(Path(images_folder).glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No se encontraron imágenes en {images_folder}")
        return

    print(f"Encontradas {len(image_files)} imágenes para procesar")

    results_summary = []
    start_time = time.time()

    for i, image_path in enumerate(image_files):
        print(f"Procesando {i + 1}/{len(image_files)}: {image_path.name}")

        try:
            # Analizar imagen
            annotated_image, damage_info = analyze_single_image(model, str(image_path), conf_threshold)

            if annotated_image is not None:
                # Guardar imagen procesada
                output_path = os.path.join(output_folder, f"processed_{image_path.name}")
                cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

                # Guardar información
                results_summary.append({
                    'image': image_path.name,
                    'num_potholes': damage_info['num_potholes'],
                    'total_area': damage_info['total_area'],
                    'percentage_damage': damage_info['percentage_damage']
                })

                print(f"  - Baches detectados: {damage_info['num_potholes']}")
                print(f"  - Daño: {damage_info['percentage_damage']:.2f}%")

        except Exception as e:
            print(f"Error procesando {image_path.name}: {e}")

    # Guardar resumen
    if results_summary:
        try:
            import pandas as pd
            df = pd.DataFrame(results_summary)
            summary_path = os.path.join(output_folder, 'processing_summary.csv')
            df.to_csv(summary_path, index=False)

            processing_time = time.time() - start_time

            print(f"\n Procesamiento en lote completado!")
            print(f"  Tiempo total: {processing_time:.2f} segundos")
            print(f"  Imágenes procesadas: {len(results_summary)}")
            print(f" Estadísticas:")
            print(f"   - Promedio de baches por imagen: {df['num_potholes'].mean():.2f}")
            print(f"   - Promedio de daño: {df['percentage_damage'].mean():.2f}%")
            print(
                f"   - Imagen con más daño: {df.loc[df['percentage_damage'].idxmax(), 'image']} ({df['percentage_damage'].max():.2f}%)")
            print(f" Resumen guardado en: {summary_path}")

        except ImportError:
            print("Pandas no está disponible. Instalalo con: pip install pandas")


def convert_video_format(input_path, output_path, target_format='mp4'):
    """Convertir video a formato específico usando ffmpeg"""
    try:
        import subprocess

        if target_format.lower() == 'mp4':
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-c:v', 'libx264', '-c:a', 'aac',
                '-movflags', 'faststart',
                '-preset', 'medium',
                '-crf', '23',
                output_path
            ]
        else:
            cmd = ['ffmpeg', '-y', '-i', input_path, output_path]

        print(f"Convirtiendo video a formato {target_format}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f" Video convertido exitosamente: {output_path}")
            return True
        else:
            print(f" Error en conversión: {result.stderr}")
            return False

    except FileNotFoundError:
        print("  FFmpeg no está instalado. Usa el archivo original.")
        return False
    except Exception as e:
        print(f" Error en conversión: {e}")
        return False


def create_damage_report(stats, video_name, output_path="damage_report.txt"):
    """Crear reporte detallado de daños"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE ANÁLISIS DE DAÑOS EN CARRETERA\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Video analizado: {video_name}\n")
        f.write(f"Fecha de análisis: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("RESUMEN EJECUTIVO:\n")
        f.write("-" * 20 + "\n")
        f.write(f"• Daño promedio detectado: {stats['avg_damage']:.2f}%\n")
        f.write(f"• Daño máximo registrado: {stats['max_damage']:.2f}%\n")
        f.write(f"• Promedio de baches por frame: {stats['avg_potholes']:.2f}\n")
        f.write(f"• Tiempo con daño significativo: {stats.get('significant_damage_time', 0):.1f}s\n\n")

        # Clasificación del estado de la carretera
        avg_damage = stats['avg_damage']
        if avg_damage < 1:
            condition = "EXCELENTE"
        elif avg_damage < 3:
            condition = "BUENO"
        elif avg_damage < 5:
            condition = "REGULAR"
        elif avg_damage < 10:
            condition = "MALO"
        else:
            condition = "CRÍTICO"

        f.write(f"ESTADO GENERAL DE LA CARRETERA: {condition}\n\n")

        f.write("ESTADÍSTICAS DETALLADAS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"• Total de frames procesados: {stats['total_frames']}\n")
        f.write(f"• Tiempo de procesamiento: {stats['processing_time']:.2f}s\n")
        f.write(f"• Frames con daño severo (>5%): {len(stats.get('severe_damage_frames', []))}\n")
        f.write(f"• Máximo de baches simultáneos: {stats['max_potholes']}\n\n")

        if stats.get('severe_damage_frames'):
            f.write("MOMENTOS CON DAÑO SEVERO:\n")
            f.write("-" * 25 + "\n")
            for frame_info in stats['severe_damage_frames'][:10]:  # Mostrar solo los primeros 10
                f.write(
                    f"• Tiempo {frame_info['time']:.1f}s: {frame_info['damage']:.2f}% de daño, {frame_info['potholes']} baches\n")
            if len(stats['severe_damage_frames']) > 10:
                f.write(f"... y {len(stats['severe_damage_frames']) - 10} momentos más\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"📄 Reporte detallado guardado en: {output_path}")


def main():
    """Función principal mejorada"""
    parser = argparse.ArgumentParser(description='Procesador mejorado de videos para detección de baches')
    parser.add_argument('--model', '-m', help='Ruta al modelo entrenado (.pt) - opcional')
    parser.add_argument('--input', '-i', required=True, help='Ruta al video o imagen de entrada')
    parser.add_argument('--output', '-o', help='Ruta del archivo de salida')
    parser.add_argument('--conf', '-c', type=float, default=0.25, help='Umbral de confianza (default: 0.25)')
    parser.add_argument('--preview', '-p', action='store_true', help='Mostrar preview durante procesamiento')
    parser.add_argument('--batch', '-b', help='Procesar múltiples imágenes (especifica carpeta)')
    parser.add_argument('--format', '-f', default='mp4', help='Formato de salida para video (default: mp4)')
    parser.add_argument('--report', '-r', action='store_true', help='Generar reporte detallado')

    args = parser.parse_args()

    print("=" * 60)
    print("PROCESADOR MEJORADO DE VIDEOS - DETECCIÓN DE BACHES")
    print("Soporta YOLOv8 y YOLOv11 con selección automática")
    print("=" * 60)

    # Cargar modelo
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f" Error cargando modelo: {e}")
        return

    # Determinar tipo de procesamiento
    input_path = args.input

    # Procesamiento en lote de imágenes
    if args.batch:
        if not args.output:
            args.output = 'processed_images'
        batch_process_images(model, args.batch, args.output, args.conf)
        return

    # Verificar si es imagen o video
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

    file_ext = Path(input_path).suffix.lower()

    if file_ext in image_extensions:
        # Procesar imagen individual
        print("  Procesando imagen individual...")

        if not args.output:
            args.output = f"processed_{Path(input_path).name}"

        annotated_image, damage_info = analyze_single_image(model, input_path, args.conf)

        if annotated_image is not None:
            # Guardar imagen
            cv2.imwrite(args.output, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            # Mostrar resultados
            plt.figure(figsize=(12, 8))
            plt.imshow(annotated_image)
            plt.title(f'Detección de Baches - {Path(input_path).name}')
            plt.axis('off')

            # Añadir información de daño
            info_text = f"Baches detectados: {damage_info['num_potholes']}\n"
            info_text += f"Área total dañada: {damage_info['total_area']:.0f} píxeles\n"
            info_text += f"Porcentaje de daño: {damage_info['percentage_damage']:.2f}%"

            # Clasificar severidad
            if damage_info['percentage_damage'] > 10:
                severity = "SEVERO"
                color = "red"
            elif damage_info['percentage_damage'] > 5:
                severity = "MODERADO"
                color = "orange"
            elif damage_info['percentage_damage'] > 1:
                severity = "LEVE"
                color = "yellow"
            else:
                severity = "MÍNIMO"
                color = "green"

            info_text += f"\nSeveridad: {severity}"

            plt.figtext(0.02, 0.02, info_text, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))

            plt.tight_layout()
            analysis_filename = f"analysis_{Path(input_path).stem}.png"
            plt.savefig(analysis_filename, dpi=150, bbox_inches='tight')
            plt.show()

            print(f" Imagen procesada guardada en: {args.output}")
            print(f" Análisis guardado en: {analysis_filename}")

    elif file_ext in video_extensions:
        # Procesar video
        print(" Procesando video...")

        if not args.output:
            args.output = f"processed_{Path(input_path).stem}.{args.format}"

        try:
            # Procesar video
            stats = process_video_with_damage_assessment(
                model, input_path, args.output, args.conf, args.preview
            )

            # Crear análisis gráfico
            if stats['damage_history']:
                analysis_filename = f"damage_analysis_{Path(input_path).stem}.png"
                plot_damage_analysis(stats, analysis_filename)

            # Generar reporte si se solicita
            if args.report:
                report_filename = f"damage_report_{Path(input_path).stem}.txt"
                create_damage_report(stats, Path(input_path).name, report_filename)

            # Convertir formato si es necesario
            if args.format.lower() != Path(args.output).suffix[1:].lower():
                final_output = f"final_{Path(input_path).stem}.{args.format}"
                if convert_video_format(args.output, final_output, args.format):
                    print(f" Video final guardado en: {final_output}")
                else:
                    print(f" Video guardado en: {args.output}")
            else:
                print(f" Video procesado guardado en: {args.output}")

        except Exception as e:
            print(f" Error procesando video: {e}")

    else:
        print(f" Formato de archivo no soportado: {file_ext}")
        print("Formatos soportados:")
        print(f"- Imágenes: {', '.join(image_extensions)}")
        print(f"- Videos: {', '.join(video_extensions)}")


if __name__ == "__main__":
    # Si se ejecuta sin argumentos, mostrar ayuda mejorada
    import sys

    if len(sys.argv) == 1:
        print("=" * 60)
        print("PROCESADOR MEJORADO DE VIDEOS - DETECCIÓN DE BACHES")
        print("Soporta YOLOv8 y YOLOv11 con selección automática")
        print("=" * 60)
        print("\n Uso básico:")
        print("python process_video.py --input video.mp4")
        print("\n Ejemplos:")
        print("\n# Procesar un video con modelo entrenado:")
        print("python process_video.py -m best.pt -i sample_video.mp4 -o resultado.mp4")
        print("\n# Procesar con modelo automático (sin modelo entrenado):")
        print("python process_video.py -i video.mp4 --preview")
        print("\n# Procesar una imagen:")
        print("python process_video.py -i imagen.jpg -o resultado.jpg")
        print("\n# Procesamiento en lote de imágenes:")
        print("python process_video.py --batch carpeta_imagenes -o resultados")
        print("\n# Con preview en tiempo real y reporte:")
        print("python process_video.py -i video.mp4 --preview --report")
        print("\n# Configurar umbral de confianza:")
        print("python process_video.py -i video.mp4 --conf 0.5")
        print("\n💡 Características nuevas:")
        print("-  Soporte para YOLOv11 (mejor precisión)")
        print("-  Selección automática del mejor modelo según tu hardware")
        print("-  Análisis mejorado con estadísticas detalladas")
        print("-  Reportes automáticos de estado de carretera")
        print("-  Visualización en tiempo real mejorada")
        print("-  Procesamiento en lote optimizado")
        print("\n Para más opciones, usa: python process_video.py --help")
    else:
        main()