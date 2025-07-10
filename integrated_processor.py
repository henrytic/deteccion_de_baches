
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
import json
from datetime import datetime

# Importar nuestro clasificador
from pothole_classifier import PotholeClassifier


class IntegratedPotholeProcessor:

    def __init__(self, model, pixel_to_cm_ratio=None):

        self.model = model
        self.classifier = PotholeClassifier(pixel_to_cm_ratio)

        # Estadísticas de procesamiento
        self.total_potholes_detected = 0
        self.severity_history = []
        self.frame_analyses = []

    def process_single_frame(self, frame, conf_threshold=0.25):

        # Realizar detección con YOLO
        results = self.model.predict(source=frame, imgsz=640, conf=conf_threshold, verbose=False)

        if len(results) == 0 or results[0].masks is None:
            return {
                'frame_analyzed': True,
                'potholes_detected': 0,
                'analyses': [],
                'severity_summary': {},
                'annotated_frame': frame.copy()
            }

        # Analizar baches con clasificador
        image_shape = frame.shape[:2]  # (height, width)
        analyses = self.classifier.analyze_potholes_in_image(results[0], image_shape)

        # Crear resumen de severidad
        severity_summary = self.classifier.create_severity_summary(analyses)

        # Dibujar clasificación en el frame
        annotated_frame = self.classifier.draw_classified_potholes(frame, analyses)

        # Agregar overlay de información general
        annotated_frame = self._add_frame_overlay(annotated_frame, severity_summary)

        return {
            'frame_analyzed': True,
            'potholes_detected': len(analyses),
            'analyses': analyses,
            'severity_summary': severity_summary,
            'annotated_frame': annotated_frame
        }

    def _add_frame_overlay(self, frame, severity_summary):
        overlay = frame.copy()

        # Configuración del overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Posición del panel de información
        panel_x, panel_y = 20, 20
        panel_width, panel_height = 350, 200

        # Fondo semi-transparente
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Información general
        y_offset = panel_y + 30
        cv2.putText(frame, f"Baches detectados: {severity_summary['total_potholes']}",
                    (panel_x + 10, y_offset), font, 0.6, (255, 255, 255), thickness)

        if severity_summary['total_potholes'] > 0:
            y_offset += 25
            cv2.putText(frame, f"Diametro promedio: {severity_summary['average_diameter']:.1f}{self.classifier.unit}",
                        (panel_x + 10, y_offset), font, 0.5, (255, 255, 255), thickness)

            # Mostrar distribución por severidad
            y_offset += 30
            cv2.putText(frame, "Severidad:", (panel_x + 10, y_offset), font, 0.6, (255, 255, 255), thickness)

            for severity, count in severity_summary['severity_counts'].items():
                y_offset += 20
                color = self.classifier.severity_categories[severity]['color']
                cv2.putText(frame, f"  {severity}: {count}",
                            (panel_x + 15, y_offset), font, 0.4, color, thickness)

        return frame

    def process_video_with_classification(self, video_path, output_path, conf_threshold=0.25, show_preview=False):

        print(f" Procesando video con clasificación de severidad...")
        print(f" Entrada: {video_path}")
        print(f" Salida: {output_path}")

        # Abrir video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        # Propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f" Propiedades: {width}x{height}, {fps}FPS, {total_frames} frames")

        # Configurar writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Variables de seguimiento
        frame_count = 0
        start_time = time.time()

        # Estadísticas globales
        global_severity_counts = {}
        all_diameters = []
        critical_moments = []

        print("\n Iniciando procesamiento frame por frame...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Mostrar progreso
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                eta = (elapsed / frame_count) * (total_frames - frame_count)
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                print(
                    f" Progreso: {progress:.1f}% - Frame {frame_count}/{total_frames} - ETA: {eta:.1f}s - FPS: {fps_processing:.1f}")

            # Procesar frame
            frame_analysis = self.process_single_frame(frame, conf_threshold)

            # Actualizar estadísticas globales
            if frame_analysis['potholes_detected'] > 0:
                severity_summary = frame_analysis['severity_summary']

                # Acumular conteos de severidad
                for severity, count in severity_summary['severity_counts'].items():
                    global_severity_counts[severity] = global_severity_counts.get(severity, 0) + count

                # Recopilar diámetros
                all_diameters.extend(severity_summary['diameters'])

                # Detectar momentos críticos (muchos baches severos/críticos)
                critical_count = severity_summary['severity_counts'].get('CRITICO', 0) + \
                                 severity_summary['severity_counts'].get('SEVERO', 0)

                if critical_count >= 2:  # 2 o más baches críticos/severos
                    critical_moments.append({
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'critical_potholes': critical_count,
                        'total_potholes': severity_summary['total_potholes'],
                        'avg_diameter': severity_summary['average_diameter']
                    })

            # Guardar análisis del frame
            self.frame_analyses.append(frame_analysis)

            # Escribir frame procesado
            out.write(frame_analysis['annotated_frame'])

            # Mostrar preview
            if show_preview:
                cv2.imshow('Clasificación de Severidad en Tiempo Real', frame_analysis['annotated_frame'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("  Procesamiento interrumpido por el usuario")
                    break

        # Limpiar recursos
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

        # Calcular estadísticas finales
        total_time = time.time() - start_time
        total_potholes = sum(global_severity_counts.values())
        avg_diameter = np.mean(all_diameters) if all_diameters else 0
        max_diameter = max(all_diameters) if all_diameters else 0

        # Crear estadísticas completas
        final_stats = {
            'processing_info': {
                'total_frames': frame_count,
                'processing_time': total_time,
                'fps_processing': frame_count / total_time,
                'video_duration': frame_count / fps
            },
            'detection_summary': {
                'total_potholes': total_potholes,
                'severity_distribution': global_severity_counts,
                'average_diameter': avg_diameter,
                'max_diameter': max_diameter,
                'unit': self.classifier.unit
            },
            'critical_moments': critical_moments,
            'frame_analyses': self.frame_analyses
        }

        print(f"\n Procesamiento completado!")
        print(f"  Tiempo total: {total_time:.2f}s")
        print(f" Baches detectados: {total_potholes}")
        print(f" Diámetro promedio: {avg_diameter:.1f}{self.classifier.unit}")
        print(f"  Momentos críticos: {len(critical_moments)}")

        return final_stats

    def generate_comprehensive_report(self, stats, video_name, save_path=None):

        if save_path is None:
            save_path = f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE COMPREHENSIVO DE ANÁLISIS DE BACHES CON CLASIFICACIÓN DE SEVERIDAD\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Video analizado: {video_name}\n")
            f.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Unidad de medida: {self.classifier.unit}\n\n")

            # Información del procesamiento
            proc_info = stats['processing_info']
            f.write("INFORMACIÓN DEL PROCESAMIENTO:\n")
            f.write("-" * 35 + "\n")
            f.write(f"• Duración del video: {proc_info['video_duration']:.2f} segundos\n")
            f.write(f"• Frames procesados: {proc_info['total_frames']}\n")
            f.write(f"• Tiempo de procesamiento: {proc_info['processing_time']:.2f} segundos\n")
            f.write(f"• Velocidad de procesamiento: {proc_info['fps_processing']:.2f} FPS\n\n")

            # Resumen de detección
            detection = stats['detection_summary']
            f.write("RESUMEN DE DETECCIÓN:\n")
            f.write("-" * 25 + "\n")
            f.write(f"• Total de baches detectados: {detection['total_potholes']}\n")
            f.write(f"• Diámetro promedio: {detection['average_diameter']:.2f} {detection['unit']}\n")
            f.write(f"• Diámetro máximo: {detection['max_diameter']:.2f} {detection['unit']}\n")
            f.write(
                f"• Densidad de baches: {detection['total_potholes'] / proc_info['video_duration']:.2f} baches/segundo\n\n")

            # Distribución por severidad
            f.write("DISTRIBUCIÓN POR SEVERIDAD:\n")
            f.write("-" * 30 + "\n")
            total_baches = detection['total_potholes']
            for severity, count in detection['severity_distribution'].items():
                percentage = (count / total_baches * 100) if total_baches > 0 else 0
                action = self.classifier.severity_categories[severity]['action']
                f.write(f"• {severity}: {count} baches ({percentage:.1f}%) - {action}\n")
            f.write("\n")

            # Momentos críticos
            critical_moments = stats['critical_moments']
            if critical_moments:
                f.write("MOMENTOS CRÍTICOS (≥2 baches severos/críticos):\n")
                f.write("-" * 50 + "\n")
                for moment in critical_moments[:10]:  # Mostrar los primeros 10
                    f.write(f"• Tiempo {moment['time']:.1f}s: {moment['critical_potholes']} baches críticos/severos "
                            f"de {moment['total_potholes']} total (Ø{moment['avg_diameter']:.1f}{detection['unit']})\n")
                if len(critical_moments) > 10:
                    f.write(f"... y {len(critical_moments) - 10} momentos críticos más\n")
                f.write("\n")

            # Evaluación del estado de la carretera
            f.write("EVALUACIÓN DEL ESTADO DE LA CARRETERA:\n")
            f.write("-" * 40 + "\n")

            # Calcular índices de severidad
            critical_count = detection['severity_distribution'].get('CRITICO', 0)
            severe_count = detection['severity_distribution'].get('SEVERO', 0)
            moderate_count = detection['severity_distribution'].get('MODERADO', 0)

            critical_ratio = (critical_count + severe_count) / total_baches if total_baches > 0 else 0

            if critical_ratio > 0.5:
                road_condition = "CRÍTICO"
                priority = "INMEDIATA"
            elif critical_ratio > 0.3:
                road_condition = "MALO"
                priority = "URGENTE"
            elif critical_ratio > 0.1:
                road_condition = "REGULAR"
                priority = "ALTA"
            elif moderate_count > 0:
                road_condition = "ACEPTABLE"
                priority = "MEDIA"
            else:
                road_condition = "BUENO"
                priority = "BAJA"

            f.write(f"• Estado general: {road_condition}\n")
            f.write(f"• Prioridad de intervención: {priority}\n")
            f.write(f"• Porcentaje de baches críticos/severos: {critical_ratio * 100:.1f}%\n\n")

            # Recomendaciones específicas
            f.write("RECOMENDACIONES ESPECÍFICAS:\n")
            f.write("-" * 30 + "\n")

            if critical_count > 0:
                f.write(f" ACCIÓN INMEDIATA: Reparar {critical_count} baches críticos\n")
            if severe_count > 0:
                f.write(f"  ACCIÓN URGENTE: Reparar {severe_count} baches severos en 48h\n")
            if moderate_count > 0:
                f.write(f" PROGRAMAR: Reparar {moderate_count} baches moderados en 1-2 semanas\n")

            if len(critical_moments) > 5:
                f.write(f" INSPECCIÓN: {len(critical_moments)} zonas requieren inspección detallada\n")

            # Estimación de costos (valores aproximados)
            f.write(f"\nESTIMACIÓN DE COSTOS (aproximado):\n")
            f.write(f"• Baches críticos: ${critical_count * 200:.0f} (${200}/bache)\n")
            f.write(f"• Baches severos: ${severe_count * 150:.0f} (${150}/bache)\n")
            f.write(f"• Baches moderados: ${moderate_count * 100:.0f} (${100}/bache)\n")
            total_cost = critical_count * 200 + severe_count * 150 + moderate_count * 100
            f.write(f"• TOTAL ESTIMADO: ${total_cost:.0f}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f" Reporte comprehensivo guardado en: {save_path}")
        return save_path

    def create_severity_timeline(self, stats, save_path=None):

        if save_path is None:
            save_path = f"severity_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        # Extraer datos por frame
        frame_numbers = []
        critical_counts = []
        severe_counts = []
        moderate_counts = []
        total_counts = []

        for i, analysis in enumerate(self.frame_analyses):
            if analysis['frame_analyzed'] and analysis['potholes_detected'] > 0:
                severity_summary = analysis['severity_summary']
                frame_numbers.append(i)
                critical_counts.append(severity_summary['severity_counts'].get('CRITICO', 0))
                severe_counts.append(severity_summary['severity_counts'].get('SEVERO', 0))
                moderate_counts.append(severity_summary['severity_counts'].get('MODERADO', 0))
                total_counts.append(severity_summary['total_potholes'])

        if not frame_numbers:
            print("No hay datos suficientes para crear timeline")
            return

        # Convertir frames a tiempo (asumiendo 30 FPS)
        fps = 30
        time_axis = np.array(frame_numbers) / fps

        # Crear gráfico
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Gráfico 1: Línea de tiempo de severidad
        ax1.plot(time_axis, critical_counts, 'r-', linewidth=2, label='Críticos', marker='o', markersize=4)
        ax1.plot(time_axis, severe_counts, 'orange', linewidth=2, label='Severos', marker='s', markersize=4)
        ax1.plot(time_axis, moderate_counts, 'yellow', linewidth=2, label='Moderados', marker='^', markersize=4)

        ax1.fill_between(time_axis, critical_counts, alpha=0.3, color='red')
        ax1.fill_between(time_axis, severe_counts, alpha=0.3, color='orange')
        ax1.fill_between(time_axis, moderate_counts, alpha=0.3, color='yellow')

        ax1.set_xlabel('Tiempo (segundos)')
        ax1.set_ylabel('Número de Baches')
        ax1.set_title('Línea de Tiempo de Severidad de Baches')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gráfico 2: Total de baches por momento
        ax2.bar(time_axis, total_counts, width=0.5, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Tiempo (segundos)')
        ax2.set_ylabel('Total de Baches')
        ax2.set_title('Distribución Temporal de Detecciones')
        ax2.grid(True, alpha=0.3)

        # Marcar momentos críticos
        critical_times = [moment['time'] for moment in stats['critical_moments']]
        for ct in critical_times:
            ax1.axvline(x=ct, color='red', linestyle='--', alpha=0.7)
            ax2.axvline(x=ct, color='red', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f" Timeline de severidad guardado en: {save_path}")


def load_model_for_processing(model_path=None):
    """Cargar modelo para procesamiento"""
    if model_path and os.path.exists(model_path):
        print(f"Cargando modelo entrenado: {model_path}")
        return YOLO(model_path)
    else:
        print("Cargando modelo preentrenado YOLOv11...")
        try:
            return YOLO('yolo11m-seg.pt')
        except:
            print("Intentando YOLOv8...")
            return YOLO('yolov8m-seg.pt')


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Procesador integrado con clasificación de severidad')
    parser.add_argument('--model', '-m', help='Ruta al modelo entrenado (.pt)')
    parser.add_argument('--input', '-i', required=True, help='Video de entrada')
    parser.add_argument('--output', '-o', help='Video de salida')
    parser.add_argument('--conf', '-c', type=float, default=0.25, help='Umbral de confianza')
    parser.add_argument('--ratio', '-r', type=float, help='Ratio píxel a cm')
    parser.add_argument('--preview', '-p', action='store_true', help='Mostrar preview')
    parser.add_argument('--calibrate', action='store_true', help='Calibrar escala')

    args = parser.parse_args()

    print("=" * 70)
    print("PROCESADOR INTEGRADO CON CLASIFICACIÓN DE SEVERIDAD DE BACHES")
    print("=" * 70)

    # Configurar ratio de píxeles
    pixel_ratio = args.ratio
    if args.calibrate and not pixel_ratio:
        from pothole_classifier import calibrate_pixel_ratio
        pixel_ratio = calibrate_pixel_ratio()

    # Cargar modelo
    try:
        model = load_model_for_processing(args.model)
    except Exception as e:
        print(f" Error cargando modelo: {e}")
        return

    # Configurar salida
    if not args.output:
        input_path = Path(args.input)
        args.output = f"classified_{input_path.stem}.mp4"

    # Crear procesador integrado
    processor = IntegratedPotholeProcessor(model, pixel_ratio)

    try:
        # Procesar video
        stats = processor.process_video_with_classification(
            args.input, args.output, args.conf, args.preview
        )

        # Generar reportes y análisis
        video_name = Path(args.input).name

        # Reporte comprehensivo
        processor.generate_comprehensive_report(stats, video_name)

        # Timeline de severidad
        processor.create_severity_timeline(stats)

        # Exportar datos
        json_path = f"classified_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            # Hacer que los datos sean serializables
            serializable_stats = {
                'processing_info': stats['processing_info'],
                'detection_summary': stats['detection_summary'],
                'critical_moments': stats['critical_moments']
                # Omitir frame_analyses por ser muy grande
            }
            json.dump(serializable_stats, f, indent=2)

        print(f"\n Procesamiento completo!")
        print(f" Video clasificado: {args.output}")
        print(f" Datos exportados: {json_path}")
        print(f" Consulta los reportes generados para análisis detallado")

    except Exception as e:
        print(f" Error durante el procesamiento: {e}")


if __name__ == "__main__":
    main()