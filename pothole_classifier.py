
import warnings

warnings.filterwarnings('ignore')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, pi
import json
from datetime import datetime
from pathlib import Path
import argparse


class PotholeClassifier:

    def __init__(self, pixel_to_cm_ratio=None):

        self.pixel_to_cm_ratio = pixel_to_cm_ratio

        # Definir categorías de severidad según estándares internacionales
        # Basado en normas de ASTM D6433 y PCI (Pavement Condition Index)
        if pixel_to_cm_ratio:
            # Categorías en centímetros
            self.severity_categories = {
                'MINIMO': {'min': 0, 'max': 10, 'color': (0, 255, 0), 'action': 'Monitoreo'},
                'LEVE': {'min': 10, 'max': 30, 'color': (0, 255, 255), 'action': 'Mantenimiento preventivo'},
                'MODERADO': {'min': 30, 'max': 60, 'color': (0, 165, 255), 'action': 'Reparación programada'},
                'SEVERO': {'min': 60, 'max': 100, 'color': (0, 69, 255), 'action': 'Reparación urgente'},
                'CRITICO': {'min': 100, 'max': float('inf'), 'color': (0, 0, 255), 'action': 'Reparación inmediata'}
            }
            self.unit = 'cm'
        else:
            self.severity_categories = {
                'MINIMO': {'min': 0, 'max': 50, 'color': (0, 255, 0), 'action': 'Monitoreo'},
                'LEVE': {'min': 50, 'max': 150, 'color': (0, 255, 255), 'action': 'Mantenimiento preventivo'},
                'MODERADO': {'min': 150, 'max': 300, 'color': (0, 165, 255), 'action': 'Reparación programada'},
                'SEVERO': {'min': 300, 'max': 500, 'color': (0, 69, 255), 'action': 'Reparación urgente'},
                'CRITICO': {'min': 500, 'max': float('inf'), 'color': (0, 0, 255), 'action': 'Reparación inmediata'}
            }
            self.unit = 'px'

    def calculate_diameter_from_mask(self, mask):

        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        area_pixels = cv2.contourArea(largest_contour)

        if area_pixels < 10:  # Filtrar áreas muy pequeñas
            return None

        radius_from_area = sqrt(area_pixels / pi)
        diameter_from_area = 2 * radius_from_area

        (x, y), radius_enclosing = cv2.minEnclosingCircle(largest_contour)
        diameter_enclosing = 2 * radius_enclosing

        x, y, w, h = cv2.boundingRect(largest_contour)
        diameter_bounding = max(w, h)

        leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
        rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
        topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

        horizontal_dist = sqrt((rightmost[0] - leftmost[0]) ** 2 + (rightmost[1] - leftmost[1]) ** 2)
        vertical_dist = sqrt((bottommost[0] - topmost[0]) ** 2 + (bottommost[1] - topmost[1]) ** 2)
        diameter_extreme = max(horizontal_dist, vertical_dist)

        diameter_avg = np.mean([diameter_from_area, diameter_enclosing, diameter_bounding, diameter_extreme])

        if self.pixel_to_cm_ratio:
            diameter_real = diameter_avg * self.pixel_to_cm_ratio
            area_real = area_pixels * (self.pixel_to_cm_ratio ** 2)
        else:
            diameter_real = diameter_avg
            area_real = area_pixels

        return {
            'diameter_pixels': diameter_avg,
            'diameter_real': diameter_real,
            'area_pixels': area_pixels,
            'area_real': area_real,
            'methods': {
                'from_area': diameter_from_area,
                'enclosing_circle': diameter_enclosing,
                'bounding_box': diameter_bounding,
                'extreme_points': diameter_extreme
            },
            'contour': largest_contour,
            'center': (int(x + w / 2), int(y + h / 2)),
            'bounding_rect': (x, y, w, h)
        }

    def classify_severity(self, diameter):

        for severity, params in self.severity_categories.items():
            if params['min'] <= diameter < params['max']:
                return {
                    'severity': severity,
                    'color': params['color'],
                    'action': params['action'],
                    'range': f"{params['min']}-{params['max']} {self.unit}",
                    'diameter': diameter
                }

        return {
            'severity': 'CRITICO',
            'color': self.severity_categories['CRITICO']['color'],
            'action': self.severity_categories['CRITICO']['action'],
            'range': f">{self.severity_categories['CRITICO']['min']} {self.unit}",
            'diameter': diameter
        }

    def analyze_potholes_in_image(self, results, image_shape):

        analyses = []

        if results.masks is None:
            return analyses

        masks = results.masks.data.cpu().numpy()

        for i, mask in enumerate(masks):
            # Redimensionar máscara al tamaño de la imagen
            mask_resized = cv2.resize(mask, (image_shape[1], image_shape[0]))

            # Calcular diámetro
            diameter_info = self.calculate_diameter_from_mask(mask_resized)

            if diameter_info is None:
                continue

            # Clasificar severidad
            classification = self.classify_severity(diameter_info['diameter_real'])

            # Combinar información
            analysis = {
                'id': i,
                'diameter_info': diameter_info,
                'classification': classification,
                'confidence': float(results.boxes.conf[i]) if results.boxes is not None else 1.0
            }

            analyses.append(analysis)

        return analyses

    def draw_classified_potholes(self, image, analyses):

        annotated_image = image.copy()

        for analysis in analyses:
            diameter_info = analysis['diameter_info']
            classification = analysis['classification']

            # Obtener información
            center = diameter_info['center']
            diameter = diameter_info['diameter_real']
            severity = classification['severity']
            color = classification['color']
            confidence = analysis['confidence']

            cv2.drawContours(annotated_image, [diameter_info['contour']], -1, color, 3)

            # Dibujar centro
            cv2.circle(annotated_image, center, 5, color, -1)

            # Dibujar círculo indicando tamaño
            radius_visual = int(diameter_info['diameter_pixels'] / 2)
            cv2.circle(annotated_image, center, radius_visual, color, 2)

            # Crear etiqueta
            if self.pixel_to_cm_ratio:
                label = f"{severity}\n{diameter:.1f}cm\nConf:{confidence:.2f}"
            else:
                label = f"{severity}\n{diameter:.0f}px\nConf:{confidence:.2f}"

            # Configurar texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # Calcular tamaño del texto para el fondo
            lines = label.split('\n')
            line_height = 25
            max_width = 0

            for line in lines:
                (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
                max_width = max(max_width, text_width)

            # Posición del texto
            text_x = center[0] + 20
            text_y = center[1] - 20

            # Dibujar fondo del texto
            cv2.rectangle(annotated_image,
                          (text_x - 5, text_y - len(lines) * line_height),
                          (text_x + max_width + 10, text_y + 10),
                          (0, 0, 0), -1)

            # Dibujar texto línea por línea
            for i, line in enumerate(lines):
                y_position = text_y - (len(lines) - 1 - i) * line_height
                cv2.putText(annotated_image, line, (text_x, y_position),
                            font, font_scale, (255, 255, 255), thickness)

        return annotated_image

    def create_severity_summary(self, analyses):

        if not analyses:
            return {
                'total_potholes': 0,
                'severity_counts': {},
                'average_diameter': 0,
                'max_diameter': 0,
                'min_diameter': 0,
                'total_area': 0,
                'severity_distribution': {}
            }

        # Contar por severidad
        severity_counts = {}
        diameters = []
        areas = []

        for analysis in analyses:
            severity = analysis['classification']['severity']
            diameter = analysis['diameter_info']['diameter_real']
            area = analysis['diameter_info']['area_real']

            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            diameters.append(diameter)
            areas.append(area)

        # Calcular distribución porcentual
        total_potholes = len(analyses)
        severity_distribution = {
            severity: (count / total_potholes) * 100
            for severity, count in severity_counts.items()
        }

        return {
            'total_potholes': total_potholes,
            'severity_counts': severity_counts,
            'average_diameter': np.mean(diameters),
            'max_diameter': max(diameters),
            'min_diameter': min(diameters),
            'total_area': sum(areas),
            'severity_distribution': severity_distribution,
            'diameters': diameters,
            'areas': areas
        }

    def plot_severity_analysis(self, summary, save_path=None):

        if summary['total_potholes'] == 0:
            print("No hay baches para analizar")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Gráfico 1: Distribución por severidad
        severities = list(summary['severity_counts'].keys())
        counts = list(summary['severity_counts'].values())
        colors = [self.severity_categories[sev]['color'] for sev in severities]
        colors_rgb = [(c[2] / 255, c[1] / 255, c[0] / 255) for c in colors]  # BGR to RGB

        axes[0, 0].pie(counts, labels=severities, colors=colors_rgb, autopct='%1.1f%%')
        axes[0, 0].set_title('Distribución de Baches por Severidad')

        # Gráfico 2: Histograma de diámetros
        axes[0, 1].hist(summary['diameters'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel(f'Diámetro ({self.unit})')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].set_title('Distribución de Diámetros')
        axes[0, 1].axvline(summary['average_diameter'], color='red', linestyle='--',
                           label=f'Promedio: {summary["average_diameter"]:.1f}{self.unit}')
        axes[0, 1].legend()

        # Gráfico 3: Barras por severidad
        axes[1, 0].bar(severities, counts, color=colors_rgb)
        axes[1, 0].set_xlabel('Severidad')
        axes[1, 0].set_ylabel('Cantidad de Baches')
        axes[1, 0].set_title('Cantidad por Categoría de Severidad')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Gráfico 4: Relación área vs diámetro
        areas = summary['areas']
        diameters = summary['diameters']
        axes[1, 1].scatter(diameters, areas, alpha=0.6, s=50)
        axes[1, 1].set_xlabel(f'Diámetro ({self.unit})')
        axes[1, 1].set_ylabel(f'Área ({self.unit}²)')
        axes[1, 1].set_title('Relación Diámetro vs Área')

        # Línea de tendencia
        z = np.polyfit(diameters, areas, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(diameters, p(diameters), "r--", alpha=0.8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Análisis de severidad guardado en: {save_path}")

        plt.show()

    def generate_detailed_report(self, analyses, summary, image_name, save_path=None):

        if save_path is None:
            save_path = f"severity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("REPORTE DETALLADO DE CLASIFICACIÓN DE BACHES POR SEVERIDAD\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Imagen analizada: {image_name}\n")
            f.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Unidad de medida: {self.unit}\n\n")

            # Resumen ejecutivo
            f.write("RESUMEN EJECUTIVO:\n")
            f.write("-" * 20 + "\n")
            f.write(f"• Total de baches detectados: {summary['total_potholes']}\n")
            f.write(f"• Diámetro promedio: {summary['average_diameter']:.2f} {self.unit}\n")
            f.write(f"• Diámetro máximo: {summary['max_diameter']:.2f} {self.unit}\n")
            f.write(f"• Área total afectada: {summary['total_area']:.2f} {self.unit}²\n\n")

            # Distribución por severidad
            f.write("DISTRIBUCIÓN POR SEVERIDAD:\n")
            f.write("-" * 30 + "\n")
            for severity, count in summary['severity_counts'].items():
                percentage = summary['severity_distribution'][severity]
                action = self.severity_categories[severity]['action']
                f.write(f"• {severity}: {count} baches ({percentage:.1f}%) - {action}\n")
            f.write("\n")

            # Criterios de clasificación
            f.write("CRITERIOS DE CLASIFICACIÓN:\n")
            f.write("-" * 30 + "\n")
            for severity, params in self.severity_categories.items():
                if params['max'] == float('inf'):
                    range_str = f">{params['min']} {self.unit}"
                else:
                    range_str = f"{params['min']}-{params['max']} {self.unit}"
                f.write(f"• {severity}: {range_str} - {params['action']}\n")
            f.write("\n")

            # Análisis detallado por bache
            if analyses:
                f.write("ANÁLISIS DETALLADO POR BACHE:\n")
                f.write("-" * 35 + "\n")
                for i, analysis in enumerate(analyses, 1):
                    diameter = analysis['diameter_info']['diameter_real']
                    area = analysis['diameter_info']['area_real']
                    severity = analysis['classification']['severity']
                    confidence = analysis['confidence']
                    action = analysis['classification']['action']

                    f.write(f"Bache #{i}:\n")
                    f.write(f"  - Diámetro: {diameter:.2f} {self.unit}\n")
                    f.write(f"  - Área: {area:.2f} {self.unit}²\n")
                    f.write(f"  - Severidad: {severity}\n")
                    f.write(f"  - Confianza: {confidence:.2f}\n")
                    f.write(f"  - Acción recomendada: {action}\n\n")

            # Recomendaciones
            f.write("RECOMENDACIONES:\n")
            f.write("-" * 17 + "\n")

            # Generar recomendaciones basadas en la distribución
            critical_count = summary['severity_counts'].get('CRITICO', 0)
            severe_count = summary['severity_counts'].get('SEVERO', 0)
            moderate_count = summary['severity_counts'].get('MODERADO', 0)

            if critical_count > 0:
                f.write(f" URGENTE: {critical_count} baches críticos requieren reparación inmediata\n")
            if severe_count > 0:
                f.write(f"  IMPORTANTE: {severe_count} baches severos requieren reparación urgente\n")
            if moderate_count > 0:
                f.write(f" PROGRAMAR: {moderate_count} baches moderados para reparación programada\n")

            # Prioridad de intervención
            total_severe = critical_count + severe_count
            if total_severe > summary['total_potholes'] * 0.3:
                f.write("\n PRIORIDAD ALTA: Más del 30% de baches requieren atención urgente\n")
            elif total_severe > 0:
                f.write("\n PRIORIDAD MEDIA: Algunos baches requieren atención urgente\n")
            else:
                f.write("\n PRIORIDAD BAJA: Mantenimiento preventivo recomendado\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f" Reporte detallado guardado en: {save_path}")
        return save_path

    def export_data_to_json(self, analyses, summary, save_path=None):

        if save_path is None:
            save_path = f"pothole_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Preparar datos para JSON
        data = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'classifier_unit': self.unit,
                'pixel_to_cm_ratio': self.pixel_to_cm_ratio,
                'total_potholes': summary['total_potholes']
            },
            'summary': summary,
            'potholes': []
        }

        # Convertir análisis a formato serializable
        for analysis in analyses:
            pothole_data = {
                'id': analysis['id'],
                'diameter': float(analysis['diameter_info']['diameter_real']),
                'area': float(analysis['diameter_info']['area_real']),
                'severity': analysis['classification']['severity'],
                'confidence': float(analysis['confidence']),
                'action_required': analysis['classification']['action'],
                'center_coordinates': analysis['diameter_info']['center'],
                'bounding_rect': analysis['diameter_info']['bounding_rect']
            }
            data['potholes'].append(pothole_data)

        # Guardar JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f" Datos exportados a JSON: {save_path}")
        return save_path


def calibrate_pixel_ratio():

    print("=" * 50)
    print("CALIBRACIÓN DE ESCALA PÍXEL-CENTÍMETRO")
    print("=" * 50)
    print("\nPara obtener medidas precisas, necesitas calibrar la escala.")
    print("Opciones:")
    print("1. Usar una referencia conocida en la imagen")
    print("2. Estimar basado en altura de cámara")
    print("3. Usar configuración estándar")
    print("4. Saltar calibración (usar píxeles)")

    choice = input("\nSelecciona una opción (1-4): ").strip()

    if choice == '1':
        print("\nMide un objeto conocido en tu imagen (ej: línea de carretera, objeto de referencia)")
        pixels = float(input("Longitud en píxeles: "))
        cm = float(input("Longitud real en centímetros: "))
        ratio = cm / pixels
        print(f"Ratio calculado: {ratio:.4f} cm/píxel")
        return ratio

    elif choice == '2':
        print("\nEstimación basada en altura de cámara:")
        height = float(input("Altura de la cámara (metros): "))
        # Fórmula aproximada: a mayor altura, mayor cobertura por píxel
        estimated_ratio = height * 0.3  # Aproximación empírica
        print(f"Ratio estimado: {estimated_ratio:.4f} cm/píxel")
        return estimated_ratio

    elif choice == '3':
        print("Usando configuración estándar para cámaras de vehículo (0.2 cm/píxel)")
        return 0.2

    else:
        print("Usando píxeles como unidad de medida")
        return None


def main():
    parser = argparse.ArgumentParser(description='Clasificador de severidad de baches')
    parser.add_argument('--calibrate', '-c', action='store_true', help='Calibrar escala píxel-centímetro')
    parser.add_argument('--ratio', '-r', type=float, help='Ratio píxel a cm (opcional)')
    parser.add_argument('--demo', '-d', action='store_true', help='Ejecutar demo con datos sintéticos')

    args = parser.parse_args()

    print("=" * 60)
    print("CLASIFICADOR DE SEVERIDAD DE BACHES POR DIÁMETRO")
    print("=" * 60)

    # Configurar ratio
    pixel_ratio = None
    if args.calibrate:
        pixel_ratio = calibrate_pixel_ratio()
    elif args.ratio:
        pixel_ratio = args.ratio
        print(f"Usando ratio especificado: {pixel_ratio} cm/píxel")

    # Crear clasificador
    classifier = PotholeClassifier(pixel_to_cm_ratio=pixel_ratio)

    if args.demo:
        # Demo con datos sintéticos
        print("\n Ejecutando demo con datos sintéticos...")

        # Simular análisis de baches
        demo_analyses = []
        demo_diameters = [5, 15, 25, 45, 75, 120]  # Diferentes severidades

        for i, diameter in enumerate(demo_diameters):
            # Simular información de diámetro
            if pixel_ratio:
                diameter_pixels = diameter / pixel_ratio
            else:
                diameter_pixels = diameter * 10  # Para demo en píxeles

            diameter_info = {
                'diameter_pixels': diameter_pixels,
                'diameter_real': diameter if pixel_ratio else diameter * 10,
                'area_pixels': np.pi * (diameter_pixels / 2) ** 2,
                'area_real': np.pi * (diameter / 2) ** 2 if pixel_ratio else np.pi * ((diameter * 10) / 2) ** 2,
                'center': (100 + i * 50, 100),
                'contour': np.array([[[100 + i * 50, 100]]]),  # Punto simple para demo
                'bounding_rect': (90 + i * 50, 90, 20, 20)
            }

            # Clasificar
            classification = classifier.classify_severity(diameter_info['diameter_real'])

            analysis = {
                'id': i,
                'diameter_info': diameter_info,
                'classification': classification,
                'confidence': 0.9 - i * 0.1
            }

            demo_analyses.append(analysis)

        # Crear resumen
        summary = classifier.create_severity_summary(demo_analyses)

        # Mostrar resultados
        print(f"\n Resultados del demo:")
        print(f"Total de baches: {summary['total_potholes']}")
        print(f"Diámetro promedio: {summary['average_diameter']:.1f} {classifier.unit}")

        print(f"\nDistribución por severidad:")
        for severity, count in summary['severity_counts'].items():
            print(f"  {severity}: {count} baches")

        # Generar gráficos
        classifier.plot_severity_analysis(summary, 'demo_severity_analysis.png')

        # Generar reporte
        classifier.generate_detailed_report(demo_analyses, summary, 'demo_image.jpg', 'demo_severity_report.txt')

        # Exportar a JSON
        classifier.export_data_to_json(demo_analyses, summary, 'demo_pothole_data.json')

        print("\n Demo completado! Archivos generados:")
        print("  - demo_severity_analysis.png")
        print("  - demo_severity_report.txt")
        print("  - demo_pothole_data.json")

    else:
        print("\n Uso del clasificador:")
        print("Este clasificador se integra con los scripts de procesamiento de video.")
        print("Para usarlo:")
        print("1. python pothole_classifier.py --demo  # Ver demo")
        print("2. python pothole_classifier.py --calibrate  # Calibrar escala")
        print("3. Integrar con process_video.py para clasificación automática")

        # Mostrar categorías configuradas
        print(f"\n Categorías de severidad configuradas ({classifier.unit}):")
        for severity, params in classifier.severity_categories.items():
            if params['max'] == float('inf'):
                range_str = f">{params['min']} {classifier.unit}"
            else:
                range_str = f"{params['min']}-{params['max']} {classifier.unit}"
            print(f"  {severity}: {range_str} - {params['action']}")


if __name__ == "__main__":
    main()