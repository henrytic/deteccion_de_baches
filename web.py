#!/usr/bin/env python3
"""
Interfaz Gráfica Interactiva para Análisis de Baches
Aplicación web usando Streamlit para análisis completo de videos
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
import io
import zipfile
from datetime import datetime
import json
from pathlib import Path
import base64
import time

# Configurar página
st.set_page_config(
    page_title="🕳️ Analizador Inteligente de Baches",
    page_icon="🕳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }

    .severity-critical {
        background: linear-gradient(135deg, #ff416c 0%, #ff4757 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }

    .severity-severe {
        background: linear-gradient(135deg, #ff9500 0%, #ff6b35 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }

    .severity-moderate {
        background: linear-gradient(135deg, #ffc048 0%, #ff9500 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }

    .severity-good {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }

    .status-panel {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }

    .upload-zone {
        border: 2px dashed #007bff;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Importaciones de nuestros módulos (simuladas para el ejemplo)
# En el archivo real, estas serían las importaciones reales
@st.cache_resource
def load_yolo_model(model_choice):
    """Cargar modelo YOLO con cache"""
    try:
        from ultralytics import YOLO
        if model_choice == "Automático":
            model = YOLO('yolo11m-seg.pt')
        elif model_choice == "Rápido":
            model = YOLO('yolo11s-seg.pt')
        elif model_choice == "Preciso":
            model = YOLO('yolo11l-seg.pt')
        else:
            model = YOLO('yolo11m-seg.pt')
        return model
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None


class PotholeAnalyzerGUI:
    """Clase principal para la interfaz gráfica"""

    def __init__(self):
        self.model = None
        self.pixel_ratio = None
        self.severity_categories = {
            'CRÍTICO': {'color': '#FF0000', 'min_diameter': 100},
            'SEVERO': {'color': '#FF4500', 'min_diameter': 60},
            'MODERADO': {'color': '#FFA500', 'min_diameter': 30},
            'LEVE': {'color': '#FFFF00', 'min_diameter': 10},
            'MÍNIMO': {'color': '#00FF00', 'min_diameter': 0}
        }

    def render_header(self):
        """Renderizar encabezado de la aplicación"""
        st.markdown('<h1 class="main-header">🕳️ Analizador Inteligente de Baches</h1>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                         border-radius: 10px; color: white; margin-bottom: 2rem;">
                <h4>🚀 Análisis Automático con IA • 📊 Reportes Detallados • 🎯 Clasificación por Severidad</h4>
            </div>
            """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Renderizar panel lateral de configuración"""
        st.sidebar.header("⚙️ Configuración")

        # Configuración del modelo
        st.sidebar.subheader("🤖 Modelo de IA")
        model_choice = st.sidebar.selectbox(
            "Seleccionar modelo:",
            ["Automático", "Rápido", "Preciso"],
            help="Automático: Balance óptimo, Rápido: Procesamiento veloz, Preciso: Mayor exactitud"
        )

        # Configuración de escala
        st.sidebar.subheader("📏 Calibración de Escala")
        use_real_measurements = st.sidebar.checkbox("Usar medidas reales (cm)",
                                                    help="Calibrar para obtener medidas en centímetros")

        if use_real_measurements:
            calibration_method = st.sidebar.radio(
                "Método de calibración:",
                ["Manual", "Altura de cámara", "Estándar"],
                help="Manual: Especificar ratio, Altura: Basado en altura de cámara, Estándar: 0.2 cm/píxel"
            )

            if calibration_method == "Manual":
                pixels = st.sidebar.number_input("Píxeles de referencia:", min_value=1, value=100)
                cm = st.sidebar.number_input("Centímetros reales:", min_value=0.1, value=20.0)
                self.pixel_ratio = cm / pixels
            elif calibration_method == "Altura de cámara":
                height = st.sidebar.number_input("Altura de cámara (metros):", min_value=0.5, value=2.0)
                self.pixel_ratio = height * 0.3  # Fórmula aproximada
            else:
                self.pixel_ratio = 0.2

            st.sidebar.info(f"Ratio: {self.pixel_ratio:.4f} cm/píxel")
        else:
            self.pixel_ratio = None

        # Configuración de detección
        st.sidebar.subheader("🎯 Parámetros de Detección")
        confidence = st.sidebar.slider("Umbral de confianza:", 0.1, 0.9, 0.25,
                                       help="Mayor valor = menos detecciones pero más precisas")

        # Configuración de severidad personalizada
        st.sidebar.subheader("⚠️ Umbrales de Severidad")
        if st.sidebar.checkbox("Personalizar umbrales"):
            unit = "cm" if self.pixel_ratio else "píxeles"

            critical_threshold = st.sidebar.number_input(f"Crítico (>{unit}):", value=100 if self.pixel_ratio else 500)
            severe_threshold = st.sidebar.number_input(f"Severo (>{unit}):", value=60 if self.pixel_ratio else 300)
            moderate_threshold = st.sidebar.number_input(f"Moderado (>{unit}):", value=30 if self.pixel_ratio else 150)
            mild_threshold = st.sidebar.number_input(f"Leve (>{unit}):", value=10 if self.pixel_ratio else 50)

        return model_choice, confidence

    def render_upload_section(self):
        """Renderizar sección de carga de archivos"""
        st.header("📤 Cargar Video para Análisis")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            <div class="upload-zone">
                <h4>🎥 Arrastra tu video aquí o haz clic para seleccionar</h4>
                <p>Formatos soportados: MP4, AVI, MOV, MKV</p>
                <p>Tamaño máximo: 500MB</p>
            </div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Seleccionar video",
                type=['mp4', 'avi', 'mov', 'mkv'],
                label_visibility="collapsed"
            )

        with col2:
            if uploaded_file is not None:
                st.success("✅ Video cargado correctamente")

                # Mostrar información del archivo
                file_details = {
                    "Nombre": uploaded_file.name,
                    "Tamaño": f"{uploaded_file.size / (1024 * 1024):.2f} MB",
                    "Tipo": uploaded_file.type
                }

                for key, value in file_details.items():
                    st.info(f"**{key}:** {value}")

        return uploaded_file

    def process_video(self, uploaded_file, model_choice, confidence):
        """Procesar video y generar análisis"""
        if uploaded_file is None:
            return None

        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name

        # Simular procesamiento (en la implementación real, aquí iría el código de procesamiento)
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Simulación de procesamiento
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 20:
                status_text.text("🤖 Cargando modelo de IA...")
            elif i < 40:
                status_text.text("🎥 Analizando frames del video...")
            elif i < 70:
                status_text.text("🕳️ Detectando y clasificando baches...")
            elif i < 90:
                status_text.text("📊 Generando reportes y gráficos...")
            else:
                status_text.text("✅ Finalizando análisis...")

            time.sleep(0.05)  # Simular procesamiento

        # Limpiar elementos de progreso
        progress_bar.empty()
        status_text.empty()

        # Generar datos simulados para el demo
        analysis_results = self.generate_demo_results()

        # Limpiar archivo temporal
        os.unlink(temp_video_path)

        return analysis_results

    def generate_demo_results(self):
        """Generar resultados de demostración"""
        # Datos simulados para el demo
        return {
            'total_potholes': 47,
            'video_duration': 120.5,
            'processing_time': 45.2,
            'severity_distribution': {
                'CRÍTICO': 5,
                'SEVERO': 8,
                'MODERADO': 15,
                'LEVE': 12,
                'MÍNIMO': 7
            },
            'diameter_stats': {
                'mean': 35.4,
                'max': 125.8,
                'min': 8.2,
                'std': 24.7
            },
            'critical_moments': [
                {'time': 23.5, 'potholes': 3, 'severity': 'CRÍTICO'},
                {'time': 67.2, 'potholes': 2, 'severity': 'SEVERO'},
                {'time': 89.1, 'potholes': 4, 'severity': 'CRÍTICO'},
                {'time': 105.8, 'potholes': 2, 'severity': 'SEVERO'}
            ],
            'timeline_data': np.random.poisson(2, 120),  # Simulación de detecciones por segundo
            'diameter_distribution': np.random.gamma(2, 15, 47),  # Distribución de diámetros
            'cost_estimation': {
                'critical': 5 * 200,
                'severe': 8 * 150,
                'moderate': 15 * 100,
                'total': 5 * 200 + 8 * 150 + 15 * 100
            }
        }

    def render_results_dashboard(self, results):
        """Renderizar dashboard de resultados"""
        if results is None:
            return

        st.header("📊 Dashboard de Resultados")

        # Métricas principales
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{results['total_potholes']}</h3>
                <p>🕳️ Baches Detectados</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="severity-critical">
                <h3>{results['severity_distribution']['CRÍTICO']}</h3>
                <p>🚨 Críticos</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="severity-severe">
                <h3>{results['severity_distribution']['SEVERO']}</h3>
                <p>⚠️ Severos</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            unit = "cm" if self.pixel_ratio else "px"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{results['diameter_stats']['mean']:.1f}{unit}</h3>
                <p>📏 Diámetro Promedio</p>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="severity-good">
                <h3>${results['cost_estimation']['total']:,}</h3>
                <p>💰 Costo Estimado</p>
            </div>
            """, unsafe_allow_html=True)

        # Estado general de la carretera
        critical_ratio = (results['severity_distribution']['CRÍTICO'] +
                          results['severity_distribution']['SEVERO']) / results['total_potholes']

        if critical_ratio > 0.3:
            road_status = "🔴 CRÍTICO - Requiere intervención inmediata"
            status_class = "severity-critical"
        elif critical_ratio > 0.15:
            road_status = "🟠 MALO - Requiere reparación urgente"
            status_class = "severity-severe"
        elif critical_ratio > 0.05:
            road_status = "🟡 REGULAR - Programar mantenimiento"
            status_class = "severity-moderate"
        else:
            road_status = "🟢 BUENO - Mantenimiento preventivo"
            status_class = "severity-good"

        st.markdown(f"""
        <div class="{status_class}">
            <h3>Estado General de la Carretera</h3>
            <h2>{road_status}</h2>
        </div>
        """, unsafe_allow_html=True)

    def render_interactive_charts(self, results):
        """Renderizar gráficos interactivos"""
        st.header("📈 Análisis Visual Interactivo")

        tab1, tab2, tab3, tab4 = st.tabs(["🥧 Distribución", "📊 Timeline", "🗺️ Mapa de Calor", "📏 Estadísticas"])

        with tab1:
            # Gráfico de distribución por severidad
            col1, col2 = st.columns(2)

            with col1:
                # Gráfico de pastel
                severities = list(results['severity_distribution'].keys())
                counts = list(results['severity_distribution'].values())
                colors = ['#FF0000', '#FF4500', '#FFA500', '#FFFF00', '#00FF00']

                fig_pie = go.Figure(data=[go.Pie(
                    labels=severities,
                    values=counts,
                    marker=dict(colors=colors),
                    hole=0.4,
                    textinfo='label+percent+value',
                    textfont=dict(size=12)
                )])

                fig_pie.update_layout(
                    title=dict(text="Distribución por Severidad", x=0.5),
                    font=dict(size=14),
                    height=400
                )

                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                # Gráfico de barras
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=severities,
                        y=counts,
                        marker=dict(color=colors),
                        text=counts,
                        textposition='auto'
                    )
                ])

                fig_bar.update_layout(
                    title=dict(text="Cantidad por Severidad", x=0.5),
                    xaxis=dict(title="Severidad"),
                    yaxis=dict(title="Cantidad de Baches"),
                    font=dict(size=14),
                    height=400
                )

                st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            # Timeline de detecciones
            timeline_data = results['timeline_data']
            time_axis = np.arange(len(timeline_data))

            fig_timeline = go.Figure()

            # Línea principal
            fig_timeline.add_trace(go.Scatter(
                x=time_axis,
                y=timeline_data,
                mode='lines+markers',
                name='Detecciones por segundo',
                line=dict(color='#007bff', width=2),
                marker=dict(size=4)
            ))

            # Marcar momentos críticos
            for moment in results['critical_moments']:
                fig_timeline.add_vline(
                    x=moment['time'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"🚨 {moment['potholes']} baches",
                    annotation_position="top"
                )

            # Área sombreada para promedio
            avg_detections = np.mean(timeline_data)
            fig_timeline.add_hline(
                y=avg_detections,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Promedio: {avg_detections:.1f}"
            )

            fig_timeline.update_layout(
                title=dict(text="Timeline de Detecciones", x=0.5),
                xaxis=dict(title="Tiempo (segundos)"),
                yaxis=dict(title="Baches detectados"),
                height=400,
                showlegend=True
            )

            st.plotly_chart(fig_timeline, use_container_width=True)

            # Tabla de momentos críticos
            if results['critical_moments']:
                st.subheader("⚠️ Momentos Críticos Detectados")

                critical_df = pd.DataFrame(results['critical_moments'])
                critical_df['time'] = critical_df['time'].apply(lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}")
                critical_df.columns = ['Tiempo (mm:ss)', 'Baches', 'Severidad']

                st.dataframe(critical_df, use_container_width=True)

        with tab3:
            # Mapa de calor simulado
            st.subheader("🗺️ Mapa de Calor de Severidad")

            # Generar datos de mapa de calor simulado
            np.random.seed(42)  # Para resultados consistentes
            heatmap_data = np.random.rand(20, 30) * 50

            # Agregar algunos puntos críticos
            heatmap_data[5:8, 10:15] = np.random.rand(3, 5) * 30 + 70  # Zona crítica
            heatmap_data[15:18, 5:10] = np.random.rand(3, 5) * 20 + 60  # Zona severa

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                colorscale=[
                    [0.0, '#00FF00'],  # Verde (bueno)
                    [0.2, '#FFFF00'],  # Amarillo (leve)
                    [0.4, '#FFA500'],  # Naranja (moderado)
                    [0.7, '#FF4500'],  # Naranja oscuro (severo)
                    [1.0, '#FF0000']  # Rojo (crítico)
                ],
                colorbar=dict(
                    title=dict(text="Severidad (%)", side="right")
                )
            ))

            fig_heatmap.update_layout(
                title=dict(text="Mapa de Calor - Distribución de Severidad en la Carretera", x=0.5),
                xaxis=dict(title="Posición Lateral (m)"),
                yaxis=dict(title="Posición Longitudinal (m)"),
                height=500
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Leyenda del mapa de calor
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **🎨 Leyenda del Mapa de Calor:**
                - 🟢 **Verde**: Estado bueno (0-20%)
                - 🟡 **Amarillo**: Daño leve (20-40%)
                - 🟠 **Naranja**: Daño moderado (40-70%)
                - 🔴 **Rojo**: Daño severo/crítico (70-100%)
                """)

            with col2:
                st.markdown("""
                **📍 Zonas de Atención:**
                - **Zona 1** (10-15m, 5-8m): Crítica - Reparación inmediata
                - **Zona 2** (5-10m, 15-18m): Severa - Reparación urgente
                - **Otras áreas**: Monitoreo continuo
                """)

        with tab4:
            # Estadísticas detalladas
            col1, col2 = st.columns(2)

            with col1:
                # Histograma de diámetros
                diameters = results['diameter_distribution']
                unit = "cm" if self.pixel_ratio else "px"

                fig_hist = go.Figure(data=[go.Histogram(
                    x=diameters,
                    nbinsx=20,
                    marker=dict(color='skyblue', opacity=0.7),
                    name='Distribución'
                )])

                # Línea de promedio
                mean_diameter = np.mean(diameters)
                fig_hist.add_vline(
                    x=mean_diameter,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Promedio: {mean_diameter:.1f}{unit}",
                    annotation_position="top right"
                )

                fig_hist.update_layout(
                    title=dict(text=f"Distribución de Diámetros ({unit})", x=0.5),
                    xaxis=dict(title=f"Diámetro ({unit})"),
                    yaxis=dict(title="Frecuencia"),
                    height=400,
                    showlegend=False
                )

                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                # Estadísticas numéricas
                unit = "cm" if self.pixel_ratio else "px"
                stats_data = {
                    'Métrica': ['Total de Baches', 'Diámetro Promedio', 'Diámetro Máximo',
                                'Diámetro Mínimo', 'Desviación Estándar', 'Duración del Video',
                                'Tiempo de Procesamiento', 'Baches por Segundo'],
                    'Valor': [
                        f"{results['total_potholes']} baches",
                        f"{results['diameter_stats']['mean']:.1f} {unit}",
                        f"{results['diameter_stats']['max']:.1f} {unit}",
                        f"{results['diameter_stats']['min']:.1f} {unit}",
                        f"{results['diameter_stats']['std']:.1f} {unit}",
                        f"{results['video_duration']:.1f} segundos",
                        f"{results['processing_time']:.1f} segundos",
                        f"{results['total_potholes'] / results['video_duration']:.2f}"
                    ]
                }

                stats_df = pd.DataFrame(stats_data)
                st.subheader("📊 Estadísticas Detalladas")
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

    def render_download_section(self, results):
        """Renderizar sección de descarga de reportes"""
        st.header("📥 Descargar Reportes")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Generar reporte PDF simulado
            pdf_report = self.generate_pdf_report(results)
            st.download_button(
                label="📄 Descargar Reporte PDF",
                data=pdf_report,
                file_name=f"reporte_baches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                help="Reporte completo en formato PDF"
            )

        with col2:
            # Generar datos JSON
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="💾 Descargar Datos JSON",
                data=json_data,
                file_name=f"datos_baches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Datos en formato JSON para integración"
            )

        with col3:
            # Generar Excel
            excel_data = self.generate_excel_report(results)
            st.download_button(
                label="📊 Descargar Excel",
                data=excel_data,
                file_name=f"analisis_baches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Datos en formato Excel"
            )

        # Video procesado (simulado)
        st.subheader("🎥 Video Procesado")
        st.info(
            "📹 El video con las anotaciones de baches estará disponible para descarga una vez completado el procesamiento.")

        # Botón para video procesado (simulado)
        if st.button("📥 Descargar Video Procesado", disabled=True):
            st.info("Funcionalidad disponible en la versión completa")

    def generate_pdf_report(self, results):
        """Generar reporte PDF simulado"""
        # En la implementación real, aquí se generaría un PDF con reportlab o similar
        report_content = f"""
        REPORTE DE ANÁLISIS DE BACHES
        =============================

        Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        RESUMEN EJECUTIVO:
        - Total de baches detectados: {results['total_potholes']}
        - Baches críticos: {results['severity_distribution']['CRÍTICO']}
        - Baches severos: {results['severity_distribution']['SEVERO']}
        - Costo estimado de reparación: ${results['cost_estimation']['total']:,}

        DISTRIBUCIÓN POR SEVERIDAD:
        - Crítico: {results['severity_distribution']['CRÍTICO']} baches
        - Severo: {results['severity_distribution']['SEVERO']} baches
        - Moderado: {results['severity_distribution']['MODERADO']} baches
        - Leve: {results['severity_distribution']['LEVE']} baches
        - Mínimo: {results['severity_distribution']['MÍNIMO']} baches
        """

        return report_content.encode('utf-8')

    def generate_excel_report(self, results):
        """Generar reporte Excel simulado"""
        # Crear DataFrame con los datos
        data = []
        for severity, count in results['severity_distribution'].items():
            data.append({
                'Severidad': severity,
                'Cantidad': count,
                'Porcentaje': f"{(count / results['total_potholes'] * 100):.1f}%"
            })

        df = pd.DataFrame(data)

        # Convertir a Excel en memoria
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Distribución Severidad', index=False)

            # Agregar hoja de estadísticas
            stats_data = {
                'Métrica': ['Total Baches', 'Diámetro Promedio', 'Diámetro Máximo', 'Costo Total'],
                'Valor': [results['total_potholes'], results['diameter_stats']['mean'],
                          results['diameter_stats']['max'], results['cost_estimation']['total']]
            }
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Estadísticas', index=False)

        output.seek(0)
        return output.getvalue()

    def run(self):
        """Ejecutar la aplicación principal"""
        # Renderizar componentes principales
        self.render_header()

        # Panel lateral de configuración
        model_choice, confidence = self.render_sidebar()

        # Cargar modelo (simulado)
        if self.model is None:
            with st.spinner("🤖 Cargando modelo de IA..."):
                self.model = load_yolo_model(model_choice)

        # Sección principal
        uploaded_file = self.render_upload_section()

        # Botón de procesamiento
        if uploaded_file is not None:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🚀 Iniciar Análisis", type="primary", use_container_width=True):
                    st.session_state.processing = True

        # Procesar y mostrar resultados
        if uploaded_file is not None and st.session_state.get('processing', False):
            with st.container():
                results = self.process_video(uploaded_file, model_choice, confidence)

                if results:
                    st.session_state.results = results
                    st.session_state.processing = False
                    st.success("✅ ¡Análisis completado exitosamente!")
                    st.balloons()

        # Mostrar dashboard de resultados
        if st.session_state.get('results'):
            results = st.session_state.results

            # Dashboard principal
            self.render_results_dashboard(results)

            # Gráficos interactivos
            self.render_interactive_charts(results)

            # Sección de descarga
            self.render_download_section(results)

            # Panel de acciones recomendadas
            self.render_action_panel(results)

    def render_action_panel(self, results):
        """Renderizar panel de acciones recomendadas"""
        st.header("🎯 Acciones Recomendadas")

        critical_count = results['severity_distribution']['CRÍTICO']
        severe_count = results['severity_distribution']['SEVERO']
        moderate_count = results['severity_distribution']['MODERADO']

        # Prioridades
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🚨 Prioridad Inmediata")
            if critical_count > 0:
                st.error(f"**{critical_count} baches críticos** requieren reparación inmediata (24-48 horas)")
                st.markdown("- Cerrar al tráfico si es necesario")
                st.markdown("- Reparación con concreto asfáltico")
                st.markdown(f"- Costo estimado: ${critical_count * 200:,}")
            else:
                st.success("✅ No hay baches críticos detectados")

            st.subheader("⚠️ Prioridad Urgente")
            if severe_count > 0:
                st.warning(f"**{severe_count} baches severos** requieren reparación en 1-2 semanas")
                st.markdown("- Señalización temporal")
                st.markdown("- Programar cuadrilla de reparación")
                st.markdown(f"- Costo estimado: ${severe_count * 150:,}")
            else:
                st.success("✅ No hay baches severos detectados")

        with col2:
            st.subheader("📋 Programación de Mantenimiento")
            if moderate_count > 0:
                st.info(f"**{moderate_count} baches moderados** para mantenimiento programado")
                st.markdown("- Incluir en plan mensual")
                st.markdown("- Sellado y bacheo preventivo")
                st.markdown(f"- Costo estimado: ${moderate_count * 100:,}")

            # Cronograma sugerido
            st.subheader("📅 Cronograma Sugerido")
            timeline_data = []

            if critical_count > 0:
                timeline_data.append(
                    {"Acción": "Reparar baches críticos", "Plazo": "24-48 horas", "Prioridad": "🔴 Inmediata"})
            if severe_count > 0:
                timeline_data.append(
                    {"Acción": "Reparar baches severos", "Plazo": "1-2 semanas", "Prioridad": "🟠 Urgente"})
            if moderate_count > 0:
                timeline_data.append({"Acción": "Mantenimiento programado", "Plazo": "1 mes", "Prioridad": "🟡 Media"})

            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                st.dataframe(timeline_df, use_container_width=True, hide_index=True)

        # Resumen de costos
        st.subheader("💰 Resumen de Inversión")

        total_cost = results['cost_estimation']['total']

        cost_breakdown = pd.DataFrame({
            'Categoría': ['Baches Críticos', 'Baches Severos', 'Baches Moderados', 'TOTAL'],
            'Cantidad': [critical_count, severe_count, moderate_count, critical_count + severe_count + moderate_count],
            'Costo Unitario': ['$200', '$150', '$100', '-'],
            'Costo Total': [f"${critical_count * 200:,}", f"${severe_count * 150:,}",
                            f"${moderate_count * 100:,}", f"${total_cost:,}"]
        })

        st.dataframe(cost_breakdown, use_container_width=True, hide_index=True)

        # Comparación con costos de no reparar
        st.subheader("⚖️ Análisis Costo-Beneficio")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **💸 Costo de Reparación Inmediata:**
            - Inversión total: ${total_cost:,}
            - Vida útil extendida: 5-7 años
            - Prevención de daños mayores
            """)

        with col2:
            potential_cost = total_cost * 3  # Estimación de costo futuro
            st.markdown(f"""
            **📈 Costo de No Reparar:**
            - Deterioro acelerado: ${potential_cost:,}
            - Daños a vehículos: Responsabilidad civil
            - Reducción de vida útil: 70%
            """)

        # Recomendación final
        savings = potential_cost - total_cost
        st.success(f"""
        **💡 Recomendación:** Reparar inmediatamente puede ahorrar hasta **${savings:,}** 
        en costos futuros y evitar responsabilidades por daños a vehículos.
        """)


def main():
    """Función principal de la aplicación"""
    # Inicializar estado de sesión
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # Crear y ejecutar aplicación
    app = PotholeAnalyzerGUI()
    app.run()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🕳️ <strong>Analizador Inteligente de Baches</strong> - Desarrollado con ❤️ usando Streamlit y YOLOv11</p>
        <p>📧 soporte@analizadorbaches.com | 📞 +1-800-BACHES | 🌐 www.analizadorbaches.com</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()