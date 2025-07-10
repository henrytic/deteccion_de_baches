import warnings

warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
import yaml
import torch
from PIL import Image
from ultralytics import YOLO

# Configurar estilo visual
sns.set(rc={'axes.facecolor': '#ffe4de'}, style='darkgrid')


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
            recommended_size = 'small'  # Cambiado de nano a small para mejor rendimiento
            print("Hardware limitado detectado. Recomendación: small")
        elif gpu_memory < 6:
            recommended_size = 'medium'
            print("GPU moderada detectada. Recomendación: medium")
        elif gpu_memory < 8:
            recommended_size = 'medium'
            print("GPU buena detectada. Recomendación: medium")
        elif gpu_memory < 12:
            recommended_size = 'large'
            print("GPU potente detectada. Recomendación: large")
        else:
            recommended_size = 'large'  # No extra_large por defecto para evitar tiempos excesivos
            print("GPU muy potente detectada. Recomendación: large")
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
                print("Usando YOLOv11 - ¡La versión más reciente con mejores resultados!")
            else:
                print("Usando YOLOv8 - Versión estable")

            return model, model_name

        except Exception as e:
            print(f"Error cargando {model_name}: {e}")
            continue

    # Si no se pudo cargar ningún modelo
    raise Exception("No se pudo cargar ningún modelo YOLO")


def setup_model_interactive():
    print("\n" + "=" * 50)
    print("SELECCIÓN DEL MODELO YOLO")
    print("=" * 50)

    print("\nOpciones disponibles:")
    print("1. Automático (recomendado) - Detecta tu hardware y selecciona el mejor modelo")
    print("2. Small (yolo11s/yolov8s) - Rápido, buena precisión")
    print("3. Medium (yolo11m/yolov8m) - Balance óptimo (RECOMENDADO)")
    print("4. Large (yolo11l/yolov8l) - Alta precisión, requiere más GPU")
    print("5. Nano (yolo11n/yolov8n) - Muy rápido, menor precisión")
    print("6. Extra Large (yolo11x/yolov8x) - Máxima precisión, muy lento")

    print("\n Recomendación: Medium (opción 3) ofrece el mejor balance para detección de baches")

    while True:
        try:
            choice = input("\nSelecciona una opción (1-6): ").strip()

            size_map = {
                '1': 'auto',
                '2': 'small',
                '3': 'medium',
                '4': 'large',
                '5': 'nano',
                '6': 'extra_large'
            }

            if choice in size_map:
                return select_yolo_model(size_map[choice])
            else:
                print("Opción inválida. Por favor selecciona 1-6.")

        except KeyboardInterrupt:
            print("\nSelección cancelada.")
            return None, None


def setup_model():
    print("Configurando modelo YOLO...")

    # Preguntar al usuario si quiere selección automática o manual
    print("\n¿Cómo quieres seleccionar el modelo?")
    print("1. Automático (recomendado) - Detecta hardware automáticamente")
    print("2. Manual - Seleccionar manualmente")

    choice = input("Selecciona (1-2): ").strip()

    if choice == '2':
        return setup_model_interactive()
    else:
        print("Usando selección automática...")
        return select_yolo_model('auto')


def load_dataset_info(dataset_path):
    yaml_file_path = os.path.join(dataset_path, 'data.yaml')

    print("Cargando información del dataset...")
    with open(yaml_file_path, 'r') as file:
        yaml_content = yaml.load(file, Loader=yaml.FullLoader)
        print(yaml.dump(yaml_content, default_flow_style=False))

    return yaml_file_path


def analyze_dataset(dataset_path):
    train_images_path = os.path.join(dataset_path, 'train', 'images')
    valid_images_path = os.path.join(dataset_path, 'valid', 'images')

    num_train_images = 0
    num_valid_images = 0
    train_image_sizes = set()
    valid_image_sizes = set()

    # Verificar imágenes de entrenamiento
    for filename in os.listdir(train_images_path):
        if filename.endswith('.jpg'):
            num_train_images += 1
            image_path = os.path.join(train_images_path, filename)
            with Image.open(image_path) as img:
                train_image_sizes.add(img.size)

    # Verificar imágenes de validación
    for filename in os.listdir(valid_images_path):
        if filename.endswith('.jpg'):
            num_valid_images += 1
            image_path = os.path.join(valid_images_path, filename)
            with Image.open(image_path) as img:
                valid_image_sizes.add(img.size)

    print(f"Número de imágenes de entrenamiento: {num_train_images}")
    print(f"Número de imágenes de validación: {num_valid_images}")

    if len(train_image_sizes) == 1:
        print(f"Todas las imágenes de entrenamiento tienen el mismo tamaño: {train_image_sizes.pop()}")
    else:
        print("Las imágenes de entrenamiento tienen tamaños variables.")

    if len(valid_image_sizes) == 1:
        print(f"Todas las imágenes de validación tienen el mismo tamaño: {valid_image_sizes.pop()}")
    else:
        print("Las imágenes de validación tienen tamaños variables.")

    return train_images_path, valid_images_path


def show_sample_images(train_images_path):
    random.seed(0)
    image_files = [f for f in os.listdir(train_images_path) if f.endswith('.jpg')]
    random_images = random.sample(image_files, min(15, len(image_files)))

    plt.figure(figsize=(19, 12))

    for i, image_file in enumerate(random_images):
        image_path = os.path.join(train_images_path, image_file)
        image = Image.open(image_path)
        plt.subplot(3, 5, i + 1)
        plt.imshow(image)
        plt.axis('off')

    plt.suptitle('Muestra Aleatoria de Imágenes del Dataset', fontsize=24)
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()


def get_training_config(model_name):
    configs = {
        'nano': {
            'epochs': 200,
            'batch': 32,
            'imgsz': 640,
            'patience': 25,
            'lr0': 0.001,
            'lrf': 0.01
        },
        'small': {
            'epochs': 150,
            'batch': 24,
            'imgsz': 640,
            'patience': 20,
            'lr0': 0.0005,
            'lrf': 0.01
        },
        'medium': {
            'epochs': 120,
            'batch': 16,
            'imgsz': 640,
            'patience': 15,
            'lr0': 0.0001,
            'lrf': 0.01
        },
        'large': {
            'epochs': 100,
            'batch': 12,
            'imgsz': 640,
            'patience': 12,
            'lr0': 0.0001,
            'lrf': 0.01
        },
        'extra_large': {
            'epochs': 80,
            'batch': 8,
            'imgsz': 640,
            'patience': 10,
            'lr0': 0.00005,
            'lrf': 0.01
        }
    }

    # Detectar tamaño del modelo
    if 'n' in model_name.lower():
        return configs['nano']
    elif 's' in model_name.lower():
        return configs['small']
    elif 'm' in model_name.lower():
        return configs['medium']
    elif 'l' in model_name.lower():
        return configs['large']
    elif 'x' in model_name.lower():
        return configs['extra_large']
    else:
        return configs['medium']  # default


def train_model(model, yaml_file_path, model_name):
    print("Iniciando entrenamiento del modelo...")

    # Obtener configuración optimizada
    config = get_training_config(model_name)

    print(f"Configuración optimizada para {model_name}:")
    print(f"- Épocas: {config['epochs']}")
    print(f"- Batch size: {config['batch']}")
    print(f"- Tamaño de imagen: {config['imgsz']}")
    print(f"- Paciencia: {config['patience']}")
    print(f"- Learning rate inicial: {config['lr0']}")

    # Ajustar batch size si hay problemas de memoria
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if gpu_memory < 6 and config['batch'] > 16:
            config['batch'] = 12
            print(f"  Ajustando batch size a {config['batch']} debido a limitaciones de GPU")
        elif gpu_memory < 4 and config['batch'] > 12:
            config['batch'] = 8
            print(f"  Ajustando batch size a {config['batch']} debido a limitaciones de GPU")

    # Generar nombre único para el experimento
    experiment_name = f"experiment_{model_name.replace('-seg.pt', '').replace('.', '_')}"

    results = model.train(
        data=yaml_file_path,
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        patience=config['patience'],
        batch=config['batch'],
        optimizer='auto',
        lr0=config['lr0'],
        lrf=config['lrf'],
        dropout=0.25,
        device=device_name,
        seed=42,
        save=True,
        save_period=max(10, config['epochs'] // 10),
        project='pothole_training',
        name=experiment_name,
        # Configuraciones adicionales para mejores resultados
        amp=True,  # Mixed precision training
        fraction=1.0,  # Usar todo el dataset
        profile=False,  # Desactivar profiling para mejor rendimiento
        # Augmentaciones optimizadas para detección de baches
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )
    return results, experiment_name


def plot_learning_curve(df, train_loss_col, val_loss_col, title, ylim_range=[0, 2]):

    plt.figure(figsize=(12, 4))
    sns.lineplot(data=df, x='epoch', y=train_loss_col, label='Train Loss', color='blue', linestyle='-', linewidth=2)
    sns.lineplot(data=df, x='epoch', y=val_loss_col, label='Validation Loss', color='#ed2f00', linestyle='--',
                 linewidth=2)
    plt.title(title)
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.ylim(ylim_range)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Guardar gráfico
    filename = title.lower().replace(' ', '_').replace('-', '_') + '.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


def read_and_convert_image(file_path):
    """Leer y convertir imagen para plotting"""
    if not os.path.exists(file_path):
        return None
    image = cv2.imread(file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def analyze_training_results(results_path):
    print("Analizando resultados del entrenamiento...")

    # Mostrar imagen de resultados
    results_file_path = os.path.join(results_path, 'results.png')
    if os.path.exists(results_file_path):
        image = cv2.imread(results_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(20, 8))
        plt.imshow(image)
        plt.title('Tendencias de Pérdida de Entrenamiento y Validación', fontsize=24)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('training_results_overview.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Analizar CSV de resultados
    results_csv_path = os.path.join(results_path, 'results.csv')
    if os.path.exists(results_csv_path):
        df = pd.read_csv(results_csv_path)
        df.columns = df.columns.str.strip()

        # Crear gráficos individuales de pérdida
        plot_learning_curve(df, 'train/box_loss', 'val/box_loss', 'Curva de Aprendizaje - Pérdida de Bounding Box')
        plot_learning_curve(df, 'train/cls_loss', 'val/cls_loss', 'Curva de Aprendizaje - Pérdida de Clasificación')
        plot_learning_curve(df, 'train/dfl_loss', 'val/dfl_loss', 'Curva de Aprendizaje - Pérdida Focal Distribuida')
        plot_learning_curve(df, 'train/seg_loss', 'val/seg_loss', 'Curva de Aprendizaje - Pérdida de Segmentación',
                            ylim_range=[0, 5])

        # Guardar DataFrame de resultados
        df.to_csv('training_metrics.csv', index=False)
        print("Métricas de entrenamiento guardadas en 'training_metrics.csv'")


def show_evaluation_curves(results_path):
    print("Generando curvas de evaluación...")

    box_files_titles = {
        'BoxP_curve.png': 'Curva Precisión-Confianza Bounding Box',
        'BoxR_curve.png': 'Curva Recall-Confianza Bounding Box',
        'BoxF1_curve.png': 'Curva F1-Confianza Bounding Box'
    }
    mask_files_titles = {
        'MaskP_curve.png': 'Curva Precisión-Confianza Máscara',
        'MaskR_curve.png': 'Curva Recall-Confianza Máscara',
        'MaskF1_curve.png': 'Curva F1-Confianza Máscara'
    }

    # Subplot 3x2
    fig, axs = plt.subplots(3, 2, figsize=(20, 20))

    # Plotear imágenes Box
    for i, (filename, title) in enumerate(box_files_titles.items()):
        img_path = os.path.join(results_path, filename)
        img = read_and_convert_image(img_path)
        if img is not None:
            axs[i, 0].imshow(img)
            axs[i, 0].set_title(title, fontsize=16)
        axs[i, 0].axis('off')

    # Plotear imágenes Mask
    for i, (filename, title) in enumerate(mask_files_titles.items()):
        img_path = os.path.join(results_path, filename)
        img = read_and_convert_image(img_path)
        if img is not None:
            axs[i, 1].imshow(img)
            axs[i, 1].set_title(title, fontsize=16)
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('evaluation_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def show_confusion_matrices(results_path):
    print("Mostrando matrices de confusión...")

    confusion_matrix_path = os.path.join(results_path, 'confusion_matrix.png')
    confusion_matrix_normalized_path = os.path.join(results_path, 'confusion_matrix_normalized.png')

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    cm_img = read_and_convert_image(confusion_matrix_path)
    if cm_img is not None:
        axs[0].imshow(cm_img)
        axs[0].set_title('Matriz de Confusión', fontsize=20)
    axs[0].axis('off')

    cm_norm_img = read_and_convert_image(confusion_matrix_normalized_path)
    if cm_norm_img is not None:
        axs[1].imshow(cm_norm_img)
        axs[1].set_title('Matriz de Confusión Normalizada', fontsize=20)
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()


def validate_model(results_path, model_name):
    print("Validando modelo entrenado...")

    best_model_path = os.path.join(results_path, 'weights', 'best.pt')

    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        metrics = best_model.val(split='val')

        # Mostrar métricas
        metrics_df = pd.DataFrame.from_dict(metrics.results_dict, orient='index', columns=['Valor de Métrica'])
        print("\nMétricas del modelo:")
        print(metrics_df.round(3))

        # Guardar métricas
        metrics_df.to_csv('validation_metrics.csv')
        print("Métricas de validación guardadas en 'validation_metrics.csv'")

        # Mostrar métricas clave
        key_metrics = {
            'mAP50 (Box)': metrics.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95 (Box)': metrics.results_dict.get('metrics/mAP50-95(B)', 0),
            'mAP50 (Mask)': metrics.results_dict.get('metrics/mAP50(M)', 0),
            'mAP50-95 (Mask)': metrics.results_dict.get('metrics/mAP50-95(M)', 0)
        }

        print(f"\n Métricas Clave del Modelo {model_name}:")
        print("=" * 50)
        for metric, value in key_metrics.items():
            print(f"{metric}: {value:.3f}")

        # Exportar modelo
        print("\nExportando modelo a formato ONNX...")
        try:
            best_model.export(format='onnx')
            print("Modelo exportado exitosamente a formato ONNX.")
        except Exception as e:
            print(f" Error al exportar modelo: {e}")

        return best_model, best_model_path
    else:
        print(f"No se encontró el modelo en {best_model_path}")
        return None, None


def test_model_sample(best_model, valid_images_path):
    print("Probando modelo en imágenes de validación...")

    image_files = [file for file in os.listdir(valid_images_path) if file.endswith('.jpg')]

    # Seleccionar 9 imágenes a intervalos iguales
    num_images = len(image_files)
    selected_images = [image_files[i] for i in range(0, num_images, max(1, num_images // 9))][:9]

    fig, axes = plt.subplots(3, 3, figsize=(20, 21))
    fig.suptitle('Inferencias en Conjunto de Validación', fontsize=24)

    for i, ax in enumerate(axes.flatten()):
        if i < len(selected_images):
            image_path = os.path.join(valid_images_path, selected_images[i])
            results = best_model.predict(source=image_path, imgsz=640)
            annotated_image = results[0].plot()
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            ax.imshow(annotated_image_rgb)
            ax.set_title(f'Imagen {i + 1}', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('validation_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    print("=" * 60)
    print("ENTRENAMIENTO MEJORADO DEL MODELO DE DETECCIÓN DE BACHES")
    print("Soporta YOLOv8 y YOLOv11 con selección automática")
    print("=" * 60)

    # Configurar rutas
    current_dir = os.getcwd()
    dataset_path = os.path.join(current_dir, 'Dataset')

    print(f"Directorio actual: {current_dir}")
    print(f"Ruta del dataset: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró el dataset en {dataset_path}")
        print("Asegúrate de que la carpeta 'Dataset' esté en el mismo directorio que este script.")
        return

    try:
        model, model_name = setup_model()
        if model is None:
            print("No se pudo configurar el modelo.")
            return
    except Exception as e:
        print(f"Error configurando modelo: {e}")
        return

    yaml_file_path = load_dataset_info(dataset_path)

    train_images_path, valid_images_path = analyze_dataset(dataset_path)

    print("\nMostrando muestra de imágenes del dataset...")
    show_sample_images(train_images_path)

    print("\n" + "=" * 50)
    print(f"Configuración seleccionada: {model_name}")
    if 'yolo11' in model_name:
        print("🚀 Excelente elección! YOLOv11 ofrece mejor precisión y velocidad.")

    print("\n¿Deseas continuar con el entrenamiento? (y/n): ", end="")
    train_choice = input().lower()

    if train_choice != 'y':
        print("Entrenamiento cancelado.")
        return

    print("\n" + "=" * 50)
    print("INICIANDO ENTRENAMIENTO...")
    print("=" * 50)

    try:
        results, experiment_name = train_model(model, yaml_file_path, model_name)
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        return

    results_path = os.path.join('pothole_training', experiment_name)
    if not os.path.exists(results_path):
        # Buscar la carpeta de resultados más reciente
        runs_dir = 'runs/segment'
        if os.path.exists(runs_dir):
            train_dirs = [d for d in os.listdir(runs_dir) if d.startswith('train')]
            if train_dirs:
                latest_train = max(train_dirs)
                results_path = os.path.join(runs_dir, latest_train)

    print(f"\nRuta de resultados: {results_path}")

    if os.path.exists(results_path):
        print("\n" + "=" * 50)
        print("ANALIZANDO RESULTADOS...")
        print("=" * 50)

        analyze_training_results(results_path)
        show_evaluation_curves(results_path)
        show_confusion_matrices(results_path)

        best_model, best_model_path = validate_model(results_path, model_name)

        if best_model is not None:
            test_model_sample(best_model, valid_images_path)

            print("\n" + "=" * 60)
            print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
            print("=" * 60)
            print(f"Modelo utilizado: {model_name}")
            print(f"Modelo guardado en: {best_model_path}")
            print("\nArchivos generados:")
            print("- training_metrics.csv")
            print("- validation_metrics.csv")
            print("- sample_images.png")
            print("- training_results_overview.png")
            print("- evaluation_curves.png")
            print("- confusion_matrices.png")
            print("- validation_predictions.png")
            print("- best.onnx (modelo exportado)")

            print(f"\n💡 Para procesar videos, usa:")
            print(f"python process_video.py --model {best_model_path} --input tu_video.mp4")
        else:
            print("Error: No se pudo validar el modelo entrenado.")
    else:
        print(f"Error: No se encontraron resultados de entrenamiento en {results_path}")

if __name__ == "__main__":
    main()