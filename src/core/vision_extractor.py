"""
vision_extractor.py
-------------------
Módulo de visión por computadora para análisis de bandejas de plantas.
Incluye cálculo de cobertura verde (canopy) y conteo de plantas por cuadrícula.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import cv2
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs

# ---------------------------------------------------------------------------
# Configuración inicial de PlantCV
# Desactiva el plotting interactivo para entornos headless / Raspberry Pi.
# Se sobreescribirá con args.debug más adelante.
# ---------------------------------------------------------------------------
pcv.params.debug = "None"

# ---------------------------------------------------------------------------
# Funciones auxiliares (privadas)
# ---------------------------------------------------------------------------
def _create_green_mask(image_path):
    """
    Shared pipeline: read image → ExG index → threshold → noise fill.
    Returns (original_color_image, binary_mask).
    """
    original_color_image, _, _ = pcv.readimage(filename=image_path)

    b, g, r = cv2.split(original_color_image)
    b, g, r = b.astype(float), g.astype(float), r.astype(float)

    # Índice de Exceso de Verde (ExG) para resaltar tejido vivo
    exg = np.clip((2 * g) - r - b, 0, 255).astype(np.uint8)

    mask = pcv.threshold.binary(gray_img=exg, threshold=20, object_type="light")
    mask = pcv.fill(bin_img=mask, size=200)

    return original_color_image, mask


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------
def calculate_living_canopy(image_path):
    """
    Calculate the percentage of the image covered by living green tissue.
    """
    try:
        _, mask = _create_green_mask(image_path)
        total_pixels = mask.size
        living_pixels = np.count_nonzero(mask)
        return round((living_pixels / total_pixels) * 100, 2)
    except Exception as e:
        print(f"Error processing canopy coverage: {e}")
        return None


def count_plants(image_path, rows=6, cols=4):
    """
    Segmenta, cuenta y registra métricas de las plantas en una bandeja.
    Permite parametrizar el tamaño de la cuadrícula (rows x cols).
    """

    # --- Sección 1: Input/Output variables ---
    args = WorkflowInputs(
        images=[image_path],
        names="image1",
        result="arabidopsis_results.json",
        outdir=".",
        writeimg=True,
        debug="None",
        sample_label="genotype"
    )

    # Configurar parámetros globales de PlantCV
    pcv.params.debug = args.debug
    pcv.params.dpi = 100
    pcv.params.text_size = 10
    pcv.params.text_thickness = 20

    # Leer la imagen principal
    img, path, filename = pcv.readimage(filename=args.image1)

    # --- Sección 2: Segmentación ---
    # Usamos el canal 'a' de LAB y Umbral de Otsu para automatizar la luz
    a = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    a_thresh = pcv.threshold.otsu(gray_img=a, object_type='dark')
    a_fill = pcv.fill(bin_img=a_thresh, size=200)

    # --- Sección 3: Definir ROIs (Cuadrícula) ---
    rois = pcv.roi.auto_grid(mask=a_fill, nrows=rows, ncols=cols, img=img)

    # --- Sección 4: Crear etiquetas y conteo ---
    labeled_mask, num_plants = pcv.create_labels(mask=a_fill, rois=rois, roi_type="partial")

    etiquetas_presentes = np.unique(labeled_mask)
    conteo_real = len(etiquetas_presentes) - 1

    # --- Sección 5: Cálculo de áreas (individual y total) ---
    area_total_pixeles = 0
    print("\n" + "=" * 40)
    print(f"REPORTE DETALLADO DE BANDEJA ({rows}x{cols})")
    print("-" * 40)

    for etiqueta in etiquetas_presentes:
        if etiqueta == 0: continue

        area_pld = np.sum(labeled_mask == etiqueta)
        area_total_pixeles += area_pld
        print(f"  - Planta en ROI {etiqueta}: {area_pld} píxeles")

    # Calcular % de cobertura usando la función previa
    canopy_percent = calculate_living_canopy(args.image1)

    print("-" * 40)
    print(f"CONTEO: {conteo_real} plantas detectadas.")
    print(f"ÁREA TOTAL: {area_total_pixeles} píxeles.")
    print(f"COBERTURA (Canopy %): {canopy_percent}%")
    print("=" * 40 + "\n")

    # --- Sección 6: Registro de observaciones ---
    pcv.outputs.add_observation(
        sample='image1', variable='plant_count',
        trait='number of detected plants',
        method='count', scale='plants',
        datatype=int, value=conteo_real, label='plants'
    )

    pcv.outputs.add_observation(
        sample='image1', variable='total_area_pixels',
        trait='total leaf area',
        method='pixel_sum', scale='pixels',
        datatype=int, value=int(area_total_pixeles), label='pixels'
    )

    pcv.outputs.add_observation(
        sample='image1', variable='canopy_coverage',
        trait='percentage of green coverage',
        method='exg_ratio', scale='percent',
        datatype=float, value=canopy_percent, label='percent'
    )

    # Medir tamaño y color para el JSON detallado
    pcv.analyze.size(img=img, labeled_mask=labeled_mask, n_labels=num_plants)
    pcv.analyze.color(rgb_img=img, labeled_mask=labeled_mask, n_labels=num_plants, colorspaces="HSV")

    # --- Sección 7: Guardar resultados ---
    pcv.outputs.save_results(filename=args.result, outformat="json")

    print(f"Análisis completado con éxito.")

    return conteo_real