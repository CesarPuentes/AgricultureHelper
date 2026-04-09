"""
vision_extractor.py
-------------------
Módulo de visión por computadora para análisis de bandejas de plantas.
Incluye cálculo de cobertura verde (canopy), conteo de plantas por cuadrícula,
y generación de mapas de diagnóstico visual para verificación humana.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
from datetime import datetime

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

# Default output directory for diagnostic maps
_MAP_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "debug_output")

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


def _ensure_output_dir(output_dir: str | None = None) -> str:
    """Create output directory if it doesn't exist, return its path."""
    out = output_dir or _MAP_OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Mapa de Diagnóstico Visual
# ---------------------------------------------------------------------------
def generate_diagnostic_map(
    image_path: str,
    plant_id: str = "unknown",
    output_dir: str | None = None,
) -> dict:
    """
    Genera un mapa de diagnóstico visual con overlays de colores.

    Produce un PNG anotado donde:
      - Verde semitransparente = tejido vivo detectado
      - Rojo semitransparente  = zonas NO verdes (suelo, maceta, tejido muerto)
      - Contornos blancos      = bordes de las áreas verdes
      - Texto                  = % de cobertura superpuesto

    Esto permite al desarrollador humano confirmar visualmente que
    las predicciones del pipeline de visión son correctas.

    Args:
        image_path: Ruta a la imagen de la planta/bandeja.
        plant_id: Identificador de la planta (para el nombre del archivo).
        output_dir: Directorio de salida. Default: debug_output/.

    Returns:
        dict con 'map_path', 'coverage_pct', y 'generated_at'.
    """
    out_dir = _ensure_output_dir(output_dir)
    original, mask = _create_green_mask(image_path)

    # --- Calcular cobertura ---
    total_pixels = mask.size
    living_pixels = np.count_nonzero(mask)
    coverage_pct = round((living_pixels / total_pixels) * 100, 2)

    # --- Crear canvas de overlay ---
    overlay = original.copy()
    h, w = mask.shape[:2]

    # Overlay verde sobre tejido vivo
    green_overlay = np.zeros_like(original)
    green_overlay[:] = (0, 200, 0)  # BGR: verde brillante
    living_mask_3ch = cv2.merge([mask, mask, mask]) > 0
    overlay = np.where(living_mask_3ch, 
                       cv2.addWeighted(original, 0.6, green_overlay, 0.4, 0),
                       overlay)

    # Overlay rojo sobre zonas muertas/suelo
    red_overlay = np.zeros_like(original)
    red_overlay[:] = (0, 0, 180)  # BGR: rojo
    dead_mask_3ch = ~living_mask_3ch
    overlay = np.where(dead_mask_3ch,
                       cv2.addWeighted(original, 0.7, red_overlay, 0.3, 0),
                       overlay)

    # --- Dibujar contornos ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    # --- Agregar texto con estadísticas ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Cobertura Verde: {coverage_pct}%"
    text_size = cv2.getTextSize(text, font, 1.0, 2)[0]

    # Fondo negro para legibilidad
    cv2.rectangle(overlay, (10, 10), (20 + text_size[0], 20 + text_size[1] + 10),
                  (0, 0, 0), -1)
    cv2.putText(overlay, text, (15, 15 + text_size[1]),
                font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Leyenda
    legend_y = 30 + text_size[1] + 10
    cv2.rectangle(overlay, (15, legend_y), (35, legend_y + 15), (0, 200, 0), -1)
    cv2.putText(overlay, "Tejido Vivo", (42, legend_y + 13),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(overlay, (15, legend_y + 25), (35, legend_y + 40), (0, 0, 180), -1)
    cv2.putText(overlay, "Suelo / No-verde", (42, legend_y + 38),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Guardar ---
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diagnostic_map_{plant_id}_{timestamp_str}.png"
    map_path = os.path.join(out_dir, filename)
    cv2.imwrite(map_path, overlay)

    return {
        "map_path": map_path,
        "coverage_pct": coverage_pct,
        "generated_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------
def calculate_living_canopy(image_path, save_map: bool = False, plant_id: str = "unknown", output_dir: str | None = None):
    """
    Calculate the percentage of the image covered by living green tissue.

    Args:
        image_path: Path to the plant image.
        save_map: If True, also generates a diagnostic map PNG.
        plant_id: Plant identifier (used in map filename).
        output_dir: Directory where the diagnostic map will be saved.

    Returns:
        float: coverage percentage, or None on error.
        If save_map=True, returns tuple (coverage_pct, map_info).
    """
    try:
        _, mask = _create_green_mask(image_path)
        total_pixels = mask.size
        living_pixels = np.count_nonzero(mask)
        coverage = round((living_pixels / total_pixels) * 100, 2)

        if save_map:
            map_info = generate_diagnostic_map(image_path, plant_id=plant_id, output_dir=output_dir)
            return coverage, map_info

        return coverage
    except Exception as e:
        print(f"Error processing canopy coverage: {e}")
        return None


def count_plants(image_path, rows=6, cols=4, save_map: bool = False, plant_id: str = "unknown"):
    """
    Segmenta, cuenta y registra métricas de las plantas en una bandeja.
    Permite parametrizar el tamaño de la cuadrícula (rows x cols).

    Args:
        image_path: Path to the tray image.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        save_map: If True, generates a diagnostic map PNG alongside analysis.
        plant_id: Plant identifier for the map filename.
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

    # --- Sección 8: Mapa de diagnóstico (opcional) ---
    map_info = None
    if save_map:
        map_info = generate_diagnostic_map(image_path, plant_id=plant_id)
        print(f"Mapa de diagnóstico guardado en: {map_info['map_path']}")

    print(f"Análisis completado con éxito.")

    return conteo_real if not save_map else (conteo_real, map_info)