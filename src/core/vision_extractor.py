"""
vision_extractor.py
-------------------
Módulo de visión por computadora para análisis de bandejas de plantas.
Incluye cálculo de cobertura verde (canopy), conteo de plantas por cuadrícula,
conteo de hojas individual, y generación de mapas de diagnóstico visual.
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
# ---------------------------------------------------------------------------
pcv.params.debug = "None"
_MAP_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "debug_output")

# ---------------------------------------------------------------------------
# Funciones auxiliares (privadas)
# ---------------------------------------------------------------------------
def _create_green_mask(image_path):
    """Shared pipeline: read image → ExG index → threshold → noise fill."""
    original_color_image, _, _ = pcv.readimage(filename=image_path)
    b, g, r = cv2.split(original_color_image)
    b, g, r = b.astype(float), g.astype(float), r.astype(float)
    exg = np.clip((2 * g) - r - b, 0, 255).astype(np.uint8)
    mask = pcv.threshold.binary(gray_img=exg, threshold=20, object_type="light")
    mask = pcv.fill(bin_img=mask, size=200)
    return original_color_image, mask

def _ensure_output_dir(output_dir: str | None = None) -> str:
    """Create output directory if it doesn't exist."""
    out = output_dir or _MAP_OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    return out

def _segment_tray_grid(image_path, rows, cols):
    """
    Helper compartido que realiza la lectura, segmentación base y creación
    de la cuadrícula etiquetada. Evita repetir código entre count_plants y count_leaves.
    """
    img, path, filename = pcv.readimage(filename=image_path)
    a = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    a_thresh = pcv.threshold.otsu(gray_img=a, object_type='dark')
    a_fill = pcv.fill(bin_img=a_thresh, size=200)
    rois = pcv.roi.auto_grid(mask=a_fill, nrows=rows, ncols=cols, img=img)
    labeled_mask, num_plants = pcv.create_labels(mask=a_fill, rois=rois, roi_type="partial")
    return img, labeled_mask, num_plants

# ---------------------------------------------------------------------------
# Mapa de Diagnóstico Visual
# ---------------------------------------------------------------------------
def generate_diagnostic_map(image_path: str, plant_id: str = "unknown", output_dir: str | None = None) -> dict:
    """Genera un mapa de diagnóstico visual con overlays de colores."""
    out_dir = _ensure_output_dir(output_dir)
    original, mask = _create_green_mask(image_path)
    total_pixels = mask.size
    living_pixels = np.count_nonzero(mask)
    coverage_pct = round((living_pixels / total_pixels) * 100, 2)

    overlay = original.copy()
    h, w = mask.shape[:2]

    green_overlay = np.zeros_like(original)
    green_overlay[:] = (0, 200, 0)
    living_mask_3ch = cv2.merge([mask, mask, mask]) > 0
    overlay = np.where(living_mask_3ch, cv2.addWeighted(original, 0.6, green_overlay, 0.4, 0), overlay)

    red_overlay = np.zeros_like(original)
    red_overlay[:] = (0, 0, 180)
    dead_mask_3ch = ~living_mask_3ch
    overlay = np.where(dead_mask_3ch, cv2.addWeighted(original, 0.7, red_overlay, 0.3, 0), overlay)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Cobertura Verde: {coverage_pct}%"
    text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
    cv2.rectangle(overlay, (10, 10), (20 + text_size[0], 20 + text_size[1] + 10), (0, 0, 0), -1)
    cv2.putText(overlay, text, (15, 15 + text_size[1]), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    legend_y = 30 + text_size[1] + 10
    cv2.rectangle(overlay, (15, legend_y), (35, legend_y + 15), (0, 200, 0), -1)
    cv2.putText(overlay, "Tejido Vivo", (42, legend_y + 13), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(overlay, (15, legend_y + 25), (35, legend_y + 40), (0, 0, 180), -1)
    cv2.putText(overlay, "Suelo / No-verde", (42, legend_y + 38), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diagnostic_map_{plant_id}_{timestamp_str}.png"
    map_path = os.path.join(out_dir, filename)
    cv2.imwrite(map_path, overlay)

    return {"map_path": map_path, "coverage_pct": coverage_pct, "generated_at": datetime.now().isoformat()}

# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------
def calculate_living_canopy(image_path, save_map: bool = False, plant_id: str = "unknown", output_dir: str | None = None):
    """Calculate the percentage of the image covered by living green tissue."""
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
    """Segmenta, cuenta y registra métricas de las plantas en una bandeja."""
    args = WorkflowInputs(images=[image_path], names="image1", result="arabidopsis_results.json", outdir=".", writeimg=True, debug="None", sample_label="genotype")
    pcv.params.debug = args.debug
    pcv.params.dpi = 100
    pcv.params.text_size = 10
    pcv.params.text_thickness = 20

    img, labeled_mask, num_plants = _segment_tray_grid(image_path, rows, cols)
    etiquetas_presentes = np.unique(labeled_mask)
    conteo_real = len(etiquetas_presentes) - 1

    area_total_pixeles = 0
    print("\n" + "=" * 40)
    print(f"REPORTE DETALLADO DE BANDEJA ({rows}x{cols})")
    print("-" * 40)
    for etiqueta in etiquetas_presentes:
        if etiqueta == 0: continue
        area_pld = np.sum(labeled_mask == etiqueta)
        area_total_pixeles += area_pld
        print(f"  - Planta en ROI {etiqueta}: {area_pld} píxeles")

    canopy_percent = calculate_living_canopy(image_path)

    print("-" * 40)
    print(f"CONTEO: {conteo_real} plantas detectadas.")
    print(f"ÁREA TOTAL: {area_total_pixeles} píxeles.")
    print(f"COBERTURA (Canopy %): {canopy_percent}%")
    print("=" * 40 + "\n")

    pcv.outputs.add_observation(sample='image1', variable='plant_count', trait='number of detected plants', method='count', scale='plants', datatype=int, value=conteo_real, label='plants')
    pcv.outputs.add_observation(sample='image1', variable='total_area_pixels', trait='total leaf area', method='pixel_sum', scale='pixels', datatype=int, value=int(area_total_pixeles), label='pixels')
    pcv.outputs.add_observation(sample='image1', variable='canopy_coverage', trait='percentage of green coverage', method='exg_ratio', scale='percent', datatype=float, value=canopy_percent, label='percent')

    pcv.analyze.size(img=img, labeled_mask=labeled_mask, n_labels=num_plants)
    pcv.analyze.color(rgb_img=img, labeled_mask=labeled_mask, n_labels=num_plants, colorspaces="HSV")
    pcv.outputs.save_results(filename=args.result, outformat="json")

    map_info = None
    if save_map:
        map_info = generate_diagnostic_map(image_path, plant_id=plant_id)
        print(f"Mapa de diagnóstico guardado en: {map_info['map_path']}")

    print(f"Análisis completado con éxito.")
    return conteo_real if not save_map else (conteo_real, map_info)

def count_leaves(image_path, rows=6, cols=4, watershed_dist=15, min_area=500):
    """
    Cuenta el número total de hojas en una bandeja iterando sobre la cuadrícula
    y aplicando Watershed individualmente a cada planta detectada.
    """
    pcv.outputs.clear() # Asegura que no haya datos residuales
    
    img, labeled_mask, _ = _segment_tray_grid(image_path, rows, cols)
    etiquetas_presentes = np.unique(labeled_mask)
    total_hojas_bandeja = 0

    print("\n" + "=" * 40)
    print(f"INICIANDO CONTEO DE HOJAS ({rows}x{cols})")
    print("-" * 40)

    for etiqueta in etiquetas_presentes:
        if etiqueta == 0: continue

        area_pld = np.sum(labeled_mask == etiqueta)
        
        if area_pld > min_area:
            label_str = f"hojas_roi_{etiqueta}"
            plant_mask = np.where(labeled_mask == etiqueta, 255, 0).astype(np.uint8)

            pcv.watershed_segmentation(rgb_img=img, mask=plant_mask, distance=watershed_dist, label=label_str)
            hojas_estimadas = pcv.outputs.observations[label_str]["estimated_object_count"]["value"]
            total_hojas_bandeja += hojas_estimadas
            
            print(f"  - ROI {etiqueta} | Área: {area_pld} px | Hojas: {hojas_estimadas}")
        else:
            print(f"  - ROI {etiqueta} | Área: {area_pld} px | (Área insuficiente, ignorada)")

    print("-" * 40)
    print(f"TOTAL DE HOJAS DETECTADAS: {total_hojas_bandeja}")
    print("=" * 40 + "\n")
    
    pcv.outputs.clear() # Limpia al terminar para no afectar otros flujos
    return total_hojas_bandeja