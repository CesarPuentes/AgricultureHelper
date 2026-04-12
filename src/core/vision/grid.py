"""
vision/grid.py — Segmentación de Cuadrícula y Conteo
====================================================
Funciones para segmentar bandejas en cuadrículas (6x4) y contar
plantas individuales y hojas mediante watershed segmentation.
"""

import logging
from typing import Tuple

import cv2
import numpy as np
from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs

from ..config import VisionThresholds, PlantCVConfig
from ..utils import ensure_output_dir, save_debug_image
from .canopy import calculate_canopy_coverage

logger = logging.getLogger(__name__)


def segment_tray_grid(
    image_path: str,
    rows: int | None = None,
    cols: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Segmenta la bandeja en cuadrícula etiquetada.
    
    Args:
        image_path: Ruta a la imagen de la bandeja.
        rows: Filas de la cuadrícula (default desde VisionThresholds).
        cols: Columnas de la cuadrícula (default desde VisionThresholds).
    
    Returns:
        tuple: (imagen, máscara etiquetada, número de plantas detectadas)
    """
    from ..utils.image_utils import create_lab_mask
    
    rows = rows or VisionThresholds.DEFAULT_ROWS
    cols = cols or VisionThresholds.DEFAULT_COLS
    
    img, _, _ = pcv.readimage(filename=image_path)
    
    # LAB 'a' channel: positive=red, negative=green
    a = pcv.rgb2gray_lab(rgb_img=img, channel=VisionThresholds.LAB_CHANNEL)
    a_thresh = pcv.threshold.otsu(gray_img=a, object_type='dark')
    a_fill = pcv.fill(bin_img=a_thresh, size=VisionThresholds.LAB_FILL_SIZE)
    
    rois = pcv.roi.auto_grid(mask=a_fill, nrows=rows, ncols=cols, img=img)
    labeled_mask, num_plants = pcv.create_labels(mask=a_fill, rois=rois, roi_type="partial")
    
    return img, labeled_mask, num_plants


def count_plants(
    image_path: str,
    rows: int = 6,
    cols: int = 4,
    save_map: bool = False,
    plant_id: str = "unknown",
) -> int | Tuple[int, dict]:
    """
    Segmenta, cuenta y registra métricas de las plantas en una bandeja.
    
    Args:
        image_path: Ruta a la imagen de la bandeja.
        rows: Filas de la cuadrícula (default 6).
        cols: Columnas de la cuadrícula (default 4).
        save_map: Si True, genera mapa de diagnóstico.
        plant_id: Identificador para el nombre del archivo.
    
    Returns:
        int: Número de plantas detectadas, o tupla (count, map_info).
    """
    # Configurar PlantCV
    args = WorkflowInputs(
        images=[image_path], 
        names="image1", 
        result=PlantCVConfig.RESULT_FILENAME, 
        outdir=".", 
        writeimg=True, 
        debug=PlantCVConfig.DEBUG_MODE, 
        sample_label=PlantCVConfig.SAMPLE_LABEL
    )
    pcv.params.debug = args.debug
    pcv.params.dpi = PlantCVConfig.DPI
    pcv.params.text_size = PlantCVConfig.TEXT_SIZE
    pcv.params.text_thickness = PlantCVConfig.TEXT_THICKNESS

    img, labeled_mask, num_plants = segment_tray_grid(image_path, rows, cols)
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

    canopy_percent = calculate_canopy_coverage(image_path)

    print("-" * 40)
    print(f"CONTEO: {conteo_real} plantas detectadas.")
    print(f"ÁREA TOTAL: {area_total_pixeles} píxeles.")
    print(f"COBERTURA (Canopy %): {canopy_percent}%")
    print("=" * 40 + "\n")

    pcv.outputs.add_observation(
        sample='image1', variable='plant_count', 
        trait='number of detected plants', method='count', 
        scale='plants', datatype=int, value=conteo_real, label='plants'
    )
    pcv.outputs.add_observation(
        sample='image1', variable='total_area_pixels', 
        trait='total leaf area', method='pixel_sum', 
        scale='pixels', datatype=int, value=int(area_total_pixeles), label='pixels'
    )
    pcv.outputs.add_observation(
        sample='image1', variable='canopy_coverage', 
        trait='percentage of green coverage', method='exg_ratio', 
        scale='percent', datatype=float, value=canopy_percent, label='percent'
    )

    pcv.analyze.size(img=img, labeled_mask=labeled_mask, n_labels=num_plants)
    pcv.analyze.color(rgb_img=img, labeled_mask=labeled_mask, n_labels=num_plants, colorspaces="HSV")
    pcv.outputs.save_results(filename=args.result, outformat="json")

    map_info = None
    if save_map:
        from .diagnostic_map import generate_diagnostic_map
        map_info = generate_diagnostic_map(image_path, plant_id=plant_id)
        print(f"Mapa de diagnóstico guardado en: {map_info['map_path']}")

    print("Análisis completado con éxito.")
    return conteo_real if not save_map else (conteo_real, map_info)


def count_leaves(
    image_path: str,
    rows: int = 6,
    cols: int = 4,
    watershed_dist: int | None = None,
    min_area: int | None = None,
    save_map: bool = False,
    plant_id: str = "unknown",
    output_dir: str | None = None,
) -> int | Tuple[int, str]:
    """
    Cuenta el número total de hojas en una bandeja aplicando Watershed.
    
    Args:
        image_path: Ruta a la imagen.
        rows: Filas de la cuadrícula.
        cols: Columnas de la cuadrícula.
        watershed_dist: Distancia para watershed (default desde VisionThresholds).
        min_area: Área mínima para planta (default desde VisionThresholds).
        save_map: Si True, guarda imagen resumen con cajas.
        plant_id: Identificador para el nombre del archivo.
        output_dir: Directorio de salida.
    
    Returns:
        int: Total de hojas, o tupla (total, map_path).
    """
    watershed_dist = watershed_dist or VisionThresholds.WATERSHED_DISTANCE
    min_area = min_area or VisionThresholds.MIN_LEAF_AREA
    
    out_dir = ensure_output_dir(output_dir)
    pcv.outputs.clear()

    # Guardar estado de debug para restaurar después
    estado_debug_anterior = pcv.params.debug
    pcv.params.debug = PlantCVConfig.DEBUG_MODE

    img, labeled_mask, _ = segment_tray_grid(image_path, rows, cols)
    total_hojas = 0
    etiquetas_plantas = np.unique(labeled_mask)

    # Lienzo resumen
    imagen_resumen = img.copy()

    print("\nProcesando hojas por planta... por favor espera.")

    for etiqueta in etiquetas_plantas:
        if etiqueta == 0: continue

        mascara_planta = np.where(labeled_mask == etiqueta, 255, 0).astype(np.uint8)
        area_planta = np.count_nonzero(mascara_planta)
        
        if area_planta > min_area:
            label_str = f"hojas_roi_{etiqueta}"
            
            # Watershed en silencio
            pcv.watershed_segmentation(
                rgb_img=img, 
                mask=mascara_planta, 
                distance=watershed_dist, 
                label=label_str
            )
            
            hojas_planta = pcv.outputs.observations[label_str]["estimated_object_count"]["value"]
            total_hojas += hojas_planta
            
            # Dibujar caja y texto en imagen resumen
            contornos, _ = cv2.findContours(mascara_planta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                cv2.drawContours(imagen_resumen, contornos, -1, (255, 255, 255), 1)
                x, y, w, h = cv2.boundingRect(max(contornos, key=cv2.contourArea))
                
                cv2.rectangle(imagen_resumen, (x, y), (x + w, y + h), (255, 200, 0), 2)
                texto = f"Hojas: {hojas_planta}"
                (w_txt, h_txt), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    imagen_resumen, 
                    (x, max(0, y - h_txt - 10)), 
                    (x + w_txt, max(0, y)), 
                    (0, 0, 0), 
                    -1
                )
                cv2.putText(
                    imagen_resumen, texto, (x, max(15, y - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                )
            
            print(f"  - ROI {etiqueta} | Hojas: {hojas_planta}")

    print(f"TOTAL HOJAS: {total_hojas}\n")

    # Letrero grande con el total
    cv2.rectangle(imagen_resumen, (10, 10), (450, 70), (0, 0, 0), -1)
    cv2.putText(
        imagen_resumen, f"TOTAL HOJAS: {total_hojas}", (20, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA
    )

    map_path = None
    if save_map:
        map_path = save_debug_image(imagen_resumen, f"leaf_summary_{plant_id}", out_dir)
        print(f"Resumen de hojas guardado en: {map_path}")

    # Restaurar debug y limpiar
    pcv.params.debug = estado_debug_anterior
    pcv.outputs.clear()
    
    return (total_hojas, map_path) if save_map else total_hojas
