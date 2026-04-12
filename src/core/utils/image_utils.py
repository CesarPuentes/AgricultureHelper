"""
image_utils.py — Utilidades de Procesamiento de Imágenes Compartidas
==================================================================
Funciones de visión por computadora usadas por múltiples módulos.
Unifica las máscaras duplicadas de anomaly/utils.py y vision_extractor.py.
"""

import logging
from typing import Tuple

import cv2
import numpy as np
from plantcv import plantcv as pcv

from ..config import VisionThresholds

logger = logging.getLogger(__name__)


def create_green_mask(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lee imagen y crea máscara de tejido verde usando ExG index.
    
    Pipeline unificado:
    1. Read image → 2. ExG (Excess Green) calculation → 3. Threshold → 4. Fill holes
    
    Args:
        image_path: Ruta a la imagen de entrada.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Imagen original en BGR
            - Máscara binaria del tejido verde (255=vegetación, 0=no-verde)
    
    Note:
        Reemplaza:
        - anomaly/utils._plant_mask() [LAB-based]
        - vision_extractor._create_green_mask() [ExG-based]
        
        Ahora ambas usan la misma función, asegurando consistencia.
    """
    original_color_image, _, _ = pcv.readimage(filename=image_path)
    
    b, g, r = cv2.split(original_color_image)
    b, g, r = b.astype(float), g.astype(float), r.astype(float)
    
    # ExG = 2G - R - B ( Excess Green Index )
    exg = np.clip((2 * g) - r - b, 0, 255).astype(np.uint8)
    
    # Threshold binario
    mask = pcv.threshold.binary(
        gray_img=exg,
        threshold=VisionThresholds.EXG_THRESHOLD,
        object_type="light"
    )
    
    # Rellenar ruido pequeño
    mask = pcv.fill(bin_img=mask, size=VisionThresholds.EXG_FILL_SIZE)
    
    return original_color_image, mask


def create_lab_mask(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lee imagen y crea máscara usando segmentación LAB (para bandejas/plantas).
    
    Útil para localizar plantas individuales sobre fondo oscuro.
    
    Args:
        image_path: Ruta a la imagen de entrada.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Imagen original en BGR
            - Máscara binaria de plantas (255=planta, 0=background)
    """
    img, _, _ = pcv.readimage(filename=image_path)
    
    # Canal 'a' de LAB: positivo=rojo, negativo=verde
    a = pcv.rgb2gray_lab(rgb_img=img, channel=VisionThresholds.LAB_CHANNEL)
    
    # Otsu threshold para объектов oscuros vs claros
    thresh = pcv.threshold.otsu(gray_img=a, object_type='dark')
    
    # Rellenar huecos pequeños
    mask = pcv.fill(bin_img=thresh, size=VisionThresholds.LAB_FILL_SIZE)
    
    return img, mask


def merge_mask_channels(mask: np.ndarray) -> np.ndarray:
    """
    Convierte máscara binaria 1-canál a 3-canales para blending con imagen.
    
    Args:
        mask: Máscara binaria (HxW) con valores 0 y 255.
    
    Returns:
        np.ndarray: Máscara de 3 canales (HxWx3).
    """
    return cv2.merge([mask, mask, mask])


def draw_label_pill(
    image: np.ndarray,
    x: int,
    y: int,
    text: str,
    bg_color: tuple = (0, 0, 0),
    text_color: tuple = (255, 255, 255),
    font_scale: float = 0.5,
    thickness: int = 1,
) -> np.ndarray:
    """
    Dibuja un "pill label" con fondo coloreado y texto centrado.
    
    Args:
        image: Imagen en la que dibujar (modifica in-place y retorna).
        x: Coordenada X de la esquina superior izquierda del pill.
        y: Coordenada Y de la esquina superior izquierda del pill.
        text: Texto a mostrar.
        bg_color: Color del fondo del pill (BGR).
        text_color: Color del texto (BGR).
        font_scale: Escala del texto.
        thickness: Grosor del texto.
    
    Returns:
        np.ndarray: Imagen con el label dibujado.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Obtener tamaño del texto para calcular el rectángulo
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Rectángulo del fondo: x, y-top-left → x+w+padding, y+baseline+padding
    padding_x, padding_y = 8, 4
    top_left = (x, y - text_h - padding_y)
    bottom_right = (x + text_w + padding_x * 2, y + baseline + padding_y)
    
    # Fondo negro semi-transparente
    cv2.rectangle(image, top_left, bottom_right, bg_color, -1)
    
    # Texto centrado verticalmente
    text_x = x + padding_x
    text_y = y - padding_y  # Baseline está ligeramente por encima
    
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )
    
    return image


def create_canopy_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    living_color: tuple = (0, 200, 0),
    dead_color: tuple = (0, 0, 180),
    living_alpha: float = 0.4,
    dead_alpha: float = 0.3,
) -> np.ndarray:
    """
    Crea overlay de colores superponiendo vegetación viva y muerta.
    
    Args:
        original: Imagen original BGR.
        mask: Máscara binaria (255=vegetación viva).
        living_color: Color para tejido vivo.
        dead_color: Color para suelo/no-verde.
        living_alpha: Peso de blend para tejido vivo.
        dead_alpha: Peso de blend para tejido muerto.
    
    Returns:
        np.ndarray: Imagen con overlay de colores.
    """
    overlay = original.copy()
    h, w = mask.shape[:2]
    
    living_mask_3ch = merge_mask_channels(mask) > 0
    dead_mask_3ch = ~living_mask_3ch
    
    # Crear capas de color
    green_layer = np.full_like(original, living_color)
    red_layer = np.full_like(original, dead_color)
    
    # Blend tejido vivo
    overlay = np.where(
        living_mask_3ch,
        cv2.addWeighted(original, 1 - living_alpha, green_layer, living_alpha, 0),
        overlay
    )
    
    # Blend tejido muerto
    overlay = np.where(
        dead_mask_3ch,
        cv2.addWeighted(original, 1 - dead_alpha, red_layer, dead_alpha, 0),
        overlay
    )
    
    return overlay


def draw_contours(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Dibuja contornos externos de una máscara sobre la imagen.
    
    Args:
        image: Imagen de salida.
        mask: Máscara binaria.
        color: Color de los contornos (BGR).
        thickness: Grosor de línea.
    
    Returns:
        np.ndarray: Imagen con contornos dibujados.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)
    return image
