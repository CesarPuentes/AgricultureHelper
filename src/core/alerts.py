"""
alerts.py — Capa 0: Motor Determinístico de Alertas
----------------------------------------------------
Guardián de Hardware: Watchdog (sensor desconectado) y
Deltas de Visión (marchitamiento veloz).

Diseñado para correr cada hora como tarea de fondo.
No requiere configuración del granjero — es 100% universal.
"""

from datetime import datetime, timedelta
from typing import Optional

from src.database.manager import get_latest_reading, get_reading_n_hours_ago
from src.models.models import AlertResult


# ---------------------------------------------------------------------------
# Capa 0a: Watchdog — Sensor desconectado
# ---------------------------------------------------------------------------
def check_watchdog(
    plant_id: str,
    max_silence_minutes: int = 130,
    now: Optional[datetime] = None,
) -> Optional[AlertResult]:
    """
    Verifica que el sensor esté vivo.

    Si la base de datos no ha recibido el "ping" horario de `plant_id`
    en los últimos `max_silence_minutes` (default 2h 10min = 130 min),
    genera una alerta de tipo watchdog.

    Args:
        plant_id: Identificador del sensor/planta.
        max_silence_minutes: Minutos máximos de silencio permitidos.
        now: Hora actual (inyectable para tests).

    Returns:
        AlertResult si el sensor está desconectado, None si todo OK.
    """
    now = now or datetime.now()
    latest = get_latest_reading(plant_id)

    if latest is None:
        return AlertResult(
            alert_type="watchdog",
            severity="critical",
            plant_id=plant_id,
            message=f"Alerta: Sin datos registrados para sensor '{plant_id}'. "
                    f"Verificar conexión o registro inicial.",
            triggered_at=now,
            details={"last_reading_timestamp": None},
        )

    # Parse timestamp — soporta ISO 8601 y formato SQLite
    ts_raw = latest["timestamp"]
    if isinstance(ts_raw, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                last_ts = datetime.strptime(ts_raw, fmt)
                break
            except ValueError:
                continue
        else:
            last_ts = datetime.fromisoformat(ts_raw)
    else:
        last_ts = ts_raw

    silence = now - last_ts
    if silence > timedelta(minutes=max_silence_minutes):
        return AlertResult(
            alert_type="watchdog",
            severity="critical",
            plant_id=plant_id,
            message=f"Alerta: Sensor '{plant_id}' desconectado o batería muerta. "
                    f"Sin datos por {silence.total_seconds() / 60:.0f} minutos.",
            triggered_at=now,
            details={
                "last_reading_timestamp": str(last_ts),
                "silence_minutes": round(silence.total_seconds() / 60, 1),
                "threshold_minutes": max_silence_minutes,
            },
        )

    return None


# ---------------------------------------------------------------------------
# Capa 0b: Delta de Visión — Marchitamiento veloz
# ---------------------------------------------------------------------------
def check_vision_delta(
    plant_id: str,
    hours: int = 4,
    threshold_pct: float = 5.0,
) -> Optional[AlertResult]:
    """
    Detecta pérdida rápida de cobertura verde.

    Compara el % de cobertura verde actual con la lectura de hace
    `hours` horas. Si la caída supera `threshold_pct` puntos porcentuales
    absolutos, genera alerta crítica.

    Universalidad: Ninguna planta sana pierde 5% de masa verde en 4h.

    Args:
        plant_id: Identificador del sensor/planta.
        hours: Ventana de comparación en horas.
        threshold_pct: Caída mínima (en puntos porcentuales) para alertar.

    Returns:
        AlertResult si se detecta marchitamiento, None si todo OK.
    """
    current = get_latest_reading(plant_id)
    if current is None or current.get("living_coverage_pct") is None:
        return None

    past = get_reading_n_hours_ago(plant_id, hours=hours)
    if past is None or past.get("living_coverage_pct") is None:
        return None

    current_pct = current["living_coverage_pct"]
    past_pct = past["living_coverage_pct"]
    delta = past_pct - current_pct

    if delta > threshold_pct:
        return AlertResult(
            alert_type="vision_delta",
            severity="critical",
            plant_id=plant_id,
            message=f"Alerta Visual Crítica: Marchitamiento veloz o daño estructural "
                    f"detectado en '{plant_id}'. Cobertura verde cayó "
                    f"{delta:.1f}% en {hours}h (de {past_pct:.1f}% a {current_pct:.1f}%).",
            triggered_at=datetime.now(),
            details={
                "current_coverage_pct": current_pct,
                "past_coverage_pct": past_pct,
                "delta_pct": round(delta, 2),
                "window_hours": hours,
                "threshold_pct": threshold_pct,
            },
        )

    return None


# ---------------------------------------------------------------------------
# Orquestador: Ejecuta todos los checks para un sensor
# ---------------------------------------------------------------------------
def run_all_checks(plant_id: str) -> list[AlertResult]:
    """
    Ejecuta todos los checks de Capa 0 para un plant_id.
    Retorna lista de alertas activas (vacía si todo está bien).
    """
    alerts: list[AlertResult] = []

    watchdog = check_watchdog(plant_id)
    if watchdog:
        alerts.append(watchdog)

    vision = check_vision_delta(plant_id)
    if vision:
        alerts.append(vision)

    return alerts