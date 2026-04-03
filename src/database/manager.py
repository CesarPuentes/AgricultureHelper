import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

DB_PATH = os.path.join(os.getcwd(), "agriculture.db")

def init_db():
    """Initializes the SQLite database and creates the sensor_readings table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            air_humidity_rh FLOAT,
            soil_moisture_pct FLOAT,
            temperature_c FLOAT,
            light_lux FLOAT,
            living_coverage_pct FLOAT,
            plant_count INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def save_readings(readings: List[Dict[str, Any]]):
    """
    Saves a list of readings to the database.
    Each reading should be a dictionary with keys matching the table columns.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = '''
        INSERT INTO sensor_readings (
            plant_id, timestamp, air_humidity_rh, soil_moisture_pct, 
            temperature_c, light_lux, living_coverage_pct, plant_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    '''
    
    data_to_insert = [
        (
            r.get('plant_id'),
            r.get('timestamp'),
            r.get('air_humidity_rh'),
            r.get('soil_moisture_pct'),
            r.get('temperature_c'),
            r.get('light_lux'),
            r.get('living_coverage_pct'),
            r.get('plant_count')
        ) for r in readings
    ]
    
    cursor.executemany(query, data_to_insert)
    conn.commit()
    conn.close()

def get_recent_readings(plant_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieves the most recent readings for a specific plant."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM sensor_readings 
        WHERE plant_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (plant_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_latest_reading(plant_id: str) -> Dict[str, Any] | None:
    """Returns the single most recent reading for a given plant_id."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM sensor_readings
        WHERE plant_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (plant_id,))

    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_reading_n_hours_ago(plant_id: str, hours: int = 4, reference_time: datetime | None = None) -> Dict[str, Any] | None:
    """
    Returns the reading closest to `hours` hours ago for a given plant_id.
    
    Args:
        plant_id: Sensor/plant identifier.
        hours: How many hours ago to look.
        reference_time: Reference "now" time. Defaults to datetime.now().
    """
    ref = reference_time or datetime.now()
    target_time = ref - timedelta(hours=hours)
    target_ts = target_time.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT *, ABS(
            CAST(strftime('%s', timestamp) AS INTEGER) -
            CAST(strftime('%s', ?) AS INTEGER)
        ) AS time_diff
        FROM sensor_readings
        WHERE plant_id = ?
        ORDER BY time_diff ASC
        LIMIT 1
    ''', (target_ts, plant_id))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    result = dict(row)
    result.pop('time_diff', None)
    return result


def get_all_plant_ids() -> List[str]:
    """Returns a list of all distinct plant_ids in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT DISTINCT plant_id FROM sensor_readings')
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ids


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
