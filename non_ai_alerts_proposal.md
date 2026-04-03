# 🧬 Selección de Alertas (Edge / Tier 1)

Basado en las restricciones operativas:
1.  **Frecuencia**: Mediciones cada 1 hora.
2.  **Usuario**: Agricultores (requiere máxima automatización, cero configuración compleja).
3.  **Computación**: Permite ML clásico ligero (SciKit-Learn).
4.  **Escalabilidad**: Idealmente universal, o con configuración manual mínima (Base de datos local).

El panel ha descartado las opciones que requieren alta frecuencia (como Derivadas por segundo) o infraestructura compleja (APIs externas garantizadas) y ha seleccionado las **3 estrategias ganadoras**.

---

## 🏆 Top 3 Estrategias a Implementar

### 1. El Enfoque Universal (Ganador Absoluto): "Z-Score / Isolation Forest (Carga ML Ligera)"
Esta es la solución perfecta para el requisito de **"automatización para granjeros sin configurar umbrales"** y es **100% universal** para cualquier cultivo.

*   **¿Por qué es universal?**: No necesita saber si es un cactus (que vive feliz al 10% de humedad) o un helecho (que necesita 80%). Simplemente aprende lo que es "normal" para *esa maceta específica* observando los últimos N días.
*   **Mecánica (Estadística Básica - Z-Score)**:
    Calcula la media móvil de las últimas 48 horas (48 lecturas). Si la lectura actual (ej. humedad) está a más de 2.5 desviaciones estándar de esa media, lanza alerta: *"Caída de humedad inusual respecto a los últimos 2 días."*
*   **Mecánica (ML Ligero - Isolation Forest)**:
    Si queremos cruzar variables (ej. Temperatura ALTA + Humedad BAJA), entrenamos un `IsolationForest` de SciKit-Learn con las últimas 100 lecturas. ¡Se entrena en milisegundos en una Raspberry Pi!
*   **Veredicto**: Implementación obligatoria. Es genérico, no requiere intervención del granjero y detecta fallos de los propios sensores.

---

### 2. El Enfoque Fisiológico Universitario: "VPD (Deficiencia de Presión de Vapor)"
La temperatura o la humedad solas no significan nada aisaldamene. El VPD es la "presión" real que siente la planta.

*   **¿Por qué funciona con 1 dato por hora?**: El estrés hídrico no es instantáneo. Monitorear el VPD cada hora nos da una curva perfecta del día.
*   **¿Es Universal?**: **NO totalmente, pero es categorizable**. 
    *   Casi todas las plantas de invernadero/interior comparten un rango de VPD saludable (0.8 kPa a 1.2 kPa).
    *   Podemos tener una mini base de datos SQLite con 3 perfiles genéricos: *Tropical, Templado, Seco* y pedirle al granjero que elija uno al registrar la planta.
*   **Mecánica**: Una simple fórmula matemática que cruza Temperatura y Humedad Relativa. Si excede 1.5 kPa por más de 2 lecturas consecutivas (2 horas): *"Alerta: Estrés hídrico severo inminente (VPD Crítico)".*

---

### 3. El Guardián de Hardware: "Watchdog y Deltas de Cobertura Crítica"
Para proteger la inversión física.

*   **Watchdog (Comunicaciones)**: Si la base de datos no recibe el "ping" horario del sensor en 2h y 10m -> *"Alerta: Sensor desconectado o batería muerta"*.
*   **Deltas de Visión (La última línea de defensa)**:
    *   Toma la lectura actual de % Verde.
    *   Compara con la lectura de hace 4 horas.
    *   Si cayó más de un 5% absoluto (ej. de 40% a 35%): *"Alerta Visual Crítica: Marchitamiento veloz o daño estructural detectado."*
    *   **Universalidad**: Es 100% universal. Ninguna planta sana pierde el 5% de su masa verde en 4 horas de manera natural.

---

## 🛠️ Plan de Implementación Recomendado (MVP Alertas)

Para el motor local de alertas (que podemos correr como una tarea de fondo o al momento de registrar una nueva lectura), seguiremos esta arquitectura de "Capas de Seguridad":

1.  **Capa 0 (Hardware)**: El script verifica si el dato llegó a tiempo (Watchdog).
2.  **Capa 1 (Universal - Visión)**: ¿La planta perdió masa verde inexplicablemente rápido hoy? (Delta).
3.  **Capa 2 (Botánico - Física)**: ¿El clima actual genera un "efecto secadora" mortal para este tipo de planta? (VPD según perfil en BD local).
4.  **Capa 3 (ML Universal - Sensores)**: ¿Las lecturas del suelo o ambiente están completamente desfasadas de lo que esta planta ha experimentado en la última semana? (Z-Score / Isolation Forest).

---

## 💬 Respuesta del Panel a tus Sugerencias

Has planteado dos puntos críticos que elevan el diseño del sistema. Aquí está la evaluación del panel:

### 1. Sobre la "Base de Datos de Cultivos Pre-Cargada" (Setup Cero)
**El panel está 100% de acuerdo.** Has tocado un concepto clave en UX para AgTech: el "Time-to-Value".
*   **Implementación Sugerida**: Junto a `agriculture.db`, podemos incluir un archivo `crop_profiles.json` (o una tabla estática) con unos 20-30 cultivos comunes (Tomate, Lechuga, Cannabis, Fresa, etc.) y sus umbrales óptimos de VPD, humedad y temperatura.
*   **Flujo**: Cuando el granjero registra una nueva bandeja en la UI (`frontend_test.py`), simplemente selecciona "Tomate" en un menú desplegable. El sistema carga automáticamente las tolerancias.
*   **Rol del ML ligero (Isolation Forest)**: Servirá como *red de seguridad*. Actuará sobre las variables no contempladas en el perfil estático, o ajustará los parámetros pre-cargados si nota que, en el invernadero específico del usuario, el "Tomate" se comporta ligeramente distinto al manual.

### 2. ¿Dónde cabe el Agente de IA si esto ya funciona solo?
Si el sistema determinístico es tan robusto, ¿para qué queremos un LLM/Agente? El panel responde con el **Patrón del Ruteador de Investigación (Investigation Router)**:

El motor determinístico (las Capas 0 a 3) genera **Síntomas**, no **Diagnósticos**.
*   **Lo que hace el determinismo**: Lanza una alerta: *"¡Ring ring! El Delta de Cobertura Visual cayó 8% y la Humedad del suelo está normal"*. 
*   **Lo que hace el grangero (sin IA)**: Recibe el SMS, tiene que dejar lo que está haciendo, ir al invernadero, mirar las hojas, buscar en Google imágenes de enfermedades, y adivinar qué pasa.
*   **Donde entra el Agente de IA (Supervisor/Visión)**: El Agente no está ahí para *mirar los sensores todo el día* (eso es caro y propenso a errores). El Agente **duerme** hasta que el motor determinístico lo despierta con un síntoma. 
    1.  **Despertar**: El motor detecta la caída del 8% de verdor.
    2.  **Investigación**: El Agente toma la última foto de la cámara, lee el historial de la base de datos (Z-scores, VPD) y razona: *"El suelo está mojado, pero el VPD fue altísimo ayer. La foto muestra bordes amarillos. Esto es quemadura por nutrientes o estrés por calor, no falta de agua"*.
    3.  **Acción**: El Agente envía un mensaje natural al granjero: *"Hola, detecté marchitamiento rápido en la Bandeja 4. Revisé los sensores y el riego está bien, pero el calor de ayer afectó las hojas. Te sugiero encender el extractor extra hoy. ¿Quieres que lo haga por ti?"*

**Conclusión**: El código determinístico es el **Sistema Nervioso Periférico** (reflejos rápidos, baratos, infalibles). El Agente de IA es la **Corteza Cerebral** (razonamiento lento, caro, solo se usa cuando el problema requiere inteligencia real). Un sistema no reemplaza al otro; dependen mutuamente.

*No necesitamos agentes LLM para estas decisiones, solo buenas matemáticas puras de Python que disparan una campana de alarma (SMS, e-mail o UI) para que el granjero tome el control.*
