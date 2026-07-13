## Coherence v0.1.0 — Analizador de audio en tiempo real

### Características principales

**Función de transferencia**
- Hasta 2 engines de medición simultáneos con colores independientes (picker de color por engine)
- Promedio exponencial estilo SMAART con control de profundidad
- Suavizado 1/3, 1/6, 1/12, 1/24 de octava (aplicado simultáneo a magnitud y fase)
- Overlay de coherencia γ² con umbral configurable
- Línea de AVG dominante cuando está activo

**Respuesta al impulso**
- Eje fijo ±100 ms con labels en la parte superior del panel
- Delay finder por correlación cruzada (ms / muestras / metros)
- Normalización por engine independiente

**Fase**
- Panel dividido o integrado (Magnitude + Phase simultáneos)
- Línea de AVG en blanco con coherencia aplicada como máscara

**Espectro RTA**
- 1/3 de octava (CPB), respuesta plana a ruido rosa
- Hasta 2 engines de espectro simultáneos
- Averaging depth configurable

**Transport por engine**
- Cada engine ▶ abre su propio stream de audio independientemente
- Main ▶ activa/desactiva todos los engines a la vez
- Engines permanecen activos si el otro se detiene

**Trazas de referencia**
- Guardar traza con nombre y color personalizado
- Magnitud y fase se guardan juntas y se muestran enlazadas
- Hasta 7 trazas simultáneas

**Otros**
- Calibración de micrófono (.txt con pares freq/dB)
- Espectrograma en tiempo real (colormap inferno)
- Menús completos con shortcuts de teclado
- Exportar PNG de la vista activa

### Instalación macOS
1. Descargar `Coherence-macOS.zip`
2. Descomprimir → arrastrar `Coherence.app` a `/Aplicaciones`
3. Primera apertura: clic derecho → **Abrir** (para saltar Gatekeeper)

Requiere interfaz de audio (probado con Focusrite Scarlett 18i8).
