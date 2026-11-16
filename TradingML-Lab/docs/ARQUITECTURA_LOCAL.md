# Arquitectura local — IA para trading en este equipo

Equipo: Ryzen 9 8940HX (16 hilos) · RTX 5050 8 GB · 15 GB RAM · Windows · MT5 + MetaEditor.
Principio: **Python entrena y razona; MQL5 ejecuta rápido y cerca del mercado.** Todo corre local.

## 1. Qué aporta MQL5 (y qué no)
| Capacidad | Para qué nos sirve | Límite |
|---|---|---|
| Datos nativos en tiempo real (OnTick/OnTimer, CopyRates, ticks, order book) | Alimentar el modelo con exactitud, sin capturas de pantalla | Solo variables del terminal |
| **ONNX nativo** (OnnxCreate/OnnxRun) | Ejecutar redes entrenadas en Python dentro del EA/indicador, sin red ni Python en vivo, baja latencia | Modelos compatibles; tamaños moderados |
| Strategy Tester + agentes locales | Backtest/optimización con tu CPU multinúcleo | Simula, no garantiza futuro |
| Calendario económico integrado (CalendarValueHistory) | Eventos macro (NFP, tasas, IPC) con fecha, importancia y valor real vs. previsto: **factor externo estructurado, sin internet** | Solo eventos macro, no noticias abiertas |
| WebRequest | Llamar a un servicio local (127.0.0.1) con nuestro "cerebro" en Python | Hay que autorizar la URL en opciones |
| Integración Python oficial (paquete MetaTrader5) | Descarga histórico/ticks para entrenar | Solo lectura en nuestro caso |
| OpenCL | Cálculo en GPU desde MQL5 | Más trabajo que ONNX; secundario |

## 2. Sobre "ver la pantalla"
Recomendación honesta: **no capturar pantalla como fuente principal.** Los precios exactos ya están disponibles por API (velas, ticks, spread); una captura es una versión borrosa y lenta de los mismos datos.
La captura sí tiene un uso válido como **experimento aparte**: un modelo de visión (CNN sobre imágenes de gráfico) para ver si el "patrón visual" aporta algo. Se genera renderizando las velas desde datos (no screenshots), reproducible y sin depender de la pantalla.
Para "tiempo real": un bucle que lee ticks/velas cerradas cada N segundos o en cada barra nueva.

## 3. Capas del sistema
1. **Datos de mercado** (MT5): OHLC, ticks, spread, calendario económico → `data/`.
2. **Datos externos** (Python): titulares RSS/GDELT, series macro (FRED), volatilidad (VIX), calendario. Todo con marca de tiempo de *publicación* para evitar fuga de información.
3. **Contexto con LLM local** (Ollama, ya instalado): convierte titulares en variables numéricas: sentimiento, tipo de evento (guerra, catástrofe, política monetaria), divisas afectadas, severidad 0-1. Va **a una tabla de features**, no decide operaciones.
4. **Modelos**: base (lineal/árboles) → MLP/LSTM/CNN-1D en GPU → ensemble/stacking.
5. **Evaluación**: walk-forward, spread, drawdown, registro de experimentos.
6. **Despliegue**: exportar a ONNX → EA/indicador en MQL5 → Strategy Tester → demo.

## 4. Factores externos: realismo
- Se pueden convertir en variables, pero el valor hay que **medirlo**: cada fuente entra como feature y se prueba si mejora fuera de muestra respecto del modelo sin ella (ablación).
- Los precios reaccionan en segundos a noticias; un LLM local en M5 llega tarde para el primer movimiento. Sí puede aportar **régimen** (alta incertidumbre geopolítica → volatilidad) y filtros ("no operar 30 min alrededor de NFP").
- Necesitamos históricos de noticias con fecha para entrenar (GDELT es gratuito); sin histórico no hay entrenamiento serio.

## 5. Uso del hardware
- GPU RTX 5050: entrenamiento PyTorch (CUDA 12.8+), mixed precision, lotes en GPU; LLM local cuantizado de 4-8 B parámetros cabe en 8 GB.
- Ryzen 16 hilos: feature engineering con pandas/numpy, scikit-learn con `n_jobs=-1`, agentes del Strategy Tester.
- Regla: **no entrenar y correr el LLM a la vez** en la GPU (8 GB).

## 6. Ejecución constante
- Programador de tareas de Windows (o servicio) para: descarga incremental de velas, captura de titulares cada 15 min, reentrenamiento nocturno, informe diario.
- Todo escribe a `reports/experimentos.csv` y logs; nada opera con dinero.

## 7. Riesgos
Los modelos en mercados líquidos suelen mostrar poca señal; la complejidad no garantiza edge. Más variables = más riesgo de sobreajuste. Por eso: hipótesis previa, línea base, ablación, test intocable.
