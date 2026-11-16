# Guía paso a paso (de cero a investigación presentable)

Cada paso tiene: qué aprendes, qué hace Claude, qué haces tú (lo que Claude no puede hacer desde fuera) y cómo sabes que terminó. Cada paso produce una sección de tu futuro paper.

## Paso 0 — Preparar el laboratorio  ✅ casi listo
- Hecho: carpeta, Python 3.12, librerías, módulo de conexión.
- En descarga: PyTorch con GPU (2.7 GB).
- **Tú (5 min):** en MetaTrader 5 → Archivo → *Abrir una cuenta demo* (elige cualquier broker de la lista, deja los datos por defecto). Abajo a la derecha deben verse dos barras de conexión con números, no "sin conexión". Luego: Herramientas → Opciones → Expert Advisors → marca *Permitir trading algorítmico* (solo para pruebas futuras; hoy no enviamos órdenes).
- **Prueba de éxito:** `python src/fase1_descarga.py EURUSD M5 20000` imprime velas y crea `data/raw/EURUSD_M5.csv`. Si sale `IPC timeout`, aún no hay conexión: cierra MT5, ábrelo de nuevo, espera a ver precios y reintenta.
- Sección del paper: *Datos y entorno experimental*.

## Paso 1 — Entender los datos (Fase 1)
- Aprendes: qué es una vela (OHLC), tick volume, spread, zona horaria, huecos de fin de semana.
- Claude: notebook que grafica, cuenta huecos y describe el retorno (media, desviación, colas).
- Tú: mirar 3 gráficas y responder "¿qué me sorprende?".
- Éxito: dataset limpio de ≥ 50 000 velas y tabla descriptiva.
- Paper: *Descripción del dataset*.

## Paso 2 — Línea base y regresión lineal (Fase 2)
- Aprendes: variables, target, correlación, error, sobreajuste, por qué "predecir 0" es difícil de vencer.
- Claude: construye features sin fuga de información, entrena Ridge, mide MAE/RMSE/R² y precisión de dirección con intervalo de confianza.
- Tú: leer la tabla de coeficientes y explicar con tus palabras qué variable pesa más.
- Éxito: tabla modelo vs. línea base.
- Paper: *Metodología* + *Resultados I*.

## Paso 3 — Árboles de decisión (Fase 3)
- Aprendes: reglas, profundidad, bosques, importancia de variables.
- Éxito: curva "profundidad vs. error" que muestra dónde empieza el sobreajuste.

## Paso 4 — Redes neuronales (Fase 4)
- Aprendes: neurona, capas, activación, pérdida, gradiente. Primero una red hecha a mano con NumPy (≈ 40 líneas), luego PyTorch en tu GPU.
- Éxito: curva de pérdida entrenamiento vs. validación; prueba de velocidad GPU vs. CPU.

## Paso 5 — Comparar y combinar (Fase 5)
- Aprendes: ensemble, stacking, walk-forward, significancia estadística.
- Éxito: una tabla final y una conclusión honesta sobre H0/H1 (aunque sea "no se supera la línea base").

## Paso 6 — Factores externos
- Aprendes: calendario económico de MT5, titulares (RSS/GDELT), LLM local (Ollama) que convierte noticias en números, ablación (¿mejora con y sin la variable?).
- **Tú:** encender Ollama y descargar un modelo (te doy el comando exacto cuando lleguemos).
- Paper: *Variables exógenas*.

## Paso 7 — MQL5 y MT5
- Aprendes: exportar a ONNX, EA/indicador en MetaEditor, Strategy Tester.
- **Tú:** pulsar *Compilar* (F7) en MetaEditor y ejecutar el Strategy Tester; Claude no puede manejar la interfaz de MT5 por sí solo con fiabilidad, pero te guía clic a clic o por captura.
- Paper: *Despliegue y pruebas*.

## Paso 8 — Ejecución continua
- Programador de tareas de Windows: descarga incremental, reentrenamiento nocturno, informe diario.

## Paso 9 — Redacción
`docs/paper.md`: Resumen · Introducción · Trabajos relacionados · Datos · Método · Resultados · Discusión · Limitaciones · Conclusión · Referencias.

## Qué no puede hacer Claude desde fuera (queda de tu lado)
1. Iniciar cuenta demo y dejar MT5 conectado.
2. Pulsar botones dentro de MT5/MetaEditor (compilar, Strategy Tester) salvo con captura de pantalla asistida.
3. Aprobar descargas grandes o instalaciones de programas (Ollama, modelos de IA).
4. Decidir el activo, la temporalidad y el riesgo que aceptas.

## Glosario mínimo
Vela: resumen de precios de un intervalo · Spread: costo de entrar y salir · Feature: variable de entrada · Target: lo que se predice · Sobreajuste: memorizar el pasado y fallar en lo nuevo · Fuga de información: usar datos del futuro sin querer · Walk-forward: entrenar con pasado, evaluar en lo siguiente, avanzar · ONNX: formato para llevar el modelo a MQL5.
