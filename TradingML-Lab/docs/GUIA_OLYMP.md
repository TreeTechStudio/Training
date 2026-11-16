# Guía sencilla — Olymp Trade en vivo

## La idea en una frase
El programa **lee el precio de tu pantalla de Olymp**, predice "Arriba/Abajo" para dentro
de N segundos, espera, revisa si acertó y **aprende de cada resultado**. Nunca hace clic:
es una bitácora de operaciones "en papel" para medir si el modelo sirve.

```
 Pantalla Olymp ──► lee precio (OCR de Windows) ──► variables ──► 3 modelos predicen
       ▲                                                              │
       └────────── a los N segundos: ¿acertó? ── anota ── la IA aprende ┘
```

## Cómo usarlo (2 pasos)
1. Abre Olymp, elige el activo y **maximiza la ventana**.
2. Doble clic en `INICIAR_OLYMP_VIVO.bat` → escribe la duración (5) y el pago (0.85).
   Verás cada segundo la señal de la IA y, cada 30 s, un **marcador**.

## Cómo leer el marcador
```
--- Marcador (meta > 54.05%) ---
  Repetir movimiento   50.45% ±2.59% (n=1433)
  Llevar la contraria  49.55% ±2.59% (n=1433)
  IA (aprende)         48.32% ±2.65% (n=1370)
```
- **Repetir movimiento** y **Llevar la contraria** son reglas tontas (líneas base). La IA
  solo vale algo si les gana.
- **Meta 54.05 %**: con pago 85 % ganas 0.85 si aciertas y pierdes 1 si fallas. Por debajo
  de ese acierto, a la larga se pierde.
- **±**: margen de error. Si el rango (48.32 − 2.65 hasta 48.32 + 2.65) toca 54 %, aún no
  hay evidencia. Hacen falta **cientos o miles** de predicciones para concluir algo.
- El ejemplo de arriba es con precios aleatorios simulados: ahí nadie puede ganar, y por eso
  todos quedan cerca de 50 %. Sirve para comprobar que el programa no hace trampa.

## Archivos que genera (tu material de investigación)
| Archivo | Qué contiene |
|---|---|
| `data/pantalla/ticks_FECHA.csv` | Precio de cada segundo (para entrenar modelos más grandes después, con la GPU) |
| `data/pantalla/bitacora_FECHA.csv` | Cada predicción: precio de entrada/salida, resultado y qué dijo cada modelo |

## Cosas importantes para tu investigación
1. **Quickler y los activos OTC los genera Olymp**, no son mercado real. Noticias o guerras no
   les afectan como a EUR/USD. Puedes comparar Quickler contra EUR/USD (Forex) y ver si alguno
   es más predecible: es una buena pregunta de investigación.
2. **Pago < 100 %**: la meta no es 50 %, es 1/(1+pago).
3. **Lectura de pantalla**: si mueves/achicas la ventana o cambias de monitor, el lector deja de
   ver el precio. Mantén Olymp maximizado.
4. **Para EUR/USD real** ya tenemos además 324 mil velas de 1 minuto de 2026 (Dukascopy) y el
   experimento `src/exp01_direccion.py`.

## Siguientes pasos
| Paso | Qué | Estado |
|---|---|---|
| 1 | Lector de precio por OCR | ✅ probado (3409.9971 leído exacto) |
| 2 | Predicción en vivo + bitácora + aprendizaje | ✅ probado en simulación |
| 3 | Juntar ≥ 2 horas de datos reales de Olymp | **te toca: dejarlo corriendo** |
| 4 | Entrenar red neuronal (GPU) con los ticks guardados | siguiente |
| 5 | Noticias y eventos (LLM local Ollama) para EUR/USD | después |
| 6 | Redactar resultados tipo paper | al final |

## Glosario
- **OCR**: programa que lee texto de una imagen.
- **Línea base**: regla simple que el modelo debe superar.
- **Aprendizaje en línea**: el modelo se actualiza con cada resultado nuevo.
- **Evaluación honesta**: se predice antes de saber la respuesta; nunca al revés.


---

# Actualización: segundo monitor + informe final

## Preparar el segundo monitor (una sola vez)
1. Conéctalo y pulsa **Win + P → Extender** (no "Solo segunda pantalla").
2. Configuración → Sistema → Pantalla: elige la **pantalla de la laptop** y marca **"Convertir en pantalla principal"**. El lector solo mira la pantalla principal.
3. **No cambies la resolución ni el escalado** de la pantalla de la laptop (1920×1200): el lector busca el precio en una zona fija.
4. Maximiza Olymp en la pantalla de la laptop. Todo tu trabajo va en el segundo monitor.
5. Laptop conectada a la corriente; Configuración → Sistema → Inicio/apagado: "Nunca" suspender ni apagar pantalla mientras está conectada.

## Flujo completo
| Paso | Qué haces |
|---|---|
| 1 | Doble clic en `INICIAR_OLYMP_VIVO.bat`. Escribe el nombre del activo (Quickler, EURUSD…), la duración (5) y el pago (0.85). |
| 2 | Déjalo horas. Para cambiar de activo: Ctrl+C, cambia el activo en Olymp y vuelve a iniciar con el nombre nuevo. |
| 3 | Al terminar: Ctrl+C y doble clic en `GENERAR_INFORME.bat`. Se abre `informe.html` en el navegador. |

Cada activo se guarda por separado (`bitacora_ACTIVO_h5_p85_FECHA.csv`), así el informe puede **compararlos**.

## Qué trae el informe
- **Resumen por modelo:** predicciones, acierto, intervalo de confianza realista, valor *p* y ganancia por unidad apostada.
- **Acierto en el tiempo** y **curva de capital** (apostando 1 en cada señal, sin operaciones solapadas).
- **Estabilidad por bloques** de tiempo y **por hora del día**.
- **¿La confianza de la IA sirve?**
- **El precio por dentro:** ¿un movimiento anticipa al siguiente? (autocorrelación), volatilidad, huecos de lectura.
- **Comparación entre activos** y una sección de **límites del estudio** para tu discusión.
- Tablas sueltas en `.csv` dentro de la misma carpeta, por si quieres usarlas en tu paper.

## Cómo leer los números (importante)
- **Independientes ≈ n/h:** se predice cada segundo pero cada operación dura 5 s, así que las predicciones se solapan. Solo ≈ una de cada 5 aporta información nueva. El informe usa ese número para ser honesto; el marcador en vivo ya lo hace también.
- **Valor p:** probabilidad de ver ese acierto *solo por suerte*. Menos de 0.05 (ajustado por probar 3 modelos) es el umbral mínimo para tomarlo en serio, y aun así hay que **repetirlo otro día**.
- **Ganancia por 1 apostado:** positiva = el modelo ganaría dinero (en demo); negativa = perdería.

## Cómo sabemos que el informe no engaña (validación hecha)
Se probó con datos simulados: en **azar puro** no encontró ventaja (acierto ≈ 51 %, p ajustado = 1.0), y con una **tendencia plantada** sí la detectó (80.6 % y 72.5 %, p < 0.001). Además hay 10 pruebas automáticas de la estadística (`python -m pytest tests -q`).
