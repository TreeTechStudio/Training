# Plan maestro — TradingML-Lab

Laboratorio local de Machine Learning para predicción en mercados financieros, con rigor de investigación. Solo lectura de datos; sin órdenes reales.

## 1. Pregunta de investigación
¿Puede un modelo (regresión lineal, árbol, red neuronal o su combinación) predecir el retorno de la siguiente vela de EURUSD M5 mejor que una línea base ingenua, **después de costos** (spread), y de forma estable fuera de muestra?

## 2. Hipótesis (se formulan antes de ver resultados)
- H0: ningún modelo supera a la línea base (retorno = 0 / dirección = azar) fuera de muestra. Es lo que predice la hipótesis de mercado eficiente en forma débil.
- H1: al menos un modelo tiene dirección acertada significativamente > 50 % y expectativa positiva tras spread, en varios periodos walk-forward.
- Resultado negativo = resultado válido y publicable. El objetivo es aprender a medir, no "encontrar un grial".

## 3. Reglas metodológicas (innegociables)
1. Sin fuga de información: toda feature usa solo velas cerradas hasta t; el target es el retorno de t a t+1.
2. Split cronológico: train → validación → test final intocable. Nunca aleatorio.
3. Walk-forward (ventana expansiva o rodante) para estimar estabilidad.
4. Líneas base obligatorias: retorno 0, "mismo signo que la vela previa", media móvil simple.
5. Métricas predictivas (MAE, RMSE, R² OOS, accuracy direccional con intervalo de confianza) **y** financieras (retorno neto tras spread, drawdown máx., factor de beneficio, nº de operaciones).
6. Control de múltiples pruebas: registrar TODOS los experimentos; no elegir el mejor a posteriori sin corrección (riesgo de sobreajuste de backtest).
7. Reproducibilidad: semillas fijas, versiones en requirements.txt, un registro por experimento (`reports/experimentos.csv`).
8. Demo antes que real; real fuera del alcance de este proyecto.

## 4. Fases y entregables
| Fase | Contenido | Entregable | Criterio de salida |
|---|---|---|---|
| 0 | Entorno (Python 3.12 + venv), estructura | Proyecto corriendo | `import MetaTrader5` OK |
| 1 | Conexión MT5 y dataset EURUSD M5 | CSV en data/raw + notebook de exploración | ≥ 50 000 velas sin huecos anómalos |
| 2 | Features + línea base + regresión lineal (Ridge/Lasso) | Informe con coeficientes y OOS | Métricas vs. base con IC |
| 3 | Árbol de decisión / Random Forest / Gradient Boosting | Importancia de variables, curva de profundidad | Comparación justa en mismo split |
| 4 | Red MLP (sklearn) → PyTorch en GPU RTX 5050; luego LSTM/1D-CNN | Curvas de pérdida, early stopping | Sin sobreajuste evidente |
| 5 | Ensemble y stacking con predicciones out-of-fold | Tabla comparativa final + test de significancia | Conclusión sobre H0/H1 |
| 6 | Exportar a ONNX y consumir en MQL5 (indicador), Strategy Tester, luego demo | EA/indicador experimental | Resultado coherente con Python |
| 7 | Redacción tipo paper | `docs/paper.md` | Intro, método, resultados, limitaciones |

## 5. Fase 2–5 en detalle (lo que aprenderás)
- Fase 2: variables, correlación, mínimos cuadrados, regularización, sesgo/varianza, sobreajuste.
- Fase 3: impureza Gini, profundidad, poda, bagging vs. boosting, importancia por permutación.
- Fase 4: neurona, activación, pérdida, backpropagation, SGD/Adam, normalización, dropout. Implementar una MLP mínima con NumPy para entender el gradiente antes de usar PyTorch.
- Fase 5: stacking correcto (predicciones out-of-fold para evitar fuga), calibración, ponderación.
- Fase 6: ONNX es la vía oficial para ejecutar el modelo dentro de MQL5.

## 6. Extensiones posteriores
Más activos (GBPUSD, XAUUSD), otras temporalidades, features de régimen de volatilidad, clasificación 3 clases (+/−/0) con umbral por costos, ticks para escalas de segundos.

## 7. Riesgos conocidos
Ruido alto y señal baja en M5; el spread puede borrar una ventaja pequeña; cambios de régimen; sobreajuste de backtest. Mitigación: sección 3.

## 8. Referencias
- MetaQuotes. MetaTrader 5 Python integration. https://www.mql5.com/es/docs/python_metatrader5
- MetaQuotes. Neural networks for algorithmic trading with MQL5 (NeuroBook). https://www.mql5.com/es/neurobook
- López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
- Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning. Springer.
- Goodfellow, Bengio & Courville (2016). Deep Learning. MIT Press.
