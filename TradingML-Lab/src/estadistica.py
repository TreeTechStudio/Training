"""Estadística del informe (funciones puras, probadas en tests/test_estadistica.py).

Idea central: las predicciones se hacen cada segundo pero cada operación dura h
segundos, así que se SOLAPAN. Dos predicciones consecutivas comparten casi todo su
resultado. Por eso el número de muestras independientes es ~ n/h, y los intervalos
y p-valores se calculan con ese número (si no, salen engañosamente optimistas).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def n_independiente(n: int, h: int) -> float:
    return max(n / max(h, 1), 1.0)


def wilson(aciertos: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción (mejor que ±1.96·√(p(1-p)/n))."""
    if n <= 0:
        return (np.nan, np.nan)
    p = aciertos / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - m, c + m)


def prueba_acierto(aciertos: int, n: int, h: int, objetivo: float) -> dict:
    """Prueba de una cola: ¿el acierto real es MAYOR que `objetivo`?

    Usa n efectivo = n/h (muestras independientes). Devuelve acierto, IC95 (ingenuo y
    efectivo) y p-valor binomial exacto.
    """
    if n == 0:
        return dict(n=0, n_ind=0, acierto=np.nan, ic_ing=(np.nan, np.nan), ic_ind=(np.nan, np.nan), p=np.nan)
    acc = aciertos / n
    n_ind = n_independiente(n, h)
    k_ind = acc * n_ind
    p = stats.binomtest(int(round(k_ind)), int(round(n_ind)), objetivo, alternative="greater").pvalue
    return dict(n=n, n_ind=n_ind, acierto=acc, ic_ing=wilson(aciertos, n), ic_ind=wilson(k_ind, n_ind), p=p)


def valor_esperado(acc: float, pago: float) -> float:
    """Ganancia esperada por cada 1 apostado: aciertas -> +pago, fallas -> -1."""
    return acc * pago - (1 - acc)


def acierto_minimo(pago: float) -> float:
    return 1 / (1 + pago)


def no_solapadas(df, h: int, col_tiempo: str = "hora_utc"):
    """Submuestra de operaciones que NO se solapan (una cada h segundos).

    Es lo que podrías operar de verdad con una sola posición abierta a la vez.
    """
    # No usar astype("int64"): la unidad de tiempo (s, µs, ns) cambia según la versión de pandas.
    t = (pd.to_datetime(df[col_tiempo], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    ultimo = None
    keep = []
    for i, ti in enumerate(t):
        if ultimo is None or ti - ultimo >= h:
            keep.append(i)
            ultimo = ti
    return df.iloc[keep]


def max_drawdown(curva: np.ndarray) -> float:
    if len(curva) == 0:
        return 0.0
    pico = np.maximum.accumulate(np.concatenate([[0.0], curva]))[1:]
    return float((pico - curva).max())


def autocorrelacion(x: np.ndarray, max_lag: int = 10) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    v = (x * x).sum()
    if v == 0 or len(x) <= max_lag:
        return np.full(max_lag, np.nan)
    return np.array([(x[:-k] * x[k:]).sum() / v for k in range(1, max_lag + 1)])


def veredicto(p_ajustado: float, acc: float, objetivo: float, n_ind: float, minimo_muestras: int = 300) -> str:
    """Texto neutral y prudente sobre qué permiten concluir los datos."""
    if n_ind < minimo_muestras:
        return f"Muestra insuficiente (≈{n_ind:.0f} independientes; se recomiendan ≥{minimo_muestras})."
    if acc > objetivo and p_ajustado < 0.05:
        return "Evidencia estadística de superar el acierto mínimo; replicar en otro día antes de creerlo."
    if acc > objetivo:
        return "Supera el mínimo en la muestra, pero sin significancia: puede ser azar."
    return "Sin evidencia de superar el acierto mínimo."
