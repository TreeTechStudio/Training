"""Pruebas de las funciones estadísticas con respuestas conocidas.

Uso: python -m pytest tests/test_estadistica.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import estadistica as e  # noqa: E402


def test_acierto_minimo_85():
    assert abs(e.acierto_minimo(0.85) - 0.5405) < 1e-3


def test_valor_esperado_en_punto_de_equilibrio_es_cero():
    assert abs(e.valor_esperado(e.acierto_minimo(0.85), 0.85)) < 1e-12


def test_wilson_contiene_proporcion_y_es_simetrico_en_50():
    lo, hi = e.wilson(50, 100)
    assert lo < 0.5 < hi and abs((0.5 - lo) - (hi - 0.5)) < 1e-9


def test_60_de_100_independientes_es_significativo_contra_50():
    r = e.prueba_acierto(600, 1000, h=1, objetivo=0.5)
    assert r["p"] < 1e-9


def test_mismo_acierto_con_solapamiento_pierde_significancia():
    # 6000 de 10000 con h=100 -> solo 100 muestras independientes: ya no es contundente
    sin = e.prueba_acierto(6000, 10000, h=1, objetivo=0.5)["p"]
    con = e.prueba_acierto(6000, 10000, h=100, objetivo=0.5)["p"]
    assert con > sin * 1e6 and con > 0.01


def test_50_por_ciento_no_es_significativo():
    assert e.prueba_acierto(500, 1000, h=1, objetivo=0.5)["p"] > 0.4


def test_no_solapadas():
    t = pd.to_datetime(pd.Series(np.arange(0, 100)), unit="s", utc=True)
    df = pd.DataFrame({"hora_utc": t})
    assert len(e.no_solapadas(df, 5)) == 20


def test_drawdown():
    assert e.max_drawdown(np.array([1, 2, 1, 0, 3])) == 2.0


def test_autocorrelacion_senal_alternante():
    x = np.tile([1.0, -1.0], 500)
    assert e.autocorrelacion(x, 2)[0] < -0.99


def test_veredicto_muestra_insuficiente():
    assert "insuficiente" in e.veredicto(0.001, 0.9, 0.54, n_ind=50)
