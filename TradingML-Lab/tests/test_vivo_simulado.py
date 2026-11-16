"""Prueba olymp_vivo.py con precios SIMULADOS (paseo aleatorio, sin pantalla).

En un paseo aleatorio nadie puede predecir: todos los modelos deben quedar cerca
del 50 %. Si alguno sale muy por encima, habría un error (fuga de información).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.olymp_vivo as vivo  # noqa: E402

rng = np.random.default_rng(0)
estado = {"p": 3400.0, "n": 0}


def precio_falso():
    estado["n"] += 1
    if estado["n"] > 1500:
        raise KeyboardInterrupt
    estado["p"] = round(estado["p"] * (1 + rng.normal(0, 2e-5)), 4)
    return estado["p"]


vivo.leer_precio = precio_falso
vivo.time.sleep = lambda s: None
vivo.ROOT = Path(__file__).resolve().parent / "tmp_sim"
(vivo.ROOT / "data" / "pantalla").mkdir(parents=True, exist_ok=True)
vivo.print = lambda *a, **k: None if a and "precio" in str(a[0]) else __builtins__.print(*a, **k)
vivo.main(5, 0.85)
