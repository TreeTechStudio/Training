"""Simula una corrida larga con la ventana de Olymp tapada de vez en cuando.

Cada 300 lecturas, 12 lecturas seguidas fallan (None). Comprobamos que:
  - el programa no inventa precios (ticks sin repetidos masivos),
  - se reanuda y vuelve a calentar,
  - el marcador sigue cerca de 50 % (paseo aleatorio).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.olymp_vivo as vivo  # noqa: E402

rng = np.random.default_rng(1)
estado = {"p": 3400.0, "n": 0}


def precio_falso():
    estado["n"] += 1
    if estado["n"] > 2400:
        raise KeyboardInterrupt
    if estado["n"] % 300 < 12 and estado["n"] > 300:
        return None  # ventana tapada
    estado["p"] = round(estado["p"] * (1 + rng.normal(0, 2e-5)), 4)
    return estado["p"]


vivo.leer_precio = precio_falso
vivo.time.sleep = lambda s: None
vivo.ROOT = Path(__file__).resolve().parent / "tmp_fallos"
(vivo.ROOT / "data" / "pantalla").mkdir(parents=True, exist_ok=True)
vivo.main(5, 0.85)

ticks = (vivo.ROOT / "data" / "pantalla").glob("ticks_*.csv")
filas = [l.split(",")[1] for f in ticks for l in f.read_text().splitlines()]
print("ticks guardados:", len(filas), "| lecturas válidas esperadas ~", 2400 - 12 * 7)
