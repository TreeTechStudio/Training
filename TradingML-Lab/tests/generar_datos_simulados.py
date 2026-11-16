"""Crea datos SIMULADOS con reloj realista y genera el informe para validarlo.

  AZAR       paseo aleatorio puro: el informe NO debe encontrar ventaja.
  TENDENCIA  precio con regímenes de tendencia (sube/baja ~60 s): el informe SÍ debe
             detectar ventaja en "Repetir movimiento".

Uso: python tests/generar_datos_simulados.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
import src.olymp_vivo as vivo  # noqa: E402
from src import informe  # noqa: E402

CARPETA = AQUI / "tmp_informe"
N_LECTURAS = 9000          # ≈ 2.5 h simuladas a 1 lectura por segundo
BASE = datetime(2026, 11, 16, 18, 0, 0, tzinfo=timezone.utc)


def correr(activo: str, generador):
    estado = {"n": 0}

    class RelojFalso(datetime):
        @classmethod
        def now(cls, tz=None):
            t = BASE + timedelta(seconds=estado["n"])
            return t if tz else t.replace(tzinfo=None)

    def leer():
        estado["n"] += 1
        if estado["n"] > N_LECTURAS:
            raise KeyboardInterrupt
        return generador(estado["n"])

    vivo.leer_precio, vivo.datetime = leer, RelojFalso
    vivo.time.sleep = lambda s: None
    vivo.ROOT = CARPETA.parent / "tmp_informe_raiz"
    (vivo.ROOT / "data" / "pantalla").mkdir(parents=True, exist_ok=True)
    vivo.main(5, 0.85, activo)


def azar():
    rng = np.random.default_rng(7)
    p = {"v": 3400.0}

    def g(_):
        p["v"] = round(p["v"] * (1 + rng.normal(0, 2e-5)), 4)
        return p["v"]
    return g


def tendencia():
    rng = np.random.default_rng(8)
    p = {"v": 3400.0, "deriva": 0.0}

    def g(n):
        if n % 60 == 1:   # cada ~60 s cambia el régimen: tendencia alcista o bajista
            p["deriva"] = rng.choice([-1, 1]) * 1.2e-5
        p["v"] = round(p["v"] * (1 + p["deriva"] + rng.normal(0, 2e-5)), 4)
        return p["v"]
    return g


if __name__ == "__main__":
    import shutil
    shutil.rmtree(CARPETA.parent / "tmp_informe_raiz", ignore_errors=True)
    correr("AZAR", azar())
    correr("TENDENCIA", tendencia())
    datos = CARPETA.parent / "tmp_informe_raiz" / "data" / "pantalla"
    informe.ROOT = CARPETA.parent / "tmp_informe_raiz"
    print("\nINFORME:", informe.main(datos))
