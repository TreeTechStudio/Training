"""Predicción EN VIVO sobre Olymp (operaciones en papel: nunca hace clic).

Cada segundo:
  1. Lee el precio de la pantalla (olymp_precio.py).
  2. Calcula variables con los últimos segundos (movimiento, nerviosismo...).
  3. Cada modelo dice "Arriba" o "Abajo" para dentro de H segundos.
  4. Pasados H segundos revisa si acertó, lo anota y el modelo APRENDE de ese resultado.

Así medimos de forma honesta: cada predicción se hace ANTES de conocer la respuesta
(evaluación "predecir y luego aprender"). La meta es superar el acierto mínimo
1 / (1 + pago); con pago 85 % es 54.05 %.

Uso:  python src/olymp_vivo.py 5 0.85 Quickler   (segundos, pago, nombre del activo)
Requisito: Olymp maximizado y visible en el monitor principal. Ctrl+C para parar.
"""
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.olymp_precio import leer_precio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VENTANAS = (1, 3, 5, 10, 30, 60)
MIN_HISTORIA = 61          # segundos necesarios antes de predecir
MIN_APRENDIZAJE = 60       # ejemplos antes de que el modelo de IA empiece a opinar


def variables(p: np.ndarray) -> np.ndarray:
    """Variables con los precios de cada segundo (p[-1] = ahora). Sin mirar el futuro."""
    r = np.diff(p) / p[:-1]
    f = [p[-1] / p[-1 - n] - 1 for n in VENTANAS]                 # movimiento en n s
    f += [r[-10:].std(), r[-30:].std(), r[-60:].std()]            # nerviosismo
    f += [p[-1] / p[-10:].mean() - 1, p[-1] / p[-30:].mean() - 1]  # lejos del promedio
    sube, baja = r[-14:].clip(min=0).sum(), -r[-14:].clip(max=0).sum()
    f.append(sube / (sube + baja + 1e-12))                         # fuerza (0-1)
    return np.array(f) * np.array([1e4] * 11 + [1])               # escala legible


class Marcador:
    def __init__(self, h: int = 1):
        self.h = max(h, 1)
        self.aciertos = 0
        self.total = 0

    def anotar(self, ok: bool):
        self.aciertos += ok
        self.total += 1

    def texto(self) -> str:
        if not self.total:
            return "  —  "
        a = self.aciertos / self.total
        # Se predice cada segundo pero cada operación dura h s: las predicciones se
        # solapan y NO son independientes. Solo ~ n/h cuentan como muestras distintas.
        n_ind = max(self.total / self.h, 1)
        e = 1.96 * np.sqrt(a * (1 - a) / n_ind)
        return f"{a:6.2%} ±{e:5.2%} (n={self.total}, independientes≈{n_ind:.0f})"


def main(h: int = 5, pago: float = 0.85, activo: str = "activo") -> None:
    for flujo in (sys.stdout, sys.stderr):   # que las flechas ↑↓ nunca tumben el programa
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    minimo = 1 / (1 + pago)
    ia = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=42)
    entrenado = 0
    precios = deque(maxlen=600)
    pendientes = deque()   # (segundo, precio, X, {modelo: predicción})
    marcas = {"Repetir movimiento": Marcador(h), "Llevar la contraria": Marcador(h), "IA (aprende)": Marcador(h)}
    hoy = f"{datetime.now():%Y%m%d}"
    activo = "".join(c for c in activo if c.isalnum() or c in "-") or "activo"
    tick_f = (ROOT / "data" / "pantalla" / f"ticks_{activo}_{hoy}.csv").open("a", encoding="utf-8")
    # El nombre guarda duración y pago, para que el informe no los confunda después.
    bit_path = ROOT / "data" / "pantalla" / f"bitacora_{activo}_h{h}_p{round(pago * 100)}_{hoy}.csv"
    nuevo = not bit_path.exists()
    bit = bit_path.open("a", encoding="utf-8")
    if nuevo:
        bit.write("hora_utc,precio_entrada,precio_salida,resultado,repetir,contraria,ia,prob_ia\n")
    print(f"Duración {h}s | pago {pago:.0%} -> acierto mínimo {minimo:.2%}. Calentando {MIN_HISTORIA}s...")

    seg = 0
    fallos = 0
    t0 = time.time()
    try:
        while True:
            p = leer_precio()
            ahora = datetime.now(timezone.utc)
            if p is None or (precios and abs(p / precios[-1] - 1) > 0.01):
                # Lectura fallida: NO inventamos precios. Tras 5 fallos seguidos
                # (ventana tapada, minimizada...) se descarta el historial para que
                # los datos con huecos no contaminen al modelo ni el marcador.
                fallos += 1
                if fallos == 5:
                    precios.clear()
                    pendientes.clear()
                    print(f"{ahora:%H:%M:%S} No veo el precio. ¿Olymp está maximizado y visible? "
                          f"Pausa: retomo cuando vuelva (calentará {MIN_HISTORIA}s).")
                time.sleep(1)
                t0 = time.time() - seg   # re-sincroniza el ritmo de 1 lectura/segundo
                continue
            fallos = 0
            precios.append(p)
            tick_f.write(f"{ahora.isoformat(timespec='milliseconds')},{p}\n")
            seg += 1

            # 1) Resolver las predicciones que ya cumplieron H segundos
            while pendientes and pendientes[0][0] <= seg - h:
                s, entrada, X, preds, prob = pendientes.popleft()
                if p == entrada:
                    resultado = "empate"          # en Olymp: se devuelve el importe
                else:
                    sube = int(p > entrada)
                    resultado = "sube" if sube else "baja"
                    for nombre, pr in preds.items():
                        if pr is not None:
                            marcas[nombre].anotar(pr == sube)
                    ia.partial_fit([X], [sube], classes=[0, 1])   # la IA aprende del resultado
                    entrenado += 1
                bit.write(f"{ahora.isoformat(timespec='seconds')},{entrada},{p},{resultado},"
                          + ",".join("" if v is None else str(v) for v in preds.values())
                          + f",{'' if prob is None else round(prob, 3)}\n")

            # 2) Nueva predicción
            if len(precios) >= MIN_HISTORIA:
                arr = np.array(precios)
                X = variables(arr)
                mov = arr[-1] - arr[-1 - h]
                rep = None if mov == 0 else int(mov > 0)
                prob = ia.predict_proba([X])[0, 1] if entrenado >= MIN_APRENDIZAJE else None
                preds = {"Repetir movimiento": rep,
                         "Llevar la contraria": None if rep is None else 1 - rep,
                         "IA (aprende)": None if prob is None else int(prob > 0.5)}
                pendientes.append((seg, p, X, preds, prob))
                señal = "—" if prob is None else ("ARRIBA ↑" if prob > 0.5 else "ABAJO ↓") + f" ({max(prob, 1-prob):.0%})"
                print(f"{ahora:%H:%M:%S} precio {p}  IA: {señal}")

            # 3) Marcador cada 30 s
            if seg % 30 == 0 and seg > MIN_HISTORIA:
                print(f"\n--- Marcador (meta > {minimo:.2%}) ---")
                for nombre, m in marcas.items():
                    print(f"  {nombre:20s} {m.texto()}")
                print()
            bit.flush()
            tick_f.flush()
            time.sleep(max(0, t0 + seg - time.time()))   # ritmo de 1 lectura por segundo
    except KeyboardInterrupt:
        print("\nDetenido. Resultado final:")
        for nombre, m in marcas.items():
            print(f"  {nombre:20s} {m.texto()}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 5, float(a[1]) if len(a) > 1 else 0.85, a[2] if len(a) > 2 else "activo")
