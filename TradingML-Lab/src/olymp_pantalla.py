"""Lee el gráfico de Olymp Trade desde la PANTALLA (solo mira, nunca hace clic).

Idea sencilla: el precio en el gráfico es una línea de un color. Si sabemos
qué color es y en qué zona de la pantalla está, podemos medir a qué altura
(en píxeles) está su punta derecha. Más arriba = precio subió.

Pasos:
  1) python src/olymp_pantalla.py calibrar
     - Se abre una foto de tu pantalla. Dibuja un rectángulo sobre el gráfico
       y pulsa ENTER.
     - Luego haz CLIC sobre la línea del precio (para aprender su color) y
       pulsa cualquier tecla.
  2) python src/olymp_pantalla.py registrar
     - Cada segundo mide la altura de la línea y la guarda en
       data/pantalla/olymp_FECHA.csv. Ctrl+C para parar.

Límite importante: Olymp reescala el gráfico solo. La altura en píxeles sirve
para ver si sube o baja en los últimos segundos, no como precio exacto.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "data" / "pantalla" / "config_olymp.json"
TOL = 40  # tolerancia de color (0-255)


def foto(region=None) -> np.ndarray:
    """Captura la pantalla (o una región) como imagen BGR."""
    with mss.mss() as s:
        mon = region or s.monitors[1]
        return cv2.cvtColor(np.array(s.grab(mon)), cv2.COLOR_BGRA2BGR)


def calibrar() -> None:
    img = foto()
    x, y, w, h = cv2.selectROI("1) Dibuja el rectangulo del grafico y pulsa ENTER", img, False)
    cv2.destroyAllWindows()
    zona = img[y:y + h, x:x + w].copy()
    color = {}

    def clic(ev, cx, cy, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            color["bgr"] = zona[cy, cx].tolist()
            print("Color de la línea:", color["bgr"])

    cv2.imshow("2) Haz clic sobre la linea del precio y pulsa una tecla", zona)
    cv2.setMouseCallback("2) Haz clic sobre la linea del precio y pulsa una tecla", clic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if "bgr" not in color:
        sys.exit("No hiciste clic en la línea. Repite la calibración.")
    with mss.mss() as s:  # coordenadas absolutas (monitor principal)
        m = s.monitors[1]
    cfg = {"left": m["left"] + int(x), "top": m["top"] + int(y), "width": int(w), "height": int(h),
           "bgr": color["bgr"]}
    CFG.parent.mkdir(parents=True, exist_ok=True)
    CFG.write_text(json.dumps(cfg, indent=2))
    cv2.imwrite(str(CFG.parent / "calibracion.png"), zona)
    print("Guardado", CFG)


def medir(img: np.ndarray, bgr) -> float | None:
    """Altura (0 abajo, 1 arriba) de la punta derecha de la línea del precio."""
    dif = np.abs(img.astype(int) - np.array(bgr)).max(axis=2)
    ys, xs = np.nonzero(dif < TOL)
    if len(xs) < 5:
        return None
    borde = xs >= xs.max() - 3  # las últimas columnas con línea = el precio actual
    return 1 - ys[borde].mean() / img.shape[0]


def registrar(segundos: float = 1.0) -> None:
    cfg = json.loads(CFG.read_text())
    region = {k: cfg[k] for k in ("left", "top", "width", "height")}
    out = ROOT / "data" / "pantalla" / f"olymp_{datetime.now():%Y%m%d}.csv"
    nuevo = not out.exists()
    prev = None
    with out.open("a", encoding="utf-8") as f:
        if nuevo:
            f.write("time_utc,altura\n")
        print("Registrando... Ctrl+C para parar ->", out)
        try:
            while True:
                h = medir(foto(region), cfg["bgr"])
                ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
                f.write(f"{ahora},{'' if h is None else round(h, 4)}\n")
                f.flush()
                if h is None:
                    print(ahora, "no veo la línea (¿se movió la ventana?)")
                else:
                    flecha = "=" if prev is None or abs(h - prev) < 0.002 else ("↑" if h > prev else "↓")
                    print(ahora, f"{h:.3f}", flecha)
                    prev = h
                time.sleep(segundos)
        except KeyboardInterrupt:
            print("\nDetenido.")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "registrar"
    calibrar() if modo == "calibrar" else registrar()
