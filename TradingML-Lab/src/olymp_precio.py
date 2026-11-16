"""Lee el PRECIO EXACTO de Olymp Trade desde la pantalla (solo mira, nunca hace clic).

Cómo funciona:
  1. Toma una foto de la franja derecha del gráfico (donde está la escala de precios).
  2. Busca la "burbuja" blanca que marca el precio actual (p. ej. 3409.9971).
  3. La lee con el OCR (reconocimiento de texto) que ya trae Windows.
  4. Descarta lecturas absurdas (saltos enormes = error de lectura).

Uso:  python src/olymp_precio.py            -> imprime y guarda precios en data/pantalla/
      python src/olymp_precio.py prueba     -> lee una sola vez y guarda la recorte en PNG

Requisitos: app de Olymp abierta, MAXIMIZADA y visible en el monitor principal.
"""
import asyncio
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mss
import numpy as np
from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage.streams import DataWriter

ROOT = Path(__file__).resolve().parent.parent
# Franja donde Olymp dibuja la burbuja del precio (pantalla 1920x1200, app maximizada)
FRANJA = {"left": 1500, "top": 230, "width": 115, "height": 900}
MOTOR = OcrEngine.try_create_from_user_profile_languages()


def capturar(region=FRANJA) -> np.ndarray:
    with mss.MSS() as s:
        return np.array(s.grab(region))[:, :, :3]  # BGR


def localizar_burbuja(img: np.ndarray) -> np.ndarray | None:
    """Devuelve el recorte de la burbuja blanca (filas con muchos píxeles casi blancos)."""
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blancos = (gris > 225).sum(axis=1)
    filas = np.nonzero(blancos > img.shape[1] * 0.3)[0]
    if len(filas) < 8:
        return None
    # filas casi seguidas (las de los números tienen menos blanco) = la burbuja
    cortes = np.split(filas, np.nonzero(np.diff(filas) > 4)[0] + 1)
    bloque = max(cortes, key=len)
    y0, y1 = max(bloque[0] - 2, 0), bloque[-1] + 3
    cols = np.nonzero((gris[y0:y1] > 225).sum(axis=0) > 0)[0]
    return gris[y0:y1, cols[0]:cols[-1] + 1]


async def _ocr(gris: np.ndarray) -> str:
    grande = cv2.resize(gris, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    bgra = cv2.cvtColor(grande, cv2.COLOR_GRAY2BGRA)
    h, w = bgra.shape[:2]
    dw = DataWriter()
    dw.write_bytes(bgra.tobytes())
    bmp = SoftwareBitmap.create_copy_from_buffer(
        dw.detach_buffer(), BitmapPixelFormat.BGRA8, w, h, BitmapAlphaMode.PREMULTIPLIED)
    res = await MOTOR.recognize_async(bmp)
    return res.text


def leer_precio() -> float | None:
    crop = localizar_burbuja(capturar())
    if crop is None:
        return None
    texto = asyncio.run(_ocr(crop)).replace(",", ".").replace(" ", "")
    m = re.search(r"\d+\.\d+", texto)
    return float(m.group()) if m else None


def registrar(intervalo: float = 0.25) -> None:
    out = ROOT / "data" / "pantalla" / f"olymp_precio_{datetime.now():%Y%m%d}.csv"
    nuevo = not out.exists()
    previo = None
    with out.open("a", encoding="utf-8") as f:
        if nuevo:
            f.write("time_utc,precio\n")
        print("Leyendo precio... Ctrl+C para parar ->", out)
        try:
            while True:
                p = leer_precio()
                if p is not None and (previo is None or abs(p / previo - 1) < 0.01):
                    t = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                    f.write(f"{t},{p}\n")
                    f.flush()
                    flecha = "" if previo is None else ("↑" if p > previo else "↓" if p < previo else "=")
                    print(t[11:23], p, flecha)
                    previo = p
                time.sleep(intervalo)
        except KeyboardInterrupt:
            print("\nDetenido.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prueba":
        img = capturar()
        crop = localizar_burbuja(img)
        cv2.imwrite(str(ROOT / "data" / "pantalla" / "franja.png"), img)
        if crop is None:
            print("No encontré la burbuja del precio. ¿Olymp está maximizado y visible?")
        else:
            cv2.imwrite(str(ROOT / "data" / "pantalla" / "burbuja.png"), crop)
            t = time.time()
            print("Texto leído:", repr(asyncio.run(_ocr(crop))), "| precio:", leer_precio(),
                  f"| {1000 * (time.time() - t):.0f} ms")
    else:
        registrar()
