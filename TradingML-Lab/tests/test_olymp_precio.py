"""Prueba el lector con una captura real guardada (no necesita Olymp abierto).

Uso: python tests/test_olymp_precio.py ruta_a_captura.png
"""
import asyncio
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.olymp_precio import FRANJA, _ocr, localizar_burbuja  # noqa: E402

img = cv2.imread(sys.argv[1])
f = FRANJA
crop = localizar_burbuja(img[f["top"]:f["top"] + f["height"], f["left"]:f["left"] + f["width"]])
print("Burbuja encontrada:", crop is not None)
print("Texto leído por Windows OCR:", repr(asyncio.run(_ocr(crop))))
