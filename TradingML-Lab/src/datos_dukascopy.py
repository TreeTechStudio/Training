"""Descarga velas de 1 minuto GRATIS desde Dukascopy (banco suizo) y las agrupa.

Por qué: no necesitamos MT5 ni cuenta. Son precios reales del mercado
interbancario, con años de historia: ideal para entrenar e investigar.

Uso:  python src/datos_dukascopy.py EURUSD 2025-01-01 2026-11-14 5
      (símbolo, desde, hasta, minutos por vela)

Formato del archivo: un archivo por día, comprimido LZMA, registros de 24 bytes:
segundos desde medianoche, apertura, cierre, mínimo, máximo (enteros) y volumen.
"""
import lzma
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
URL = "https://datafeed.dukascopy.com/datafeed/{s}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"


def divisor(symbol: str) -> float:
    """Los precios vienen como enteros: EURUSD 1.08123 -> 108123."""
    return 1e3 if "JPY" in symbol or symbol.startswith("XAU") else 1e5


def bajar_dia(symbol: str, dia: date) -> pd.DataFrame | None:
    """Descarga las velas de 1 minuto de un día (el mes en la URL empieza en 0)."""
    cache = ROOT / "data" / "raw" / "cache" / symbol / f"{dia}.bi5"
    if cache.exists():  # ya descargado antes: no se vuelve a pedir
        contenido = cache.read_bytes()
    else:
        url = URL.format(s=symbol, y=dia.year, m=dia.month - 1, d=dia.day)
        contenido = None
        for _ in range(3):
            try:
                r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code in (200, 404):
                    contenido = r.content if r.status_code == 200 else b""
                    break
            except requests.RequestException:
                continue
        if contenido is None:
            return None  # falló la red: se reintentará en la próxima ejecución
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(contenido)
    if not contenido:
        return None
    raw = np.frombuffer(lzma.decompress(contenido), dtype=">u4").reshape(-1, 6)
    k = divisor(symbol)
    base = pd.Timestamp(dia, tz="UTC")
    df = pd.DataFrame({
        "time": base + pd.to_timedelta(raw[:, 0].astype("int64"), unit="s"),
        "open": raw[:, 1] / k, "high": raw[:, 4] / k, "low": raw[:, 3] / k, "close": raw[:, 2] / k,
        "volume": raw[:, 5].astype(">u4").view(">f4").astype(float),
    })
    return df[df["volume"] > 0]  # minutos sin operaciones (fin de semana) fuera


def descargar(symbol: str, desde: str, hasta: str) -> pd.DataFrame:
    d0, d1 = date.fromisoformat(desde), date.fromisoformat(hasta)
    dias = [d0 + timedelta(n) for n in range((d1 - d0).days + 1)]
    dias = [d for d in dias if d.weekday() != 5]  # sábado no hay mercado
    carpeta = ROOT / "data" / "raw" / "cache" / symbol
    pendientes = dias
    for vuelta in range(1, 5):  # 1.ª descarga + reintentos de los que fallaron por red
        with ThreadPoolExecutor(max_workers=8) as ex:  # pocas conexiones: el servidor no nos frena
            for i, _ in enumerate(ex.map(lambda d: bajar_dia(symbol, d), pendientes), 1):
                if i % 50 == 0 or i == len(pendientes):
                    print(f"  vuelta {vuelta}: {i}/{len(pendientes)} días", flush=True)
        pendientes = [d for d in dias if not (carpeta / f"{d}.bi5").exists()]
        if not pendientes:
            break
    if pendientes:
        print(f"  AVISO: {len(pendientes)} días no se pudieron bajar; vuelve a ejecutar para completarlos")
    listos = [d for d in dias if (carpeta / f"{d}.bi5").exists()]
    partes = [p for p in (bajar_dia(symbol, d) for d in listos) if p is not None]  # solo caché
    return pd.concat(partes).sort_values("time").reset_index(drop=True)


def agrupar(df: pd.DataFrame, minutos: int) -> pd.DataFrame:
    """Convierte velas de 1 minuto en velas de N minutos."""
    if minutos == 1:
        return df
    g = df.set_index("time").resample(f"{minutos}min", label="left", closed="left")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(), "volume": g["volume"].sum(),
    }).dropna()
    return out.reset_index()


if __name__ == "__main__":
    a = sys.argv[1:]
    sym = a[0] if len(a) > 0 else "EURUSD"
    desde = a[1] if len(a) > 1 else "2025-01-01"
    hasta = a[2] if len(a) > 2 else str(date.today() - timedelta(1))
    tf = int(a[3]) if len(a) > 3 else 1
    m1 = descargar(sym, desde, hasta)
    m1.to_csv(ROOT / "data" / "raw" / f"{sym}_M1.csv", index=False)
    df = agrupar(m1, tf)
    out = ROOT / "data" / "raw" / f"{sym}_M{tf}.csv"
    df.to_csv(out, index=False)
    print(f"{len(m1)} velas de 1 min -> {len(df)} velas de {tf} min en {out}")
    print("Rango:", df["time"].min(), "->", df["time"].max())
    print("Chequeo: max>=min en todas:", bool((df["high"] >= df["low"]).all()))
    print(df.tail(3).to_string())
