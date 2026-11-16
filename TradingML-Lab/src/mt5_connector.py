"""Conexión de SOLO LECTURA con MetaTrader 5.

Este módulo NO envía órdenes ni usa credenciales: se conecta al terminal
MT5 ya abierto en tu PC, consulta símbolos y descarga velas históricas.

Concepto clave: en `copy_rates_from_pos`, la posición 0 es la vela ACTUAL
(aún formándose). Nosotros empezamos en la posición 1 para usar solo
velas CERRADAS y evitar usar información incompleta.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


class MT5Error(RuntimeError):
    """Error al comunicarse con MetaTrader 5."""


def connect(path: str | None = None) -> None:
    """Inicializa la conexión con el terminal MT5 (sin login ni órdenes)."""
    ok = mt5.initialize(path) if path else mt5.initialize()
    if not ok:
        raise MT5Error(f"No se pudo inicializar MT5: {mt5.last_error()}")


def disconnect() -> None:
    """Cierra la conexión con el terminal."""
    mt5.shutdown()


@contextmanager
def session(path: str | None = None):
    """Abre y cierra la conexión automáticamente, incluso si hay errores."""
    connect(path)
    try:
        yield
    finally:
        disconnect()


def terminal_summary() -> dict:
    """Datos básicos del terminal y la cuenta (solo lectura)."""
    t = mt5.terminal_info()
    a = mt5.account_info()
    if t is None:
        raise MT5Error(f"Sin info del terminal: {mt5.last_error()}")
    return {
        "terminal": t.name,
        "build": t.build,
        "connected_to_broker": t.connected,
        "company": t.company,
        "account_server": a.server if a else None,
        "account_mode": ("demo" if a.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else "real/otro") if a else None,
    }


def list_symbols(pattern: str = "*") -> list[str]:
    """Lista símbolos disponibles. `pattern` admite comodines, p. ej. '*USD*'."""
    symbols = mt5.symbols_get(pattern)
    if symbols is None:
        raise MT5Error(f"No se pudieron obtener símbolos: {mt5.last_error()}")
    return [s.name for s in symbols]


def get_candles(symbol: str, timeframe: str = "M5", n: int = 10_000) -> pd.DataFrame:
    """Descarga las últimas `n` velas CERRADAS de `symbol`.

    Devuelve DataFrame con: time (UTC), open, high, low, close,
    tick_volume, spread, real_volume.
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Temporalidad inválida: {timeframe}. Usa {list(TIMEFRAMES)}")
    if not mt5.symbol_select(symbol, True):
        raise MT5Error(f"Símbolo no disponible '{symbol}': {mt5.last_error()}")
    # start_pos=1 -> omite la vela actual (en formación)
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[timeframe], 1, n)
    if rates is None or len(rates) == 0:
        raise MT5Error(f"Sin datos para {symbol} {timeframe}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def save_csv(df: pd.DataFrame, symbol: str, timeframe: str, folder: str | Path) -> Path:
    """Guarda el DataFrame en `folder/SYMBOL_TF.csv` y devuelve la ruta."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{symbol}_{timeframe}.csv"
    df.to_csv(out, index=False)
    return out
